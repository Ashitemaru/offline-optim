import os
import sys
import numpy as np
from tqdm import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))

from src.algorithm.greedy import greedy_policy
from src.algorithm.value_iterator import TraceValueIterator
from src.utils.data_loader import trace_loader, formated_e2e_trace_loader
from src.utils.estimator import ClientProcTimeEstimator, TimestampExtrapolator
from src.utils.action_transform import action_transform, mpc_action_transform

# Hyperparams for MC
WINDOW_SIZE = 10  # In seconds


def trace_simulator(
    file_path,
    buf_size=0,
    proc_time_estimator_mode=0,
    jitter_buffer_controller_mode=0,
    timestamp_extrapolator_mode=0,
    render_queue_mode=0,
    strict_buffer=True,
    max_delay_frame_no=1,
    frame_interval=17,
    vsync_slot_threshold=4,
    tear_protect=False,
    mode="baseline",
    qoe_coefficient=0.02,
    tearing_freq_threshold=1,
    print_log=True,
):
    sim_data_len = 60 * 60 * 60 + 2400
    data = trace_loader(file_path, start_idx=0, len_limit=sim_data_len)

    if data is None or data.shape[0] < 2000:
        return None, None

    # Only simulate 60FPS traces
    recv_interval = np.mean(data[1:, 12] - data[:-1, 12])
    if recv_interval < 13 or recv_interval > 21:
        return None, None

    # Init some averages
    try:
        valid_idx = (data[60:2400, 5:12].sum(-1) > data[59:2399, 5:12].sum(-1)) + 60
    except:
        return None, None
    avg_dec_time = np.mean(data[valid_idx, 9])
    avg_dec_total_time = np.mean(data[valid_idx, 6:10].sum(-1))
    avg_render_time = np.mean(data[valid_idx, 11])
    avg_proc_time = np.mean(data[valid_idx, 6:12].sum(-1))

    # Ignore traces with vertical-sync on
    if avg_render_time > 15:
        return None, None

    data = data[2400:, :]
    anchor_recv_ts = data[0, 5]
    data[:, 5] -= anchor_recv_ts

    # initialize it with avg_dec_total_time
    proc_time_estimator = ClientProcTimeEstimator(
        avg_dec_time=avg_dec_time,
        avg_render_time=avg_render_time,
        avg_proc_time=avg_dec_total_time,
        mode=proc_time_estimator_mode,
    )

    avg_dec_time = np.mean(data[:, 9])
    avg_dec_total_time = np.mean(data[:, 6:10].sum(-1))
    avg_render_time = np.mean(data[:, 11])
    avg_proc_time = np.mean(data[:, 6:12].sum(-1))

    nearest_display_ts = np.zeros(data.shape[0])
    nearest_display_slot = np.zeros(data.shape[0])
    expect_display_ts = np.zeros(data.shape[0])
    expect_nearest_display_ts = np.zeros(data.shape[0])
    expect_display_slot = np.zeros(data.shape[0])
    expect_nearest_display_slot = np.zeros(data.shape[0])
    expect_recv_ts = np.zeros(data.shape[0])
    expect_proc_time = np.zeros(data.shape[0])
    expect_frame_ready_ts = np.zeros(data.shape[0])
    actual_display_slot = np.zeros(data.shape[0])
    actual_display_ts = np.zeros(data.shape[0])
    invoke_present_ts = np.zeros(data.shape[0])
    cur_valid_flag = np.zeros(data.shape[0])
    # add_valid_flag = np.zeros(data.shape[0])
    # slot_delayed_flag = np.zeros(data.shape[0])
    # slot_moved_flag = np.zeros(data.shape[0])
    # minus_valid_flag = np.zeros(data.shape[0])
    tear_flag = np.zeros(data.shape[0])
    # render_discard_flag = np.zeros(data.shape[0])  # 0 for render, 1 for discard, only used when render_ctrlable
    display_discard_flag = np.zeros(data.shape[0])  # 0 for display, 1 for discard
    display_delayed_flag = np.zeros(data.shape[0])  # 0 for display, 1 for discard

    dec_over_ts = data[:, 5:10].sum(-1)
    frame_ready_ts = data[:, 5:10].sum(-1)  # from receiving to finishing decoding

    # calculate the expect display ts and slot for the first frame
    # set the display refresh slot of the 1st frame, according to the vsync ts diff
    if len(data[0]) < 26:
        display_slot_shift = 8
    else:
        display_slot_shift = data[0, 25]

    # if use optim mode, need to train the optim model
    if mode == "optim":
        for idx in range(1, data.shape[0]):
            # Preprocess copied from baseline
            if data[idx, 2] == 1:
                frame_ready_ts[idx] = max(
                    frame_ready_ts[idx - 1] + data[idx - 1, 9], frame_ready_ts[idx]
                )
        iterator = TraceValueIterator(
            frame_ready_ts - nearest_display_ts[0],
            delay_threshold=2,
            tearing_protect=False,
            top_tearing_freq_threshold=0,
            qoe_coefficient=qoe_coefficient,
        )
        iterator.fit()

        action_list = np.zeros(data.shape[0])
        lowest_free_slot_list = np.zeros(data.shape[0])

    # If use the optim-buffer mode, load the action list and transform it
    if mode == "optim-buffer":
        path = file_path[:-4].replace("data", "model/action") + "_vi_toptear_bufmix.npy"
        action_list = np.load(path)
        # path = file_path[:-4].replace("data", "model/action") + "_vi_toptear_buf1.npy"
        # action_buf1 = np.load(path)
        # path = file_path[:-4].replace("data", "model/action") + "_vi_toptear_buf2.npy"
        # action_buf2 = np.load(path)
        # action_list = mpc_action_transform(frame_ready_ts - nearest_display_ts[0], action_buf1, action_buf2, start_slot=actual_display_slot[0] + 1)
        # path = file_path[:-4] + "_baseline_toptear_buf1_sim.npy"
        # action_list = action_transform(action_buf2, frame_ready_ts - nearest_display_ts[0], tear_threshold=0)

    # UPDATE!
    # if display immediately, then its nearest vsync ts is the previous one, although it may cause screen tearing
    nearest_display_ts[0] = frame_ready_ts[0] + display_slot_shift - frame_interval
    expect_nearest_display_ts[0] = nearest_display_ts[0]
    nearest_display_slot[0] = 0
    expect_display_ts[0] = nearest_display_ts[0] + frame_interval * buf_size
    expect_display_slot[0] = nearest_display_slot[0] + buf_size
    expect_nearest_display_slot[0] = nearest_display_slot[0]
    expect_proc_time[0] = proc_time_estimator.get_proc_time_estimate()
    cur_valid_flag[0] = 1
    # slot_moved_flag[0] = 1

    recv_time_estimator = TimestampExtrapolator(
        first_frame_sts=data[0, 12],
        first_frame_recv_ts=0,
        mode=timestamp_extrapolator_mode,
    )  # expect_render_time = expect_recv_time + predicted_proc_time

    # after initialization, display the first frame as soon as possible
    # UPDATE! For the first frame, if tear protect is ON and its ready ts is less than (frame_interval-vsync_slot_threshold) ms away from the next vsync ts,
    # then delay it to the next vsync ts to avoid screen tearing.
    if tear_protect and display_slot_shift < frame_interval - vsync_slot_threshold:
        invoke_present_ts[0] = nearest_display_ts[0] + frame_interval
    else:
        invoke_present_ts[0] = frame_ready_ts[0]

    # UPDATE! (for calculation, to ensure the actual display slot is calculated correctly when render time = 0)
    actual_display_ts[0] = invoke_present_ts[0] + max(data[0, 11], 1)
    actual_display_slot[0] = (
        np.ceil(
            max(0, invoke_present_ts[0] + 1 - nearest_display_ts[0]) / frame_interval
        )
        - 1
    )

    def naive_mc_controller(idx):
        # I. GMM from data array

        # II. MC from the simulated distribution

        # III. Get the final action

        pass

    def vsync_controller(idx):
        lowest_free_slot = actual_display_slot[idx - 1] + 1
        slot = np.ceil((frame_ready_ts[idx] - nearest_display_ts[0]) / 17) - 1
        relative_ts = (
            frame_ready_ts[idx]
            - nearest_display_ts[0]
            - frame_interval * lowest_free_slot
        )
        action = greedy_policy(
            relative_ts=relative_ts, index=0, avg_delay=0, max_buffer=2
        )

        if action == -1:
            actual_display_slot[idx] = actual_display_slot[idx - 1]
            invoke_present_ts[idx] = frame_ready_ts[idx]
            data[idx, 11] = -1
        else:
            if action > 0:
                next_display_slot = max(lowest_free_slot, slot + action)
                invoke_present_ts[idx] = (
                    frame_interval * next_display_slot + nearest_display_ts[0]
                )
            else:
                invoke_present_ts[idx] = frame_ready_ts[idx]
            actual_display_ts[idx] = invoke_present_ts[idx] + max(data[idx, 11], 1)
            actual_display_slot[idx] = (
                np.ceil(
                    max(0, invoke_present_ts[idx] + 1 - nearest_display_ts[0])
                    / frame_interval
                )
                - 1
            )

    def offline_optim_controller_with_buffer_released(idx):
        lowest_free_slot = actual_display_slot[idx - 1] + 1
        slot = (
            np.ceil((frame_ready_ts[idx] - nearest_display_ts[0]) / frame_interval) - 1
        )
        action = action_list[idx]

        if action == -1:
            actual_display_slot[idx] = actual_display_slot[idx - 1]
            invoke_present_ts[idx] = frame_ready_ts[idx]
            data[idx, 11] = -1
        else:
            if action > 0:
                next_display_slot = slot + action
                invoke_present_ts[idx] = (
                    frame_interval * next_display_slot + nearest_display_ts[0]
                )
            else:
                invoke_present_ts[idx] = frame_ready_ts[idx]

            # Filter out all the invalid action (like when the frame is 2 frames earlier than the LFS but the action is only 1)
            # "optim" mode should have already filtered out all the invalid actions but "optim-buffer" mode may not
            # TODO: transforming action sequences without the simulator may lead to this messy situation :(
            if (
                invoke_present_ts[idx]
                < lowest_free_slot * frame_interval + nearest_display_ts[0]
            ):
                actual_display_slot[idx] = actual_display_slot[idx - 1]
                invoke_present_ts[idx] = frame_ready_ts[idx]
                data[idx, 11] = -1
            else:
                actual_display_ts[idx] = invoke_present_ts[idx] + max(data[idx, 11], 1)
                actual_display_slot[idx] = (
                    np.ceil(
                        max(0, invoke_present_ts[idx] + 1 - nearest_display_ts[0])
                        / frame_interval
                    )
                    - 1
                )

    def offline_optim_controller(idx):
        lowest_free_slot = actual_display_slot[idx - 1] + 1
        slot = (
            np.ceil((frame_ready_ts[idx] - nearest_display_ts[0]) / frame_interval) - 1
        )
        action = iterator.get_policy(
            idx,
            frame_ready_ts[idx]
            - nearest_display_ts[0]
            - lowest_free_slot * frame_interval,
        )

        action_list[idx] = action
        lowest_free_slot_list[idx] = lowest_free_slot

        if action == -1:
            actual_display_slot[idx] = actual_display_slot[idx - 1]
            invoke_present_ts[idx] = frame_ready_ts[idx]
            data[idx, 11] = -1
        else:
            if action > 0:
                next_display_slot = max(lowest_free_slot, slot + action)
                invoke_present_ts[idx] = (
                    frame_interval * next_display_slot + nearest_display_ts[0]
                )
            else:
                invoke_present_ts[idx] = frame_ready_ts[idx]
            actual_display_ts[idx] = invoke_present_ts[idx] + max(data[idx, 11], 1)
            actual_display_slot[idx] = (
                np.ceil(
                    max(0, invoke_present_ts[idx] + 1 - nearest_display_ts[0])
                    / frame_interval
                )
                - 1
            )

        # re-calculate the render queueing time based on the timestamp that invokes present api
        assert invoke_present_ts[idx] >= frame_ready_ts[idx], "%d %d %d" % (
            " ".join([str(item) for item in data[idx, :]]),
            idx,
            invoke_present_ts[idx],
            frame_ready_ts[idx],
        )
        data[idx, 10] = invoke_present_ts[idx] - frame_ready_ts[idx]

        assert (
            actual_display_slot[idx] >= actual_display_slot[idx - 1]
        ), "%d %d %d %d %d" % (
            idx,
            nearest_display_slot[idx],
            expect_display_slot[idx],
            actual_display_slot[idx],
            actual_display_slot[idx - 1],
        )
        assert (
            actual_display_slot[idx] >= nearest_display_slot[idx]
        ), "%d %d %d %d %d" % (
            idx,
            nearest_display_slot[idx],
            expect_display_slot[idx],
            actual_display_slot[idx],
            actual_display_slot[idx - 1],
        )

    # display control
    # TODO: must be implemented in the client side SDK
    def frame_display_controller(idx, buf_size=0):
        if data[idx, 2] == 1:
            frame_ready_ts[idx] = max(
                frame_ready_ts[idx - 1] + data[idx - 1, 9], frame_ready_ts[idx]
            )

        # UPDATE! nearest_display_slot starting from 1 now
        nearest_display_slot[idx] = (
            np.ceil(
                max(frame_ready_ts[idx] - nearest_display_ts[0], 0) / frame_interval
            )
            - 1
        )
        nearest_display_ts[idx] = (
            nearest_display_slot[idx] * frame_interval + nearest_display_ts[0]
        )

        # can be calculated on the proxy in advance, no need to be calcuated in the client SDK
        expect_recv_ts[idx] = recv_time_estimator.extrapolate_local_time(data[idx, 12])
        expect_proc_time[idx] = proc_time_estimator.get_proc_time_estimate()
        expect_frame_ready_ts[idx] = expect_recv_ts[idx] + expect_proc_time[idx]

        # UPDATE!
        expect_nearest_display_slot[idx] = (
            np.ceil(
                max(expect_frame_ready_ts[idx] - nearest_display_ts[0], 0)
                / frame_interval
            )
            - 1
        )
        expect_nearest_display_ts[idx] = (
            expect_nearest_display_slot[idx] * frame_interval + nearest_display_ts[0]
        )
        expect_display_slot[idx] = expect_nearest_display_slot[idx] + buf_size
        expect_display_ts[idx] = (
            expect_nearest_display_ts[idx] + buf_size * frame_interval
        )

        if strict_buffer:
            expect_display_slot[idx] = min(
                nearest_display_slot[idx] + max_delay_frame_no,
                expect_display_slot[idx],
            )
            expect_display_ts[idx] = min(
                nearest_display_ts[idx] + max_delay_frame_no * frame_interval,
                expect_display_ts[idx],
            )

        # display queue with 1 frame buffer
        # UPDATE!
        if (
            data[idx, 0] < data[idx - 1, 0]
            or (
                idx > 1
                and max(frame_ready_ts[idx - 1], expect_display_ts[idx - 1])
                < actual_display_ts[idx - 2]
                and max(frame_ready_ts[idx], expect_display_ts[idx])
                < actual_display_ts[idx - 2]
            )
            or (
                tear_protect
                and frame_interval < 17
                and actual_display_ts[idx - 2] >= expect_display_ts[idx - 2]
                and actual_display_ts[idx - 2] < expect_display_ts[idx - 2] + 17
                and expect_display_ts[idx - 1] < expect_display_ts[idx - 2] + 17
                and max(actual_display_ts[idx - 2], frame_ready_ts[idx - 1])
                - expect_display_ts[idx - 2]
                > vsync_slot_threshold
            )
        ):
            display_discard_flag[idx - 1] = 1
            frame_ready_ts[idx - 1] = frame_ready_ts[idx - 2]
            nearest_display_ts[idx - 1] = nearest_display_ts[idx - 2]
            nearest_display_slot[idx - 1] = nearest_display_slot[idx - 2]
            expect_display_ts[idx - 1] = expect_display_ts[idx - 2]
            actual_display_slot[idx - 1] = actual_display_slot[idx - 2]
            actual_display_ts[idx - 1] = actual_display_ts[idx - 2]
            data[idx - 1, 10] = -1
            data[idx - 1, 11] = -1

        # UPDATE!
        if (
            actual_display_slot[idx - 1] >= nearest_display_slot[idx]
            and actual_display_slot[idx - 1] < expect_display_slot[idx]
        ):
            next_display_slot = actual_display_slot[idx - 1] + 1
            invoke_present_ts[idx] = (
                frame_interval * next_display_slot + nearest_display_ts[0]
            )
            display_delayed_flag[idx] = 1
        elif (
            tear_protect
            and nearest_display_slot[idx] < expect_display_slot[idx]
            and max(actual_display_ts[idx - 1], frame_ready_ts[idx])
            - nearest_display_ts[idx]
            > vsync_slot_threshold
            and max(actual_display_ts[idx - 1], frame_ready_ts[idx])
            - nearest_display_ts[idx]
            < frame_interval
        ):
            next_display_slot = max(
                actual_display_slot[idx - 1] + 1, nearest_display_slot[idx] + 1
            )
            invoke_present_ts[idx] = (
                frame_interval * next_display_slot + nearest_display_ts[0]
            )
            display_delayed_flag[idx] = 1
        else:
            # current strategy may cause screen tearing
            invoke_present_ts[idx] = max(
                actual_display_ts[idx - 1] + 1, frame_ready_ts[idx]
            )
            display_delayed_flag[idx] = 0

        actual_display_ts[idx] = invoke_present_ts[idx] + max(data[idx, 11], 1)
        actual_display_slot[idx] = (
            np.ceil(
                max(0, invoke_present_ts[idx] + 1 - nearest_display_ts[0])
                / frame_interval
            )
            - 1
        )

        display_ts_residual = (
            actual_display_ts[idx] - nearest_display_ts[idx]
        ) % frame_interval
        if (
            display_ts_residual > vsync_slot_threshold
            and display_ts_residual < frame_interval - vsync_slot_threshold
        ):
            tear_flag[idx] = 1

        # re-calculate the render queueing time based on the timestamp that invokes present api
        assert invoke_present_ts[idx] >= frame_ready_ts[idx], "%s %d %d %d" % (
            " ".join([str(item) for item in data[idx, :]]),
            idx,
            invoke_present_ts[idx],
            frame_ready_ts[idx],
        )
        data[idx, 10] = invoke_present_ts[idx] - frame_ready_ts[idx]

        assert (
            actual_display_slot[idx] >= actual_display_slot[idx - 1]
        ), "%d %d %d %d %d" % (
            idx,
            nearest_display_slot[idx],
            expect_display_slot[idx],
            actual_display_slot[idx],
            actual_display_slot[idx - 1],
        )
        assert (
            actual_display_slot[idx] >= nearest_display_slot[idx]
        ), "%d %d %d %d %d" % (
            idx,
            nearest_display_slot[idx],
            expect_display_slot[idx],
            actual_display_slot[idx],
            actual_display_slot[idx - 1],
        )

    # current log discard all queueing frames, need to repair the render time for the discarded frames
    for idx in range(1, data.shape[0]):
        if data[idx, 5:12].sum() < data[idx - 1, 5:12].sum():
            data[idx, 11] = avg_render_time

    # Start the simulation
    for idx in range(1, data.shape[0]):
        if mode == "baseline":
            frame_display_controller(idx, buf_size=buf_size)
        elif mode == "optim":
            offline_optim_controller(idx)
        elif mode == "optim-buffer":
            offline_optim_controller_with_buffer_released(idx)
        elif mode == "vsync":
            vsync_controller(idx)
        elif mode == "mc":
            naive_mc_controller(idx)
        else:
            raise ValueError

        if idx >= 3 and mode == "baseline":
            recv_time_estimator.update(data[idx - 2, 12], data[idx - 2, 5])
            proc_time_estimator.update(
                data[idx, 9], data[idx, 11], data[idx, 6:10].sum()
            )

        if actual_display_slot[idx] != actual_display_slot[idx - 1]:
            cur_valid_flag[idx] = 1

    # not very accurate in no render_ctrl mode, as the time consumed to invoke present is not taken into account,
    # which may make the min fps even lower
    # valid_no = np.unique(nearest_display_slot).size
    valid_no = np.unique(
        np.ceil((frame_ready_ts - display_slot_shift) / frame_interval)
    ).size
    tot_time = data[-1, 5] / 1000
    max_fps = data.shape[0] / tot_time
    min_fps = valid_no / tot_time

    data[-1, 11] = -1
    valid_idx = np.where(data[:, 11] != -1)
    opt_valid_no = np.unique(actual_display_slot[valid_idx]).size
    opt_fps = opt_valid_no / tot_time

    result = [
        max_fps,  # FPS upper limit
        min_fps,  # FPS lower limit
        opt_fps,  # optimized fps
        # *jitter_buffer.get_buffer_gain(tot_time=tot_time)[1:],
        # np.sum(one_more_buf_valid_flag) / tot_time, # real fps with 1 more buffer
        # np.sum(one_less_buf_valid_flag) / tot_time, # real fps with 1 less buffer
        np.mean(
            invoke_present_ts[valid_idx] - frame_ready_ts[valid_idx]
        ),  # current overhead
        # np.mean(one_more_invoke_present_ts[valid_idx] - frame_ready_ts[valid_idx]),  # overhead with 1 more buffer
        # np.mean(one_less_invoke_present_ts[valid_idx] - frame_ready_ts[valid_idx]),  # overhead with 1 less buffer
        np.sum(tear_flag) / tot_time,  # teared frame per second
        avg_dec_time,
        avg_dec_total_time,
        avg_render_time,
        avg_proc_time,
        recv_interval,
        np.sum(cur_valid_flag) / data.shape[0],  # valid frame ratio
    ]

    return file_path, result


ROOT = "../data"

if __name__ == "__main__":
    for qoe_coefficient in [0.002, 0.004, 0.006, 0.008]:
        final_log_file = open(
            f"../log/VI-alltear-buf2-normal-qoe{qoe_coefficient}.log", "w"
        )
        for directory in os.listdir(ROOT):
            if not directory.startswith("session"):
                continue

            for file_name in tqdm(os.listdir(os.path.join(ROOT, directory))):
                if not file_name.endswith(".csv"):
                    continue
                path = os.path.join(ROOT, directory, file_name)

                result = trace_simulator(
                    path,
                    buf_size=1,
                    proc_time_estimator_mode=1,
                    jitter_buffer_controller_mode=1,
                    timestamp_extrapolator_mode=1,
                    render_queue_mode=1,
                    strict_buffer=True,
                    max_delay_frame_no=2,
                    frame_interval=17,
                    mode="optim",
                    print_log=False,
                    qoe_coefficient=qoe_coefficient,
                )
                if result[0] is None:
                    continue
                final_log_file.write(
                    "%s\t%f\t%f\t%f\t%f\n"
                    % (
                        result[0].replace("\\", "/").replace("../data/", ""),
                        result[1][2],
                        result[1][3],
                        result[1][4],
                        result[1][-1],
                    )
                )
                final_log_file.flush()

        final_log_file.close()
    exit(0)
    final_log_file = open("../log/VI-tearing-addition-buffer2.log", "w")
    for directory in os.listdir("../data"):
        if not directory.startswith("session"):
            continue

        for file_name in tqdm(os.listdir(os.path.join("../data", directory))):
            if "sim" in file_name or "ipynb" in file_name:
                continue
            path = os.path.join("../data", directory, file_name)

            full_tear_result = trace_simulator(
                path,
                buf_size=1,
                proc_time_estimator_mode=1,
                jitter_buffer_controller_mode=1,
                timestamp_extrapolator_mode=1,
                render_queue_mode=1,
                strict_buffer=True,
                max_delay_frame_no=2,
                frame_interval=17,
                mode="optim",
                tear_protect=True,
                tearing_freq_threshold=1,
                print_log=True,
            )

            if full_tear_result[0] is None:
                continue

            result = []
            zero_tear_result = trace_simulator(
                path,
                buf_size=1,
                proc_time_estimator_mode=1,
                jitter_buffer_controller_mode=1,
                timestamp_extrapolator_mode=1,
                render_queue_mode=1,
                strict_buffer=True,
                max_delay_frame_no=2,
                frame_interval=17,
                mode="optim",
                tear_protect=True,
                tearing_freq_threshold=0,
                print_log=True,
            )
            result.append(full_tear_result[1][2])
            result.append(full_tear_result[1][3])
            result.append(zero_tear_result[1][2])
            result.append(zero_tear_result[1][3])

            final_log_file.write(
                "%s\t%f\t%f\t%f\t%f\n"
                % (
                    path.replace("\\", "/").replace("../data/", ""),
                    *result,
                )
            )
            final_log_file.flush()

    final_log_file.close()
