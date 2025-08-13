import os, sys
import multiprocessing
import collections

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, "../.."))

import src.ref.load_data as load_data
import numpy as np

proc_time_estimator_mode = 1
jitter_buffer_controller_mode = 1
timestamp_extrapolator_mode = 1
render_queue_mode = 1
strict_buffer = True
render_ctrlable = False
tear_protect = True
buf_size = 0
buf_size = (int)(
    tear_protect + buf_size
)  # UPDATE! tear-protect functions similarly as adding 1 buffer
max_delay_frame_no = max(1, buf_size * 2)
vsync_slot_threshold = 4
frame_interval = 17

min_buf_size = 0
max_buf_size = 2
buf_size_update_interval = 5
target_fps = 55
min_fps_gain = 2

print_log = True

load_data_func = load_data.load_formated_e2e_framerate_log


# load_data_func = load_data.load_formated_e2e_log
class Result:
    def __init__(self):
        self.res1 = []
        self.res2 = []
        self.res3 = []

    def update_result(self, result):
        log_path = result[0]
        cur_res = result[1]
        if cur_res is not None:
            self.res1.append(cur_res[2])
            self.res2.append(cur_res[4])
            self.res3.append(cur_res[5])

            print(log_path, cur_res)


# proxy side function, can be implemented in the client SDK
class TimestampExtrapolator:
    def __init__(
        self, first_frame_sts, first_frame_recv_ts, mode=0
    ):  # 0 for naive mode, 1 for Kalman filter
        self.first_frame_sts = first_frame_sts
        self.first_frame_recv_ts = first_frame_recv_ts

        self.mode = mode

        if self.mode == 1:
            self.w = [1, 0]
            self.p = [[1, 0], [0, 1e5]]

    def update(self, frame_sts, frame_recv_ts):
        if self.mode == 0:
            pass
        elif self.mode == 1:
            recv_time_diff = frame_recv_ts - self.first_frame_recv_ts
            residual = (
                frame_sts
                - self.first_frame_sts
                - recv_time_diff * self.w[0]
                - self.w[1]
            )

            k = [0, 0]
            k[0] = self.p[0][0] * recv_time_diff + self.p[0][1]
            k[1] = self.p[1][0] * recv_time_diff + self.p[1][1]

            kLambda = 1
            tpt = kLambda + recv_time_diff * k[0] + k[1]
            k[0] /= tpt
            k[1] /= tpt

            self.w[0] = self.w[0] + k[0] * residual
            self.w[1] = self.w[1] + k[1] * residual

            p00 = (
                1
                / kLambda
                * (
                    self.p[0][0]
                    - (k[0] * recv_time_diff * self.p[0][0] + k[0] * self.p[1][0])
                )
            )
            p01 = (
                1
                / kLambda
                * (
                    self.p[0][1]
                    - (k[0] * recv_time_diff * self.p[0][1] + k[0] * self.p[1][1])
                )
            )
            self.p[1][0] = (
                1
                / kLambda
                * (
                    self.p[1][0]
                    - (k[1] * recv_time_diff * self.p[0][0] + k[1] * self.p[1][0])
                )
            )
            self.p[1][1] = (
                1
                / kLambda
                * (
                    self.p[1][1]
                    - (k[1] * recv_time_diff * self.p[0][1] + k[1] * self.p[1][1])
                )
            )
            self.p[0][0] = p00
            self.p[0][1] = p01
        else:
            raise NotImplementedError

    def reset(self, first_frame_sts, first_frame_recv_ts):
        if self.mode == 0:
            pass
        elif self.mode == 1:
            self.w = [1, 0]
            self.first_frame_sts = first_frame_sts
            self.first_frame_recv_ts = first_frame_recv_ts
        else:
            raise NotImplementedError

    def extrapolate_local_time(self, frame_sts):
        if self.mode == 0:
            return self.first_frame_recv_ts + (frame_sts - self.first_frame_sts)
        elif self.mode == 1:
            sts_diff = frame_sts - self.first_frame_sts
            return (
                self.first_frame_recv_ts + (sts_diff - self.w[1]) / self.w[0]
            )  # + 0.5
        else:
            raise NotImplementedError


# proxy side function, can be implemented in the client SDK
class JitterBufferController:
    def __init__(
        self, mode=0, frame_interval=17, *args
    ):  # 0 for Windows full control, 1 for Android fuzzy control
        self.mode = mode
        self.frame_interval = frame_interval
        if self.mode == 0:
            self.latest_frame_overdue_time = collections.deque(
                [], 60 * 60 * 20
            )  # update buffer size every 10 seconds
        elif self.mode == 1:
            self.add_valid_frame_no = collections.deque([], 60 * 60 * 20)
            self.minus_valid_frame_no = collections.deque([], 60 * 60 * 20)
            self.first_frame_display_ts = args[0]

            self.valid_frame_no = 1
            self.slot_delayed = 0
            self.slot_moved = 1
            self.prev_display_ts = 0

    def reset(self):
        if self.mode == 0:
            self.latest_frame_overdue_time = collections.deque(
                [], 60 * 60 * 20
            )  # update buffer size every 10 seconds
        elif self.mode == 1:
            self.add_valid_frame_no = collections.deque([], 60 * 60 * 20)
            self.minus_valid_frame_no = collections.deque([], 60 * 60 * 20)
            self.valid_frame_no = 0
            self.slot_delayed = 0
            self.slot_moved = 1
            self.prev_display_ts = 0

    def update(self, *args):
        if self.mode == 0:
            overdue_time = args[0]
            self.latest_frame_overdue_time.append(overdue_time)
        elif self.mode == 1:
            prev_frame_ready_ts = args[0]
            prev_expected_display_ts = args[1]
            prev_actual_display_ts = args[2]
            prev_display_slot = args[3]
            cur_frame_ready_ts = args[4]
            cur_expected_display_ts = args[5]
            cur_actual_display_ts = args[6]
            cur_display_slot = args[7]
            vsync_slot_threshold = args[8]

            if cur_display_slot != prev_display_slot:
                self.valid_frame_no += 1

            if cur_display_slot >= prev_display_slot + self.slot_delayed + 1:
                self.slot_delayed = 0

            if (
                prev_display_slot == cur_display_slot
                and (
                    prev_frame_ready_ts > cur_expected_display_ts
                    or prev_expected_display_ts == cur_expected_display_ts
                )
                and cur_frame_ready_ts < cur_expected_display_ts + frame_interval
            ):
                if (
                    self.slot_delayed > 0
                    and prev_actual_display_ts >= cur_expected_display_ts
                ):
                    self.add_valid_frame_no.append(0)
                else:
                    self.add_valid_frame_no.append(1)
                    self.slot_delayed = 1
            else:
                self.add_valid_frame_no.append(0)

            if (
                cur_display_slot == prev_display_slot + 1
                and cur_frame_ready_ts < cur_expected_display_ts
                and cur_actual_display_ts > cur_expected_display_ts
                and (
                    self.slot_moved == 0
                    or prev_frame_ready_ts >= cur_expected_display_ts - frame_interval
                )
            ):
                self.minus_valid_frame_no.append(1)
                self.slot_moved = 1
            else:
                self.minus_valid_frame_no.append(0)

                if cur_frame_ready_ts >= cur_expected_display_ts or (
                    self.slot_moved == 0
                    and cur_frame_ready_ts < cur_expected_display_ts - frame_interval
                ):
                    self.slot_moved = 0
                else:
                    self.slot_moved = 1

            # cur_nearest_slot_ts = np.ceil(
            #     max(cur_frame_ready_ts - self.first_frame_display_ts, 0)/frame_interval
            # ) * frame_interval - frame_interval + self.first_frame_display_ts
            # if cur_display_slot == prev_display_slot + 1:
            #     if prev_frame_ready_ts > prev_expected_display_ts and \
            #         prev_frame_ready_ts > cur_expected_display_ts - frame_interval and \
            #         cur_frame_ready_ts < cur_expected_display_ts:
            #         self.minus_valid_frame_no.append(1)
            #         self.prev_display_ts = prev_frame_ready_ts + 1
            #     elif prev_expected_display_ts == cur_expected_display_ts and \
            #         (prev_frame_ready_ts >= cur_expected_display_ts - frame_interval or \
            #         (prev_frame_ready_ts < cur_expected_display_ts - frame_interval and \
            #         self.prev_display_ts > cur_expected_display_ts - 2*frame_interval)) and \
            #         prev_actual_display_ts < cur_expected_display_ts and \
            #         cur_actual_display_ts > cur_expected_display_ts and \
            #         cur_actual_display_ts < cur_expected_display_ts + frame_interval and \
            #         cur_frame_ready_ts < cur_expected_display_ts:
            #         self.minus_valid_frame_no.append(1)

            #         if prev_frame_ready_ts < cur_expected_display_ts - frame_interval and \
            #         self.prev_display_ts > cur_expected_display_ts - 2*frame_interval:
            #             self.prev_display_ts = max(prev_frame_ready_ts + 1, cur_expected_display_ts - frame_interval)
            #         else:
            #             self.prev_display_ts = prev_frame_ready_ts + 1
            #     else:
            #         self.minus_valid_frame_no.append(0)
            #         self.prev_display_ts = prev_frame_ready_ts + 1
            # else:
            #     self.minus_valid_frame_no.append(0)
            #     self.prev_display_ts = prev_frame_ready_ts + 1

    def get_buffer_gain(self, tot_time=0, max_buf_size=3, max_fps=60):
        if self.mode == 0:
            res = []
            for i in range(max_buf_size + 1):
                res.append(
                    np.sum(
                        np.array(self.latest_frame_overdue_time)
                        <= self.frame_interval * i
                    )
                    / len(self.latest_frame_overdue_time)
                    * max_fps
                )
        elif self.mode == 1:
            add_fps_no = sum(self.add_valid_frame_no)
            minus_fps_no = sum(self.minus_valid_frame_no)
            res = [
                self.valid_frame_no / tot_time,
                (self.valid_frame_no + add_fps_no) / tot_time,
                (self.valid_frame_no - minus_fps_no) / tot_time,
            ]

        return res

    def get_latest_flag(self):
        if self.mode == 0:
            return self.latest_frame_overdue_time[-1]
        elif self.mode == 1:
            return (
                self.add_valid_frame_no[-1],
                self.minus_valid_frame_no[-1],
                self.slot_delayed,
                self.slot_moved,
            )
            # self.minus_valid_frame_no = collections.deque([], 60 * 60 * 20)


# proxy side function, can be implemented in the client SDK
class ClientProcTimeEstimator:
    def __init__(
        self, avg_dec_time, avg_render_time, avg_proc_time, mode=0
    ):  # 0 for EWMA, 1 for Kalman filter
        self.avg_dec_time = avg_dec_time
        self.avg_render_time = avg_render_time
        self.avg_proc_time = avg_proc_time

        self.mode = mode

        if self.mode == 0:
            self.ewma_alpha = 0.05
        elif self.mode == 1:
            self.w = [1, 0]
            self.p = [[1, 0], [0, 1e5]]

    def update(self, dec_time, render_time, proc_time):
        if self.mode == 0:
            self.avg_dec_time = (
                self.ewma_alpha * dec_time + (1 - self.ewma_alpha) * self.avg_dec_time
            )
            self.avg_render_time = (
                self.ewma_alpha * render_time
                + (1 - self.ewma_alpha) * self.avg_render_time
            )
            self.avg_proc_time = (
                self.ewma_alpha * proc_time + (1 - self.ewma_alpha) * self.avg_proc_time
            )
        elif self.mode == 1:
            proc_time_diff = proc_time - self.avg_proc_time
            residual = -proc_time_diff * self.w[0] - self.w[1]

            k = [0, 0]
            k[0] = self.p[0][0] * proc_time_diff + self.p[0][1]
            k[1] = self.p[1][0] * proc_time_diff + self.p[1][1]

            kLambda = 1
            tpt = kLambda + proc_time_diff * k[0] + k[1]
            k[0] /= tpt
            k[1] /= tpt

            self.w[0] = self.w[0] + k[0] * residual
            self.w[1] = self.w[1] + k[1] * residual

            p00 = (
                1
                / kLambda
                * (
                    self.p[0][0]
                    - (k[0] * proc_time_diff * self.p[0][0] + k[0] * self.p[1][0])
                )
            )
            p01 = (
                1
                / kLambda
                * (
                    self.p[0][1]
                    - (k[0] * proc_time_diff * self.p[0][1] + k[0] * self.p[1][1])
                )
            )
            self.p[1][0] = (
                1
                / kLambda
                * (
                    self.p[1][0]
                    - (k[1] * proc_time_diff * self.p[0][0] + k[1] * self.p[1][0])
                )
            )
            self.p[1][1] = (
                1
                / kLambda
                * (
                    self.p[1][1]
                    - (k[1] * proc_time_diff * self.p[0][1] + k[1] * self.p[1][1])
                )
            )
            self.p[0][0] = p00
            self.p[0][1] = p01
        else:
            raise NotImplementedError

    def get_proc_time_estimate(self):
        if self.mode == 0:
            return self.avg_proc_time
        elif self.mode == 1:
            return self.avg_proc_time - self.w[1] / self.w[0]  # + 0.5


def cal_valid_frame(
    file_path,
    buf_size=0,
    proc_time_estimator_mode=0,
    jitter_buffer_controller_mode=0,
    timestamp_extrapolator_mode=0,
    render_queue_mode=0,
    strict_buffer=True,
    render_ctrlable=False,
    max_delay_frame_no=1,
    frame_interval=17,
):
    """
    Simulate the display process with a frame trace and a single set of parameters.
    :param buf_size: maximum No. of display slots that one frame can be delayed
    :param proc_time_estimator_mode: methods used to predict the expected proc_ts
    :param jitter_buffer_controller_mode: methods used to calculate the FPS gain
    :param timestamp_extrapolator_mode: methods used to predict the expected recv_ts
    :param render_queue_mode: only useful when render/present is controllable, 0 for no buffer, 1 for 1 frame buffer
    :param strict_buffer: if True, one frame can be delayed at most buf_size display slot, otherwise it can be delayed to the expected display slot
    :param render_ctrlable: if True, the timestamp that SDK invokes the present function is the ts that a frame is displayed. Current render is not ctrlable, thus this variable is False
    """
    sim_data_len = 60 * 60 * 60 + 2400
    data, info = load_data_func(
        file_path, start_idx=0, len_limit=sim_data_len
    )  # sim for 20min

    if data is None or data.shape[0] < 2000:
        return None, None

    # only simulate 60FPS traces
    recv_interval = np.mean(data[1:, 12] - data[:-1, 12])
    if recv_interval < 13 or recv_interval > 21:
        return None, None

    # initialization: calculate the average decode and render time
    def init_controller(samp_len):
        valid_idx = (
            data[60:samp_len, 5:12].sum(-1) > data[59 : samp_len - 1, 5:12].sum(-1)
        ) + 60
        avg_dec_time = np.mean(data[valid_idx, 9])
        avg_dec_total_time = np.mean(data[valid_idx, 6:10].sum(-1))
        avg_render_time = np.mean(data[valid_idx, 11])
        avg_proc_time = np.mean(data[valid_idx, 6:12].sum(-1))

        return avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time

    samp_len = 2400
    avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time = init_controller(
        samp_len
    )
    if avg_render_time > 15:  # ignore traces with vertical-sync on
        return None, None

    data = data[samp_len:, :]
    anchor_recv_ts = data[0, 5]
    data[:, 5] -= anchor_recv_ts

    # if render_ctrlable, initialize the ClientProcTimeEstimator with avg_proc_time
    # otherwise, initialize it with avg_dec_total_time
    if render_ctrlable:
        proc_time_estimator = ClientProcTimeEstimator(
            avg_dec_time=avg_dec_time,
            avg_render_time=avg_render_time,
            avg_proc_time=avg_proc_time,
            mode=proc_time_estimator_mode,
        )
    else:
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
    tear_flag = np.zeros(data.shape[0])

    render_discard_flag = np.zeros(
        data.shape[0]
    )  # 0 for render, 1 for discard, only used when render_ctrlable
    display_discard_flag = np.zeros(data.shape[0])  # 0 for display, 1 for discard
    display_delayed_flag = np.zeros(data.shape[0])  # 0 for display, 1 for discard

    dec_over_ts = data[:, 5:10].sum(-1)
    if render_ctrlable:
        frame_ready_ts = data[:, 5:12].sum(-1)  # from receiving to finishing rendering
    else:
        frame_ready_ts = data[:, 5:10].sum(-1)  # from receiving to finishing decoding

    # calculate the expect display ts and slot for the first frame
    # set the display refresh slot of the 1st frame, according to the vsync ts diff
    if len(data[0]) < 26:
        display_slot_shift = 8
    else:
        display_slot_shift = data[0, 25]
    # if display immediately, then its nearest vsync ts is the previous one, although it may cause screen tearing
    nearest_display_ts[0] = frame_ready_ts[0] + display_slot_shift - frame_interval
    expect_nearest_display_ts[0] = nearest_display_ts[0]
    nearest_display_slot[0] = 0
    expect_display_ts[0] = nearest_display_ts[0] + frame_interval * buf_size
    expect_display_slot[0] = nearest_display_slot[0] + buf_size
    expect_nearest_display_slot[0] = nearest_display_slot[0]
    expect_proc_time[0] = proc_time_estimator.get_proc_time_estimate()
    cur_valid_flag[0] = 1

    recv_time_estimator = TimestampExtrapolator(
        first_frame_sts=data[0, 12],
        first_frame_recv_ts=0,
        mode=timestamp_extrapolator_mode,
    )  # expect_render_time = expect_recv_time + predicted_proc_time

    # after initialization, display the first frame as soon as possible
    if render_ctrlable:
        actual_display_slot[0] = nearest_display_slot[0]
    else:
        if tear_protect and display_slot_shift < frame_interval - vsync_slot_threshold:
            invoke_present_ts[0] = nearest_display_ts[0] + frame_interval
        else:
            invoke_present_ts[0] = frame_ready_ts[0]
        actual_display_ts[0] = invoke_present_ts[0] + max(data[0, 11], 1)
        actual_display_slot[0] = (
            np.ceil(
                max(0, invoke_present_ts[0] + 1 - nearest_display_ts[0])
                / frame_interval
            )
            - 1
        )

    # old simulation, current render is not ctrlable
    def render_queue_controller(idx, mode=0):  # 0 for no buffer, 1 for 1 buffer
        if mode == 0 and dec_over_ts[idx] < frame_ready_ts[idx - 1]:
            render_discard_flag[idx] = 1
        elif (
            mode == 1
            and dec_over_ts[idx] < frame_ready_ts[idx - 1]
            and dec_over_ts[idx + 1] < frame_ready_ts[idx - 1]
        ):
            render_discard_flag[idx] = 1
        else:
            return NotImplementedError

        if render_discard_flag[idx] == 1:
            frame_ready_ts[idx] = frame_ready_ts[
                idx - 1
            ]  # inherit previous ts for following frame check
            data[idx, 10] = -1
            data[idx, 11] = -1
        elif (
            frame_ready_ts[idx] < frame_ready_ts[idx - 1]
        ):  # current log discard all queueing frames, therefore need to repair the data
            frame_ready_ts[idx] = frame_ready_ts[idx - 1] + avg_render_time
            data[idx, 10] = max(frame_ready_ts[idx - 1] - dec_over_ts[idx], 0)
            data[idx, 11] = avg_render_time
            # print('Calibrate frame_ready_ts for frame %d' % idx)

        return render_discard_flag[idx]

    # display control
    # TODO: must be implemented in the client side SDK
    def frame_display_controller(idx, buf_size=0):
        if render_ctrlable and render_discard_flag[idx]:
            return

        if data[idx, 2] == 1:
            frame_ready_ts[idx] = max(
                frame_ready_ts[idx - 1] + data[idx - 1, 9], frame_ready_ts[idx]
            )
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
                nearest_display_slot[idx] + max_delay_frame_no, expect_display_slot[idx]
            )
            expect_display_ts[idx] = min(
                nearest_display_ts[idx] + max_delay_frame_no * frame_interval,
                expect_display_ts[idx],
            )
        if render_ctrlable:
            if (
                nearest_display_slot[idx] < expect_display_slot[idx]
                and actual_display_slot[idx - 1] < expect_display_slot[idx]
            ):
                actual_display_slot[idx] = max(
                    actual_display_slot[idx - 1] + 1, nearest_display_slot[idx]
                )
                display_delayed_flag[idx] = 1
            else:
                actual_display_slot[idx] = max(
                    actual_display_slot[idx - 1], nearest_display_slot[idx]
                )  # TODO
                display_delayed_flag[idx] = 0
        else:
            # display queue with 1 frame buffer
            if (
                idx > 1
                and max(frame_ready_ts[idx - 1], expect_display_ts[idx - 1])
                < actual_display_ts[idx - 2]
                and max(frame_ready_ts[idx], expect_display_ts[idx])
                < actual_display_ts[idx - 2]
            ):
                display_discard_flag[idx - 1] = 1
                frame_ready_ts[idx - 1] = frame_ready_ts[idx - 2]
                nearest_display_ts[idx - 1] = nearest_display_ts[idx - 2]
                nearest_display_slot[idx - 1] = nearest_display_slot[idx - 2]
                # expect_display_ts[idx-1] = expect_display_ts[idx-2]
                actual_display_slot[idx - 1] = actual_display_slot[idx - 2]
                actual_display_ts[idx - 1] = actual_display_ts[idx - 2]
                data[idx - 1, 10] = -1
                data[idx - 1, 11] = -1

            cur_slot_ts = (
                np.ceil(
                    max(frame_ready_ts[idx] - nearest_display_ts[0], 0) / frame_interval
                )
                * frame_interval
                + nearest_display_ts[0]
            )
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
    if not render_ctrlable:
        for idx in range(1, data.shape[0]):
            if data[idx, 5:12].sum() < data[idx - 1, 5:12].sum():
                data[idx, 11] = avg_render_time

    # start the simulation
    for idx in range(1, data.shape[0] - 1):
        if render_ctrlable:
            ret = render_queue_controller(idx, mode=render_queue_mode)
            if ret == 1:
                nearest_display_slot[idx] = -1
                expect_display_ts[idx] = -1
                actual_display_slot[idx] = -1

        frame_display_controller(idx, buf_size=buf_size)

        if idx >= 3:
            recv_time_estimator.update(data[idx - 2, 12], data[idx - 2, 5])
            if render_ctrlable:
                proc_time_estimator.update(
                    data[idx, 9], data[idx, 11], data[idx, 6:12].sum()
                )
            else:
                proc_time_estimator.update(
                    data[idx, 9], data[idx, 11], data[idx, 6:10].sum()
                )

        if actual_display_slot[idx] != actual_display_slot[idx - 1]:
            cur_valid_flag[idx] = 1

    return cur_valid_flag, invoke_present_ts


def cal_single_para_result(
    file_path,
    buf_size=0,
    proc_time_estimator_mode=0,
    jitter_buffer_controller_mode=0,
    timestamp_extrapolator_mode=0,
    render_queue_mode=0,
    strict_buffer=True,
    render_ctrlable=False,
    max_delay_frame_no=1,
    frame_interval=17,
    print_log=False,
):
    """
    Simulate the display process with a frame trace and a single set of parameters.
    :param buf_size: maximum No. of display slots that one frame can be delayed
    :param proc_time_estimator_mode: methods used to predict the expected proc_ts
    :param jitter_buffer_controller_mode: methods used to calculate the FPS gain
    :param timestamp_extrapolator_mode: methods used to predict the expected recv_ts
    :param render_queue_mode: only useful when render/present is controllable, 0 for no buffer, 1 for 1 frame buffer
    :param strict_buffer: if True, one frame can be delayed at most buf_size display slot, otherwise it can be delayed to the expected display slot
    :param render_ctrlable: if True, the timestamp that SDK invokes the present function is the ts that a frame is displayed. Current render is not ctrlable, thus this variable is False
    """
    sim_data_len = 60 * 60 * 60 + 2400
    data, info = load_data_func(
        file_path, start_idx=0, len_limit=sim_data_len
    )  # sim for 20min

    if data is None or data.shape[0] < 2000:
        return None, None

    # only simulate 60FPS traces
    recv_interval = np.mean(data[1:, 12] - data[:-1, 12])
    if recv_interval < 13 or recv_interval > 21:
        return None, None

    # initialization: calculate the average decode and render time
    def init_controller(samp_len):
        valid_idx = (
            data[60:samp_len, 5:12].sum(-1) > data[59 : samp_len - 1, 5:12].sum(-1)
        ) + 60
        avg_dec_time = np.mean(data[valid_idx, 9])
        avg_dec_total_time = np.mean(data[valid_idx, 6:10].sum(-1))
        avg_render_time = np.mean(data[valid_idx, 11])
        avg_proc_time = np.mean(data[valid_idx, 6:12].sum(-1))

        return avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time

    samp_len = 2400
    avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time = init_controller(
        samp_len
    )
    if avg_render_time > 15:  # ignore traces with vertical-sync on
        return None, None

    data = data[samp_len:, :]
    anchor_recv_ts = data[0, 5]
    data[:, 5] -= anchor_recv_ts

    # if render_ctrlable, initialize the ClientProcTimeEstimator with avg_proc_time
    # otherwise, initialize it with avg_dec_total_time
    if render_ctrlable:
        proc_time_estimator = ClientProcTimeEstimator(
            avg_dec_time=avg_dec_time,
            avg_render_time=avg_render_time,
            avg_proc_time=avg_proc_time,
            mode=proc_time_estimator_mode,
        )
    else:
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
    add_valid_flag = np.zeros(data.shape[0])
    slot_delayed_flag = np.zeros(data.shape[0])
    slot_moved_flag = np.zeros(data.shape[0])
    minus_valid_flag = np.zeros(data.shape[0])
    tear_flag = np.zeros(data.shape[0])

    render_discard_flag = np.zeros(
        data.shape[0]
    )  # 0 for render, 1 for discard, only used when render_ctrlable
    display_discard_flag = np.zeros(data.shape[0])  # 0 for display, 1 for discard
    display_delayed_flag = np.zeros(data.shape[0])  # 0 for display, 1 for discard

    dec_over_ts = data[:, 5:10].sum(-1)
    if render_ctrlable:
        frame_ready_ts = data[:, 5:12].sum(-1)  # from receiving to finishing rendering
    else:
        frame_ready_ts = data[:, 5:10].sum(-1)  # from receiving to finishing decoding
    original_display_ts = data[:, 5:12].sum(-1)

    # calculate the expect display ts and slot for the first frame
    # set the display refresh slot of the 1st frame, according to the vsync ts diff
    if len(data[0]) < 26:
        display_slot_shift = 8
    else:
        display_slot_shift = data[0, 25]
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
    slot_moved_flag[0] = 1

    recv_time_estimator = TimestampExtrapolator(
        first_frame_sts=data[0, 12],
        first_frame_recv_ts=0,
        mode=timestamp_extrapolator_mode,
    )  # expect_render_time = expect_recv_time + predicted_proc_time
    jitter_buffer = JitterBufferController(
        jitter_buffer_controller_mode, frame_interval, nearest_display_ts[0]
    )

    # after initialization, display the first frame as soon as possible
    if render_ctrlable:
        actual_display_slot[0] = nearest_display_slot[0]
    else:
        # UPDATE! For the first frame, if tear protect is ON and its ready ts is less than (frame_interval-vsync_slot_threshold) ms away from the next vsync ts,
        # then delay it to the next vsync ts to avoid screen tearing.
        if tear_protect and display_slot_shift < frame_interval - vsync_slot_threshold:
            invoke_present_ts[0] = nearest_display_ts[0] + frame_interval
        else:
            invoke_present_ts[0] = frame_ready_ts[0]
        actual_display_ts[0] = invoke_present_ts[0] + max(
            data[0, 11], 1
        )  # UPDATE! (for calculation, to ensure the actual display slot is calculated correctly when render time = 0)
        actual_display_slot[0] = (
            np.ceil(
                max(0, invoke_present_ts[0] + 1 - nearest_display_ts[0])
                / frame_interval
            )
            - 1
        )  # UPDATE!

    # old simulation, current render is not ctrlable
    def render_queue_controller(idx, mode=0):  # 0 for no buffer, 1 for 1 buffer
        if mode == 0 and dec_over_ts[idx] < frame_ready_ts[idx - 1]:
            render_discard_flag[idx] = 1
        elif (
            mode == 1
            and dec_over_ts[idx] < frame_ready_ts[idx - 1]
            and dec_over_ts[idx + 1] < frame_ready_ts[idx - 1]
        ):
            render_discard_flag[idx] = 1
        else:
            return NotImplementedError

        if render_discard_flag[idx] == 1:
            frame_ready_ts[idx] = frame_ready_ts[
                idx - 1
            ]  # inherit previous ts for following frame check
            data[idx, 10] = -1
            data[idx, 11] = -1
        elif (
            frame_ready_ts[idx] < frame_ready_ts[idx - 1]
        ):  # current log discard all queueing frames, therefore need to repair the data
            frame_ready_ts[idx] = frame_ready_ts[idx - 1] + avg_render_time
            data[idx, 10] = max(frame_ready_ts[idx - 1] - dec_over_ts[idx], 0)
            data[idx, 11] = avg_render_time
            # print('Calibrate frame_ready_ts for frame %d' % idx)

        return render_discard_flag[idx]

    # display control
    # TODO: must be implemented in the client side SDK
    def frame_display_controller(idx, buf_size=0):
        if render_ctrlable and render_discard_flag[idx]:
            return

        if data[idx, 2] == 1:
            frame_ready_ts[idx] = max(
                frame_ready_ts[idx - 1] + data[idx - 1, 9], frame_ready_ts[idx]
            )
        # UPDATE!
        nearest_display_slot[idx] = (
            np.ceil(
                max(frame_ready_ts[idx] - nearest_display_ts[0], 0) / frame_interval
            )
            - 1
        )
        nearest_display_ts[idx] = (
            nearest_display_slot[idx] * frame_interval + nearest_display_ts[0]
        )  # UPDATE! nearest_display_slot starting from 1 now

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
                nearest_display_slot[idx] + max_delay_frame_no, expect_display_slot[idx]
            )
            expect_display_ts[idx] = min(
                nearest_display_ts[idx] + max_delay_frame_no * frame_interval,
                expect_display_ts[idx],
            )
        if render_ctrlable:
            if (
                nearest_display_slot[idx] < expect_display_slot[idx]
                and actual_display_slot[idx - 1] < expect_display_slot[idx]
            ):
                actual_display_slot[idx] = max(
                    actual_display_slot[idx - 1] + 1, nearest_display_slot[idx]
                )
                display_delayed_flag[idx] = 1
            else:
                actual_display_slot[idx] = max(
                    actual_display_slot[idx - 1], nearest_display_slot[idx]
                )  # TODO
                display_delayed_flag[idx] = 0
        else:
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
    if not render_ctrlable:
        for idx in range(1, data.shape[0]):
            if data[idx, 5:12].sum() < data[idx - 1, 5:12].sum():
                data[idx, 11] = avg_render_time

    # start the simulation
    for idx in range(1, data.shape[0]):
        if render_ctrlable:
            ret = render_queue_controller(idx, mode=render_queue_mode)
            if ret == 1:
                nearest_display_slot[idx] = -1
                expect_display_ts[idx] = -1
                actual_display_slot[idx] = -1

        frame_display_controller(idx, buf_size=buf_size)

        if idx >= 3:
            recv_time_estimator.update(data[idx - 2, 12], data[idx - 2, 5])
            if render_ctrlable:
                proc_time_estimator.update(
                    data[idx, 9], data[idx, 11], data[idx, 6:12].sum()
                )
            else:
                proc_time_estimator.update(
                    data[idx, 9], data[idx, 11], data[idx, 6:10].sum()
                )

        if jitter_buffer_controller_mode == 0:
            jitter_buffer.update(
                max(frame_ready_ts[idx] - expect_nearest_display_ts[idx], 0)
            )
            (
                add_valid_flag[idx],
                minus_valid_flag[idx],
                slot_delayed_flag[idx],
                slot_moved_flag[idx],
            ) = jitter_buffer.get_latest_flag()
        else:
            jitter_buffer.update(
                frame_ready_ts[idx - 1],
                expect_display_ts[idx - 1],
                actual_display_ts[idx - 1],
                actual_display_slot[idx - 1],
                frame_ready_ts[idx],
                expect_display_ts[idx],
                actual_display_ts[idx],
                actual_display_slot[idx],
                vsync_slot_threshold,
            )
            (
                add_valid_flag[idx],
                minus_valid_flag[idx],
                slot_delayed_flag[idx],
                slot_moved_flag[idx],
            ) = jitter_buffer.get_latest_flag()

        if actual_display_slot[idx] != actual_display_slot[idx - 1]:
            cur_valid_flag[idx] = 1

    one_more_buf_valid_flag, one_more_invoke_present_ts = cal_valid_frame(
        file_path,
        buf_size=buf_size + 1,
        proc_time_estimator_mode=proc_time_estimator_mode,
        jitter_buffer_controller_mode=jitter_buffer_controller_mode,
        timestamp_extrapolator_mode=timestamp_extrapolator_mode,
        render_queue_mode=render_queue_mode,
        strict_buffer=strict_buffer,
        render_ctrlable=render_ctrlable,
        max_delay_frame_no=max(1, (buf_size + 1) * 2),
        frame_interval=frame_interval,
    )
    one_less_buf_valid_flag, one_less_invoke_present_ts = cal_valid_frame(
        file_path,
        buf_size=max(0, buf_size - 1),
        proc_time_estimator_mode=proc_time_estimator_mode,
        jitter_buffer_controller_mode=jitter_buffer_controller_mode,
        timestamp_extrapolator_mode=timestamp_extrapolator_mode,
        render_queue_mode=render_queue_mode,
        strict_buffer=strict_buffer,
        render_ctrlable=render_ctrlable,
        max_delay_frame_no=max(1, (buf_size - 1) * 2),
        frame_interval=frame_interval,
    )

    if print_log:
        log_file = open(file_path[:-4] + "_buf%d_sim.csv" % buf_size, "w")
        if len(data[0]) < 29:
            log_file.write(
                ",".join(
                    [
                        "sim_index",
                        "render_index",
                        "frame_index",
                        "frame_type",
                        "size",
                        "loss_type",
                        "receive_timestamp",
                        "receive_and_unpack",
                        "decoder_outside_queue",
                        "decoder_insided_queue",
                        "decode",
                        "render_queue",
                        "display",
                        "capture_timestamp",
                        "send_ts",
                        "net_ts",
                        "proc_ts",
                        "tot_ts",
                        "basic_frame_ts",
                        "ul_jitter",
                        "dl_jitter",
                        "recv_ts",
                        "expect_recv_ts",
                        "expect_proc_time",
                        "actual_proc_time",
                        "dec_over_ts",
                        "frame_ready_ts",
                        "expect_frame_ready_ts",
                        "nearest_display_ts",
                        "expect_nearest_display_ts",
                        "delay_time",
                        "expect_display_ts",
                        "invoke_present_ts",
                        "actual_display_ts",
                        "nearest_display_slot",
                        "expect_display_slot",
                        "actual_display_slot",
                        "valid_frame_flag",
                        "add_valid_flag",
                        "one_more_pred_flag",
                        "one_more_actual_flag",
                        "one_more_actual_slot",
                        "slot_delayed",
                        "minus_valid_flag",
                        "one_less_pred_flag",
                        "one_less_actual_flag",
                        "one_less_actual_slot",
                        "one_less_actual_display_ts",
                        "slot_moved",
                        "display_delayed_flag",
                    ]
                )
                + "\n"
            )
            for idx in range(data.shape[0]):
                log_file.write(
                    ",".join(
                        str(item)
                        for item in [idx]
                        + data[idx].tolist()
                        + [
                            data[idx, 5],
                            expect_recv_ts[idx],
                            expect_proc_time[idx],
                            data[idx, 6:10].sum(),
                            dec_over_ts[idx],
                            frame_ready_ts[idx],
                            expect_frame_ready_ts[idx].astype(int),
                            nearest_display_ts[idx],
                            expect_nearest_display_ts[idx],
                            max(
                                0, frame_ready_ts[idx] - expect_nearest_display_ts[idx]
                            ),
                            expect_display_ts[idx],
                            invoke_present_ts[idx],
                            actual_display_ts[idx],
                            nearest_display_slot[idx],
                            expect_display_slot[idx],
                            actual_display_slot[idx],
                            cur_valid_flag[idx],
                            add_valid_flag[idx],
                            np.logical_or(
                                cur_valid_flag[idx], add_valid_flag[idx]
                            ).astype(int),
                            (int)(one_more_buf_valid_flag[idx]),
                            np.ceil(
                                max(
                                    0,
                                    one_more_invoke_present_ts[idx]
                                    + 1
                                    - nearest_display_ts[0],
                                )
                                / frame_interval
                            ).astype(int)
                            - 1,
                            slot_delayed_flag[idx],
                            minus_valid_flag[idx],
                            np.logical_and(
                                cur_valid_flag[idx], not minus_valid_flag[idx]
                            ).astype(int),
                            (int)(one_less_buf_valid_flag[idx]),
                            np.ceil(
                                max(
                                    0,
                                    one_less_invoke_present_ts[idx]
                                    + 1
                                    - nearest_display_ts[0],
                                )
                                / frame_interval
                            ).astype(int)
                            - 1,
                            one_less_invoke_present_ts[idx],
                            slot_moved_flag[idx],
                            display_delayed_flag[idx],
                        ]
                    )
                    + "\n"
                )
        else:
            log_file.write(
                ",".join(
                    [
                        "sim_index",
                        "render_index",
                        "frame_index",
                        "frame_type",
                        "size",
                        "loss_type",
                        "receive_timestamp",
                        "receive_and_unpack",
                        "decoder_outside_queue",
                        "decoder_insided_queue",
                        "decode",
                        "render_queue",
                        "display",
                        "capture_timestamp",
                        "send_ts",
                        "net_ts",
                        "proc_ts",
                        "tot_ts",
                        "basic_frame_ts",
                        "ul_jitter",
                        "dl_jitter",
                        "recv_ts",
                        "expect_recv_ts",
                        "server_expect_recv_ts",
                        "server_expect_proc_time",
                        "expect_proc_time",
                        "actual_proc_time",
                        "dec_over_ts",
                        "frame_ready_ts",
                        "expect_frame_ready_ts",
                        "server_expect_frame_ready_ts",
                        "nearest_display_ts",
                        "expect_nearest_display_ts",
                        "delay_time",
                        "expect_display_ts",
                        "invoke_present_ts",
                        "actual_display_ts",
                        "nearest_display_slot",
                        "expect_display_slot",
                        "actual_display_slot",
                        "actual_display_slot_ts",
                        "valid_frame_flag",
                        "add_valid_flag",
                        "one_more_pred_flag",
                        "one_more_actual_flag",
                        "one_more_actual_slot",
                        "minus_valid_flag",
                        "one_less_pred_flag",
                        "one_less_actual_flag",
                        "one_less_actual_slot",
                        "display_delayed_flag",
                    ]
                )
                + "\n"
            )
            for idx in range(data.shape[0]):
                log_file.write(
                    ",".join(
                        str(item)
                        for item in [idx]
                        + data[idx, :-9].tolist()
                        + [
                            data[idx, 5] + anchor_recv_ts,
                            expect_recv_ts[idx] + anchor_recv_ts,
                            data[idx, 20],
                            data[idx, 21],
                            expect_proc_time[idx],
                            data[idx, 6:10].sum(),
                            dec_over_ts[idx] + anchor_recv_ts,
                            frame_ready_ts[idx] + anchor_recv_ts,
                            expect_frame_ready_ts[idx].astype(int) + anchor_recv_ts,
                            data[idx, 20] + data[idx, 21],
                            nearest_display_ts[idx] + anchor_recv_ts,
                            expect_nearest_display_ts[idx] + anchor_recv_ts,
                            max(
                                0, frame_ready_ts[idx] - expect_nearest_display_ts[idx]
                            ),
                            expect_display_ts[idx] + anchor_recv_ts,
                            invoke_present_ts[idx] + anchor_recv_ts,
                            actual_display_ts[idx] + anchor_recv_ts,
                            nearest_display_slot[idx],
                            expect_display_slot[idx],
                            actual_display_slot[idx],
                            np.ceil(
                                max(
                                    actual_display_ts[idx - 1] - nearest_display_ts[0],
                                    0,
                                )
                                / frame_interval
                            )
                            * frame_interval
                            + nearest_display_ts[0]
                            + anchor_recv_ts,
                            cur_valid_flag[idx],
                            add_valid_flag[idx],
                            np.logical_or(
                                cur_valid_flag[idx], add_valid_flag[idx]
                            ).astype(int),
                            one_more_buf_valid_flag[idx],
                            one_more_invoke_present_ts[
                                idx
                            ],  # np.ceil(max(0, one_more_invoke_present_ts[idx]+1-nearest_display_ts[0]) / frame_interval) - 1,
                            minus_valid_flag[idx],
                            np.logical_and(
                                cur_valid_flag[idx], not minus_valid_flag[idx]
                            ).astype(int),
                            one_less_buf_valid_flag[idx],
                            one_less_invoke_present_ts[
                                idx
                            ],  # np.ceil(max(0, one_less_invoke_present_ts[idx]+1-nearest_display_ts[0]) / frame_interval) - 1,
                            display_delayed_flag[idx],
                        ]
                    )
                    + "\n"
                )

    # not very acurate in no render_ctrl mode, as the time consumed to invoke present is not taken into account,
    # which may make the min fps even lower
    # valid_no = np.unique(nearest_display_slot).size
    valid_no = np.unique(
        np.ceil(
            (frame_ready_ts - display_slot_shift) / frame_interval
        )  # TODO: 替换为 frame_ready_ts
    ).size
    tot_time = data[-1, 5] / 1000
    max_fps = data.shape[0] / tot_time
    min_fps = valid_no / tot_time

    data[-1, 11] = -1
    valid_idx = np.where(data[:, 11] != -1)
    opt_valid_no = np.unique(actual_display_slot[valid_idx]).size
    opt_fps = opt_valid_no / tot_time

    result = [
        max_fps,
        min_fps,
        opt_fps,
        *jitter_buffer.get_buffer_gain(tot_time=tot_time)[
            1:
        ],  # FPS upper limit, FPS lower limit, optimized fps, estimated fps with 1 more buffer, estimated fps with 1 less buffer,
        np.sum(one_more_buf_valid_flag) / tot_time,
        np.sum(one_less_buf_valid_flag)
        / tot_time,  # real fps with 1 more buffer, real fps with 1 less buffer,
        np.mean(
            invoke_present_ts[valid_idx] - frame_ready_ts[valid_idx]
        ),  # current overhead,
        np.mean(
            one_more_invoke_present_ts[valid_idx] - frame_ready_ts[valid_idx]
        ),  # overhead with 1 more buffer,
        np.mean(
            one_less_invoke_present_ts[valid_idx] - frame_ready_ts[valid_idx]
        ),  # overhead with 1 less buffer,
        np.sum(tear_flag) / tot_time,
        avg_dec_time,
        avg_dec_total_time,
        avg_render_time,
        avg_proc_time,
        recv_interval,
    ]  # teared frame per second, avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time, recv_interval
    output_file = open(
        "result_bufSize%d_procTime%d_jitterBuffer%d_tsExtrapolator%d_renderQueue%d_strict%d_renderCtrlable%d_frameInterval%d.csv"
        % (
            buf_size,
            proc_time_estimator_mode,
            jitter_buffer_controller_mode,
            timestamp_extrapolator_mode,
            render_queue_mode,
            strict_buffer,
            render_ctrlable,
            frame_interval,
        ),
        "a",
    )
    output_file.write(
        file_path.replace(",", "_")
        + ", "
        + ", ".join([str(item) for item in result])
        + "\n"
    )
    output_file.close()

    return file_path, result


def cal_adapt_buf_result(
    file_path,
    min_buf_size=0,
    max_buf_size=2,
    buf_size_update_interval=5,
    target_fps=55,
    min_fps_gain=2,
    proc_time_estimator_mode=0,
    jitter_buffer_controller_mode=0,
    timestamp_extrapolator_mode=0,
    render_queue_mode=0,
    strict_buffer=True,
    render_ctrlable=False,
    frame_interval=17,
    print_log=False,
):
    """
    Simulate the display process with a frame trace and a single set of parameters.
    :param buf_size: maximum No. of display slots that one frame can be delayed
    :param proc_time_estimator_mode: methods used to predict the expected proc_ts
    :param jitter_buffer_controller_mode: methods used to calculate the FPS gain
    :param timestamp_extrapolator_mode: methods used to predict the expected recv_ts
    :param render_queue_mode: only useful when render/present is controllable, 0 for no buffer, 1 for 1 frame buffer
    :param strict_buffer: if True, one frame can be delayed at most buf_size display slot, otherwise it can be delayed to the expected display slot
    :param render_ctrlable: if True, the timestamp that SDK invokes the present function is the ts that a frame is displayed. Current render is not ctrlable, thus this variable is False
    """
    sim_data_len = 60 * 60 * 60 + 2400
    data, info = load_data_func(
        file_path, start_idx=0, len_limit=sim_data_len
    )  # sim for 20min

    if data is None or data.shape[0] < 2000:
        return None, None

    # only simulate 60FPS traces
    recv_interval = np.mean(data[1:, 12] - data[:-1, 12])
    if recv_interval < 13 or recv_interval > 20:
        return None, None

    # initialization: calculate the average decode and render time
    def init_controller(samp_len):
        valid_idx = (
            data[60:samp_len, 5:12].sum(-1) > data[59 : samp_len - 1, 5:12].sum(-1)
        ) + 60
        avg_dec_time = np.mean(data[valid_idx, 9])
        avg_dec_total_time = np.mean(data[valid_idx, 6:10].sum(-1))
        avg_render_time = np.mean(data[valid_idx, 11])
        avg_proc_time = np.mean(data[valid_idx, 6:12].sum(-1))

        return avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time

    samp_len = 2400
    avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time = init_controller(
        samp_len
    )
    if avg_render_time > 15:  # ignore traces with vertical-sync on
        return None, None

    data = data[samp_len:, :]
    anchor_recv_ts = data[0, 5]
    data[:, 5] -= anchor_recv_ts

    # if render_ctrlable, initialize the ClientProcTimeEstimator with avg_proc_time
    # otherwise, initialize it with avg_dec_total_time
    if render_ctrlable:
        proc_time_estimator = ClientProcTimeEstimator(
            avg_dec_time=avg_dec_time,
            avg_render_time=avg_render_time,
            avg_proc_time=avg_proc_time,
            mode=proc_time_estimator_mode,
        )
    else:
        proc_time_estimator = ClientProcTimeEstimator(
            avg_dec_time=avg_dec_time,
            avg_render_time=avg_render_time,
            avg_proc_time=avg_dec_total_time,
            mode=proc_time_estimator_mode,
        )

    buf_size = 0
    max_delay_frame_no = max(1, buf_size * 2)

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
    add_valid_flag = np.zeros(data.shape[0])
    slot_delayed_flag = np.zeros(data.shape[0])
    slot_moved_flag = np.zeros(data.shape[0])
    minus_valid_flag = np.zeros(data.shape[0])
    real_time_buf_size = np.zeros(data.shape[0])
    tear_flag = np.zeros(data.shape[0])

    render_discard_flag = np.zeros(
        data.shape[0]
    )  # 0 for render, 1 for discard, only used when render_ctrlable
    display_discard_flag = np.zeros(data.shape[0])  # 0 for display, 1 for discard
    display_delayed_flag = np.zeros(data.shape[0])  # 0 for display, 1 for discard

    dec_over_ts = data[:, 5:10].sum(-1)
    if render_ctrlable:
        frame_ready_ts = data[:, 5:12].sum(-1)  # from receiving to finishing rendering
    else:
        frame_ready_ts = data[:, 5:10].sum(-1)  # from receiving to finishing decoding
    original_display_ts = data[:, 5:12].sum(-1)

    # calculate the expect display ts and slot for the first frame
    # set the display refresh slot of the 1st frame, according to the vsync ts diff
    if len(data[0]) < 26:
        display_slot_shift = 8
    else:
        display_slot_shift = data[0, 25]
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
    slot_moved_flag[0] = 1

    recv_time_estimator = TimestampExtrapolator(
        first_frame_sts=data[0, 12],
        first_frame_recv_ts=0,
        mode=timestamp_extrapolator_mode,
    )  # expect_render_time = expect_recv_time + predicted_proc_time
    jitter_buffer = JitterBufferController(
        jitter_buffer_controller_mode, frame_interval, nearest_display_ts[0]
    )

    # after initialization, display the first frame as soon as possible
    if render_ctrlable:
        actual_display_slot[0] = nearest_display_slot[0]
    else:
        # UPDATE! For the first frame, if tear protect is ON and its ready ts is less than (frame_interval-vsync_slot_threshold) ms away from the next vsync ts,
        # then delay it to the next vsync ts to avoid screen tearing.
        if tear_protect and display_slot_shift < frame_interval - vsync_slot_threshold:
            invoke_present_ts[0] = nearest_display_ts[0] + frame_interval
        else:
            invoke_present_ts[0] = frame_ready_ts[0]
        actual_display_ts[0] = invoke_present_ts[0] + max(
            data[0, 11], 1
        )  # UPDATE! (for calculation, to ensure the actual display slot is calculated correctly when render time = 0)
        actual_display_slot[0] = (
            np.ceil(
                max(0, invoke_present_ts[0] + 1 - nearest_display_ts[0])
                / frame_interval
            )
            - 1
        )  # UPDATE!

    # old simulation, current render is not ctrlable
    def render_queue_controller(idx, mode=0):  # 0 for no buffer, 1 for 1 buffer
        if mode == 0 and dec_over_ts[idx] < frame_ready_ts[idx - 1]:
            render_discard_flag[idx] = 1
        elif (
            mode == 1
            and dec_over_ts[idx] < frame_ready_ts[idx - 1]
            and dec_over_ts[idx + 1] < frame_ready_ts[idx - 1]
        ):
            render_discard_flag[idx] = 1
        else:
            return NotImplementedError

        if render_discard_flag[idx] == 1:
            frame_ready_ts[idx] = frame_ready_ts[
                idx - 1
            ]  # inherit previous ts for following frame check
            data[idx, 10] = -1
            data[idx, 11] = -1
        elif (
            frame_ready_ts[idx] < frame_ready_ts[idx - 1]
        ):  # current log discard all queueing frames, therefore need to repair the data
            frame_ready_ts[idx] = frame_ready_ts[idx - 1] + avg_render_time
            data[idx, 10] = max(frame_ready_ts[idx - 1] - dec_over_ts[idx], 0)
            data[idx, 11] = avg_render_time
            # print('Calibrate frame_ready_ts for frame %d' % idx)

        return render_discard_flag[idx]

    # display control
    # TODO: must be implemented in the client side SDK
    def frame_display_controller(idx, buf_size=0):
        if render_ctrlable and render_discard_flag[idx]:
            return

        if data[idx, 2] == 1:
            frame_ready_ts[idx] = max(
                frame_ready_ts[idx - 1] + data[idx - 1, 9], frame_ready_ts[idx]
            )
        # UPDATE!
        nearest_display_slot[idx] = (
            np.ceil(
                max(frame_ready_ts[idx] - nearest_display_ts[0], 0) / frame_interval
            )
            - 1
        )
        nearest_display_ts[idx] = (
            nearest_display_slot[idx] * frame_interval + nearest_display_ts[0]
        )  # UPDATE! nearest_display_slot starting from 1 now

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
                nearest_display_slot[idx] + max_delay_frame_no, expect_display_slot[idx]
            )
            expect_display_ts[idx] = min(
                nearest_display_ts[idx] + max_delay_frame_no * frame_interval,
                expect_display_ts[idx],
            )
        if render_ctrlable:
            if (
                nearest_display_slot[idx] < expect_display_slot[idx]
                and actual_display_slot[idx - 1] < expect_display_slot[idx]
            ):
                actual_display_slot[idx] = max(
                    actual_display_slot[idx - 1] + 1, nearest_display_slot[idx]
                )
                display_delayed_flag[idx] = 1
            else:
                actual_display_slot[idx] = max(
                    actual_display_slot[idx - 1], nearest_display_slot[idx]
                )  # TODO
                display_delayed_flag[idx] = 0
        else:
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
    if not render_ctrlable:
        for idx in range(1, data.shape[0]):
            if data[idx, 5:12].sum() < data[idx - 1, 5:12].sum():
                data[idx, 11] = avg_render_time

    # start the simulation
    prev_idx = 0
    bufsize_need_reduce_cnt = 0
    bufsize_increase_cnt = 0
    bufsize_decrease_cnt = 0
    bufsize_update_cnt = 0
    for idx in range(1, data.shape[0]):
        if render_ctrlable:
            ret = render_queue_controller(idx, mode=render_queue_mode)
            if ret == 1:
                nearest_display_slot[idx] = -1
                expect_display_ts[idx] = -1
                actual_display_slot[idx] = -1

        frame_display_controller(idx, buf_size=buf_size)

        if idx >= 3:
            recv_time_estimator.update(data[idx - 2, 12], data[idx - 2, 5])
            if render_ctrlable:
                proc_time_estimator.update(
                    data[idx, 9], data[idx, 11], data[idx, 6:12].sum()
                )
            else:
                proc_time_estimator.update(
                    data[idx, 9], data[idx, 11], data[idx, 6:10].sum()
                )

        if jitter_buffer_controller_mode == 0:
            jitter_buffer.update(
                max(frame_ready_ts[idx] - expect_nearest_display_ts[idx], 0)
            )
            (
                add_valid_flag[idx],
                minus_valid_flag[idx],
                slot_delayed_flag[idx],
                slot_moved_flag[idx],
            ) = jitter_buffer.get_latest_flag()
        else:
            jitter_buffer.update(
                frame_ready_ts[idx - 1],
                expect_display_ts[idx - 1],
                actual_display_ts[idx - 1],
                actual_display_slot[idx - 1],
                frame_ready_ts[idx],
                expect_display_ts[idx],
                actual_display_ts[idx],
                actual_display_slot[idx],
                vsync_slot_threshold,
            )
            (
                add_valid_flag[idx],
                minus_valid_flag[idx],
                slot_delayed_flag[idx],
                slot_moved_flag[idx],
            ) = jitter_buffer.get_latest_flag()

        if actual_display_slot[idx] != actual_display_slot[idx - 1]:
            cur_valid_flag[idx] = 1

        real_time_buf_size[idx] = buf_size
        if data[idx, 5] - data[prev_idx, 5] >= buf_size_update_interval * 1000:
            bufsize_update_cnt += 1
            cur_valid_fps, cur_1more_fps, cur_1less_fps = jitter_buffer.get_buffer_gain(
                tot_time=(data[idx, 5] - data[prev_idx, 5]) / 1000
            )
            if (
                cur_valid_fps < target_fps
                and cur_1more_fps - cur_valid_fps > min_fps_gain
            ):
                print(
                    cur_valid_fps,
                    target_fps,
                    cur_1more_fps - cur_valid_fps,
                    max_buf_size,
                )
                buf_size = min(buf_size + 1, max_buf_size)
                bufsize_need_reduce_cnt = 0
                bufsize_increase_cnt += 1
            elif (
                cur_1less_fps >= target_fps
                or cur_valid_fps - cur_1less_fps < min_fps_gain
            ):
                bufsize_need_reduce_cnt += 1
                if bufsize_need_reduce_cnt >= 3:
                    buf_size = max(buf_size - 1, min_buf_size)
                    bufsize_need_reduce_cnt = 0
                    bufsize_decrease_cnt += 1
            else:
                bufsize_need_reduce_cnt = 0
            print(
                cur_valid_fps,
                cur_1more_fps,
                cur_1less_fps,
                buf_size,
                bufsize_increase_cnt,
                bufsize_decrease_cnt,
            )
            max_delay_frame_no = max(1, buf_size * 2)
            prev_idx = idx
            jitter_buffer.reset()

    buf0_valid_flag, buf0_invoke_present_ts = cal_valid_frame(
        file_path,
        buf_size=0,
        proc_time_estimator_mode=proc_time_estimator_mode,
        jitter_buffer_controller_mode=jitter_buffer_controller_mode,
        timestamp_extrapolator_mode=timestamp_extrapolator_mode,
        render_queue_mode=render_queue_mode,
        strict_buffer=strict_buffer,
        render_ctrlable=render_ctrlable,
        max_delay_frame_no=1,
        frame_interval=frame_interval,
    )
    buf1_valid_flag, buf1_invoke_present_ts = cal_valid_frame(
        file_path,
        buf_size=1,
        proc_time_estimator_mode=proc_time_estimator_mode,
        jitter_buffer_controller_mode=jitter_buffer_controller_mode,
        timestamp_extrapolator_mode=timestamp_extrapolator_mode,
        render_queue_mode=render_queue_mode,
        strict_buffer=strict_buffer,
        render_ctrlable=render_ctrlable,
        max_delay_frame_no=2,
        frame_interval=frame_interval,
    )
    buf2_valid_flag, buf2_invoke_present_ts = cal_valid_frame(
        file_path,
        buf_size=2,
        proc_time_estimator_mode=proc_time_estimator_mode,
        jitter_buffer_controller_mode=jitter_buffer_controller_mode,
        timestamp_extrapolator_mode=timestamp_extrapolator_mode,
        render_queue_mode=render_queue_mode,
        strict_buffer=strict_buffer,
        render_ctrlable=render_ctrlable,
        max_delay_frame_no=4,
        frame_interval=frame_interval,
    )

    if print_log:
        log_file = open(file_path[:-4] + "_adapt_buf_sim.csv", "w")
        if len(data[0]) < 29:
            log_file.write(
                ",".join(
                    [
                        "sim_index",
                        "render_index",
                        "frame_index",
                        "frame_type",
                        "size",
                        "loss_type",
                        "receive_timestamp",
                        "receive_and_unpack",
                        "decoder_outside_queue",
                        "decoder_insided_queue",
                        "decode",
                        "render_queue",
                        "display",
                        "capture_timestamp",
                        "send_ts",
                        "net_ts",
                        "proc_ts",
                        "tot_ts",
                        "basic_frame_ts",
                        "ul_jitter",
                        "dl_jitter",
                        "recv_ts",
                        "expect_recv_ts",
                        "expect_proc_time",
                        "actual_proc_time",
                        "dec_over_ts",
                        "frame_ready_ts",
                        "expect_frame_ready_ts",
                        "nearest_display_ts",
                        "expect_nearest_display_ts",
                        "delay_time",
                        "expect_display_ts",
                        "invoke_present_ts",
                        "actual_display_ts",
                        "nearest_display_slot",
                        "expect_display_slot",
                        "actual_display_slot",
                        "valid_frame_flag",
                        "buf_size",
                        "display_delayed_flag",
                        "buf0_invoke_present_ts",
                        "buf1_invoke_present_ts",
                        "buf2_invoke_present_ts",
                    ]
                )
                + "\n"
            )
            for idx in range(data.shape[0]):
                log_file.write(
                    ",".join(
                        str(item)
                        for item in [idx]
                        + data[idx].tolist()
                        + [
                            data[idx, 5],
                            expect_recv_ts[idx],
                            expect_proc_time[idx],
                            data[idx, 6:10].sum(),
                            dec_over_ts[idx],
                            frame_ready_ts[idx],
                            expect_frame_ready_ts[idx].astype(int),
                            nearest_display_ts[idx],
                            expect_nearest_display_ts[idx],
                            max(
                                0, frame_ready_ts[idx] - expect_nearest_display_ts[idx]
                            ),
                            expect_display_ts[idx],
                            invoke_present_ts[idx],
                            actual_display_ts[idx],
                            nearest_display_slot[idx],
                            expect_display_slot[idx],
                            actual_display_slot[idx],
                            cur_valid_flag[idx],
                            real_time_buf_size[idx],
                            display_delayed_flag[idx],
                            buf0_invoke_present_ts[idx],
                            buf1_invoke_present_ts[idx],
                            buf2_invoke_present_ts[idx],
                        ]
                    )
                    + "\n"
                )
        else:
            log_file.write(
                ",".join(
                    [
                        "sim_index",
                        "render_index",
                        "frame_index",
                        "frame_type",
                        "size",
                        "loss_type",
                        "receive_timestamp",
                        "receive_and_unpack",
                        "decoder_outside_queue",
                        "decoder_insided_queue",
                        "decode",
                        "render_queue",
                        "display",
                        "capture_timestamp",
                        "send_ts",
                        "net_ts",
                        "proc_ts",
                        "tot_ts",
                        "basic_frame_ts",
                        "ul_jitter",
                        "dl_jitter",
                        "recv_ts",
                        "expect_recv_ts",
                        "expect_proc_time",
                        "actual_proc_time",
                        "dec_over_ts",
                        "frame_ready_ts",
                        "expect_frame_ready_ts",
                        "nearest_display_ts",
                        "expect_nearest_display_ts",
                        "delay_time",
                        "expect_display_ts",
                        "invoke_present_ts",
                        "actual_display_ts",
                        "nearest_display_slot",
                        "expect_display_slot",
                        "actual_display_slot",
                        "valid_frame_flag",
                        "buf_size",
                        "display_delayed_flag",
                    ]
                )
                + "\n"
            )
            for idx in range(data.shape[0]):
                log_file.write(
                    ",".join(
                        str(item)
                        for item in [idx]
                        + data[idx, :-9].tolist()
                        + [
                            data[idx, 5],
                            expect_recv_ts[idx],
                            expect_proc_time[idx],
                            data[idx, 6:10].sum(),
                            dec_over_ts[idx],
                            frame_ready_ts[idx],
                            expect_frame_ready_ts[idx].astype(int),
                            nearest_display_ts[idx],
                            expect_nearest_display_ts[idx],
                            max(
                                0, frame_ready_ts[idx] - expect_nearest_display_ts[idx]
                            ),
                            expect_display_ts[idx],
                            invoke_present_ts[idx],
                            actual_display_ts[idx],
                            nearest_display_slot[idx],
                            expect_display_slot[idx],
                            actual_display_slot[idx],
                            cur_valid_flag[idx],
                            real_time_buf_size[idx],
                            display_delayed_flag[idx],
                        ]
                    )
                    + "\n"
                )

    # not very acurate in no render_ctrl mode, as the time consumed to invoke present is not taken into account,
    # which may make the min fps even lower
    # valid_no = np.unique(nearest_display_slot).size
    valid_no = np.unique(
        np.ceil((original_display_ts - display_slot_shift) / frame_interval)
    ).size
    tot_time = data[-1, 5] / 1000
    max_fps = data.shape[0] / tot_time
    min_fps = valid_no / tot_time

    valid_idx = np.where(data[:-1, 11] != -1)
    opt_valid_no = np.unique(actual_display_slot[valid_idx]).size
    opt_fps = opt_valid_no / tot_time

    result = [
        max_fps,
        min_fps,
        opt_fps,
        np.sum(buf0_valid_flag) / tot_time,
        np.sum(buf1_valid_flag) / tot_time,
        np.sum(buf2_valid_flag) / tot_time,
        np.mean(invoke_present_ts[valid_idx] - frame_ready_ts[valid_idx]),
        np.mean(buf0_invoke_present_ts[valid_idx] - frame_ready_ts[valid_idx]),
        np.mean(buf1_invoke_present_ts[valid_idx] - frame_ready_ts[valid_idx]),
        np.mean(buf2_invoke_present_ts[valid_idx] - frame_ready_ts[valid_idx]),
        bufsize_update_cnt,
        bufsize_increase_cnt,
        bufsize_decrease_cnt,
        avg_dec_time,
        avg_dec_total_time,
        avg_render_time,
        avg_proc_time,
        recv_interval,
    ]
    output_file = open(
        "result_adaptBuf_procTime%d_jitterBuffer%d_tsExtrapolator%d_renderQueue%d_strict%d_renderCtrlable%d_frameInterval%d.csv"
        % (
            proc_time_estimator_mode,
            jitter_buffer_controller_mode,
            timestamp_extrapolator_mode,
            render_queue_mode,
            strict_buffer,
            render_ctrlable,
            frame_interval,
        ),
        "a",
    )
    output_file.write(
        file_path.replace(",", "_")
        + ", "
        + ", ".join([str(item) for item in result])
        + "\n"
    )
    output_file.close()

    return file_path, result


def process_all_data(root_path):
    res1 = []
    res2 = []
    res3 = []
    for data_folder in os.listdir(root_path):
        if not data_folder.startswith("2023-"):
            continue

        data_path = os.path.join(root_path, data_folder)

        for session_folder in os.listdir(data_path):
            if not session_folder.startswith("session_info"):
                continue

            session_path = os.path.join(data_path, session_folder)

            for log_name in os.listdir(session_path):
                if not log_name.endswith(".csv"):
                    continue

                log_path = os.path.join(session_path, log_name)
                _, cur_res = cal_single_para_result(
                    log_path,
                    buf_size=buf_size,
                    proc_time_estimator_mode=proc_time_estimator_mode,
                    jitter_buffer_controller_mode=jitter_buffer_controller_mode,
                    timestamp_extrapolator_mode=timestamp_extrapolator_mode,
                    render_queue_mode=render_queue_mode,
                    strict_buffer=strict_buffer,
                    render_ctrlable=render_ctrlable,
                    max_delay_frame_no=max_delay_frame_no,
                    frame_interval=frame_interval,
                )

                if cur_res is not None:
                    res1.append(cur_res[2])
                    res2.append(cur_res[4])
                    res3.append(cur_res[5])

    print(np.mean(res1), np.min(res1), np.max(res1))
    print(np.mean(res2), np.min(res2), np.max(res2))
    print(np.mean(res3), np.min(res3), np.max(res3))


def process_all_data_multithread(root_path, num_proc=8):
    p = multiprocessing.Pool(processes=num_proc)
    result = Result()
    for data_folder in os.listdir(root_path):
        if not data_folder.startswith("2023-"):
            continue

        data_path = os.path.join(root_path, data_folder)

        for session_folder in os.listdir(data_path):
            if not session_folder.startswith("session_info"):
                continue

            session_path = os.path.join(data_path, session_folder)

            for log_name in os.listdir(session_path):
                if not log_name.endswith(".csv"):
                    continue
                if log_name.endswith("sim.csv"):
                    continue

                log_path = os.path.join(session_path, log_name)

                p.apply_async(
                    cal_single_para_result,
                    args=(
                        log_path,
                        buf_size,
                        proc_time_estimator_mode,
                        jitter_buffer_controller_mode,
                        timestamp_extrapolator_mode,
                        render_queue_mode,
                        strict_buffer,
                        render_ctrlable,
                        max_delay_frame_no,
                    ),
                    callback=result.update_result,
                )
                # p.apply_async(cal_adapt_buf_result, args=(log_path, min_buf_size, max_buf_size, buf_size_update_interval, target_fps, min_fps_gain,
                #     proc_time_estimator_mode, jitter_buffer_controller_mode, timestamp_extrapolator_mode, render_queue_mode, strict_buffer,
                #     render_ctrlable), callback=result.update_result)

    p.close()
    p.join()
    print(np.mean(result.res1), np.min(result.res1), np.max(result.res1))
    print(np.mean(result.res2), np.min(result.res2), np.max(result.res2))
    print(np.mean(result.res3), np.min(result.res3), np.max(result.res3))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        for line in open(r"test_data/unit_test.txt").readlines():
            if len(line) <= 1:
                break
            input_path = os.path.join("test_data", line.strip())
            # print(cal_single_para_result(input_path, buf_size = buf_size, proc_time_estimator_mode = proc_time_estimator_mode,
            # jitter_buffer_controller_mode = jitter_buffer_controller_mode, timestamp_extrapolator_mode = timestamp_extrapolator_mode,
            # render_queue_mode = render_queue_mode, strict_buffer=strict_buffer, render_ctrlable=render_ctrlable,
            # max_delay_frame_no = max_delay_frame_no, frame_interval = frame_interval, print_log = print_log))

            print(
                cal_adapt_buf_result(
                    input_path,
                    min_buf_size=min_buf_size,
                    max_buf_size=max_buf_size,
                    buf_size_update_interval=buf_size_update_interval,
                    target_fps=target_fps,
                    min_fps_gain=min_fps_gain,
                    proc_time_estimator_mode=proc_time_estimator_mode,
                    jitter_buffer_controller_mode=jitter_buffer_controller_mode,
                    timestamp_extrapolator_mode=timestamp_extrapolator_mode,
                    render_queue_mode=render_queue_mode,
                    strict_buffer=strict_buffer,
                    render_ctrlable=render_ctrlable,
                    frame_interval=frame_interval,
                    print_log=print_log,
                )
            )

        exit()

    input_path = sys.argv[1]
    if os.path.isdir(input_path):
        # process_all_data(sys.argv[1])
        process_all_data_multithread(sys.argv[1])
    elif os.path.isfile(input_path):
        with open("../../log/baseline.log", "a") as f:
            f.write(
                str(
                    cal_single_para_result(
                        input_path,
                        buf_size=buf_size,
                        proc_time_estimator_mode=proc_time_estimator_mode,
                        jitter_buffer_controller_mode=jitter_buffer_controller_mode,
                        timestamp_extrapolator_mode=timestamp_extrapolator_mode,
                        render_queue_mode=render_queue_mode,
                        strict_buffer=strict_buffer,
                        render_ctrlable=render_ctrlable,
                        max_delay_frame_no=max_delay_frame_no,
                        frame_interval=frame_interval,
                        print_log=False,
                    )
                )
                + "\n"
            )
            f.flush()

        # print(cal_single_para_result(input_path, buf_size = buf_size, proc_time_estimator_mode = proc_time_estimator_mode,
        # jitter_buffer_controller_mode = jitter_buffer_controller_mode, timestamp_extrapolator_mode = timestamp_extrapolator_mode,
        # render_queue_mode = render_queue_mode, strict_buffer=strict_buffer, render_ctrlable=render_ctrlable,
        # max_delay_frame_no = max_delay_frame_no, frame_interval = frame_interval, print_log = print_log))

        # print(cal_adapt_buf_result(input_path, min_buf_size = min_buf_size, max_buf_size = max_buf_size,
        # buf_size_update_interval = buf_size_update_interval, target_fps = target_fps, min_fps_gain = min_fps_gain,
        # proc_time_estimator_mode = proc_time_estimator_mode,
        # jitter_buffer_controller_mode = jitter_buffer_controller_mode, timestamp_extrapolator_mode = timestamp_extrapolator_mode,
        # render_queue_mode = render_queue_mode, strict_buffer=strict_buffer, render_ctrlable=render_ctrlable,
        # frame_interval = frame_interval, print_log = print_log))
    else:
        pass
