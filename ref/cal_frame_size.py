import os, sys

import numpy as np
import matplotlib.pyplot as plt

import load_data


def plot_cdf(datas, xlabel, ylabel, labels=None, xlim=None):
    fig = plt.figure()
    for idx, data in enumerate(datas):
        data = np.sort(data)
        p = 1.0 * np.arange(len(data)) / (len(data) - 1)
        if labels is not None:
            plt.plot(data, p, label=labels[idx])
        else:
            plt.plot(data, p)

    if xlim is not None:
        plt.xlim(xlim)
    if labels is not None:
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show()
    plt.grid()
    plt.savefig(xlabel + ".jpg", dpi=200)


#     0 'render_index', 'frame_index', 'frame_type', 'size', 'loss_type',
#     5 'receive_timestamp', 'receive_and_unpack',
#     7 'decoder_outside_queue', 'decoder_insided_queue', 'decode', 'render_queue', 'display',
#     12 'capture_timestamp', 'send_ts', 'net_ts', 'proc_ts', 'tot_ts',
#     17 'basic_net_ts', 'ul_jitter', 'dl_jitter'
#     20 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,vsync_diff
#     26 jitter_buf_size,server_optim_enabled,client_optim_enabled
#     29 pts, ets, dts, Mrts0ToRtsOffset, packet_lossed_perK
#     34 encoding_rate, cc_rate, smoothrate, width, height, sqoe, ori_sqoe, target_sqoe,
#     42 recomm_bitrate, actual_bitrate, scene_change, encoding_fps,
#     46 satd, qp, mvx, mvy, intra_mb, inter_mb, cur_cgs_pause_cnt
def analyze_single_log(
    file_path, frame_interval=16.666666667, window_lth=60, min_fps_thr=55
):
    file_name = os.path.basename(file_path)
    info = [int(item) for item in file_name.split("_")[-1].split(".")[0].split(",")]
    data, info = load_data.load_formated_e2e_framerate_log_with_netinfo(
        file_path, start_idx=0
    )  # sim for 20min

    if data is None:
        return [], [], []

    if np.any(data[:, 41]) > 0:
        return [], [], []

    data = data[data[:, 43] != 0]

    data = data[data[:, 3] / data[:, 43] / 2 > 0.25]

    # I frame
    i_frame_data = data[data[:, 2] == 2, :]
    i_size_ratio = i_frame_data[:, 3] / (
        i_frame_data[:, 43] * 1024 / i_frame_data[:, 45] / 8
    )

    # P frame
    p_frame_data = data[data[:, 2] == 1, :]
    p_size_ratio = p_frame_data[:, 3] / (
        p_frame_data[:, 43] * 1024 / p_frame_data[:, 45] / 8
    )

    no_bitrate_change_data = data[
        np.where(
            np.logical_and.reduce(
                (data[1:, 2] == 1, data[:-1, 2] == 1, data[1:, 43] == data[:-1, 43])
            )
        )[0]
        + 1,
        :,
    ]
    diff_ratio = (
        np.abs(no_bitrate_change_data[1:, 3] - no_bitrate_change_data[:-1, 3])
        / no_bitrate_change_data[1:, 43]
        / 2
    )

    # plot_cdf([p_size_ratio, i_size_ratio], xlabel='frame_size vs bitrate ratio', ylabel='CDF', labels=['P frame', 'I Frame'])
    return i_size_ratio.tolist(), p_size_ratio.tolist(), diff_ratio.tolist()


def analyze_all_log(root_path):
    i_size_ratios = []
    p_size_ratios = []
    diff_ratios = []

    cnt = 0
    for data_folder in os.listdir(root_path):
        if not data_folder.startswith("2024-"):
            continue

        data_path = os.path.join(root_path, data_folder)

        for log_name in os.listdir(data_path):
            if not log_name.endswith(".csv"):
                continue

            file_path = os.path.join(data_path, log_name)
            print(cnt, file_path)
            i_size_ratio, p_size_ratio, diff_ratio = analyze_single_log(file_path)

            if len(p_size_ratio) > 1000:
                cnt += 1

            i_size_ratios += i_size_ratio
            p_size_ratios += p_size_ratio
            diff_ratios += diff_ratio

            if cnt >= 100:
                break

    plot_cdf(
        [p_size_ratios, i_size_ratios],
        xlabel="frame_size to bitrate ratio",
        ylabel="CDF",
        labels=["P frame", "I Frame"],
        xlim=[0, 5],
    )
    plot_cdf(
        [diff_ratios],
        xlabel="consecutive_frame_size_diff to bitrate ratio",
        ylabel="CDF",
        labels=["P frame"],
        xlim=[0, 2],
    )
    print(len(p_size_ratios), len(i_size_ratios), len(diff_ratios))


if __name__ == "__main__":
    input_path = sys.argv[1]
    if os.path.isdir(input_path):
        analyze_all_log(sys.argv[1])
    elif os.path.isfile(input_path):
        pass
