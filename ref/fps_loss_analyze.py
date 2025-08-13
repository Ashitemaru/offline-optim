import os, sys
import linecache
import tracemalloc

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

import load_data

DRAW_WINDOW_FIGURE = False


def display_top(snapshot, key_type="lineno", limit=3):
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print(
            "#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def plot_cdf(
    datas,
    filename,
    xlabel,
    ylabel,
    output_folder="figures",
    title=None,
    labels=None,
    xlim=None,
):
    fig = plt.figure()
    for idx, data in enumerate(datas):
        data = np.sort(data)
        if data.size > 1:
            # if xlim is not None:
            #     xlim[1] = min(xlim[1], np.max(data))
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
    if title is None:
        plt.title(title)

    output_dir = os.path.join("test_data", output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename + ".jpg"), dpi=400)

    plt.close(fig)


def plot_histogram(data, filename, xlabel, title, output_folder="figures"):
    z_scores = stats.zscore(data)
    filtered_data = data[(z_scores > -2) & (z_scores < 2) & (data < 1000)]

    plt.figure()
    ax = sns.histplot(data=filtered_data, stat="probability", kde=True)
    ax.set(xlabel=xlabel, title=title)
    ax.grid(visible=True)
    ax.set_axisbelow(True)
    fig = ax.get_figure()

    output_dir = os.path.join("test_data", output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, filename + ".jpg"), dpi=400)

    plt.close(fig)


def cal_kl_div(array1, array2):
    # Determine the range for the histogram bins
    bins = np.histogram_bin_edges(np.concatenate((array1, array2)), bins="auto")

    # Create histograms and normalize them to get probability distributions
    hist1, _ = np.histogram(array1, bins=bins, density=True)
    hist2, _ = np.histogram(array2, bins=bins, density=True)

    # Normalize the histograms to ensure they sum to 1
    prob_dist1 = hist1 / hist1.sum()
    prob_dist2 = hist2 / hist2.sum()

    # Handle zero probabilities by adding a small constant (smoothing)
    epsilon = 1e-10
    prob_dist1 += epsilon
    prob_dist2 += epsilon

    # Renormalize after smoothing
    prob_dist1 /= prob_dist1.sum()
    prob_dist2 /= prob_dist2.sum()

    # Calculate the KL divergence
    kl_divergence = stats.entropy(prob_dist1, prob_dist2)

    return kl_divergence


def plot_multi_lines(
    datas, filename, xlabels, ylabels, output_folder="figures", title=None, flag=None
):
    subplot_no = len(datas)

    fig, axs = plt.subplots(
        subplot_no, 1, layout="constrained", figsize=(14, 3 * subplot_no)
    )

    for i in range(subplot_no):
        axs[i].scatter(range(datas[i].shape[0]), datas[i])
        axs[i].plot(datas[i])
        if flag is not None:
            axs[i].scatter(np.where(flag), datas[i][flag], color="r")
        axs[i].set_xlabel(xlabels[i])
        axs[i].set_ylabel(ylabels[i])
        axs[i].grid(True)

    if title is not None:
        fig.suptitle(title)

    output_dir = os.path.join("test_data", output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, filename + ".jpg"), dpi=400)

    plt.close(fig)


# 0 'render_index', 'frame_index', 'frame_type', 'size', 'loss_type',
# 5 'receive_timestamp', 'receive_and_unpack',
# 7 'decoder_outside_queue', 'decoder_insided_queue', 'decode', 'render_queue', 'display',
# 12 'capture_timestamp', 'send_ts', 'net_ts', 'proc_ts', 'tot_ts',
# 17 'basic_net_ts', 'ul_jitter', 'dl_jitter'
# 20 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,vsync_diff
# 26 jitter_buf_size,server_optim_enabled,client_optim_enabled
# 29 pts, ets, dts, Mrts0ToRtsOffset, packet_lossed_perK,
# 34 encoding_rate, cc_rate, smoothrate, width, height, sqoe, ori_sqoe, target_sqoe,
# 42 recomm_bitrate, actual_bitrate, scene_change, encoding_fps,
# 46 satd, qp, mvx, mvy, intra_mb, inter_mb, cur_cgs_pause_cnt
# 53 delay_time,valid_frame_flag
def analyze_single_log_with_optimized_fps_info(
    file_path,
    start_idx=0,
    sim_data_len=60 * 60,
    frame_interval=16.666666667,
    window_lth=60,
    min_good_fps_thr=55,
):
    file_name = os.path.basename(file_path)
    info = [int(item) for item in file_name.split("_")[-1].split(".")[0].split(",")]
    data, _ = load_data.load_formated_e2e_framerate_log_with_netinfo_and_flag(
        file_path, start_idx=start_idx, len_limit=sim_data_len
    )  # sim for 20min

    if data is None:
        return None

    # only simulate 60FPS traces
    render_interval = np.mean(data[1:, 29] - data[:-1, 29])
    # render_interval = np.mean((data[1:, 29] - data[:-1, 29]) / (data[1:, 52] - data[:-1, 52] + 1))
    if render_interval < 10 or render_interval > 30:
        print(file_path, start_idx, start_idx + sim_data_len)
        print("render_interval:", render_interval)
        print()
        return None

    data_lth = data.shape[0] // window_lth * window_lth
    data = data[:data_lth]

    frame_cgs_render_ts = data[:, 29].reshape(-1, window_lth)
    frame_cgs_encode_ts = data[:, 31].reshape(-1, window_lth)
    frame_proxy_recv_ts = data[:, 12].reshape(-1, window_lth)
    frame_client_rec_ts = data[:, 5].reshape(-1, window_lth)
    frame_client_dec_ts = data[:, 5:10].sum(-1).reshape(-1, window_lth)
    frame_display_ts = (data[:, 5:10].sum(-1) + data[:, 53]).reshape(-1, window_lth)
    cgs_pause_cnt = data[:, 52].reshape(-1, window_lth)

    def cal_window_fps(data, window_lth, refresh_slot_offset):
        frame_cgs_render_slot = np.ceil(
            (frame_cgs_render_ts - frame_cgs_render_ts[0, 0] - refresh_slot_offset)
            / frame_interval
        )
        frame_cgs_encode_slot = np.ceil(
            (frame_cgs_encode_ts - frame_cgs_encode_ts[0, 0] - refresh_slot_offset)
            / frame_interval
        )
        frame_proxy_recv_slot = np.ceil(
            (frame_proxy_recv_ts - frame_proxy_recv_ts[0, 0] - refresh_slot_offset)
            / frame_interval
        )
        frame_client_rec_slot = np.ceil(
            (frame_client_rec_ts - frame_client_rec_ts[0, 0] - refresh_slot_offset)
            / frame_interval
        )
        frame_client_dec_slot = np.ceil(
            (frame_client_dec_ts - frame_client_dec_ts[0, 0] - refresh_slot_offset)
            / frame_interval
        )

        frame_cgs_render_fps = (
            np.count_nonzero(
                np.diff(np.sort(frame_cgs_render_slot, axis=1), axis=1), axis=1
            )
            + 1
        )
        frame_cgs_encode_fps = (
            np.count_nonzero(
                np.diff(np.sort(frame_cgs_encode_slot, axis=1), axis=1), axis=1
            )
            + 1
        )
        frame_proxy_recv_fps = (
            np.count_nonzero(
                np.diff(np.sort(frame_proxy_recv_slot, axis=1), axis=1), axis=1
            )
            + 1
        )
        frame_client_rec_fps = (
            np.count_nonzero(
                np.diff(np.sort(frame_client_rec_slot, axis=1), axis=1), axis=1
            )
            + 1
        )
        frame_client_dec_fps = (
            np.count_nonzero(
                np.diff(np.sort(frame_client_dec_slot, axis=1), axis=1), axis=1
            )
            + 1
        )

        return (
            frame_cgs_render_fps,
            frame_cgs_encode_fps,
            frame_proxy_recv_fps,
            frame_client_rec_fps,
            frame_client_dec_fps,
        )

    # refresh_slot_offsets = [0]
    refresh_slot_offsets = [-6, -3, 0, 3, 6]

    cgs_render_fpses = []
    cgs_encode_fpses = []
    proxy_recv_fpses = []
    client_rec_fpses = []
    client_dec_fpses = []
    for slot_offset in refresh_slot_offsets:
        (
            cgs_render_fps_loss,
            cgs_encode_fps_loss,
            proxy_recv_fps_loss,
            client_rec_fps_loss,
            client_dec_fps_loss,
        ) = cal_window_fps(data, window_lth, slot_offset)

        cgs_render_fpses.append(cgs_render_fps_loss)
        cgs_encode_fpses.append(cgs_encode_fps_loss)
        proxy_recv_fpses.append(proxy_recv_fps_loss)
        client_rec_fpses.append(client_rec_fps_loss)
        client_dec_fpses.append(client_dec_fps_loss)

    max_fps = (
        (window_lth + cgs_pause_cnt[:, -1] - cgs_pause_cnt[:, 0])
        / (frame_cgs_render_ts[:, -1] - frame_cgs_render_ts[:, 0])
        * 1000
    )
    cgs_render_max_fps = (
        window_lth / (frame_cgs_render_ts[:, -1] - frame_cgs_render_ts[:, 0]) * 1000
    )
    cgs_render_fps = (
        np.array(cgs_render_fpses).max(0)
        / (frame_cgs_render_ts[:, -1] - frame_cgs_render_ts[:, 0])
        * 1000
    )
    cgs_encode_fps = (
        np.array(cgs_encode_fpses).max(0)
        / (frame_cgs_encode_ts[:, -1] - frame_cgs_encode_ts[:, 0])
        * 1000
    )
    proxy_recv_fps = (
        np.array(proxy_recv_fpses).max(0)
        / (frame_proxy_recv_ts[:, -1] - frame_proxy_recv_ts[:, 0])
        * 1000
    )
    client_rec_fps = (
        np.array(client_rec_fpses).max(0)
        / (frame_client_rec_ts[:, -1] - frame_client_rec_ts[:, 0])
        * 1000
    )
    client_dec_fps = (
        np.array(client_dec_fpses).max(0)
        / (frame_client_dec_ts[:, -1] - frame_client_dec_ts[:, 0])
        * 1000
    )
    optimized_fps = (
        data[:, 54].reshape(-1, window_lth).sum(-1)
        / (frame_display_ts[:, -1] - frame_display_ts[:, 0])
        * 1000
    )
    # print(cgs_render_fps_loss)
    # print(cgs_encode_fps_loss)
    # print(proxy_recv_fps_loss)
    # print(client_rec_fps_loss)
    # print(client_dec_fps_loss)

    big_fps_diff_thr = 5
    ignorable_fps_diff_thr = 1
    high_bitrate_thr = 40000
    low_bitrate_thr = 10000
    low_latency_thr = 10
    frame_stall_thr = 100

    big_cgs_render_fps_loss = np.greater_equal(
        cgs_render_max_fps - cgs_render_fps, big_fps_diff_thr
    )
    no_cgs_render_fps_loss = np.less_equal(
        cgs_render_max_fps - cgs_render_fps, ignorable_fps_diff_thr
    )
    big_cgs_encode_fps_loss = np.greater_equal(
        cgs_render_fps - cgs_encode_fps, big_fps_diff_thr
    )
    no_cgs_encode_fps_loss = np.less_equal(
        cgs_render_fps - cgs_encode_fps, ignorable_fps_diff_thr
    )
    big_proxy_recv_fps_loss = np.greater_equal(
        cgs_encode_fps - proxy_recv_fps, big_fps_diff_thr
    )
    no_proxy_recv_fps_loss = np.less_equal(
        cgs_encode_fps - proxy_recv_fps, ignorable_fps_diff_thr
    )
    big_client_rec_fps_loss = np.greater_equal(
        proxy_recv_fps - client_rec_fps, big_fps_diff_thr
    )
    no_client_rec_fps_loss = np.less_equal(
        proxy_recv_fps - client_rec_fps, ignorable_fps_diff_thr
    )
    big_client_dec_fps_loss = np.greater_equal(
        client_rec_fps - client_dec_fps, big_fps_diff_thr
    )

    # bitrate related flag
    high_bitrate_frame_flag = np.greater_equal(data[:, 43], high_bitrate_thr).reshape(
        -1, window_lth
    )
    high_bitrate_window_flag = high_bitrate_frame_flag.sum(-1) >= window_lth
    low_bitrate_frame_flag = np.less_equal(data[:, 43], low_bitrate_thr).reshape(
        -1, window_lth
    )
    low_bitrate_window_flag = low_bitrate_frame_flag.sum(-1) >= window_lth

    bitrate_change_window_flag = np.logical_or.reduce(
        (
            np.any(data[:, 2].reshape(-1, window_lth) == 2, axis=-1),
            np.any(data[:, 44].reshape(-1, window_lth) == 1, axis=-1),
            np.max(data[:, 43].reshape(-1, window_lth), axis=-1)
            - np.min(data[:, 43].reshape(-1, window_lth), axis=-1)
            >= 10000,
        )
    )

    # latency related flag
    min_rtt = np.min(data[:, 14])
    stable_min_latency_frame_flag = np.less_equal(data[:, 14] - min_rtt, 3).reshape(
        -1, window_lth
    )
    low_latency_frame_flag = np.less_equal(data[:, 14], low_latency_thr).reshape(
        -1, window_lth
    )
    unstable_latency_window_flag = np.logical_or.reduce(
        (
            np.max(data[:, 14].reshape(-1, window_lth), axis=-1)
            - np.min(data[:, 14].reshape(-1, window_lth), axis=-1)
            >= 10,
            np.std(data[:, 14].reshape(-1, window_lth), axis=-1) > 5,
        )
    )

    # loss related flag
    lossy_frame_flag = np.logical_and(
        data[:, 33] > 0, data[:, 14] >= 2 * min_rtt
    ).reshape(-1, window_lth)
    lossy_window_flag = lossy_frame_flag.sum(-1) >= 1
    # lossy_window_flag = lossy_frame_flag.sum(-1) >= window_lth // 10

    # jitter related flag
    stall_frame_flag = np.logical_and(
        data[:, 14] >= frame_stall_thr, data[:, 17] < frame_stall_thr
    ).reshape(-1, window_lth)
    stall_window_flag = stall_frame_flag.sum(-1) >= 1

    # network_induced_fps_loss = np.logical_and(no_proxy_recv_fps_loss, big_client_rec_fps_loss)
    network_induced_fps_loss = big_client_rec_fps_loss

    network_induced_fps_loss_with_bitrate_change = np.logical_and.reduce(
        (bitrate_change_window_flag, network_induced_fps_loss)
    )

    network_loss_induced_fps_loss = np.logical_and.reduce(
        (lossy_window_flag, network_induced_fps_loss)
    )

    network_unstable_induced_fps_loss = np.logical_and.reduce(
        (
            unstable_latency_window_flag,
            np.logical_not(lossy_window_flag),
            np.logical_not(stall_window_flag),
            network_induced_fps_loss,
        )
    )

    network_stall_induced_fps_loss = np.logical_and.reduce(
        (stall_window_flag, np.logical_not(lossy_window_flag), network_induced_fps_loss)
    )

    network_discard_induced_fps_loss = np.logical_and.reduce(
        (
            max_fps - cgs_render_max_fps > big_fps_diff_thr,
            cgs_render_fps < min_good_fps_thr,
        )
    )

    render_jitter_induced_fps_loss = np.logical_and.reduce(
        (big_cgs_render_fps_loss, cgs_render_fps < min_good_fps_thr)
    )

    origin_fps_above_thr = np.greater_equal(client_rec_fps, min_good_fps_thr)
    big_optimized_client_recv_fps_gain = np.logical_and.reduce(
        (network_induced_fps_loss, optimized_fps - client_rec_fps > big_fps_diff_thr)
    )
    big_optimized_fps_gain_and_above_thr = np.logical_and.reduce(
        (
            network_induced_fps_loss,
            big_optimized_client_recv_fps_gain,
            optimized_fps > min_good_fps_thr,
        )
    )

    print(file_path, start_idx, start_idx + sim_data_len)
    print(
        "cgs_render_max_fps: %.2f cgs_render_fps: %.2f cgs_encode_fps: %.2f proxy_recv_fps: %.2f client_rec_fps: %.2f client_dec_fps: %.2f"
        % (
            np.mean(cgs_render_max_fps),
            np.mean(cgs_render_fps),
            np.mean(cgs_encode_fps),
            np.mean(proxy_recv_fps),
            np.mean(client_rec_fps),
            np.mean(client_dec_fps),
        )
    )
    print(
        "window_no: %d above_thr_no: %d discard_loss_no: %d render_loss_no: %d network_fps_loss_no: %d big_fps_gain_no: %d big_gain_and_above_thr_no: %d"
        % (
            network_induced_fps_loss.size,
            origin_fps_above_thr.sum(),
            network_discard_induced_fps_loss.sum(),
            render_jitter_induced_fps_loss.sum(),
            network_induced_fps_loss.sum(),
            big_optimized_client_recv_fps_gain.sum(),
            big_optimized_fps_gain_and_above_thr.sum(),
        )
    )
    print(
        "loss_induced_no: %d big_fps_gain_no: %d big_gain_and_above_thr_no: %d"
        % (
            network_loss_induced_fps_loss.sum(),
            np.logical_and(
                network_loss_induced_fps_loss, big_optimized_client_recv_fps_gain
            ).sum(),
            np.logical_and.reduce(
                (
                    network_loss_induced_fps_loss,
                    big_optimized_client_recv_fps_gain,
                    big_optimized_fps_gain_and_above_thr,
                )
            ).sum(),
        )
    )
    print(
        "unstable_induced_no: %d big_fps_gain_no: %d big_gain_and_above_thr_no: %d"
        % (
            network_unstable_induced_fps_loss.sum(),
            np.logical_and(
                network_unstable_induced_fps_loss, big_optimized_client_recv_fps_gain
            ).sum(),
            np.logical_and.reduce(
                (
                    network_unstable_induced_fps_loss,
                    big_optimized_client_recv_fps_gain,
                    big_optimized_fps_gain_and_above_thr,
                )
            ).sum(),
        )
    )
    print(
        "stall_induced_no: %d big_fps_gain_no: %d big_gain_and_above_thr_no: %d"
        % (
            network_stall_induced_fps_loss.sum(),
            np.logical_and(
                network_stall_induced_fps_loss, big_optimized_client_recv_fps_gain
            ).sum(),
            np.logical_and.reduce(
                (
                    network_stall_induced_fps_loss,
                    big_optimized_client_recv_fps_gain,
                    big_optimized_fps_gain_and_above_thr,
                )
            ).sum(),
        )
    )
    print()

    # input()
    # unknown_idx = np.where(np.logical_and.reduce((
    #     np.logical_not(network_induced_fps_loss), np.logical_not(origin_fps_above_thr)
    #     # network_induced_fps_loss, big_optimized_client_recv_fps_gain,
    #     # np.logical_not(network_induced_fps_loss_with_bitrate_change), np.logical_not(network_loss_induced_fps_loss), np.logical_not(network_stall_induced_fps_loss)
    # )))[0]
    # # for idx in unknown_idx:
    # for idx in [52]:
    #     print('index: %d max_fps: %.2f render_fps: %.2f encode_fps: %.2f proxy_fps: %.2f recv_fps: %.2f dec_fps: %.2f optimized_fps: %.2f'
    #           %(idx, max_fps[idx], cgs_render_fps[idx], cgs_encode_fps[idx], proxy_recv_fps[idx], client_rec_fps[idx], client_dec_fps[idx], optimized_fps[idx]))
    #     print('loss_induced: %d jitter_induced: %d stall_induced: %d discard_induced: %d'
    #           %(network_loss_induced_fps_loss[idx], network_unstable_induced_fps_loss[idx],
    #             network_stall_induced_fps_loss[idx], network_discard_induced_fps_loss[idx]
    #         ))
    #     print('bitrate')
    #     print('\t'.join([str(int(item)) for item in data[:, 43].reshape(-1, window_lth)[idx]]))
    #     print('avg_rtt', '%.1f' %data[:, 14].reshape(-1, window_lth)[idx].mean())
    #     print('rtt')
    #     print('\t'.join([str(int(item)) for item in data[:, 14].reshape(-1, window_lth)[idx]]))
    #     print('min_rtt')
    #     print('\t'.join([str(int(item)) for item in data[:, 17].reshape(-1, window_lth)[idx]]))
    #     print('packet_time')
    #     print('\t'.join([str(int(item)) for item in data[:, 32].reshape(-1, window_lth)[idx]]))
    #     print('loss')
    #     print('\t'.join([str(int(item)) for item in data[:, 33].reshape(-1, window_lth)[idx]]))
    #     input()

    # t_cgs_interval = (data[1:, 29] - data[:-1, 29]) // (data[1:, 52] - data[:-1, 52] + 1)
    t_cgs_interval = data[1:, 29] - data[:-1, 29]
    t_encode_time = data[1:, 31] - data[1:, 30]
    t_network_time = data[1:, 14]
    t_decode_time = data[1:, 9]
    t_render_time = data[1:, 11]
    t_display_interval = data[1:, 5:10].sum(-1) - data[:-1, 5:10].sum(-1)

    if DRAW_WINDOW_FIGURE:
        plot_histogram(
            t_cgs_interval,
            "%d_%d_" % (start_idx, start_idx + sim_data_len) + "1_cgs_render_interval",
            "CGS render interval (ms)",
            title="CGS render valid FPS: %.2f $\\rightarrow$ %.2f"
            % (cgs_render_max_fps.mean(), cgs_render_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )
        plot_histogram(
            t_encode_time,
            "%d_%d_" % (start_idx, start_idx + sim_data_len) + "2_cgs_encoder_time",
            "CGS encode time (ms)",
            title="CGS encode valid FPS: %.2f $\\rightarrow$ %.2f"
            % (cgs_render_fps.mean(), cgs_encode_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )
        plot_histogram(
            t_network_time,
            "%d_%d_" % (start_idx, start_idx + sim_data_len) + "3_network_time",
            "Network time (ms)",
            title="Client recv valid FPS: %.2f $\\rightarrow$ %.2f"
            % (cgs_encode_fps.mean(), client_rec_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )
        plot_histogram(
            t_decode_time,
            "%d_%d_" % (start_idx, start_idx + sim_data_len) + "4_client_decode_time",
            "Client decode time (ms)",
            title="Client decode valid FPS: %.2f $\\rightarrow$ %.2f"
            % (client_rec_fps.mean(), client_dec_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )
        plot_histogram(
            t_render_time,
            "%d_%d_" % (start_idx, start_idx + sim_data_len) + "5_client_render_time",
            "Client render time (ms)",
            title="Client optimized valid FPS: %.2f $\\rightarrow$ %.2f"
            % (client_dec_fps.mean(), optimized_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )
        plot_histogram(
            t_display_interval,
            "%d_%d_" % (start_idx, start_idx + sim_data_len)
            + "6_client_display_interval",
            "Client display interval (ms)",
            title="Client optimized valid FPS: %.2f $\\rightarrow$ %.2f"
            % (client_dec_fps.mean(), optimized_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )

        plot_multi_lines(
            [
                t_cgs_interval,
                t_encode_time,
                t_network_time,
                t_decode_time,
                t_render_time,
                t_display_interval,
            ],
            "%d_%d_" % (start_idx, start_idx + sim_data_len) + "7_e2e_time",
            xlabels=[
                "CGS render valid FPS: %.2f $\\rightarrow$ %.2f"
                % (cgs_render_max_fps.mean(), cgs_render_fps.mean()),
                "CGS encode valid FPS: %.2f $\\rightarrow$ %.2f"
                % (cgs_render_fps.mean(), cgs_encode_fps.mean()),
                "Client recv valid FPS: %.2f $\\rightarrow$ %.2f"
                % (cgs_encode_fps.mean(), client_rec_fps.mean()),
                "Client decode valid FPS: %.2f $\\rightarrow$ %.2f"
                % (client_rec_fps.mean(), client_dec_fps.mean()),
                "Client optimized valid FPS: %.2f $\\rightarrow$ %.2f"
                % (client_dec_fps.mean(), optimized_fps.mean()),
                "Client optimized valid FPS: %.2f $\\rightarrow$ %.2f"
                % (client_dec_fps.mean(), optimized_fps.mean()),
            ],
            ylabels=[
                "CGS render interval (ms)",
                "CGS encoder time (ms)",
                "Network time (ms)",
                "Client decode time (ms)",
                "Client render time (ms)",
                "Client display interval (ms)",
            ],
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
            title="E2E Time Cost",
        )

    return (
        t_cgs_interval,
        t_encode_time,
        t_network_time,
        t_decode_time,
        t_render_time,
        t_display_interval,
    )


# 0 'render_index', 'frame_index', 'frame_type', 'size', 'loss_type',
# 5 'receive_timestamp', 'receive_and_unpack',
# 7 'decoder_outside_queue', 'decoder_insided_queue', 'decode', 'render_queue', 'display',
# 12 'capture_timestamp', 'send_ts', 'net_ts', 'proc_ts', 'tot_ts',
# 17 'basic_net_ts', 'ul_jitter', 'dl_jitter'
# 20 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,vsync_diff
# 26 jitter_buf_size,server_optim_enabled,client_optim_enabled
# 29 pts, ets, dts, Mrts0ToRtsOffset, packet_lossed_perK,
# 34 encoding_rate, cc_rate, smoothrate, width, height, sqoe, ori_sqoe, target_sqoe,
# 42 recomm_bitrate, actual_bitrate, scene_change, encoding_fps,
# 46 satd, qp, mvx, mvy, intra_mb, inter_mb, cur_cgs_pause_cnt
def analyze_single_log(
    file_path,
    raw_data,
    start_idx=0,
    sim_data_len=60 * 60,
    frame_interval=16.666666667,
    window_lth=60,
    min_good_fps_thr=55,
):
    file_name = os.path.basename(file_path)
    info = [int(item) for item in file_name.split("_")[-1].split(".")[0].split(",")]
    # data, _ = load_data.load_formated_e2e_framerate_log_with_netinfo(file_path, start_idx=start_idx, len_limit=sim_data_len) # sim for 20min
    data = raw_data[start_idx : start_idx + sim_data_len, :]

    if data is None:
        return None

    # only simulate 60FPS traces
    frame_render_interval = data[1:, 29] - data[:-1, 29]
    avg_render_interval = np.mean(frame_render_interval[frame_render_interval < 100])
    # render_interval = np.mean((data[1:, 29] - data[:-1, 29]) / (data[1:, 52] - data[:-1, 52] + 1))
    if avg_render_interval < 10 or avg_render_interval > 30:
        print(file_path, start_idx, start_idx + sim_data_len)
        print("avg_render_interval:", avg_render_interval)
        print()
        return None

    data_lth = data.shape[0] // window_lth * window_lth
    data = data[:data_lth]

    frame_cgs_render_ts = data[:, 29].reshape(-1, window_lth)
    frame_cgs_encode_ts = data[:, 31].reshape(-1, window_lth)
    frame_proxy_recv_ts = data[:, 12].reshape(-1, window_lth)
    frame_client_rec_ts = data[:, 5].reshape(-1, window_lth)
    frame_client_dec_ts = data[:, 5:10].sum(-1).reshape(-1, window_lth)
    cgs_pause_cnt = data[:, 52].reshape(-1, window_lth)

    def cal_window_fps(data, window_lth, refresh_slot_offset):
        frame_cgs_render_slot = np.ceil(
            (frame_cgs_render_ts - frame_cgs_render_ts[0, 0] - refresh_slot_offset)
            / frame_interval
        )
        frame_cgs_encode_slot = np.ceil(
            (frame_cgs_encode_ts - frame_cgs_encode_ts[0, 0] - refresh_slot_offset)
            / frame_interval
        )
        frame_proxy_recv_slot = np.ceil(
            (frame_proxy_recv_ts - frame_proxy_recv_ts[0, 0] - refresh_slot_offset)
            / frame_interval
        )
        frame_client_rec_slot = np.ceil(
            (frame_client_rec_ts - frame_client_rec_ts[0, 0] - refresh_slot_offset)
            / frame_interval
        )
        frame_client_dec_slot = np.ceil(
            (frame_client_dec_ts - frame_client_dec_ts[0, 0] - refresh_slot_offset)
            / frame_interval
        )

        frame_cgs_render_fps = (
            np.count_nonzero(
                np.diff(np.sort(frame_cgs_render_slot, axis=1), axis=1), axis=1
            )
            + 1
        )
        frame_cgs_encode_fps = (
            np.count_nonzero(
                np.diff(np.sort(frame_cgs_encode_slot, axis=1), axis=1), axis=1
            )
            + 1
        )
        frame_proxy_recv_fps = (
            np.count_nonzero(
                np.diff(np.sort(frame_proxy_recv_slot, axis=1), axis=1), axis=1
            )
            + 1
        )
        frame_client_rec_fps = (
            np.count_nonzero(
                np.diff(np.sort(frame_client_rec_slot, axis=1), axis=1), axis=1
            )
            + 1
        )
        frame_client_dec_fps = (
            np.count_nonzero(
                np.diff(np.sort(frame_client_dec_slot, axis=1), axis=1), axis=1
            )
            + 1
        )

        return (
            frame_cgs_render_fps,
            frame_cgs_encode_fps,
            frame_proxy_recv_fps,
            frame_client_rec_fps,
            frame_client_dec_fps,
        )

    # refresh_slot_offsets = [0]
    refresh_slot_offsets = [-6, -3, 0, 3, 6]

    cgs_render_fpses = []
    cgs_encode_fpses = []
    proxy_recv_fpses = []
    client_rec_fpses = []
    client_dec_fpses = []
    for slot_offset in refresh_slot_offsets:
        (
            cgs_render_fps_loss,
            cgs_encode_fps_loss,
            proxy_recv_fps_loss,
            client_rec_fps_loss,
            client_dec_fps_loss,
        ) = cal_window_fps(data, window_lth, slot_offset)

        cgs_render_fpses.append(cgs_render_fps_loss)
        cgs_encode_fpses.append(cgs_encode_fps_loss)
        proxy_recv_fpses.append(proxy_recv_fps_loss)
        client_rec_fpses.append(client_rec_fps_loss)
        client_dec_fpses.append(client_dec_fps_loss)

    max_fps = (
        (window_lth + cgs_pause_cnt[:, -1] - cgs_pause_cnt[:, 0])
        / (frame_cgs_render_ts[:, -1] - frame_cgs_render_ts[:, 0])
        * 1000
    )
    cgs_render_max_fps = (
        window_lth / (frame_cgs_render_ts[:, -1] - frame_cgs_render_ts[:, 0]) * 1000
    )
    cgs_render_fps = (
        np.array(cgs_render_fpses).max(0)
        / (frame_cgs_render_ts[:, -1] - frame_cgs_render_ts[:, 0])
        * 1000
    )
    cgs_encode_fps = (
        np.array(cgs_encode_fpses).max(0)
        / (frame_cgs_encode_ts[:, -1] - frame_cgs_encode_ts[:, 0])
        * 1000
    )
    proxy_recv_fps = (
        np.array(proxy_recv_fpses).max(0)
        / (frame_proxy_recv_ts[:, -1] - frame_proxy_recv_ts[:, 0])
        * 1000
    )
    client_rec_fps = (
        np.array(client_rec_fpses).max(0)
        / (frame_client_rec_ts[:, -1] - frame_client_rec_ts[:, 0])
        * 1000
    )
    client_dec_fps = (
        np.array(client_dec_fpses).max(0)
        / (frame_client_dec_ts[:, -1] - frame_client_dec_ts[:, 0])
        * 1000
    )
    # print(cgs_render_fps_loss)
    # print(cgs_encode_fps_loss)
    # print(proxy_recv_fps_loss)
    # print(client_rec_fps_loss)
    # print(client_dec_fps_loss)

    big_fps_diff_thr = 5
    ignorable_fps_diff_thr = 1
    high_bitrate_thr = 40000
    low_bitrate_thr = 10000
    low_latency_thr = 10
    frame_stall_thr = 100

    big_cgs_render_fps_loss = np.greater_equal(
        cgs_render_max_fps - cgs_render_fps, big_fps_diff_thr
    )
    no_cgs_render_fps_loss = np.less_equal(
        cgs_render_max_fps - cgs_render_fps, ignorable_fps_diff_thr
    )
    big_cgs_encode_fps_loss = np.greater_equal(
        cgs_render_fps - cgs_encode_fps, big_fps_diff_thr
    )
    no_cgs_encode_fps_loss = np.less_equal(
        cgs_render_fps - cgs_encode_fps, ignorable_fps_diff_thr
    )
    big_proxy_recv_fps_loss = np.greater_equal(
        cgs_encode_fps - proxy_recv_fps, big_fps_diff_thr
    )
    no_proxy_recv_fps_loss = np.less_equal(
        cgs_encode_fps - proxy_recv_fps, ignorable_fps_diff_thr
    )
    big_client_rec_fps_loss = np.greater_equal(
        proxy_recv_fps - client_rec_fps, big_fps_diff_thr
    )
    no_client_rec_fps_loss = np.less_equal(
        proxy_recv_fps - client_rec_fps, ignorable_fps_diff_thr
    )
    big_client_dec_fps_loss = np.greater_equal(
        client_rec_fps - client_dec_fps, big_fps_diff_thr
    )

    # bitrate related flag
    high_bitrate_frame_flag = np.greater_equal(data[:, 43], high_bitrate_thr).reshape(
        -1, window_lth
    )
    high_bitrate_window_flag = high_bitrate_frame_flag.sum(-1) >= window_lth
    low_bitrate_frame_flag = np.less_equal(data[:, 43], low_bitrate_thr).reshape(
        -1, window_lth
    )
    low_bitrate_window_flag = low_bitrate_frame_flag.sum(-1) >= window_lth

    bitrate_change_window_flag = np.logical_or.reduce(
        (
            np.any(data[:, 2].reshape(-1, window_lth) == 2, axis=-1),
            np.any(data[:, 44].reshape(-1, window_lth) == 1, axis=-1),
            np.max(data[:, 43].reshape(-1, window_lth), axis=-1)
            - np.min(data[:, 43].reshape(-1, window_lth), axis=-1)
            >= 10000,
        )
    )

    # latency related flag
    min_rtt = np.min(data[:, 14][data[:, 14] > 0])
    stable_min_latency_frame_flag = np.less_equal(data[:, 14] - min_rtt, 3).reshape(
        -1, window_lth
    )
    low_latency_frame_flag = np.less_equal(data[:, 14], low_latency_thr).reshape(
        -1, window_lth
    )
    unstable_latency_window_flag = np.logical_or.reduce(
        (
            np.max(data[:, 14].reshape(-1, window_lth), axis=-1)
            - np.min(data[:, 14].reshape(-1, window_lth), axis=-1)
            >= 10,
            np.std(data[:, 14].reshape(-1, window_lth), axis=-1) > 5,
        )
    )

    # loss related flag
    lossy_frame_flag = np.logical_and(
        data[:, 33] > 0, data[:, 14] >= 2 * min_rtt
    ).reshape(-1, window_lth)
    lossy_window_flag = lossy_frame_flag.sum(-1) >= 1
    # lossy_window_flag = lossy_frame_flag.sum(-1) >= window_lth // 10

    # jitter related flag
    stall_frame_flag = np.logical_and(
        data[:, 14] >= frame_stall_thr, data[:, 17] < frame_stall_thr
    ).reshape(-1, window_lth)
    stall_window_flag = stall_frame_flag.sum(-1) >= 1

    # network_induced_fps_loss = np.logical_and(no_proxy_recv_fps_loss, big_client_rec_fps_loss)
    network_induced_fps_loss = big_client_rec_fps_loss

    network_induced_fps_loss_with_bitrate_change = np.logical_and.reduce(
        (bitrate_change_window_flag, network_induced_fps_loss)
    )

    network_loss_induced_fps_loss = np.logical_and.reduce(
        (lossy_window_flag, network_induced_fps_loss)
    )

    network_unstable_induced_fps_loss = np.logical_and.reduce(
        (
            unstable_latency_window_flag,
            np.logical_not(lossy_window_flag),
            np.logical_not(stall_window_flag),
            network_induced_fps_loss,
        )
    )

    network_stall_induced_fps_loss = np.logical_and.reduce(
        (stall_window_flag, np.logical_not(lossy_window_flag), network_induced_fps_loss)
    )

    network_discard_induced_fps_loss = np.logical_and.reduce(
        (
            max_fps - cgs_render_max_fps > big_fps_diff_thr,
            cgs_render_fps < min_good_fps_thr,
        )
    )

    render_jitter_induced_fps_loss = np.logical_and.reduce(
        (big_cgs_render_fps_loss, cgs_render_fps < min_good_fps_thr)
    )

    origin_fps_above_thr = np.greater_equal(client_rec_fps, min_good_fps_thr)

    print(file_path, start_idx, start_idx + sim_data_len)
    print(
        "cgs_render_max_fps: %.2f cgs_render_fps: %.2f cgs_encode_fps: %.2f proxy_recv_fps: %.2f client_rec_fps: %.2f client_dec_fps: %.2f"
        % (
            np.mean(cgs_render_max_fps),
            np.mean(cgs_render_fps),
            np.mean(cgs_encode_fps),
            np.mean(proxy_recv_fps),
            np.mean(client_rec_fps),
            np.mean(client_dec_fps),
        )
    )
    print(
        "window_no: %d above_thr_no: %d discard_loss_no: %d render_loss_no: %d network_fps_loss_no: %d"
        % (
            network_induced_fps_loss.size,
            origin_fps_above_thr.sum(),
            network_discard_induced_fps_loss.sum(),
            render_jitter_induced_fps_loss.sum(),
            network_induced_fps_loss.sum(),
        )
    )
    print(
        "loss_induced_no: %d unstable_induced_no: %d stall_induced_no: %d"
        % (
            network_loss_induced_fps_loss.sum(),
            network_unstable_induced_fps_loss.sum(),
            network_stall_induced_fps_loss.sum(),
        )
    )
    print()

    # input()
    # unknown_idx = np.where(np.logical_and.reduce((
    #     np.logical_not(network_induced_fps_loss), np.logical_not(origin_fps_above_thr)
    #     # network_induced_fps_loss, big_optimized_client_recv_fps_gain,
    #     # np.logical_not(network_induced_fps_loss_with_bitrate_change), np.logical_not(network_loss_induced_fps_loss), np.logical_not(network_stall_induced_fps_loss)
    # )))[0]
    # # for idx in unknown_idx:
    # for idx in [52]:
    #     print('index: %d max_fps: %.2f render_fps: %.2f encode_fps: %.2f proxy_fps: %.2f recv_fps: %.2f dec_fps: %.2f optimized_fps: %.2f'
    #           %(idx, max_fps[idx], cgs_render_fps[idx], cgs_encode_fps[idx], proxy_recv_fps[idx], client_rec_fps[idx], client_dec_fps[idx], optimized_fps[idx]))
    #     print('loss_induced: %d jitter_induced: %d stall_induced: %d discard_induced: %d'
    #           %(network_loss_induced_fps_loss[idx], network_unstable_induced_fps_loss[idx],
    #             network_stall_induced_fps_loss[idx], network_discard_induced_fps_loss[idx]
    #         ))
    #     print('bitrate')
    #     print('\t'.join([str(int(item)) for item in data[:, 43].reshape(-1, window_lth)[idx]]))
    #     print('avg_rtt', '%.1f' %data[:, 14].reshape(-1, window_lth)[idx].mean())
    #     print('rtt')
    #     print('\t'.join([str(int(item)) for item in data[:, 14].reshape(-1, window_lth)[idx]]))
    #     print('min_rtt')
    #     print('\t'.join([str(int(item)) for item in data[:, 17].reshape(-1, window_lth)[idx]]))
    #     print('packet_time')
    #     print('\t'.join([str(int(item)) for item in data[:, 32].reshape(-1, window_lth)[idx]]))
    #     print('loss')
    #     print('\t'.join([str(int(item)) for item in data[:, 33].reshape(-1, window_lth)[idx]]))
    #     input()

    # t_cgs_interval = (data[1:, 29] - data[:-1, 29]) // (data[1:, 52] - data[:-1, 52] + 1)
    t_cgs_interval = data[1:, 29] - data[:-1, 29]
    t_encode_time = data[1:, 31] - data[1:, 30]
    t_network_time = data[1:, 17] / 2 + data[1:, 19]
    t_decode_time = data[1:, 9]
    t_render_time = data[1:, 11]
    t_display_interval = data[1:, 5:10].sum(-1) - data[:-1, 5:10].sum(-1)

    t_cgs_jitter = data[1:, 29][t_cgs_interval > 2 * frame_interval]
    t_cgs_jitter_interval = t_cgs_jitter[1:] - t_cgs_jitter[:-1]

    network_jitter_idx = np.where(np.logical_and(data[1:, 19] > 10, data[:-1, 19] <= 2))
    t_network_jitter = data[1:, 12][network_jitter_idx]
    t_network_jitter_interval = t_network_jitter[1:] - t_network_jitter[:-1]

    i_frame_cnt = np.sum(data[:, 2] == 2)
    i_frame_jitter_cnt = np.logical_and.reduce(
        (
            data[:, 2] == 2,
            data[:, 19] >= 10,
        )
    ).sum()
    p_frame_cnt = np.sum(data[:, 2] == 1)
    p_frame_jitter_cnt = np.logical_and.reduce(
        (
            data[:, 2] == 1,
            data[:, 19] >= 10,
        )
    ).sum()

    if DRAW_WINDOW_FIGURE:
        plot_histogram(
            t_cgs_interval,
            "%d_%d_" % (start_idx, start_idx + sim_data_len) + "1_cgs_render_interval",
            "CGS render interval (ms)",
            title="CGS render valid FPS: %.2f $\\rightarrow$ %.2f"
            % (cgs_render_max_fps.mean(), cgs_render_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )
        plot_histogram(
            t_encode_time,
            "%d_%d_" % (start_idx, start_idx + sim_data_len) + "2_cgs_encoder_time",
            "CGS encode time (ms)",
            title="CGS encode valid FPS: %.2f $\\rightarrow$ %.2f"
            % (cgs_render_fps.mean(), cgs_encode_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )
        plot_histogram(
            t_network_time,
            "%d_%d_" % (start_idx, start_idx + sim_data_len) + "3_network_time",
            "Network time (ms)",
            title="Client recv valid FPS: %.2f $\\rightarrow$ %.2f"
            % (cgs_encode_fps.mean(), client_rec_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )
        plot_histogram(
            t_decode_time,
            "%d_%d_" % (start_idx, start_idx + sim_data_len) + "4_client_decode_time",
            "Client decode time (ms)",
            title="Client decode valid FPS: %.2f $\\rightarrow$ %.2f"
            % (client_rec_fps.mean(), client_dec_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )
        plot_histogram(
            t_render_time,
            "%d_%d_" % (start_idx, start_idx + sim_data_len) + "5_client_render_time",
            "Client render time (ms)",
            title="Client optimized valid FPS: %.2f" % (client_dec_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )
        plot_histogram(
            t_display_interval,
            "%d_%d_" % (start_idx, start_idx + sim_data_len)
            + "6_client_display_interval",
            "Client display interval (ms)",
            title="Client optimized valid FPS: %.2f" % (client_dec_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )
        plot_cdf(
            [t_cgs_jitter_interval],
            "%d_%d_" % (start_idx, start_idx + sim_data_len) + "7_cgs_jitter_interval",
            "CGS jitter interval (ms)",
            "CDF",
            xlim=[0, 1000],
            title="CGS render valid FPS: %.2f $\\rightarrow$ %.2f"
            % (cgs_render_max_fps.mean(), cgs_render_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )
        plot_cdf(
            [t_network_jitter_interval],
            "%d_%d_" % (start_idx, start_idx + sim_data_len)
            + "8_network_jitter_interval",
            "Network jitter interval (ms)",
            "CDF",
            xlim=[0, 1000],
            title="Client recv valid FPS: %.2f $\\rightarrow$ %.2f"
            % (cgs_encode_fps.mean(), client_rec_fps.mean()),
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
        )

        plot_multi_lines(
            [
                t_cgs_interval,
                t_encode_time,
                t_network_time,
                t_decode_time,
                t_render_time,
                t_display_interval,
            ],
            "%d_%d_" % (start_idx, start_idx + sim_data_len) + "9_e2e_time",
            xlabels=[
                "CGS render valid FPS: %.2f $\\rightarrow$ %.2f"
                % (cgs_render_max_fps.mean(), cgs_render_fps.mean()),
                "CGS encode valid FPS: %.2f $\\rightarrow$ %.2f"
                % (cgs_render_fps.mean(), cgs_encode_fps.mean()),
                (
                    "Client recv valid FPS: %.2f $\\rightarrow$ %.2f I frame cnt: %d (%.2f%%) P frame cnt: %d (%.2f%%)"
                    % (
                        cgs_encode_fps.mean(),
                        client_rec_fps.mean(),
                        i_frame_cnt,
                        (
                            i_frame_jitter_cnt / i_frame_cnt * 100
                            if i_frame_cnt > 0
                            else 0
                        ),
                        p_frame_cnt,
                        p_frame_jitter_cnt / p_frame_cnt * 100,
                    )
                    if p_frame_cnt > 0
                    else 0
                ),
                "Client decode valid FPS: %.2f $\\rightarrow$ %.2f"
                % (client_rec_fps.mean(), client_dec_fps.mean()),
                "Client optimized valid FPS: %.2f" % (client_dec_fps.mean()),
                "Client optimized valid FPS: %.2f" % (client_dec_fps.mean()),
            ],
            ylabels=[
                "CGS render interval (ms)",
                "CGS encoder time (ms)",
                "Network time (ms)",
                "Client decode time (ms)",
                "Client render time (ms)",
                "Client display interval (ms)",
            ],
            output_folder="figures_%d/%s" % (sim_data_len, file_name[:-22]),
            title="E2E Time Cost",
            flag=data[1:, 2] == 2,
        )

    del data
    return (
        t_cgs_interval,
        t_encode_time,
        t_network_time,
        t_decode_time,
        t_render_time,
        t_display_interval,
        t_cgs_jitter_interval,
        t_network_jitter_interval,
        i_frame_cnt,
        i_frame_jitter_cnt,
        p_frame_cnt,
        p_frame_jitter_cnt,
        np.mean(cgs_render_max_fps),
        np.mean(cgs_render_fps),
        np.mean(cgs_encode_fps),
        np.mean(proxy_recv_fps),
        np.mean(client_rec_fps),
        np.mean(client_dec_fps),
        network_induced_fps_loss.size,
        origin_fps_above_thr.sum(),
        network_discard_induced_fps_loss.sum(),
        render_jitter_induced_fps_loss.sum(),
        network_induced_fps_loss.sum(),
        network_loss_induced_fps_loss.sum(),
        network_unstable_induced_fps_loss.sum(),
        network_stall_induced_fps_loss.sum(),
    )


def analyze_single_log_by_window(
    file_path, tot_data_len=60 * 60 * 20, sim_data_len=60 * 60, start_idx=2400
):
    file_name = os.path.basename(file_path)
    info = [int(item) for item in file_name.split("_")[-1].split(".")[0].split(",")]
    data, _ = load_data.load_formated_e2e_framerate_log_with_netinfo(
        file_path, start_idx=start_idx, len_limit=tot_data_len
    )

    if data is None:
        print("None data")
        return None, None

    if data.shape[0] < 3600:
        print("length too small: %d" % data.shape[0])
        return None, None

    # only simulate 60FPS traces
    render_interval = np.mean(data[1:, 29] - data[:-1, 29])
    if render_interval < 10 or render_interval > 25:
        print("render_interval:", render_interval)
        return None, None

    tot_data_len = data.shape[0]
    sim_window_cnt = tot_data_len // sim_data_len

    kl_divs = [[] for i in range(6)]
    adf_pvalues = [[] for i in range(6)]
    kpss_pvalues = [[] for i in range(6)]
    intervals = [[] for i in range(2)]
    prev_window_ts = None
    for i in range(sim_window_cnt):
        cur_window_ts = analyze_single_log(
            file_path,
            data,
            start_idx=i * sim_data_len,
            sim_data_len=sim_data_len,
            window_lth=min(sim_data_len, 60),
        )

        if prev_window_ts is not None and cur_window_ts is not None:
            for i in range(6):
                kl_divs[i].append(cal_kl_div(prev_window_ts[i], cur_window_ts[i]))
                adf_pvalues[i].append(adfuller(cur_window_ts[i], regression="ct")[1])
                kpss_pvalues[i].append(kpss(cur_window_ts[i], regression="ct")[1])

            for i in range(2):
                intervals[i].append(cur_window_ts[6 + i])

        prev_window_ts = cur_window_ts

    plot_cdf(
        kl_divs,
        "kl_div_%d_" % sim_data_len + file_name[:-22],
        "KL Divergence",
        "CDF",
        labels=[
            "cgs_render_interval",
            "cgs_encode_time",
            "network_time",
            "decode_time",
            "client_render_time",
            "client_display_interval",
        ],
        output_folder="figures_%d" % sim_data_len,
    )
    plot_cdf(
        adf_pvalues,
        "adf_pvalues_%d_" % sim_data_len + file_name[:-22],
        "ADF Test P-values",
        "CDF",
        labels=[
            "cgs_render_interval",
            "cgs_encode_time",
            "network_time",
            "decode_time",
            "client_render_time",
            "client_display_interval",
        ],
        output_folder="figures_%d" % sim_data_len,
        xlim=[0, 0.25],
    )
    plot_cdf(
        kpss_pvalues,
        "kpss_pvalues_%d_" % sim_data_len + file_name[:-22],
        "KPSS Test P-values",
        "CDF",
        labels=[
            "cgs_render_interval",
            "cgs_encode_time",
            "network_time",
            "decode_time",
            "client_render_time",
            "client_display_interval",
        ],
        output_folder="figures_%d" % sim_data_len,
        xlim=[0, 0.25],
    )

    plot_histogram(
        np.concatenate(intervals[0]),
        "cgs_jitter_interval_hist_%d_" % sim_data_len + file_name[:-22],
        "CGS jitter interval (ms)",
        title="CGS jitter interval",
        output_folder="figures_%d" % sim_data_len,
    )
    plot_histogram(
        np.concatenate(intervals[1]),
        "network_jitter_interval_hist_%d_" % sim_data_len + file_name[:-22],
        "Network jitter interval (ms)",
        title="Network jitter interval",
        output_folder="figures_%d" % sim_data_len,
    )
    plot_cdf(
        [np.concatenate(intervals[0])],
        "cgs_jitter_interval_cdf_%d_" % sim_data_len + file_name[:-22],
        "CGS jitter interval (ms)",
        "CDF",
        title="CGS jitter interval",
        output_folder="figures_%d" % sim_data_len,
        xlim=[0, 1000],
    )
    plot_cdf(
        [np.concatenate(intervals[1])],
        "network_jitter_interval_cdf_%d_" % sim_data_len + file_name[:-22],
        "Network jitter interval (ms)",
        "CDF",
        title="Network jitter interval",
        output_folder="figures_%d" % sim_data_len,
        xlim=[0, 1000],
    )


def cal_log_stats(root_path, tot_data_len=60 * 60 * 30, start_idx=2400):
    output_file = open(os.path.join(root_path, "result.csv"), "w")
    output_file.write(
        "filepath,i_frame_cnt,i_frame_jitter_cnt,p_frame_cnt,p_frame_jitter_cnt,cgs_render_max_fps,cgs_render_fps,cgs_encode_fps,proxy_recv_fps,client_rec_fps,client_dec_fps,window_no,above_thr_no,discard_loss_no,render_loss_no,network_fps_loss_no,loss_induced_no,unstable_induced_no,stall_induced_no\n"
    )
    # cgs_render_interval,cgs_encode_time,network_time,client_decode_time,client_render_time,client_display_interval,cgs_jitter_interval,network_jitter_interval,

    for data_folder in os.listdir(root_path):
        if not data_folder.startswith("2024-"):
            continue

        data_path = os.path.join(root_path, data_folder)
        for log_name in os.listdir(data_path):
            if not log_name.endswith(".csv"):
                continue

            file_path = os.path.join(data_path, log_name)
            file_name = os.path.basename(file_path)
            info = [
                int(item) for item in file_name.split("_")[-1].split(".")[0].split(",")
            ]
            data, _ = load_data.load_formated_e2e_framerate_log_with_netinfo(
                file_path, start_idx=start_idx, len_limit=tot_data_len
            )

            if data is None:
                print("None data")

            if data.shape[0] < 3600:
                print("length too small: %d" % data.shape[0])

            # only simulate 60FPS traces
            frame_render_interval = data[1:, 29] - data[:-1, 29]
            avg_render_interval = np.mean(
                frame_render_interval[frame_render_interval < 200]
            )
            # render_interval = np.mean((data[1:, 29] - data[:-1, 29]) / (data[1:, 52] - data[:-1, 52] + 1))
            if avg_render_interval < 12 or avg_render_interval > 22:
                print(file_path, "avg_render_interval:", avg_render_interval)
                print()

            cur_window_ts = analyze_single_log(
                file_path, data, start_idx=0, sim_data_len=tot_data_len, window_lth=60
            )
            if cur_window_ts is not None:
                output_file.write(
                    file_path.replace(",", "_")
                    + ", "
                    + ", ".join([str(item) for item in cur_window_ts[8:]])
                    + "\n"
                )

    output_file.close()


if __name__ == "__main__":
    sim_data_lens = [600]
    sim_data_nos = [120]
    if len(sys.argv) == 1:
        # tracemalloc.start()
        for sim_data_len in sim_data_lens:
            for sim_data_no in sim_data_nos:
                # analyze_single_log_by_window('test_data/11.177.33.31_2024-04-14_2000607348_1,0,2,3,1,1,0,1,0.csv', tot_data_len=sim_data_len*sim_data_no, sim_data_len=sim_data_len, start_idx=0) # cgs render jitter
                analyze_single_log_by_window(
                    "test_data/11.177.33.27_2024-04-14_2000625488_2,0,3,2,1,1,0,1,0.csv",
                    tot_data_len=sim_data_len * sim_data_no,
                    sim_data_len=sim_data_len,
                    start_idx=4200,
                )  # cgs render jitter
                # analyze_single_log_by_window('test_data/11.177.33.17_2024-04-15_2000646093_1,0,2,3,0,0,0,1,0.csv', tot_data_len=sim_data_len*sim_data_no, sim_data_len=sim_data_len, start_idx=0) # network jitter

                # analyze_single_log_by_window('test_data/11.177.33.27_2024-04-15_2000647151_1,0,2,3,1,1,0,1,0.csv', tot_data_len=sim_data_len*sim_data_no, sim_data_len=sim_data_len, start_idx=2400) # client decoder jitter

        # snapshot = tracemalloc.take_snapshot()
        # display_top(snapshot)
        # top_stats = snapshot.statistics('traceback')
        # # pick the biggest memory block
        # stat = top_stats[0]
        # print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
        # for line in stat.traceback.format():
        #     print(line)

    else:
        input_path = sys.argv[1]
        if os.path.isdir(input_path):
            cal_log_stats(sys.argv[1])
        elif os.path.isfile(input_path):
            for sim_data_len in sim_data_lens:
                for sim_data_no in sim_data_nos:
                    analyze_single_log_by_window(
                        sys.argv[1],
                        tot_data_len=sim_data_len * sim_data_no,
                        sim_data_len=sim_data_len,
                        start_idx=0,
                    )
        else:
            pass
