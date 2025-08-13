import os
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

stall_thr = 0.001
min_fps_thr = 55
max_fps_thr = 65
render_problem_thr = 1 / 60


def plot_histogram(x, y, xlabel, ylabel, postfix="", output_dir="test_data/figures"):
    cnt = x.size

    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist2d(x, y, bins=100, range=np.array([[0, 3], [0, 3]]))

    plt.xlabel(xlabel + " (flow_cnt: %d)" % cnt)
    plt.ylabel(ylabel)
    # plt.show()
    # plt.grid()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, "Fig " + xlabel + postfix + ".jpg"), dpi=200)
    plt.close()


def plot_cdf(
    datas,
    xlabel,
    ylabel,
    labels=None,
    xlim=None,
    postfix="",
    output_dir="test_data/figures",
    x_logscale=False,
):
    fig = plt.figure(tight_layout=True)
    cnt = 0
    for idx, data in enumerate(datas):
        cnt = max(cnt, data.size)
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

    if x_logscale:
        plt.xscale("log")

    plt.xlabel(xlabel + " (flow_cnt: %d)" % cnt)
    plt.ylabel(ylabel)
    # plt.show()
    plt.grid()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, "Fig " + xlabel + postfix + ".jpg"), dpi=100)
    plt.close()


def plot_result(file_paths, buf_size):
    res_file = open(file_paths[0])

    max_fps_res = []
    min_fps_res = []

    opted_fps = [[] for i in range(len(file_paths))]
    gains = [[] for i in range(len(file_paths))]
    overheads = [[] for i in range(len(file_paths))]

    # opt_fps_res = []
    # opt1_fps_res = []
    # overhead1 = []
    # overhead2 = []
    # gain1 = []
    # gain2 = []

    buffer_gain = []
    rec_fps = {}

    for line in res_file.readlines():
        items = line.split(",")
        filename = items[0]
        items = [float(i.strip()) for i in items[1:]]
        max_fps, min_fps, opt_fps, gain, overhead = (
            items[0],
            items[1],
            items[3],
            items[4],
            items[5],
        )

        if overhead < (buf_size[0] + 0.5) * 17:
            max_fps_res.append(max_fps)
            min_fps_res.append(min_fps)
            # opt_fps_res.append(opt_fps)

            opted_fps[0].append(opt_fps)
            gains[0].append(gain)
            overheads[0].append(overhead)

            rec_fps[filename] = opt_fps

    for i in range(1, len(file_paths)):
        res_file = open(file_paths[i])
        for line in res_file.readlines():
            items = line.split(",")
            filename = items[0]
            items = [float(i.strip()) for i in items[1:]]
            opt_fps, gain, overhead = items[3], items[4], items[5]

            if overhead < (buf_size[1] + 0.5) * 17:
                opted_fps[i].append(opt_fps)
                gains[i].append(gain)
                overheads[i].append(overhead)

                if filename in rec_fps:
                    buffer_gain.append(opt_fps - rec_fps[filename])

    plot_cdf(
        [max_fps_res, min_fps_res] + opted_fps,
        "FPS",
        "CDF",
        labels=[
            "Max FPS",
            "Current FPS",
            "1 Frame buffer FPS",
            "1 Frame buffer FPS Rend_Ctrlable",
        ],
        xlim=[35, 60],
    )
    plot_cdf(
        [*gains, buffer_gain],
        "Frame-rate Gain",
        "CDF",
        labels=["1 Frame buffer", "1 Frame buffer Rend_Ctrlable", "Marginal gain"],
    )
    # plot_cdf(overheads, 'Overhead (ms)', 'CDF', labels=['Rectified', '1 Frame buffer', '2 Frame buffer'])


def plot_adapt_buf_result(file_path):
    res_file = open(file_path)
    max_fps_res = []
    min_fps_res = []

    opted_fps = [[] for i in range(3)]
    gains = [[] for i in range(2)]
    overheads = [[] for i in range(3)]
    top_tears = [[] for i in range(5)]
    top_tear_ratio = [[] for i in range(3)]
    top_tear_window1 = [[] for i in range(4)]
    top_tear_window3 = [[] for i in range(4)]
    top_tear_window6 = [[] for i in range(4)]
    fps_loss = [[] for i in range(3)]

    for line in res_file.readlines():
        items = line.split(",")
        items = [float(i.strip()) for i in items[1:]]
        (
            max_fps,
            min_fps,
            opt_fps,
            p_1more_fps,
            p_1less_fps,
            a_1more_fps,
            a_1less_fps,
            overhead,
            overhead_1more,
            overhead_1less,
            avg_dec_time,
            avg_dec_total_time,
            avg_render_time,
            avg_proc_time,
            recv_interval,
            middle_tear_fps,
            top_tear_fps,
            max_top_tear_fps,
            max95_top_tear_fps,
            max80_top_tear_fps,
            max50_top_tear_fps,
            min_top_tear_fps,
            mean_top_tear_fps,
            top_tear_fps_over1_ratio,
            top_tear_fps_over3_ratio,
            top_tear_fps_over6_ratio,
            per_window_1frame_tear_95,
            per_window_1frame_tear_80,
            per_window_1frame_tear_50,
            per_window_1frame_tear_mean,
            per_window_3frame_tear_95,
            per_window_3frame_tear_80,
            per_window_3frame_tear_50,
            per_window_3frame_tear_mean,
            per_window_6frame_tear_95,
            per_window_6frame_tear_80,
            per_window_6frame_tear_50,
            per_window_6frame_tear_mean,
        ) = (
            items[0],
            items[1],
            items[2],
            items[3],
            items[4],
            items[5],
            items[6],
            items[7],
            items[8],
            items[9],
            items[10],
            items[11],
            items[12],
            items[13],
            items[14],
            items[15],
            items[16],
            items[17],
            items[18],
            items[19],
            items[20],
            items[21],
            items[22],
            items[23],
            items[24],
            items[25],
            items[26],
            items[27],
            items[28],
            items[29],
            items[30],
            items[31],
            items[32],
            items[33],
            items[34],
            items[35],
            items[36],
            items[37],
        )

        if opt_fps > min_fps and max_fps <= 60 and avg_render_time <= 5:
            max_fps_res.append(max_fps)
            min_fps_res.append(min_fps)

            opted_fps[0].append(opt_fps)
            opted_fps[1].append(a_1more_fps)
            opted_fps[2].append(a_1less_fps)

            # gains[0].append(abs(min(p_1more_fps, max_fps) - a_1more_fps))
            # gains[1].append(abs(max(p_1less_fps, min_fps) - a_1less_fps))
            # gains[0].append(min(p_1more_fps, max_fps) - a_1more_fps)
            # gains[1].append(max(p_1less_fps, min_fps) - a_1less_fps)
            gains[0].append(p_1more_fps - a_1more_fps)
            gains[1].append(p_1less_fps - a_1less_fps)

            overheads[0].append(overhead)
            overheads[1].append(overhead_1more)
            overheads[2].append(overhead_1less)

            top_tears[0].append(max_top_tear_fps)
            top_tears[1].append(max95_top_tear_fps)
            top_tears[2].append(max80_top_tear_fps)
            top_tears[3].append(max50_top_tear_fps)
            top_tears[4].append(top_tear_fps)

            top_tear_ratio[0].append(top_tear_fps_over1_ratio)
            top_tear_ratio[1].append(top_tear_fps_over3_ratio)
            top_tear_ratio[2].append(top_tear_fps_over6_ratio)

            top_tear_window1[0].append(per_window_1frame_tear_95)
            top_tear_window1[1].append(per_window_1frame_tear_80)
            top_tear_window1[2].append(per_window_1frame_tear_50)
            top_tear_window1[3].append(per_window_1frame_tear_mean)

            top_tear_window3[0].append(per_window_3frame_tear_95)
            top_tear_window3[1].append(per_window_3frame_tear_80)
            top_tear_window3[2].append(per_window_3frame_tear_50)
            top_tear_window3[3].append(per_window_3frame_tear_mean)

            top_tear_window6[0].append(per_window_6frame_tear_95)
            top_tear_window6[1].append(per_window_6frame_tear_80)
            top_tear_window6[2].append(per_window_6frame_tear_50)
            top_tear_window6[3].append(per_window_6frame_tear_mean)

            fps_loss[0].append((max_fps - min_fps) / max_fps * 100)
            fps_loss[1].append((max_fps - opt_fps) / max_fps * 100)
            fps_loss[2].append((max_fps - a_1more_fps) / max_fps * 100)

    plot_cdf(
        [max_fps_res, min_fps_res] + opted_fps,
        "FPS",
        "CDF",
        labels=["Max FPS", "Current FPS", "1Buf FPS", "2Buf FPS", "0Buf FPS"],
        xlim=[35, 60],
    )
    plot_cdf(gains, "Error", "CDF", labels=["1MoreBuf", "1LessBuf"])
    plot_cdf(overheads, "Overhead (ms)", "CDF", labels=["1Buf", "2Buf", "0Buf"])
    plot_cdf(
        top_tears,
        "Teared FPS",
        "CDF",
        labels=["Max", "95Pct.", "80Pct.", "50Pct.", "Mean"],
    )
    plot_cdf(
        top_tear_ratio,
        "Teared Ratio",
        "CDF",
        labels=["Over 1FPS", "Over 3FPS", "Over 6FPS"],
    )
    plot_cdf(
        top_tear_window1,
        "1Teared Second Per Window",
        "CDF",
        labels=["95Pct.", "80Pct.", "50Pct.", "Mean"],
    )
    plot_cdf(
        top_tear_window3,
        "3Teared Second Per Window",
        "CDF",
        labels=["95Pct.", "80Pct.", "50Pct.", "Mean"],
    )
    plot_cdf(
        top_tear_window6,
        "6Teared Second Per Window",
        "CDF",
        labels=["95Pct.", "80Pct.", "50Pct.", "Mean"],
    )
    plot_cdf(fps_loss, "FPS Loss", "CDF", labels=["Current", "1Buf", "2Buf"])


def plot_comparison_result(file_paths):

    all_fps = []
    for file_path in file_paths:
        res_file = open(file_path)
        max_fps_res = []
        min_fps_res = []

        opted_fps = [[] for i in range(3)]
        gains = [[] for i in range(2)]
        overheads = [[] for i in range(3)]
        top_tears = [[] for i in range(5)]
        top_tear_ratio = [[] for i in range(3)]
        top_tear_window1 = [[] for i in range(4)]
        top_tear_window3 = [[] for i in range(4)]
        top_tear_window6 = [[] for i in range(4)]

        for line in res_file.readlines():
            items = line.split(",")
            items = [float(i.strip()) for i in items[1:]]
            (
                max_fps,
                min_fps,
                opt_fps,
                p_1more_fps,
                p_1less_fps,
                a_1more_fps,
                a_1less_fps,
                overhead,
                overhead_1more,
                overhead_1less,
                avg_dec_time,
                avg_dec_total_time,
                avg_render_time,
                avg_proc_time,
                recv_interval,
                middle_tear_fps,
                top_tear_fps,
                max_top_tear_fps,
                max95_top_tear_fps,
                max80_top_tear_fps,
                max50_top_tear_fps,
                min_top_tear_fps,
                mean_top_tear_fps,
                top_tear_fps_over1_ratio,
                top_tear_fps_over3_ratio,
                top_tear_fps_over6_ratio,
                per_window_1frame_tear_95,
                per_window_1frame_tear_80,
                per_window_1frame_tear_50,
                per_window_1frame_tear_mean,
                per_window_3frame_tear_95,
                per_window_3frame_tear_80,
                per_window_3frame_tear_50,
                per_window_3frame_tear_mean,
                per_window_6frame_tear_95,
                per_window_6frame_tear_80,
                per_window_6frame_tear_50,
                per_window_6frame_tear_mean,
            ) = (
                items[0],
                items[1],
                items[2],
                items[3],
                items[4],
                items[5],
                items[6],
                items[7],
                items[8],
                items[9],
                items[10],
                items[11],
                items[12],
                items[13],
                items[14],
                items[15],
                items[16],
                items[17],
                items[18],
                items[19],
                items[20],
                items[21],
                items[22],
                items[23],
                items[24],
                items[25],
                items[26],
                items[27],
                items[28],
                items[29],
                items[30],
                items[31],
                items[32],
                items[33],
                items[34],
                items[35],
                items[36],
                items[37],
            )

            if opt_fps > min_fps and max_fps <= 60 and avg_render_time <= 5:
                max_fps_res.append(max_fps)
                min_fps_res.append(min_fps)

                opted_fps[0].append(opt_fps)
                opted_fps[1].append(a_1more_fps)
                opted_fps[2].append(a_1less_fps)

                # gains[0].append(abs(min(p_1more_fps, max_fps) - a_1more_fps))
                # gains[1].append(abs(max(p_1less_fps, min_fps) - a_1less_fps))
                # gains[0].append(min(p_1more_fps, max_fps) - a_1more_fps)
                # gains[1].append(max(p_1less_fps, min_fps) - a_1less_fps)
                gains[0].append(p_1more_fps - a_1more_fps)
                gains[1].append(p_1less_fps - a_1less_fps)

                overheads[0].append(overhead)
                overheads[1].append(overhead_1more)
                overheads[2].append(overhead_1less)

                top_tears[0].append(max_top_tear_fps)
                top_tears[1].append(max95_top_tear_fps)
                top_tears[2].append(max80_top_tear_fps)
                top_tears[3].append(max50_top_tear_fps)
                top_tears[4].append(top_tear_fps)

                top_tear_ratio[0].append(top_tear_fps_over1_ratio)
                top_tear_ratio[1].append(top_tear_fps_over3_ratio)
                top_tear_ratio[2].append(top_tear_fps_over6_ratio)

                top_tear_window1[0].append(per_window_1frame_tear_95)
                top_tear_window1[1].append(per_window_1frame_tear_80)
                top_tear_window1[2].append(per_window_1frame_tear_50)
                top_tear_window1[3].append(per_window_1frame_tear_mean)

                top_tear_window3[0].append(per_window_3frame_tear_95)
                top_tear_window3[1].append(per_window_3frame_tear_80)
                top_tear_window3[2].append(per_window_3frame_tear_50)
                top_tear_window3[3].append(per_window_3frame_tear_mean)

                top_tear_window6[0].append(per_window_6frame_tear_95)
                top_tear_window6[1].append(per_window_6frame_tear_80)
                top_tear_window6[2].append(per_window_6frame_tear_50)
                top_tear_window6[3].append(per_window_6frame_tear_mean)

        all_fps.append(opted_fps[0])
    plot_cdf(
        all_fps,
        "FPS",
        "CDF",
        labels=["Baseline", "New"],
        xlim=[35, 60],
        postfix="_comparison",
    )
    # plot_cdf(gains, 'Error', 'CDF', labels=['1MoreBuf', '1LessBuf'])
    # plot_cdf(overheads, 'Overhead (ms)', 'CDF', labels=['1Buf', '2Buf', '0Buf'])
    # plot_cdf(top_tears, 'Teared FPS', 'CDF', labels=['Max', '95Pct.', '80Pct.', '50Pct.', 'Mean'])
    # plot_cdf(top_tear_ratio, 'Teared Ratio', 'CDF', labels=['Over 1FPS', 'Over 3FPS', 'Over 6FPS'])
    # plot_cdf(top_tear_window1, '1Teared Second Per Window', 'CDF', labels=['95Pct.', '80Pct.', '50Pct.', 'Mean'])
    # plot_cdf(top_tear_window3, '3Teared Second Per Window', 'CDF', labels=['95Pct.', '80Pct.', '50Pct.', 'Mean'])
    # plot_cdf(top_tear_window6, '6Teared Second Per Window', 'CDF', labels=['95Pct.', '80Pct.', '50Pct.', 'Mean'])


def plot_multi_param_results(log_path):
    data = pd.read_csv(log_path)

    results = data.iloc[:, 1:].to_numpy()
    # valid_idx = np.where(np.logical_and.reduce((results[:,2]==1, results[:,5]<=10, results[:, 7] > 50)))[0]
    valid_idx = np.where(
        np.logical_and.reduce((results[:, 5] <= 10, results[:, 7] >= min_fps_thr))
    )[0]
    print(results.shape[0], np.sum(results[:, 2] == 1), valid_idx.shape[0])
    vsync_results = results[valid_idx]

    valid_fps = [
        vsync_results[:, 9],
        vsync_results[:, 11],
        vsync_results[:, 13],
        vsync_results[:, 15],
    ]
    render_queue = [
        vsync_results[:, 10],
        vsync_results[:, 12],
        vsync_results[:, 14],
        vsync_results[:, 16],
    ]

    fps_gain = [
        vsync_results[:, 13] - vsync_results[:, 11],
        vsync_results[:, 15] - vsync_results[:, 11],
    ]
    render_queue_reduce = [
        vsync_results[:, 12] - vsync_results[:, 14],
        vsync_results[:, 12] - vsync_results[:, 16],
    ]

    plot_cdf(
        valid_fps,
        "FPS",
        "CDF",
        labels=["Online", "Naive_VSync", "Perio_drop", "Optimal"],
        postfix="_comparison",
    )
    plot_cdf(
        render_queue,
        "Render queue (ms)",
        "CDF",
        labels=["Online", "Naive_VSync", "Perio_drop", "Optimal"],
        xlim=[0, 30],
        postfix="_comparison",
    )
    plot_cdf(
        fps_gain,
        "FPS gain",
        "CDF",
        labels=["Perio_drop", "Optimal"],
        xlim=[-1, 0.5],
        postfix="_comparison",
    )
    plot_cdf(
        render_queue_reduce,
        "Render queue reduce (ms)",
        "CDF",
        labels=["Perio_drop", "Optimal"],
        xlim=[-5, 30],
        postfix="_comparison",
    )


def plot_multi_log_results(
    log_paths,
    fps_prefix="optimized_noloss_fps",
    postfix="_comparison",
    find_potential_gain=False,
    fps_gain_thr=15,
    draw=True,
    draw_fps_only=False,
):
    dfs = []
    for log_path in log_paths:
        data = pd.read_csv(log_path)
        dfs.append(data)

    new_df = functools.reduce(lambda left, right: pd.merge(left, right), dfs)
    # print([str(item) for item in new_df.columns], len(new_df))

    # valid_idx = np.where(np.logical_and.reduce((new_df['client_vsync_enabled'].to_numpy() == 1, new_df['avg_render_time'].to_numpy() <= 10, new_df['max_fps'].to_numpy() > min_fps_thr)))[0]
    valid_idx = np.where(
        np.logical_and.reduce(
            (
                new_df["client_vsync_enabled"] == 1,
                new_df["avg_render_time"].to_numpy() <= 10,
                new_df["max_fps"].to_numpy() >= min_fps_thr,
                new_df["max_fps"].to_numpy() <= max_fps_thr,
                new_df["netts_over100_cnt"] / new_df["tot_frame_no"] < stall_thr,
            )
        )
    )[0]
    print(
        new_df.shape[0], np.sum(new_df["client_vsync_enabled"] == 1), valid_idx.shape[0]
    )
    new_df = new_df.iloc[valid_idx, :]

    fps_results = []
    fps_keys = []
    fps_diff = []
    fps_diff_keys = []

    queue_results = []
    queue_keys = []
    queue_diff = []
    queue_diff_keys = []

    extra_results = []
    extra_keys = []
    extra_diff = []
    extra_diff_keys = []

    tot_queue_results = []
    tot_queue_keys = []
    tot_queue_diff = []
    tot_queue_diff_keys = []

    base_fps_key = None
    base_queue_key = None
    base_extra_key = None
    base_total_key = None

    for key in new_df.columns:
        if key.startswith(fps_prefix) or key.startswith("origin_fps"):
            fps_keys.append(key if key.startswith("origin_fps") else key[10:])
            fps_results.append(new_df[key].to_numpy())

            if "naiveVsync" in key:
                base_fps_key = key

        if key.startswith("optimized_render_queue") or key.startswith(
            "origin_render_queue"
        ):
            queue_keys.append(
                key if key.startswith("origin_render_queue") else key[23:]
            )
            queue_results.append(new_df[key].to_numpy())

            if "naiveVsync" in key:
                base_queue_key = key

        if key.startswith("extra_display"):
            extra_results.append(new_df[key].to_numpy())
            extra_keys.append(key[17:])
            if "naiveVsync" in key:
                base_extra_key = key
                base_total_key = key[17:]

        if key.startswith("optimized_total_render_queue"):
            tot_queue_keys.append(key[29:])
            tot_queue_results.append(new_df[key].to_numpy())
            if "naiveVsync" in key:
                base_total_key = key

    for key in new_df.columns:
        if key.startswith(fps_prefix) and "naiveVsync" not in key:
            fps_diff.append(new_df[key].to_numpy() - new_df[base_fps_key].to_numpy())
            fps_diff_keys.append(key[10:])

        if key.startswith("optimized_render_queue") and "naiveVsync" not in key:
            queue_diff.append(
                new_df[base_queue_key].to_numpy() - new_df[key].to_numpy()
            )
            queue_diff_keys.append(key[23:])

        if key.startswith("extra_display") and "naiveVsync" not in key:
            extra_diff.append(
                new_df[base_extra_key].to_numpy() - new_df[key].to_numpy()
            )
            extra_diff_keys.append(key[17:])

        if key.startswith("optimized_total_render_queue") and "naiveVsync" not in key:
            tot_queue_diff.append(
                new_df[base_total_key].to_numpy() - new_df[key].to_numpy()
            )
            tot_queue_diff_keys.append(key[29:])

        if key.startswith("optimized_render_queue") and "naiveVsync" in key:
            vsync_key = key

    if draw:
        plot_cdf(
            fps_results,
            "FPS",
            "CDF",
            labels=fps_keys,
            postfix="_" + fps_prefix + postfix,
        )
        plot_cdf(
            fps_diff,
            "FPS gain",
            "CDF",
            labels=fps_diff_keys,
            xlim=[-1, 1],
            postfix="_" + fps_prefix + postfix,
        )

        if not draw_fps_only:
            plot_cdf(
                queue_results,
                "Render queue (ms)",
                "CDF",
                labels=queue_keys,
                xlim=[0, 30],
                postfix=postfix,
            )
            plot_cdf(
                extra_results,
                "Extra display ts (ms)",
                "CDF",
                labels=extra_keys,
                xlim=[5, 15],
                postfix=postfix,
            )
            plot_cdf(
                tot_queue_results,
                "Total render queue (ms)",
                "CDF",
                labels=tot_queue_keys,
                xlim=[0, 30],
                postfix=postfix,
            )
            plot_cdf(
                queue_diff,
                "Render queue reduce (ms)",
                "CDF",
                labels=queue_diff_keys,
                xlim=[-1, 20],
                postfix=postfix,
            )
            plot_cdf(
                extra_diff,
                "Extra display ts reduce (ms)",
                "CDF",
                labels=extra_diff_keys,
                xlim=[-1, 10],
                postfix=postfix,
            )
            plot_cdf(
                tot_queue_diff,
                "Total render queue reduce (ms)",
                "CDF",
                labels=tot_queue_diff_keys,
                xlim=[-1, 20],
                postfix=postfix,
            )

    print("avg_fps")
    for key, value in zip(fps_keys, fps_results):
        print("%s, %.3f" % (key, np.mean(value)))

    print("avg_render_queue")
    for key, value in zip(queue_keys, queue_results):
        print("%s, %.3f" % (key, np.mean(value)))

    print("avg_extra_display_ts")
    for key, value in zip(extra_keys, extra_results):
        print("%s, %.3f" % (key, np.mean(value)))

    print("tot_queue_results")
    for key, value in zip(tot_queue_keys, tot_queue_results):
        print("%s, %.3f" % (key, np.mean(value)))

    print("fps_loss")
    for key, value in zip(fps_diff_keys, fps_diff):
        print("%s, %.3f" % (key, np.mean(value)))

    queue_diff_res = []
    print("queue_reduce")
    for key, value in zip(queue_diff_keys, queue_diff):
        print("%s, %.3f" % (key, np.mean(value)))
        queue_diff_res.append(np.mean(value))

    print("extra_display_ts_reduce")
    for key, value in zip(extra_diff_keys, extra_diff):
        print("%s, %.3f" % (key, np.mean(value)))

    print("tot_queue_diff")
    for key, value in zip(tot_queue_diff_keys, tot_queue_diff):
        print("%s, %.3f" % (key, np.mean(value)))

    if not find_potential_gain:
        return

    sorted_idx = np.argsort(queue_diff_res)
    min_key = "optimized_render_queue-" + queue_diff_keys[sorted_idx[-1]]
    if len(log_paths) == 2:
        max_key = vsync_key
    else:
        max_key = "optimized_render_queue-" + queue_diff_keys[sorted_idx[-2]]
    potential_gain_log = new_df.loc[new_df[max_key] - new_df[min_key] > fps_gain_thr]
    print(potential_gain_log.shape)

    potential_gain_log.to_csv(
        os.path.join("test_data", "result-potential_gain%.1f.csv" % fps_gain_thr),
        index=False,
    )


def compare_two_log(
    log_paths,
    fps_prefix="optimized_fps",
    postfix="_comparison",
    find_potential_gain=False,
    fps_gain_thr=15,
    draw=True,
):
    dfs = []
    for log_path in log_paths:
        data = pd.read_csv(log_path)
        dfs.append(data)

    new_df = functools.reduce(
        lambda left, right: pd.merge(left, right, on="file_name"), dfs
    )
    print(new_df.columns, len(new_df))

    # valid_idx = np.where(np.logical_and.reduce((new_df['client_vsync_enabled'].to_numpy() == 1, new_df['avg_render_time'].to_numpy() <= 10, new_df['max_fps'].to_numpy() > min_fps_thr)))[0]
    valid_idx = np.where(
        np.logical_and.reduce(
            (
                new_df["client_vsync_enabled_x"] == 1,
                new_df["avg_render_time_x"].to_numpy() <= 10,
                new_df["max_fps_x"].to_numpy() >= min_fps_thr,
                new_df["max_fps_x"].to_numpy() <= max_fps_thr,
                new_df["netts_over100_cnt_x"] / new_df["tot_frame_no_x"] < stall_thr,
            )
        )
    )[0]
    print(
        new_df.shape[0],
        np.sum(new_df["client_vsync_enabled_x"] == 1),
        valid_idx.shape[0],
    )
    new_df = new_df.iloc[valid_idx, :]

    fps_results = []
    fps_keys = []
    fps_diff = []
    fps_diff_keys = []

    queue_results = []
    queue_keys = []
    queue_diff = []
    queue_diff_keys = []

    extra_results = []
    extra_keys = []
    extra_diff = []
    extra_diff_keys = []

    tot_queue_results = []
    tot_queue_keys = []
    tot_queue_diff = []
    tot_queue_diff_keys = []

    base_fps_key = None
    base_queue_key = None
    base_extra_key = None
    base_total_key = None

    for key in new_df.columns:
        if key.startswith(fps_prefix) or key.startswith("origin_fps"):
            fps_keys.append(key if key.startswith("origin_fps") else key[10:])
            fps_results.append(new_df[key].to_numpy())

            if "naiveVsync" in key:
                base_fps_key = key

        if key.startswith("optimized_render_queue") or key.startswith(
            "origin_render_queue"
        ):
            queue_keys.append(
                key if key.startswith("origin_render_queue") else key[23:]
            )
            queue_results.append(new_df[key].to_numpy())

        if key.startswith("extra_display"):
            extra_results.append(new_df[key].to_numpy())
            extra_keys.append(key[17:])
            if "naiveVsync" in key:
                base_extra_key = key
                base_total_key = key[17:]

        if key.startswith("optimized_total_render_queue"):
            tot_queue_keys.append(key[29:])
            tot_queue_results.append(new_df[key].to_numpy())
            if "naiveVsync" in key:
                base_total_key = key

    for key in new_df.columns:
        if key.startswith(fps_prefix) and key.endswith("_x"):
            fps_diff.append(new_df[key].to_numpy() - new_df[key[:-2] + "_y"].to_numpy())
            fps_diff_keys.append(key[10:])

        if key.startswith("optimized_render_queue") and key.endswith("_x"):
            queue_diff.append(
                new_df[key[:-2] + "_y"].to_numpy() - new_df[key].to_numpy()
            )
            queue_diff_keys.append(key[23:])

        if key.startswith("extra_display") and key.endswith("_x"):
            extra_diff.append(
                new_df[key[:-2] + "_y"].to_numpy() - new_df[key].to_numpy()
            )
            extra_diff_keys.append(key[17:])

        if key.startswith("optimized_total_render_queue") and key.endswith("_x"):
            tot_queue_diff.append(
                new_df[key[:-2] + "_y"].to_numpy() - new_df[key].to_numpy()
            )
            tot_queue_diff_keys.append(key[29:])

    if draw:
        plot_cdf(fps_results, "FPS", "CDF", labels=fps_keys, postfix=postfix)
        plot_cdf(
            queue_results,
            "Render queue (ms)",
            "CDF",
            labels=queue_keys,
            xlim=[0, 30],
            postfix=postfix,
        )
        plot_cdf(
            extra_results,
            "Extra display ts (ms)",
            "CDF",
            labels=extra_keys,
            xlim=[5, 15],
            postfix=postfix,
        )
        plot_cdf(
            tot_queue_results,
            "Total render queue (ms)",
            "CDF",
            labels=tot_queue_keys,
            xlim=[0, 30],
            postfix=postfix,
        )

        plot_cdf(
            fps_diff,
            "FPS gain",
            "CDF",
            labels=fps_diff_keys,
            xlim=[-1.5, 1.5],
            postfix=postfix,
        )
        plot_cdf(
            queue_diff,
            "Render queue reduce (ms)",
            "CDF",
            labels=queue_diff_keys,
            xlim=[-1, 20],
            postfix=postfix,
        )
        plot_cdf(
            extra_diff,
            "Extra display ts reduce (ms)",
            "CDF",
            labels=extra_diff_keys,
            xlim=[-1, 10],
            postfix=postfix,
        )
        plot_cdf(
            tot_queue_diff,
            "Total render queue reduce (ms)",
            "CDF",
            labels=tot_queue_diff_keys,
            xlim=[-1, 20],
            postfix=postfix,
        )

    print("avg_fps")
    for key, value in zip(fps_keys, fps_results):
        print("%s, %.3f" % (key, np.mean(value)))

    print("avg_render_queue")
    for key, value in zip(queue_keys, queue_results):
        print("%s, %.3f" % (key, np.mean(value)))

    print("avg_extra_display_ts")
    for key, value in zip(extra_keys, extra_results):
        print("%s, %.3f" % (key, np.mean(value)))

    print("tot_queue_results")
    for key, value in zip(tot_queue_keys, tot_queue_results):
        print("%s, %.3f" % (key, np.mean(value)))

    print("fps_loss")
    for key, value in zip(fps_diff_keys, fps_diff):
        print("%s, %.3f" % (key, np.mean(value)))

    queue_diff_res = []
    print("queue_reduce")
    for key, value in zip(queue_diff_keys, queue_diff):
        print("%s, %.3f" % (key, np.mean(value)))
        queue_diff_res.append(np.mean(value))

    print("extra_display_ts_reduce")
    for key, value in zip(extra_keys, extra_diff):
        print("%s, %.3f" % (key, np.mean(value)))

    print("tot_queue_diff")
    for key, value in zip(tot_queue_diff_keys, tot_queue_diff):
        print("%s, %.3f" % (key, np.mean(value)))

    if not find_potential_gain:
        return

    min_key = "optimized_render_queue-" + queue_diff_keys[0]
    max_key = "optimized_render_queue-" + queue_diff_keys[0][:-2] + "_y"
    potential_gain_log = new_df.loc[new_df[max_key] - new_df[min_key] > fps_gain_thr]
    print(potential_gain_log.shape)

    potential_gain_log.to_csv(
        os.path.join("test_data", "result-potential_gain%.1f.csv" % fps_gain_thr),
        index=False,
    )


def plot_e2e_jitter_analyze(log_path):
    data = pd.read_csv(log_path)

    results = data.iloc[:, 1:].to_numpy()
    valid_idx = np.where(
        np.logical_and.reduce(
            (results[:, 4] > 0, results[:, 7] / (results[:, 4] + 1) < 0.8)
        )
    )[0]
    effective_results = results[valid_idx]
    print(
        results.shape[0],
        np.sum(results[:, 2] == 1),
        valid_idx.shape[0],
        effective_results.shape[0],
    )

    factor_ratios = [
        effective_results[:, 5] / effective_results[:, 4],
        effective_results[:, 6] / effective_results[:, 4],
        effective_results[:, 7] / effective_results[:, 4],
        effective_results[:, 8] / effective_results[:, 4],
        effective_results[:, 9] / effective_results[:, 4],
        effective_results[:, 10] / effective_results[:, 4],
        effective_results[:, 11] / effective_results[:, 4],
    ]

    plot_cdf(
        factor_ratios,
        "Jitter ratio",
        "CDF",
        labels=[
            "Big_frame",
            "I_frame",
            "DL_jitter",
            "Stall",
            "Packet_loss",
            "Render_jitter",
            "Decoder_jitter",
        ],
        postfix="_comparison",
    )


def plot_buffer_jitter_analyze(
    log_path, postfix="", filter=None, filter_render_problem=False
):
    df = pd.read_csv(log_path)

    if filter_render_problem:
        valid_idx = np.where(
            np.logical_and.reduce(
                (
                    df["avg_render_time"].to_numpy() <= 10,
                    df["max_fps"].to_numpy() >= min_fps_thr,
                    df["max_fps"].to_numpy() <= max_fps_thr,
                    df["netts_over100_cnt"] / df["tot_frame_no"] < stall_thr,
                    df["large_renderinterval_cnt"] / df["tot_frame_no"]
                    < render_problem_thr,
                )
            )
        )[0]
    else:
        valid_idx = np.where(
            np.logical_and.reduce(
                (
                    df["avg_render_time"].to_numpy() <= 10,
                    df["max_fps"].to_numpy() >= min_fps_thr,
                    df["max_fps"].to_numpy() <= max_fps_thr,
                    df["netts_over100_cnt"] / df["tot_frame_no"] < stall_thr,
                )
            )
        )[0]

    # print(df.shape[0], np.sum(df['client_vsync_enabled']==1), valid_idx.shape[0])
    new_df = df.iloc[valid_idx, :]

    if filter is not None:
        valid_idx = np.where(new_df["file_name"].str.contains(filter))[0]
        new_df = new_df.iloc[valid_idx, :]

    flow_cnt = new_df.shape[0]

    factor_keys = [
        "network_big_frame_induced_queue-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo",
        # 'network_i_frame_induced_queue-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo',
        "network_dl_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo",
        "network_stall_induced_queue-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo",
        "network_packet_loss_induced_queue-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo",
        "render_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo",
        "server_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo",
        "decoder_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo",
        "display_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo",
        "near_vsync_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo",
    ]
    factor_ratios = []
    for key in factor_keys:
        factor_ratios.append(
            new_df[key]
            / new_df["tot_queue_cnt-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo"]
        )

    # network_total = 0
    # for key in factor_keys:
    #     if key.startswith('network'):
    #         network_total += new_df[key]/new_df['tot_queue_cnt-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo']
    # factor_ratios.append(network_total)

    unknown_jitter = new_df[
        "tot_queue_cnt-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo"
    ]
    for key in factor_keys:
        unknown_jitter = unknown_jitter - new_df[key]
    factor_ratios.append(
        unknown_jitter
        / new_df["tot_queue_cnt-simpleCtrl-periodrop2_quickdrop1_bonusfps30_lifo"]
    )

    new_keys = []
    for key in factor_keys:
        new_keys.append(key[:-63])
    # new_keys.append('network_total')
    new_keys.append("unknown_jitter")

    plot_cdf(factor_ratios, "Ratio" + postfix, "CDF", labels=new_keys, postfix=postfix)


def plot_good_render_buffer_jitter_analyze(log_path, postfix="", filter_file=None):
    df = pd.read_csv(log_path)

    valid_idx = np.where(
        np.logical_and.reduce(
            (
                df["avg_render_time"].to_numpy() <= 10,
                df["max_fps"].to_numpy() >= min_fps_thr,
                df["max_fps"].to_numpy() <= max_fps_thr,
                df["netts_over100_cnt"] / df["tot_frame_no"] < stall_thr,
            )
        )
    )[0]

    # print(df.shape[0], np.sum(df['client_vsync_enabled']==1), valid_idx.shape[0])
    new_df = df.iloc[valid_idx, :]
    # print(new_df['file_name'].iloc[0])
    if filter_file is not None:
        valid_idx = np.zeros(new_df.shape[0])
        filter_data = open(filter_file).readlines()[1:]
        filter_set = []
        for line in filter_data:
            items = line.strip().split(",")
            if int(items[8]) == 1:
                filter_set.append(items[0] + "-" + items[3] + "-" + items[6])
        # print(len(filter_set))
        # print(filter_set[:3])

        valid_cnt = 0
        for idx in range(new_df.shape[0]):
            file_name = new_df["file_name"].iloc[idx]
            items = file_name.strip().split("/")[1:]
            sid = "sid:" + items[5].split("_")[1]
            ip = items[4].split("_")[2]
            thedate = "".join(items[3].split("-"))
            reformated_file_name = sid + "-" + ip + "-" + thedate

            if reformated_file_name in filter_set:
                valid_idx[idx] = 1
                valid_cnt += 1

        new_df = new_df.iloc[np.where(valid_idx)[0], :]
    flow_cnt = new_df.shape[0]

    factor_keys = [
        "network_big_frame_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo",
        # 'network_i_frame_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo',
        "network_dl_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo",
        "network_stall_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo",
        "network_packet_loss_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo",
        "render_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo",
        "server_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo",
        "decoder_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo",
        "display_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo",
        "near_vsync_jitter_induced_queue-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo",
    ]
    factor_ratios = []
    for key in factor_keys:
        factor_ratios.append(
            new_df[key]
            / new_df["tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"]
        )

    # network_total = 0
    # for key in factor_keys:
    #     if key.startswith('network'):
    #         network_total += new_df[key]/new_df['tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo']
    # factor_ratios.append(network_total)

    unknown_jitter = new_df[
        "tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"
    ]
    for key in factor_keys:
        unknown_jitter = unknown_jitter - new_df[key]
    factor_ratios.append(
        unknown_jitter
        / new_df["tot_queue_cnt-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"]
    )

    new_keys = []
    for key in factor_keys:
        new_keys.append(key[:-63])
    # new_keys.append('network_total')
    new_keys.append("unknown_jitter")

    plot_cdf(
        factor_ratios,
        "Ratio (Exclusive GPU)" + postfix,
        "CDF",
        labels=new_keys,
        postfix=postfix,
    )

    print(new_df["large_renderinterval_cnt"].sum() / new_df["tot_frame_no"].sum())
    new_df.to_csv(
        os.path.join("test_data", "result-exclusive_gpu%s.csv" % postfix), index=False
    )


def plot_buffer_exploitable_jitter_analyze(log_path, postfix=""):
    df = pd.read_csv(log_path)

    valid_idx = np.where(
        np.logical_and.reduce(
            (
                df["avg_render_time"].to_numpy() <= 10,
                df["max_fps"].to_numpy() >= min_fps_thr,
                df["max_fps"].to_numpy() <= max_fps_thr,
                df["netts_over100_cnt"] / df["tot_frame_no"] < stall_thr,
            )
        )
    )[0]
    # print(df.shape[0], np.sum(df['client_vsync_enabled']==1), valid_idx.shape[0])
    new_df = df.iloc[valid_idx, :]

    factor_keys = [
        "exploitable_network_big_frame_induced_queue-optimal-periodrop2_quickdrop0_bonusfps30_lifo",
        "exploitable_network_i_frame_induced_queue-optimal-periodrop2_quickdrop0_bonusfps30_lifo",
        "exploitable_network_dl_jitter_induced_queue-optimal-periodrop2_quickdrop0_bonusfps30_lifo",
        "exploitable_network_stall_induced_queue-optimal-periodrop2_quickdrop0_bonusfps30_lifo",
        "exploitable_network_packet_loss_induced_queue-optimal-periodrop2_quickdrop0_bonusfps30_lifo",
        "exploitable_render_jitter_induced_queue-optimal-periodrop2_quickdrop0_bonusfps30_lifo",
        "exploitable_server_jitter_induced_queue-optimal-periodrop2_quickdrop0_bonusfps30_lifo",
        "exploitable_decoder_jitter_induced_queue-optimal-periodrop2_quickdrop0_bonusfps30_lifo",
        "exploitable_display_jitter_induced_queue-optimal-periodrop2_quickdrop0_bonusfps30_lifo",
        "exploitable_near_vsync_jitter_induced_queue-optimal-periodrop2_quickdrop0_bonusfps30_lifo",
    ]
    factor_ratios = []
    for key in factor_keys:
        tmp_value = (
            new_df[key]
            / new_df["tot_queue_cnt-optimal-periodrop2_quickdrop0_bonusfps30_lifo"]
        )
        tmp_value.iloc[
            np.where(
                new_df["tot_queue_cnt-optimal-periodrop2_quickdrop0_bonusfps30_lifo"]
                == 0
            )[0]
        ] = 0
        factor_ratios.append(tmp_value)

    network_total = 0
    for key in factor_keys:
        if key.startswith("exploitable_network"):
            tmp_value = (
                new_df[key]
                / new_df["tot_queue_cnt-optimal-periodrop2_quickdrop0_bonusfps30_lifo"]
            )
            tmp_value.iloc[
                np.where(
                    new_df[
                        "tot_queue_cnt-optimal-periodrop2_quickdrop0_bonusfps30_lifo"
                    ]
                    == 0
                )[0]
            ] = 0
            network_total += tmp_value
    factor_ratios.append(network_total)

    new_keys = []
    for key in factor_keys:
        new_keys.append(key[12:-60])
    new_keys.append("network_total")

    plot_cdf(
        factor_ratios,
        "Exploitable jitter ratio" + postfix,
        "CDF",
        labels=new_keys,
        xlim=[0, 1],
        postfix=postfix,
    )

    exploitable_total_cnt = 0
    for key in factor_keys:
        exploitable_total_cnt = exploitable_total_cnt + new_df[key]

    factor_ratios = []
    for key in factor_keys:
        tmp_value = new_df[key] / exploitable_total_cnt
        tmp_value.iloc[np.where(exploitable_total_cnt == 0)[0]] = 0
        factor_ratios.append(tmp_value)

    new_keys = []
    for key in factor_keys:
        new_keys.append(key[12:-60])

    plot_cdf(
        factor_ratios,
        "Exploitable jitter ratio divide by category total" + postfix,
        "CDF",
        labels=new_keys,
        xlim=[0, 1],
        postfix=postfix,
    )

    factor_ratios = []
    for key in factor_keys:
        tmp_value = new_df[key] / new_df[key[12:]]
        tmp_value.iloc[np.where(new_df[key[12:]] == 0)[0]] = 0
        factor_ratios.append(tmp_value)

    new_keys = []
    for key in factor_keys:
        new_keys.append(key[12:-60])

    plot_cdf(
        factor_ratios,
        "Exploitable jitter ratio by category" + postfix,
        "CDF",
        labels=new_keys,
        xlim=[0, 1],
        postfix=postfix,
    )


def plot_buffer_jitter_interval(
    log_dir,
    filename="optimal_quickdrop0_periodrop2_maxbuf2_renderTime_ewma_original_fps_lifo_sim.csv",
    postfix="_comparison",
):
    factor_keys = [
        "network_big_frame_induced_queue",
        "network_i_frame_induced_queue",
        "network_dl_jitter_induced_queue",
        "network_stall_induced_queue",
        "network_packet_loss_induced_queue",
        "render_jitter_induced_queue",
        "server_jitter_induced_queue",
        "decoder_jitter_induced_queue",
        "display_jitter_induced_queue",
        "near_vsync_jitter_induced_queue",
    ]

    jitter_interval = {}
    exploitable_jitter_interval = {}
    jitter_amplitude = {}
    exploitable_jitter_amplitude = {}

    all_jitter_key = "all_jitter_induced_queue"
    jitter_interval[all_jitter_key] = []
    for log_name in os.listdir(log_dir):
        if not log_name.endswith(filename):
            continue

        log_path = os.path.join(log_dir, log_name)
        df = pd.read_csv(log_path)

        valid_idx = np.where(df["frame_queue_flag"].to_numpy() == 1)[0]
        new_df = df.iloc[valid_idx, :]
        dec_over_ts = new_df["dec_over_ts"].to_numpy()
        cur_interval = dec_over_ts[1:] - dec_over_ts[:-1]
        jitter_interval[all_jitter_key] += cur_interval.tolist()

        for key in factor_keys:
            if key not in jitter_interval:
                jitter_interval[key] = []
                exploitable_jitter_interval[key] = []
                jitter_amplitude[key] = []
                exploitable_jitter_amplitude[key] = []

            valid_idx = np.where(df[key].to_numpy() == 1)[0]
            new_df = df.iloc[valid_idx, :]
            dec_over_ts = new_df["dec_over_ts"].to_numpy()
            cur_interval = dec_over_ts[1:] - dec_over_ts[:-1]
            jitter_interval[key] += cur_interval.tolist()

            valid_idx = np.where(
                np.logical_and(
                    df[key].to_numpy() == 1, df["buf_change_flag"].to_numpy() == 5
                )
            )[0]
            new_df = df.iloc[valid_idx, :]
            dec_over_ts = new_df["dec_over_ts"].to_numpy()
            cur_interval = dec_over_ts[1:] - dec_over_ts[:-1]
            exploitable_jitter_interval[key] += cur_interval.tolist()

    results1 = []
    results2 = []
    results1.append(jitter_interval[all_jitter_key])
    for key in factor_keys:
        results1.append(jitter_interval[key])
        results2.append(exploitable_jitter_interval[key])

    new_keys = []
    for key in factor_keys:
        new_keys.append(key[:-14])

    plot_cdf(
        results1,
        "Jitter interval (ms)" + postfix,
        "CDF",
        labels=[all_jitter_key] + new_keys,
        xlim=[0, 3000],
        postfix=postfix,
    )
    plot_cdf(
        results2,
        "Exploitable jitter interval (ms)" + postfix,
        "CDF",
        labels=new_keys,
        xlim=[0, 3000],
        postfix=postfix,
    )


def plot_quick_drop_results(log_path, postfix="", filter=None):
    df = pd.read_csv(log_path)

    valid_idx = np.where(
        np.logical_and.reduce(
            (
                df["avg_render_time"].to_numpy() <= 10,
                df["max_fps"].to_numpy() >= min_fps_thr,
                df["max_fps"].to_numpy() <= max_fps_thr,
                df["netts_over100_cnt"] / df["tot_frame_no"] < stall_thr,
            )
        )
    )[0]
    # print(df.shape[0], np.sum(df['client_vsync_enabled']==1), valid_idx.shape[0])
    new_df = df.iloc[valid_idx, :]

    if filter is not None:
        valid_idx = np.where(new_df["file_name"].str.contains(filter))[0]
        new_df = new_df.iloc[valid_idx, :]

    simple_factor_keys = [
        # 'quick_drop_failed_cnt',
        "quick_drop_fail_ratio",
        # 'quick_drop_missed_cnt',
        "quick_drop_miss_ratio",
    ]
    factor_keys = []
    for key in new_df.columns:
        for simple_key in simple_factor_keys:
            if key.startswith(simple_key):
                factor_keys.append(key)

    factor_ratios = []
    new_keys = []
    for key in factor_keys:
        factor_ratios.append(new_df[key])
        new_keys.append(key[:-55])

    plot_cdf(
        factor_ratios,
        "Quick drop ratio" + postfix,
        "CDF",
        labels=new_keys,
        postfix=postfix,
    )

    # valid_idx = np.where(new_df['quick_drop_missed_cnt-simpleCtrl-periodrop2_quickdrop3_bonusfps30_lifo']+new_df['quick_drop_total_cnt-simpleCtrl-periodrop2_quickdrop3_bonusfps30_lifo']-new_df['quick_drop_failed_cnt-simpleCtrl-periodrop2_quickdrop3_bonusfps30_lifo'] > 0)[0]

    # new_df = new_df.iloc[valid_idx, :]
    # for key in factor_keys:
    #     factor_ratios.append(new_df[key])
    #     new_keys.append(key[:-55]+'_valid')

    # plot_cdf(factor_ratios, 'Quick drop ratio valid'+postfix, 'CDF', labels=new_keys, postfix=postfix)


def plot_jitter_analysis_results(log_path, postfix="", filter=None):
    df = pd.read_csv(log_path)

    # valid_idx = np.where(np.logical_and.reduce((df['avg_render_time'].to_numpy() <= 10, df['max_fps'].to_numpy() >= min_fps_thr, df['max_fps'].to_numpy() <= max_fps_thr, df['netts_over100_cnt'] / df['tot_frame_no'] < stall_thr)))[0]
    # df = df.iloc[valid_idx, :]

    if filter is not None:
        valid_idx = np.where(new_df["file_name"].str.contains(filter))[0]
        df = df.iloc[valid_idx, :]

    jitter_names = ["render", "dl", "dec", "display"]
    pctile_nos = [5, 25, 75, 95]
    amp_thrs = [5, 10, 20]
    interval_thrs = [10, 20, 30]

    # for jitter_name in jitter_names:
    #     keys = []
    #     values = []
    #     for pctile_no in pctile_nos:
    #         cur_key = '%s_jitter_amp_pct%d' %(jitter_name, pctile_no)
    #         keys.append(cur_key)
    #         values.append(df[cur_key])
    #     plot_cdf(values, 'Jitter analysis: %s jitter amp' %jitter_name, 'CDF', labels=keys)

    for amp_thr in amp_thrs:
        jitter_no_key = "dl_jitter_over%d_no" % (amp_thr)
        valid_idx = np.where(df[jitter_no_key].to_numpy() > 100)[0]
        new_df = df.iloc[valid_idx, :]

        keys = (
            "dl_jitter_over%d_stall,dl_jitter_over%d_bigframe,dl_jitter_over%d_loss,dl_jitter_over%d_bitrate,dl_jitter_over%d_random"
            % (amp_thr, amp_thr, amp_thr, amp_thr, amp_thr)
        ).split(",")
        values = []
        for cur_key in keys:
            values.append(new_df[cur_key] / new_df[jitter_no_key])

        plot_cdf(
            values,
            "Jitter analysis: dl jitter amp over%d type no" % (amp_thr),
            "CDF",
            labels=keys,
        )  # , xlim=[0, 40])

    for jitter_name in jitter_names:
        keys = []
        values = []
        for amp_thr in amp_thrs:
            jitter_interval_no_key = "%s_jitter_over%d_interval_no" % (
                jitter_name,
                amp_thr,
            )
            valid_idx = np.where(df[jitter_interval_no_key].to_numpy() > 100)[0]
            new_df = df.iloc[valid_idx, :]

            cur_key1 = "%s_jitter_over%d_interval_pct%d" % (jitter_name, amp_thr, 40)
            cur_key2 = "%s_jitter_over%d_interval_pct%d" % (jitter_name, amp_thr, 60)

            type1_flow_cnt = np.logical_and(
                new_df[cur_key1].to_numpy() < 10, new_df[cur_key2].to_numpy() > 30
            ).sum()

            type2_flow_cnt = (
                new_df[
                    "%s_jitter_over%d_interval_pct%d" % (jitter_name, amp_thr, 20)
                ].to_numpy()
                > 30
            ).sum()

            # if amp_thr == 5:
            #     sorted_idx = np.where(np.logical_and(
            #         new_df[cur_key1].to_numpy() < 10, new_df[cur_key2].to_numpy() > 30
            #     ))[0]
            #     save_df = new_df.iloc[sorted_idx, :]
            #     save_df.to_csv('test_data/result_jitter_analysis_%s_jitter_sample.csv' %jitter_name)

            print(
                jitter_name,
                amp_thr,
                new_df.shape[0],
                type1_flow_cnt,
                type1_flow_cnt / new_df.shape[0],
                type2_flow_cnt,
                type2_flow_cnt / new_df.shape[0],
            )

    for jitter_name in jitter_names:
        for amp_thr in amp_thrs:
            keys = []
            values = []
            for interval_thr in interval_thrs:
                jitter_interval_no_key = "%s_jitter_over%d_interval_no" % (
                    jitter_name,
                    amp_thr,
                )
                valid_idx = np.where(df[jitter_interval_no_key].to_numpy() > 100)[0]
                new_df = df.iloc[valid_idx, :]

                for pct_tile in [30, 95]:
                    cur_key = "%s_jitter_amp_over%d_interval_over%d_same_no_pct%d" % (
                        jitter_name,
                        amp_thr,
                        interval_thr,
                        pct_tile,
                    )
                    keys.append(cur_key)
                    values.append(new_df[cur_key])

                # if amp_thr == 20 and interval_thr == 30:
                #     sorted_idx = np.argsort(-new_df[cur_key])[:20]
                #     save_df = new_df.iloc[sorted_idx, :]
                #     save_df.to_csv('test_data/result_jitter_analysis_%s_jitter_sample.csv' %jitter_name)

            plot_cdf(
                values,
                "Jitter analysis: %s jitter amp over%d interval same no"
                % (jitter_name, amp_thr),
                "CDF",
                labels=keys,
                xlim=[0, 40],
            )

    return

    for jitter_name in jitter_names:
        keys = []
        values = []
        for amp_thr in amp_thrs:
            jitter_interval_no_key = "%s_jitter_over%d_interval_no" % (
                jitter_name,
                amp_thr,
            )
            valid_idx = np.where(df[jitter_interval_no_key].to_numpy() > 100)[0]
            new_df = df.iloc[valid_idx, :]

            diffs = []
            for pct in range(5, 50, 5):
                cur_key1 = "%s_jitter_over%d_interval_pct%d" % (
                    jitter_name,
                    amp_thr,
                    pct,
                )
                cur_key2 = "%s_jitter_over%d_interval_pct%d" % (
                    jitter_name,
                    amp_thr,
                    pct + 50,
                )
                diff = new_df[cur_key2] - new_df[cur_key1]
                diffs.append(diff.to_numpy())

            diffs = np.stack(diffs, axis=1)
            diff_min = np.min(diffs, axis=1)

            values.append(diff_min)
            keys.append(
                "%s_jitter_over%d_interval_50pct_min_range" % (jitter_name, amp_thr)
            )

        plot_cdf(
            values,
            "Jitter analysis: %s jitter interval 50pct min range" % (jitter_name),
            "CDF",
            labels=keys,
            xlim=[0, 100],
        )

    for jitter_name in jitter_names:
        pct_tile_no = 20
        for pct_thr in [10, 20, 30]:
            keys = []
            values = []
            for amp_thr in amp_thrs:
                jitter_pct20_key = "%s_jitter_over%d_interval_pct%d" % (
                    jitter_name,
                    amp_thr,
                    pct_tile_no,
                )
                jitter_interval_no_key = "%s_jitter_over%d_interval_no" % (
                    jitter_name,
                    amp_thr,
                )
                valid_idx = np.where(
                    np.logical_and(
                        df[jitter_interval_no_key].to_numpy() > 100,
                        df[jitter_pct20_key].to_numpy() > pct_thr,
                    )
                )[0]
                new_df = df.iloc[valid_idx, :]

                diffs = []
                for pct in range(5, 50, 5):
                    cur_key1 = "%s_jitter_over%d_interval_pct%d" % (
                        jitter_name,
                        amp_thr,
                        pct,
                    )
                    cur_key2 = "%s_jitter_over%d_interval_pct%d" % (
                        jitter_name,
                        amp_thr,
                        pct + 50,
                    )
                    diff = new_df[cur_key2] - new_df[cur_key1]
                    diffs.append(diff.to_numpy())

                diffs = np.stack(diffs, axis=1)
                diff_min = np.min(diffs, axis=1)

                # if pct_thr == 20 and amp_thr == 10:
                #     sorted_idx = np.argsort(diff_min)[:20]
                #     save_df = new_df.iloc[sorted_idx, :]
                #     save_df.to_csv('test_data/result_jitter_analysis_%s_jitter_sample.csv' %jitter_name)

                values.append(diff_min)
                keys.append(
                    "%s_jitter_over%d_interval_50pct_min_range" % (jitter_name, amp_thr)
                )

            plot_cdf(
                values,
                "Jitter analysis: %s jitter interval %dpct over%d 50pct min range"
                % (jitter_name, pct_tile_no, pct_thr),
                "CDF",
                labels=keys,
                xlim=[0, 100],
            )

    # for jitter_name in jitter_names:
    #     keys = []
    #     values = []
    #     for amp_thr in amp_thrs:
    #         jitter_consec_no_key = '%s_jitter_over%d_consec_no' %(jitter_name, amp_thr)
    #         jitter_consec_mean_key = '%s_jitter_over%d_consec_mean' %(jitter_name, amp_thr)
    #         jitter_consec_std_key = '%s_jitter_over%d_consec_std' %(jitter_name, amp_thr)

    #         jitter_interval_no_key = '%s_jitter_over%d_interval_no' %(jitter_name, amp_thr)
    #         jitter_interval_mean_key = '%s_jitter_over%d_interval_mean' %(jitter_name, amp_thr)
    #         jitter_interval_std_key = '%s_jitter_over%d_interval_std' %(jitter_name, amp_thr)

    #         valid_idx = np.where(df[jitter_consec_no_key].to_numpy() > 100)[0]
    #         new_df = df.iloc[valid_idx, :]

    #         plot_histogram(new_df[jitter_consec_std_key]/new_df[jitter_consec_mean_key], new_df[jitter_interval_std_key]/new_df[jitter_interval_mean_key],
    #                        'Jitter analysis: %s jitter amp over%d CV consec' %(jitter_name, amp_thr), 'Jitter analysis: %s jitter amp over%d CV interval' %(jitter_name, amp_thr))

    for jitter_name in jitter_names:
        keys = []
        values = []
        for amp_thr in amp_thrs:

            cur_key = "%s_jitter_over%d_frame_no" % (jitter_name, amp_thr)
            keys.append(cur_key[:-3] + "_ratio")
            values.append(df[cur_key] / df["tot_frame_no"])

        plot_cdf(
            values,
            "Jitter analysis: %s jitter amp frame ratio" % (jitter_name),
            "CDF",
            labels=keys,
            xlim=[0, 1],
        )

    for jitter_name in jitter_names:
        keys = []
        values = []
        for amp_thr in amp_thrs:
            jitter_interval_no_key = "%s_jitter_over%d_interval_no" % (
                jitter_name,
                amp_thr,
            )
            cur_key1 = "%s_jitter_over%d_interval_pct75" % (jitter_name, amp_thr)
            cur_key2 = "%s_jitter_over%d_interval_pct25" % (jitter_name, amp_thr)

            valid_idx = np.where(df[jitter_interval_no_key].to_numpy() > 100)[0]
            new_df = df.iloc[valid_idx, :]

            keys.append("%s_jitter_over%d_interval" % (jitter_name, amp_thr))
            values.append(new_df[cur_key1] - new_df[cur_key2])

        plot_cdf(
            values,
            "Jitter analysis: %s jitter interval IQR" % (jitter_name),
            "CDF",
            labels=keys,
            x_logscale=True,
        )

    for jitter_name in jitter_names:
        for amp_thr in amp_thrs:
            keys = []
            values = []

            jitter_interval_no_key = "%s_jitter_over%d_interval_no" % (
                jitter_name,
                amp_thr,
            )
            valid_idx = np.where(df[jitter_interval_no_key].to_numpy() > 100)[0]
            new_df = df.iloc[valid_idx, :]

            for interval_thr in interval_thrs:
                cur_key = "%s_jitter_amp_over%d_interval_over%d_no" % (
                    jitter_name,
                    amp_thr,
                    interval_thr,
                )
                keys.append(cur_key[:-3] + "_ratio")
                values.append(new_df[cur_key] / new_df[jitter_interval_no_key])

            plot_cdf(
                values,
                "Jitter analysis: %s jitter amp over%d interval ratio"
                % (jitter_name, amp_thr),
                "CDF",
                labels=keys,
                xlim=[0, 1],
            )

    # for jitter_name in jitter_names:
    #     for amp_thr in amp_thrs:
    #         keys = []
    #         values = []
    #         for pctile_no in pctile_nos:
    #             cur_key = '%s_jitter_over%d_consec_pct%d' %(jitter_name, amp_thr, pctile_no)
    #             keys.append(cur_key)
    #             values.append(df[cur_key])
    #         plot_cdf(values, 'Jitter analysis: %s jitter amp over%d consec' %(jitter_name, amp_thr), 'CDF', labels=keys, xlim=[0,30])

    #         keys = []
    #         values = []
    #         for pctile_no in pctile_nos:
    #             cur_key = '%s_jitter_over%d_interval_pct%d' %(jitter_name, amp_thr, pctile_no)
    #             keys.append(cur_key)
    #             values.append(df[cur_key])
    #         plot_cdf(values, 'Jitter analysis: %s jitter amp over%d interval' %(jitter_name, amp_thr), 'CDF', labels=keys)


def plot_buffer_jitter_analyze_by_amplitude(
    log_path, postfix="", filter=None, filter_render_problem=False
):
    df = pd.read_csv(log_path)

    if filter_render_problem:
        valid_idx = np.where(
            np.logical_and.reduce(
                (
                    df["avg_render_time"].to_numpy() <= 10,
                    df["max_fps"].to_numpy() >= min_fps_thr,
                    df["max_fps"].to_numpy() <= max_fps_thr,
                    df["netts_over100_cnt"] / df["tot_frame_no"] < stall_thr,
                    df["large_renderinterval_cnt"] / df["tot_frame_no"]
                    < render_problem_thr,
                )
            )
        )[0]
    else:
        valid_idx = np.where(
            np.logical_and.reduce(
                (
                    df["avg_render_time"].to_numpy() <= 10,
                    df["max_fps"].to_numpy() >= min_fps_thr,
                    df["max_fps"].to_numpy() <= max_fps_thr,
                    df["netts_over100_cnt"] / df["tot_frame_no"] < stall_thr,
                )
            )
        )[0]

    # print(df.shape[0], np.sum(df['client_vsync_enabled']==1), valid_idx.shape[0])
    new_df = df.iloc[valid_idx, :]

    if filter is not None:
        valid_idx = np.where(new_df["file_name"].str.contains(filter))[0]
        new_df = new_df.iloc[valid_idx, :]

    key_postfix = "-simpleCtrl-periodrop2_quickdrop0_bonusfps30_lifo"
    tot_queue_key = "tot_queue_cnt%s" % key_postfix
    valid_idx = np.where(new_df[tot_queue_key].to_numpy() > 0)[0]
    new_df = new_df.iloc[valid_idx, :]

    flow_cnt = new_df.shape[0]

    factor_key_without_postfix = "biggest_render_jitter_queue,biggest_network_jitter_queue,biggest_decode_jitter_queue,biggest_display_jitter_queue".split(
        ","
    )

    factor_keys = (
        "biggest_render_jitter_queue%s,biggest_network_jitter_queue%s,biggest_decode_jitter_queue%s,biggest_display_jitter_queue%s"
        % (key_postfix, key_postfix, key_postfix, key_postfix)
    )
    factor_keys = factor_keys.split(",")

    factor_ratios = []
    for key in factor_keys:
        factor_ratios.append(new_df[key] / new_df[tot_queue_key])

    # unknown_jitter = new_df[tot_queue_key]
    # for key in factor_keys:
    #     unknown_jitter = unknown_jitter - new_df[key]
    # factor_ratios.append(unknown_jitter/new_df[tot_queue_key])

    new_keys = factor_key_without_postfix
    # new_keys.append('unknown_jitter')

    plot_cdf(
        factor_ratios, "Ratio by amp" + postfix, "CDF", labels=new_keys, postfix=postfix
    )


if __name__ == "__main__":

    # plot_multi_log_results([
    #     'test_data/result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-optimal-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    # ], postfix='_bonusfps30', find_potential_gain=True, fps_gain_thr = -20, draw_fps_only=True)

    # plot_multi_log_results([
    #     'test_data/result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    # ], postfix='_tmp', find_potential_gain=True, fps_gain_thr = -5, draw=False)

    # plot_multi_log_results([
    #     'test_data/result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps10_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps10_lifo.csv',
    #     'test_data/result-optimal-periodrop2_quickdrop0_maxbuf2_bonusfps10_lifo.csv',
    # ], postfix='_bonusfps10', find_potential_gain=False, draw_fps_only=True)

    # plot_multi_log_results([
    #     'test_data/result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-optimal-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv'
    # ], postfix='_optimal_gain_comparison', find_potential_gain=True, fps_gain_thr = 3.5, draw=False)

    # plot_multi_log_results([
    #     'test_data/result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps10_lifo.csv'
    # ], postfix='_optimal_gain_comparison', find_potential_gain=True, fps_gain_thr = -10, draw=False)

    # plot_multi_log_results([
    #     'test_data/result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps10_lifo.csv',
    #     'test_data/result-optimal-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-optimal-periodrop2_quickdrop0_maxbuf2_bonusfps10_lifo.csv',
    # ], find_potential_gain=False)

    # plot_buffer_jitter_analyze('test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv')
    # plot_buffer_jitter_analyze('test_data/result-potential_gain3.5.csv', postfix='_potential_gain3.5')

    # plot_good_render_buffer_jitter_analyze('test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv', postfix='_no_filter')
    # plot_good_render_buffer_jitter_analyze('test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv', filter_file='test_data/filter.csv', postfix='_filtered')

    # plot_buffer_jitter_analyze('test_data/result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo.csv', postfix='_ETH', filter='2_3')
    # plot_buffer_jitter_analyze('test_data/result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo.csv', postfix='_WiFi24G', filter='3_1')
    # plot_buffer_jitter_analyze('test_data/result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo.csv', postfix='_WiFi5G', filter='3_2')
    # plot_buffer_jitter_analyze('test_data/result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo.csv', postfix='_ETH_good_render', filter='2_3', filter_render_problem=True)
    # plot_buffer_jitter_analyze('test_data/result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo.csv', postfix='_WiFi24G_good_render', filter='3_1', filter_render_problem=True)
    # plot_buffer_jitter_analyze('test_data/result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo.csv', postfix='_WiFi5G_good_render', filter='3_2', filter_render_problem=True)

    # plot_quick_drop_results('test_data/result-simpleCtrl-periodrop2_quickdrop5_maxbuf2_bonusfps30_lifo.csv', postfix='quickdrop5')
    # plot_quick_drop_results('test_data/result-simpleCtrl-periodrop2_quickdrop7_maxbuf2_bonusfps30_lifo_v2.csv', postfix='quickdrop7v2')
    # plot_quick_drop_results('test_data/result-simpleCtrl-periodrop2_quickdrop7_maxbuf2_bonusfps30_lifo.csv', postfix='quickdrop7')
    # plot_quick_drop_results('test_data/result-simpleCtrl-periodrop2_quickdrop8_maxbuf2_bonusfps30_lifo.csv', postfix='quickdrop8')
    # plot_quick_drop_results('test_data/result-simpleCtrl-periodrop2_quickdrop9_maxbuf2_bonusfps30_lifo.csv', postfix='quickdrop9')

    # plot_buffer_exploitable_jitter_analyze('test_data/result-optimal-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv')
    # plot_buffer_exploitable_jitter_analyze('test_data/result-potential_gain3.5.csv', postfix='_potential_gain3.5')

    # plot_multi_log_results([
    #     'test_data/result-naiveVsync-periodrop1_quickdrop0_maxbuf4_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-optimal-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    # ], postfix='_bufsize4_comparison', find_potential_gain=False)

    # plot_multi_log_results([
    #     'test_data/result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop5_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop7_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop8_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-simpleCtrl-periodrop2_quickdrop9_maxbuf2_bonusfps30_lifo.csv',
    #     'test_data/result-optimal-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv',
    # ], postfix='_quickdrop_comparison', find_potential_gain=False)

    # plot_buffer_jitter_interval('/mydata/clwwwu/frame_log/sample_gain3.5', postfix='_gain3.5_comparsion')
    # plot_buffer_jitter_interval('/mydata/clwwwu/frame_log/sample_gain2', postfix='_gain2.0_comparsion')

    # compare_two_log([
    #     'test_data/result-optimal-periodrop2_quickdrop0_maxbuf2_bonusfps30_objective_fps_lifo.csv',
    #     'test_data/result-optimal-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv'
    # ], postfix='_optimal_gain_comparison', find_potential_gain=True, fps_gain_thr = -10, draw=False)

    plot_jitter_analysis_results(
        "test_data/result_jitter_analysis_start3600_len72000.csv"
    )
    plot_buffer_jitter_analyze_by_amplitude(
        "test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv"
    )
    plot_buffer_jitter_analyze_by_amplitude(
        "test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv",
        postfix="_ETH",
        filter="2_3.csv",
    )
    plot_buffer_jitter_analyze_by_amplitude(
        "test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv",
        postfix="_WiFi24G",
        filter="3_1.csv",
    )
    plot_buffer_jitter_analyze_by_amplitude(
        "test_data/result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo.csv",
        postfix="_WiFi5G",
        filter="3_2.csv",
    )
