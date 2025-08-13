import numpy as np
import matplotlib.pyplot as plt


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
    plt.savefig(xlabel + ".jpg", dpi=300)


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

    opted_fps = [[] for i in range(4)]
    gains = [[] for i in range(4)]
    overheads = [[] for i in range(4)]

    for line in res_file.readlines():
        items = line.split(",")
        items = [float(i.strip()) for i in items[1:]]
        (
            max_fps,
            min_fps,
            opt_fps,
            buf0_fps,
            buf1_fps,
            buf2_fps,
            overhead,
            buf0_overhead,
            buf1_overhead,
            buf2_overhead,
            avg_dec_time,
            avg_dec_total_time,
            avg_render_time,
            avg_proc_time,
            recv_interval,
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
        )

        if opt_fps > min_fps:
            max_fps_res.append(max_fps)
            min_fps_res.append(min_fps)

            opted_fps[0].append(opt_fps)
            opted_fps[1].append(buf0_fps)
            opted_fps[2].append(buf1_fps)
            opted_fps[3].append(buf2_fps)

            # gains[0].append(abs(min(p_1more_fps, max_fps) - a_1more_fps))
            # gains[1].append(abs(max(p_1less_fps, min_fps) - a_1less_fps))
            # gains[0].append(min(p_1more_fps, max_fps) - a_1more_fps)
            # gains[1].append(max(p_1less_fps, min_fps) - a_1less_fps)

            overheads[0].append(overhead)
            overheads[1].append(buf0_overhead)
            overheads[2].append(buf1_overhead)
            overheads[3].append(buf2_overhead)

    plot_cdf(
        [max_fps_res, min_fps_res] + opted_fps,
        "FPS",
        "CDF",
        labels=[
            "Max FPS",
            "Current FPS",
            "AdaptBuf FPS",
            "Rectified FPS",
            "Buf1 FPS",
            "Buf2 FPS",
        ],
        xlim=[35, 60],
    )
    # plot_cdf(gains, 'Error', 'CDF', labels=['1MoreBuf', '1LessBuf'])
    plot_cdf(
        overheads,
        "Overhead (ms)",
        "CDF",
        labels=["AdaptBuf", "Rectified", "1 Frame buffer", "2 Frame buffer"],
    )


if __name__ == "__main__":
    plot_adapt_buf_result(
        "result_adaptBuf_procTime1_jitterBuffer1_tsExtrapolator1_renderQueue1_strict1_renderCtrlable0_frameInterval17.csv"
    )
