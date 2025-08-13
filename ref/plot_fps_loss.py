import os, sys

import numpy as np
import matplotlib.pyplot as plt


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
    plt.grid()
    if title is None:
        plt.title(title)

    output_dir = os.path.join("test_data", output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(os.path.join(output_dir, filename + ".jpg"))
    plt.savefig(os.path.join(output_dir, filename + ".jpg"), dpi=200)
    plt.close(fig)


def plot_fps_loss(file_path):
    lines = open(file_path).readlines()[1:]

    i_jitter_ratios = []
    p_jitter_ratios = []
    for idx, line in enumerate(lines):
        items = line.strip().split(",")
        try:
            i_jitter_ratios.append(
                int(items[2]) / int(items[1]) if int(items[1]) > 0 else 0
            )
            p_jitter_ratios.append(
                int(items[4]) / int(items[3]) if int(items[3]) > 0 else 0
            )
        except:
            print(idx, items[1:5])

    plot_cdf(
        [i_jitter_ratios, p_jitter_ratios],
        "frame_type_jitter_ratio",
        "Jitter Frame Ratio",
        "CDF",
        labels=["I Frame", "P Frame"],
    )


if __name__ == "__main__":
    plot_fps_loss(sys.argv[1])
