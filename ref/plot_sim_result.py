import numpy as np
import matplotlib.pyplot as plt


def plot_result(file_path):
    lines = open(file_path).readlines()
    net_time = []
    render_queue = []
    buf_size = []
    valid_fps = []

    cnt = 0
    for line in lines:
        cnt += 1
        if cnt < 1200:
            continue
        items = line.strip().split()
        net_time.append(int(items[79]))
        render_queue.append(int(items[73]))
        buf_size.append(int(items[115]))
        valid_fps.append(float(items[121]))

        if cnt > 6200:
            break

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.subplots_adjust(left=0.04, right=0.8)

    twin1 = ax.twinx()
    twin2 = ax.twinx()

    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    twin2.spines.right.set_position(("axes", 1.1))

    (p1,) = ax.plot(net_time, "C0", label="Netts")
    (p4,) = ax.plot(render_queue, "C3", label="Render queue")
    (p2,) = twin1.plot(buf_size, "C1", label="Buf size")
    (p3,) = twin2.plot(valid_fps, "C2", label="Valid FPS")

    ax.set(ylim=(0, 50), xlabel="Frame Index", ylabel="Time (ms)")
    twin1.set(ylim=(-10, 2), ylabel="No. of Frame")
    twin2.set(ylim=(0, 65), ylabel="FPS")

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    ax.tick_params(axis="y", colors=p1.get_color())
    twin1.tick_params(axis="y", colors=p2.get_color())
    twin2.tick_params(axis="y", colors=p3.get_color())

    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.legend(
        handles=[p1, p2, p3, p4],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=4,
    )

    plt.tight_layout()
    # plt.show()
    plt.grid()
    plt.savefig("sim_result.jpg", dpi=300)


if __name__ == "__main__":
    plot_result("test_data/frame_echo.log")
