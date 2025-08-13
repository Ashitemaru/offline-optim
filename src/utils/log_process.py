import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, "../.."))

from src.utils.data_loader import baseline_loader, log_loader


def calc_tail_fps(df):
    frame_ts = []
    for i in range(len(df)):
        if df["valid_frame_flag"].iloc[i] == 1:
            frame_ts.append(df["actual_display_ts"].iloc[i])

    # Use 1 second as a window
    window_fps = []
    lower_ptr = 0
    for higher_ptr in range(len(frame_ts)):
        if frame_ts[higher_ptr] - frame_ts[0] <= 1000:
            continue

        while frame_ts[higher_ptr] - frame_ts[lower_ptr] > 1000:
            lower_ptr += 1
        ts_diff = frame_ts[higher_ptr] - frame_ts[lower_ptr]
        ts_diff = 1e-3 if ts_diff == 0 else ts_diff
        window_fps.append((higher_ptr - lower_ptr + 1) / ts_diff * 1000)

    return (
        np.percentile(window_fps, 1),
        np.percentile(window_fps, 0.5),
        np.percentile(window_fps, 0.1),
    )


def calc_tail_delay(df):
    delay = []
    for i in range(len(df)):
        if df["valid_frame_flag"].iloc[i] == 1:
            delay.append(df["delay_time"].iloc[i])

    # Calc window average, window size = 60 frames
    window_delay = []
    for i in range(len(delay) - 60):
        window_delay.append(np.mean(delay[i : i + 60]))

    return (
        np.percentile(window_delay, 99),
        np.percentile(window_delay, 99.5),
        np.percentile(window_delay, 99.9),
    )


EXPERIMENT_LIST = [
    # "baseline_alltear_buf2_normal",
    # "vi_alltear_buf2_released",
    # "vi_notear_buf2_released",
    # "vi_toptear_buf2_released",
    # "vi_toptear_bufmix_released",
    "vsync_alltear_buf3_normal",
]

if __name__ == "__main__":
    for experiment in EXPERIMENT_LIST:
        root_path = f"../detailed_log/{experiment}"
        log_file = open(
            f"../log/{experiment.replace('_', '-').replace('vi', 'VI')}-tail.log", "w"
        )
        for directory in os.listdir(root_path):
            for file_name in tqdm(os.listdir(os.path.join(root_path, directory))):
                path = os.path.join(root_path, directory, file_name)
                df = pd.read_csv(path)

                tail_fps = calc_tail_fps(df)
                tail_delay = calc_tail_delay(df)

                log_file.write(
                    "%s\t%f\t%f\n"
                    % (
                        path.replace(root_path + "/", ""),
                        tail_fps[0],
                        tail_delay[0],
                    )
                )
                log_file.flush()

        log_file.close()
    exit(0)
    final_data = {}
    df = pd.read_csv("./log_summary_append.csv")

    # DF -> final_data
    for i in range(len(df)):
        trace = df["trace"].iloc[i]
        final_data[trace] = -np.ones(72)
        final_data[trace][:45] = df.iloc[i, -45:].to_numpy()

    for directory in os.listdir("../data"):
        if not directory.startswith("session"):
            continue

        for file_name in tqdm(os.listdir(os.path.join("../data", directory))):
            path = os.path.join("../data", directory, file_name)
            key = os.path.join(directory, "_".join(file_name.split("_")[:3]) + ".csv")

            if "baseline_buf1_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][45:48] = calc_tail_delay(df)
            elif "optim_buf1_tear0.05_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][48:51] = calc_tail_delay(df)
            elif "optim_buf1_tear0.10_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][51:54] = calc_tail_delay(df)
            elif "optim_buf1_tear0.15_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][54:57] = calc_tail_delay(df)
            elif "optim_buf1_tear0.20_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][57:60] = calc_tail_delay(df)
            elif "optim_buf1_tear0.25_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][60:63] = calc_tail_delay(df)
            elif "optim_buf1_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][63:66] = calc_tail_delay(df)
            elif "optim_buf1_tear1.00_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][66:69] = calc_tail_delay(df)
            elif "optim_buf1_tear0.00_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][69:72] = calc_tail_delay(df)

    final_df = pd.DataFrame(
        columns=[
            "trace",
            "baseline_fps",
            "baseline_delay",
            "5%_tearing_fps",
            "5%_tearing_delay",
            "10%_tearing_fps",
            "10%_tearing_delay",
            "15%_tearing_fps",
            "15%_tearing_delay",
            "20%_tearing_fps",
            "20%_tearing_delay",
            "25%_tearing_fps",
            "25%_tearing_delay",
            "optim_fps",
            "optim_delay",
            "baseline_1%_tail_fps",
            "baseline_0.5%_tail_fps",
            "baseline_0.1%_tail_fps",
            "5%_tearing_1%_tail_fps",
            "5%_tearing_0.5%_tail_fps",
            "5%_tearing_0.1%_tail_fps",
            "10%_tearing_1%_tail_fps",
            "10%_tearing_0.5%_tail_fps",
            "10%_tearing_0.1%_tail_fps",
            "15%_tearing_1%_tail_fps",
            "15%_tearing_0.5%_tail_fps",
            "15%_tearing_0.1%_tail_fps",
            "20%_tearing_1%_tail_fps",
            "20%_tearing_0.5%_tail_fps",
            "20%_tearing_0.1%_tail_fps",
            "25%_tearing_1%_tail_fps",
            "25%_tearing_0.5%_tail_fps",
            "25%_tearing_0.1%_tail_fps",
            "optim_1%_tail_fps",
            "optim_0.5%_tail_fps",
            "optim_0.1%_tail_fps",
            "100%_tearing_fps",
            "100%_tearing_delay",
            "0%_tearing_fps",
            "0%_tearing_delay",
            "100%_tearing_1%_tail_fps",
            "100%_tearing_0.5%_tail_fps",
            "100%_tearing_0.1%_tail_fps",
            "0%_tearing_1%_tail_fps",
            "0%_tearing_0.5%_tail_fps",
            "0%_tearing_0.1%_tail_fps",
            "baseline_1%_delay",
            "baseline_0.5%_delay",
            "baseline_0.1%_delay",
            "5%_tearing_1%_delay",
            "5%_tearing_0.5%_delay",
            "5%_tearing_0.1%_delay",
            "10%_tearing_1%_delay",
            "10%_tearing_0.5%_delay",
            "10%_tearing_0.1%_delay",
            "15%_tearing_1%_delay",
            "15%_tearing_0.5%_delay",
            "15%_tearing_0.1%_delay",
            "20%_tearing_1%_delay",
            "20%_tearing_0.5%_delay",
            "20%_tearing_0.1%_delay",
            "25%_tearing_1%_delay",
            "25%_tearing_0.5%_delay",
            "25%_tearing_0.1%_delay",
            "optim_1%_delay",
            "optim_0.5%_delay",
            "optim_0.1%_delay",
            "100%_tearing_1%_delay",
            "100%_tearing_0.5%_delay",
            "100%_tearing_0.1%_delay",
            "0%_tearing_1%_delay",
            "0%_tearing_0.5%_delay",
            "0%_tearing_0.1%_delay",
        ]
    )
    # Add a row into df
    for trace in final_data.keys():
        final_df.loc[len(final_df)] = [
            trace,
            *final_data[trace],
        ]
    # Save the df
    final_df.to_csv("./log_summary_append2.csv", float_format="%.5f")
    exit(0)
    final_data = {}
    # 0: baseline FPS
    # 1: baseline avg delay
    # 2: 5% tearing FPS
    # 3: 5% tearing delay
    # 4: 10% tearing FPS
    # 5: 10% tearing delay
    # 6: 15% tearing FPS
    # 7: 15% tearing delay
    # 8: 20% tearing FPS
    # 9: 20% tearing delay
    # 10: 25% tearing FPS
    # 11: 25% tearing delay
    # 12: optim FPS
    # 13: optim avg delay

    # Get baseline FPS / delay
    baseline = log_loader("../log/baseline-enhanced.log")
    for trace in baseline.keys():
        final_data[trace] = -np.ones(35)
        final_data[trace][:2] = baseline[trace][:2]

    # Get VI FPS / delay
    vi_log = log_loader("../log/VI-tearing-buffer2.log")
    for trace in vi_log.keys():
        assert trace in final_data
        final_data[trace][2:14] = vi_log[trace]

    # Calc the tail FPS
    for directory in os.listdir("../data"):
        if not directory.startswith("session"):
            continue

        for file_name in tqdm(os.listdir(os.path.join("../data", directory))):
            path = os.path.join("../data", directory, file_name)
            key = os.path.join(directory, "_".join(file_name.split("_")[:3]) + ".csv")

            if "baseline_buf1_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][14:17] = calc_tail_fps(df)
            elif "optim_buf1_tear0.05_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][17:20] = calc_tail_fps(df)
            elif "optim_buf1_tear0.10_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][20:23] = calc_tail_fps(df)
            elif "optim_buf1_tear0.15_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][23:26] = calc_tail_fps(df)
            elif "optim_buf1_tear0.20_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][26:29] = calc_tail_fps(df)
            elif "optim_buf1_tear0.25_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][29:32] = calc_tail_fps(df)
            elif "optim_buf1_sim" in file_name:
                assert key in final_data, key
                df = pd.read_csv(path)
                final_data[key][32:35] = calc_tail_fps(df)

    df = pd.DataFrame(
        columns=[
            "trace",
            "baseline_fps",
            "baseline_delay",
            "5%_tearing_fps",
            "5%_tearing_delay",
            "10%_tearing_fps",
            "10%_tearing_delay",
            "15%_tearing_fps",
            "15%_tearing_delay",
            "20%_tearing_fps",
            "20%_tearing_delay",
            "25%_tearing_fps",
            "25%_tearing_delay",
            "optim_fps",
            "optim_delay",
            "baseline_1%_tail_fps",
            "baseline_0.5%_tail_fps",
            "baseline_0.1%_tail_fps",
            "5%_tearing_1%_tail_fps",
            "5%_tearing_0.5%_tail_fps",
            "5%_tearing_0.1%_tail_fps",
            "10%_tearing_1%_tail_fps",
            "10%_tearing_0.5%_tail_fps",
            "10%_tearing_0.1%_tail_fps",
            "15%_tearing_1%_tail_fps",
            "15%_tearing_0.5%_tail_fps",
            "15%_tearing_0.1%_tail_fps",
            "20%_tearing_1%_tail_fps",
            "20%_tearing_0.5%_tail_fps",
            "20%_tearing_0.1%_tail_fps",
            "25%_tearing_1%_tail_fps",
            "25%_tearing_0.5%_tail_fps",
            "25%_tearing_0.1%_tail_fps",
            "optim_1%_tail_fps",
            "optim_0.5%_tail_fps",
            "optim_0.1%_tail_fps",
        ]
    )
    # Add a row into df
    for trace in final_data.keys():
        df.loc[len(df)] = [
            trace,
            *final_data[trace],
        ]
    # Save the df
    df.to_csv("./log_summary.csv", float_format="%.5f")
    exit(0)
    final_log_file = open("../log/baseline-enhanced-1lowfps-clip.log", "w")
    for directory in os.listdir("../data"):
        if not directory.startswith("session"):
            continue

        for file_name in tqdm(os.listdir(os.path.join("../data", directory))):
            if "optim" in file_name or "sim" not in file_name:
                continue
            path = os.path.join("../data", directory, file_name)
            df = pd.read_csv(path)

            frame_time = []
            prev_display_ts = None
            for i in range(1, len(df)):
                if df["valid_frame_flag"].iloc[i] == 1:
                    if prev_display_ts is None:
                        prev_display_ts = df["actual_display_ts"].iloc[i]
                    else:
                        frame_time.append(
                            df["actual_display_ts"].iloc[i] - prev_display_ts
                        )
                        prev_display_ts = df["actual_display_ts"].iloc[i]

            frame_time.sort()
            frame_time = [x for x in frame_time if x < 17 * 8]
            result = []
            for ratio in [1, 0.01, 0.005, 0.001]:
                frame_time_slice = frame_time[-int(len(frame_time) * ratio) :]
                # print(frame_time_slice)
                fps = 1000 / np.mean(frame_time_slice)
                result.append(fps)

            final_log_file.write(
                "%s\t%f\t%f\t%f\t%f\n"
                % (
                    path.replace("../data/", ""),
                    *result,
                )
            )
            final_log_file.flush()

    final_log_file.close()
    exit(0)
    final_log = pd.read_csv(
        "../final_data/session_info_9.150.5.177_2023-11-28/bad_346550_0,3,3,1_final.csv"
    )
    # final_log["aligned_offset"] = final_log["frame_ready_ts"] - final_log["nearest_display_ts"].iloc[0]
    final_log["aligned_max_ready_ts"] = (
        final_log[["frame_ready_ts", "expect_display_ts"]].max(axis=1)
        - final_log["nearest_display_ts"].iloc[0]
    )
    final_log["aligned_actual_display_ts"] = (
        final_log["actual_display_ts"] - final_log["nearest_display_ts"].iloc[0]
    )
    # final_log["aligned_slot"] = final_log["aligned_offset"] // 17
    # final_log["aligned_offset"] -= final_log["aligned_slot"] * 17
    final_log.iloc[:1000, :].to_csv("./tmp.csv")
    tot_frame = len(final_log)
    baseline_valid = final_log["valid_frame_flag"].to_numpy()
    optim_valid = final_log["optimal_valid_frame_flag"].to_numpy()

    baseline_display_slot = final_log["actual_display_slot"].to_numpy()
    optim_display_slot = final_log["optimal_display_slot"].to_numpy()

    plt.figure()
    plt.plot(baseline_display_slot[:20], label="baseline")
    plt.plot(optim_display_slot[:20], label="optim")
    plt.legend()
    plt.savefig("../image/display_slot.png")
    exit(0)
    baseline = log_loader("../log/baseline-adapted.log")
    vi_log = log_loader("../log/VI-adapted-buffer2.log")
    vsync_log = log_loader("../log/vsync-buffer2.log")

    df = pd.DataFrame(
        columns=[
            "trace",
            "baseline_fps",
            "optim_fps",
            "vsync_fps",
            "baseline_delay",
            "optim_delay",
            "vsync_delay",
            "baseline_valid_ratio",
            "optim_valid_ratio",
            "vsync_valid_ratio",
        ]
    )
    # Add a row into df
    for trace in baseline.keys():
        baseline_fps, baseline_delay, _, baseline_valid_ratio = baseline[trace]
        optim_fps, optim_delay, _, optim_valid_ratio = vi_log[trace]
        vsync_fps, vsync_delay, _, vsync_valid_ratio = vsync_log[trace]

        df.loc[len(df)] = [
            trace,
            baseline_fps,
            optim_fps,
            vsync_fps,
            baseline_delay,
            optim_delay,
            vsync_delay,
            baseline_valid_ratio,
            optim_valid_ratio,
            vsync_valid_ratio,
        ]

    df.to_csv("../final_data/log_summary.csv", float_format="%.5f")
    exit(0)
    with open("../log/baseline-adapted.log", "r") as f:
        for line in tqdm(f.readlines()):
            suffix = line.split("\t")[0][:-4] + "_baseline_buf1_sim.csv"
            baseline = pd.read_csv("../data/" + suffix)

            suffix = line.split("\t")[0][:-4] + "_optim_buf1_sim.csv"
            optim = pd.read_csv("../data/" + suffix)

            baseline["optimal_display_ts"] = optim["actual_display_ts"]
            baseline["optimal_display_slot"] = optim["actual_display_slot"]
            baseline["optimal_invoke_present_ts"] = optim["invoke_present_ts"]
            baseline["optimal_delay"] = optim["delay_time"]
            baseline["optimal_valid_frame_flag"] = optim["valid_frame_flag"]

            save_path = "../final_data/" + line.split("\t")[0][:-4] + "_final.csv"
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            baseline.to_csv(save_path, index=False)
