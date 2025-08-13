import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import numpy as np
import os


# No float point numbers...
# But why and how?
# How: Add eps=1e-5, but that is way too complex for this toy
SLOT_DURATION = 17  # In ms
VALID_PORTION = 1

CSV_PATH = "./data.csv"
DECODED_TS = []  # Will be inited later
ALIGNED_TS = []  # Will be inited later
FIXED_DELAY = []  # Will be inited later
LOWEST_VALID_SLOT = []  # Will be inited later

LATENCY_PANELTY = 10


def lowest_slot(ts):
    if ts < VALID_PORTION * SLOT_DURATION:
        return 0
    else:
        return (ts - VALID_PORTION * SLOT_DURATION) // SLOT_DURATION + 1


def read_csv(path):
    trace = {}
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                keys = [x.strip() for x in line.split(",")]
                for key in keys:
                    trace[key.strip()] = []
                continue

            for j, value in enumerate(line.split(",")):
                trace[keys[j]].append(float(value.strip()))

    return trace


def init_global(path=CSV_PATH):
    global DECODED_TS, FIXED_DELAY, LOWEST_VALID_SLOT, ALIGNED_TS

    trace = read_csv(path)

    # Judge the frame rate
    estimated_frame_rate = (
        len(trace["pts"]) / (trace["pts"][-1] - trace["pts"][0]) * 1000
    )
    if estimated_frame_rate < 50 or estimated_frame_rate > 70:
        return False

    DECODED_TS = [
        recv_ts + proc_time
        for recv_ts, proc_time, loss_type in zip(
            trace["client_recv_ts"],
            trace["proc_time"],
            trace["loss_type"],
        )
        if loss_type == 0
    ]
    FIXED_DELAY = [0] * len(DECODED_TS)  # TODO

    start_ts = DECODED_TS[0]
    ALIGNED_TS = [x - start_ts + trace["vsync_diff"][0] for x in DECODED_TS]
    LOWEST_VALID_SLOT = [lowest_slot(ts) for ts in ALIGNED_TS]
    assert [
        i for i in range(len(DECODED_TS) - 1) if DECODED_TS[i] > DECODED_TS[i + 1]
    ] == []
    return True


def eval_delay(delay, debug=False):
    assert delay[0] == 0
    assert len(delay) == len(ALIGNED_TS)
    if debug:
        print(f"Received {len(ALIGNED_TS)} frames")

    play_ts = [x + y for x, y in zip(ALIGNED_TS, delay) if y != -1]
    if debug:
        print(f"Discard frames from the buffer, {len(play_ts)} frames left")

    # Discard frames with the same slot
    play_slot = [x // SLOT_DURATION for x in play_ts]
    play_ts = [
        x for i, x in enumerate(play_ts) if i == 0 or play_slot[i] != play_slot[i - 1]
    ]
    if debug:
        print(f"Discard frames with slot collision, {len(play_ts)} frames left")

    # Calculate frame rate & tearing rate
    frame_rate = len(play_ts) / (play_ts[-1] - play_ts[0]) * 1000
    tearing_rate = (
        len([x for x in play_ts if x / SLOT_DURATION - x // SLOT_DURATION >= 4 / 17])
        / (play_ts[-1] - play_ts[0])
        * 1000
    )

    return frame_rate, tearing_rate


def naive_delay(delay_slot_threshold=2):
    delay = [0]
    occupied_slot = [0]
    for i in range(1, len(ALIGNED_TS)):
        for j in range(delay_slot_threshold + 1):
            slot = LOWEST_VALID_SLOT[i] + j

            # Update this
            found = False
            for occupied in reversed(occupied_slot):
                if occupied == slot:
                    found = True
                    break
                if occupied < slot:
                    found = False
                    break
            if not found:
                delay.append(max(0, slot * SLOT_DURATION - ALIGNED_TS[i]))
                occupied_slot.append(slot)
                break

        if len(delay) != i + 1:
            delay.append(-1)

    return delay


def greedy_delay(predict_depth=5, latency_threshold=20):
    delay = [0]
    extra_delay_list = [0]
    lowest_free_slot = 1

    index = 1
    while index < len(ALIGNED_TS):
        predicted_slot = []
        for next_frame in range(index, min(index + predict_depth, len(ALIGNED_TS))):
            slot = max(lowest_free_slot, LOWEST_VALID_SLOT[next_frame])

            # `slot` can be affected by `lowest_free_slot`
            # While `LOWEST_FREE_SLOT` is a totally greedy choice
            # The cost we should pay for a more reasonable choice is the `extra_delay`
            extra_delay = max(0, slot * SLOT_DURATION - ALIGNED_TS[next_frame]) - max(
                0,
                LOWEST_VALID_SLOT[next_frame] * SLOT_DURATION - ALIGNED_TS[next_frame],
            )
            predicted_slot.append(slot if extra_delay < latency_threshold else 1e8)

        # If we cannot meet the latency requirement, we just drop the frame
        if (np.array(predicted_slot) == 1e8).all():
            delay.append(-1)
            index += 1
            continue

        predicted_slot = np.array(predicted_slot)
        selected_loss = np.argmin(predicted_slot)
        selected_slot = max(lowest_free_slot, LOWEST_VALID_SLOT[index + selected_loss])

        delay += [-1] * selected_loss
        actual_delay = max(
            0, selected_slot * SLOT_DURATION - ALIGNED_TS[index + selected_loss]
        )
        totally_greedy_delay = max(
            0,
            LOWEST_VALID_SLOT[index + selected_loss] * SLOT_DURATION
            - ALIGNED_TS[index + selected_loss],
        )
        delay.append(actual_delay)
        extra_delay_list.append(actual_delay - totally_greedy_delay)

        lowest_free_slot = selected_slot + 1
        index += 1 + selected_loss

    return delay, sum(extra_delay_list) / len(extra_delay_list)


if __name__ == "__main__":
    # init_global()
    # delay = naive_delay(delay_slot_threshold=3)
    # frame_rate = eval_delay(delay, debug=True)
    # print(f"Frame rate: {frame_rate:.2f} FPS")
    # print(f"Average delay: {np.average([x for x in delay if x != -1]):.2f} ms")
    # print(f"Discard frames: {sum(1 for x in delay if x == -1)}")
    # exit(0)

    # Create a .log file in the log folder
    # Check whether the log folder exists before creating the .log file
    if not os.path.exists("./log"):
        os.mkdir("./log")
    log_file = open("./log/allow_tearing.log", "w")

    # Traverse the data folder and init the global
    for file in os.listdir("./data"):
        if not file.startswith("session"):
            continue

        for data in tqdm(os.listdir(os.path.join("./data", file))):
            if "sim" in data:
                continue
            path = os.path.join(file, data)
            if init_global(os.path.join("./data", file, data)):
                delay = naive_delay(delay_slot_threshold=1)
                frame_rate, tearing_rate = eval_delay(delay)
                log_file.write(
                    f"{path}\t{frame_rate}\t{np.average([x for x in delay if x != -1])}\n"
                )
            else:
                log_file.write(f"{path}\tInvalid\n")
            log_file.flush()

    log_file.close()
    exit(0)

    init_global()

    delay_thresholds = list(range(1, 43))
    results = [greedy_delay(latency_threshold=x) for x in tqdm(delay_thresholds)]
    delays = [x[0] for x in results]
    extra_delays = [x[1] for x in results]
    frame_rates = [eval_delay(x) for x in delays]

    plt.figure()
    plt.plot(delay_thresholds, frame_rates)
    plt.xlabel("Latency Threshold (ms)")
    plt.ylabel("Frame Rate (fps)")
    plt.title("Frame Rate vs. Latency Threshold")
    plt.savefig("image/threshold_to_frame_rate.png")

    plt.figure()
    plt.plot(delay_thresholds, extra_delays)
    plt.plot([0, 42], [0, 42], linestyle="--", color="gray")
    plt.xlabel("Latency Threshold (ms)")
    plt.ylabel("Extra Delay (ms)")
    plt.title("Extra Delay vs. Latency Threshold")
    plt.savefig("image/threshold_to_extra_delay.png")
