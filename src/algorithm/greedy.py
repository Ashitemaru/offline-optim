import numpy as np
from random import random


def greedy_policy(index, relative_ts, avg_delay, max_buffer=2, slot_duration=17):
    delay_slot_cnt = -(np.ceil(relative_ts / slot_duration) - 1)
    if delay_slot_cnt > max_buffer:
        return -1
    else:
        return max(0, delay_slot_cnt)


def clipped_greedy_policy(
    index,
    relative_ts,
    avg_delay,
    avg_delay_threshold=10,
    delay_window_size=20,
    delay_slack=2,  # Allow to be a little bit larger than the delay threshold
    avg_tearing_threshold=2,
    tearing_slack=0.5,
    slot_slack=1,  # Loosen the constraint for discarding frames
    random_loss=0.5,
    max_buffer=1,
    slot_duration=17,
):
    for buffer in range(max_buffer + 1):
        if relative_ts < -buffer * slot_duration:
            continue

        # Only an approximate value, but maybe enough for the policy to judge
        delay = (buffer + 1) * slot_duration - relative_ts % slot_duration
        avg_delay = (avg_delay * delay_window_size + delay) / (delay_window_size + 1)

        # is_tearing = buffer == 0 and ts % slot_duration > tearing_threshold
        # tearing_rate = (tearing_frame_cnt + is_tearing) / ts if ts != 0 else 1e8

        if avg_delay < avg_delay_threshold + delay_slack:
            # and tearing_rate < avg_tearing_threshold + tearing_slack
            return buffer

    if relative_ts // slot_duration + max_buffer < slot_slack:
        return -1
    else:
        return -1 if random() < random_loss else 0


if __name__ == "__main__":
    pass
