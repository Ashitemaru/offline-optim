import os
import sys
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))

from src.algorithm.greedy import greedy_policy


def action_transform(
    action, aligned_ts, delay_threshold=2, slot_duration=17, tear_threshold=0
):
    assert len(action) == len(aligned_ts)

    ptr = len(action) - 1
    while ptr > 0:
        if action[ptr] == -1:
            pptr = ptr - 1
            while action[pptr] > 1 or (
                action[pptr] == 1 and aligned_ts[pptr] % slot_duration < tear_threshold
            ):
                action[pptr] -= 1
                pptr -= 1

            action[pptr + 1] = -1
            action[ptr] = delay_threshold
            ptr = pptr

        else:
            ptr -= 1

    return action


def mpc_action_transform(
    aligned_ts,
    action_buf1,
    action_buf2,
    start_slot,
    window_size=5 * 60,
    fps_diff_threshold=2,
):
    action_buf1 = action_transform(
        action_buf1, aligned_ts, tear_threshold=5, delay_threshold=1
    )
    action_buf2 = action_transform(
        action_buf2, aligned_ts, tear_threshold=5, delay_threshold=2
    )

    lowest_free_slot = start_slot
    action_bufmix = np.zeros(len(aligned_ts), dtype=int)
    max_buffer = 2 * np.ones(len(aligned_ts), dtype=int)
    for ptr in range(1, len(aligned_ts)):
        if max_buffer[ptr] == 2:
            fps_buf1 = sum(
                1 for x in action_buf1[ptr : ptr + window_size] if x != -1
            ) / (window_size / 60)
            fps_buf2 = sum(
                1 for x in action_buf2[ptr : ptr + window_size] if x != -1
            ) / (window_size / 60)
            if abs(fps_buf2 - fps_buf1) < fps_diff_threshold:
                max_buffer[ptr : ptr + window_size] = 1

        relative_ts = aligned_ts[ptr] - lowest_free_slot * 17
        action = greedy_policy(
            relative_ts=relative_ts, index=0, avg_delay=0, max_buffer=max_buffer[ptr]
        )
        action_bufmix[ptr] = action_buf2[ptr]

        if action != -1:
            invoke_ts = (
                aligned_ts[ptr]
                if action == 0
                else max(lowest_free_slot, np.ceil(aligned_ts[ptr] / 17) - 1 + action)
                * 17
            )
            lowest_free_slot = np.ceil(max(0, invoke_ts + 1) / 17)

    # TODO: An ugly hack to see the space for optimization, but it is not a good idea
    discard_num = int(len(action_bufmix) * 1.5 / 60)
    discard_idx = np.random.choice(len(action_bufmix), discard_num, replace=False)
    action_bufmix[discard_idx] = -1
    return action_transform(
        action_bufmix, aligned_ts, tear_threshold=5, delay_threshold=1
    )


if __name__ == "__main__":
    pass
