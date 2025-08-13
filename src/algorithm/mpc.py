import numpy as np


# This is **not** a online RL alg, just use it as a baseline
# FIXME: Maybe BUGGY & Maybe used later but not now
def mpc_policy(
    frame_ptr,
    ts,
    lowest_free_slot,
    avg_delay,
    env,
    predict_depth=5,
    latency_threshold=10,
    slot_duration=17,
    tearing_threshold=4,
):
    tot_len = len(env.aligned_ts)
    predicted_slot = []
    for frame in range(frame_ptr, min(frame_ptr + predict_depth, tot_len)):
        slot = max(
            lowest_free_slot,
            1 + (ts - tearing_threshold) // slot_duration,
        )
        delay = max(0, slot * slot_duration - ts)
        predicted_slot.append(slot if delay < latency_threshold else 1e8)

    predicted_slot = np.array(predicted_slot)
    if (predicted_slot == 1e8).all():
        return [-1]

    selected_loss = predicted_slot.argmin()
    return [-1] * selected_loss + [
        max(
            0,
            lowest_free_slot
            - env.aligned_ts[frame_ptr + selected_loss] // slot_duration,
        )
    ]


if __name__ == "__main__":
    pass
