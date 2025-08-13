import numpy as np


class TraceValueIterator:
    def __init__(
        self,
        aligned_ts,
        slot_duration=17,
        delay_threshold=2,
        qoe_coefficient=0.02,
        tearing_protect=False,
        top_tearing_space_threshold=4,
        top_tearing_freq_threshold=0.1,
    ):
        self.T = slot_duration
        self.n = delay_threshold
        self.L = len(aligned_ts)
        self.delta = np.diff(aligned_ts)

        # state = (frame_ptr, relative_ts)
        size = (self.L, (self.n + 1) * self.T)
        self.value = np.zeros(size, dtype=np.float64)
        self.policy = np.zeros(size, dtype=np.int8)

        # init
        self.value[-1, :] = 1
        for i in range(-self.n, -1):
            self.policy[-1, i * self.T : (i + 1) * self.T] = -i
        self.policy[-1, -self.T :] = 1

        self.gamma = 1
        self.alpha = qoe_coefficient

        # tearing restricted
        self.restricted = np.zeros((self.L,))
        if tearing_protect:
            top_restricted_idx = np.array(
                [
                    idx
                    for idx, ts in enumerate(aligned_ts)
                    if ts % slot_duration <= top_tearing_space_threshold
                ]
            )
            top_restricted_num = int(
                len(top_restricted_idx) * (1 - top_tearing_freq_threshold)
            )
            top_restricted_idx = np.random.choice(
                top_restricted_idx, top_restricted_num, replace=False
            )
            bottom_restricted_idx = np.array(
                [
                    idx
                    for idx, ts in enumerate(aligned_ts)
                    if ts % slot_duration > top_tearing_space_threshold
                ]
            )
            self.restricted[top_restricted_idx] = 1
            self.restricted[bottom_restricted_idx] = 1

    def get_value(self, index, ts):
        ts = int(ts)

        if ts >= 0:
            return self.value[index, ts % self.T]
        elif -self.n * self.T <= ts < 0:
            return self.value[index, ts]
        elif index < self.L - 1:
            return self.gamma * self.get_value(index + 1, ts + self.delta[index])
        else:
            return 0

    def get_policy(self, index, ts):
        ts = int(ts)

        if ts >= 0:
            return self.policy[index, ts % self.T]
        elif -self.n * self.T <= ts < 0:
            return self.policy[index, ts]
        else:
            return -1

    def fit(self):
        for index in reversed(range(self.L - 1)):
            for ts in range(-self.n * self.T, self.T):  # traverse all states
                max_reward = -1
                best_action = -1
                for action in [-1] + list(
                    reversed(range(self.n + 1))
                ):  # traverse all actions
                    next_free_slot = (
                        0 if action == -1 else np.floor(ts / self.T) + action + 1
                    )
                    if (
                        action != -1 and next_free_slot <= 0
                    ):  # if the frame is to be reserved, but it cannot meet the lowest free slot, discard
                        continue

                    if (
                        self.restricted[index] == 1 and action == 0
                    ):  # if the frame is restricted by tearing, discard action == 0
                        continue

                    next_relative_ts = ts + self.delta[index] - next_free_slot * self.T
                    if action > 0:
                        delay = action * self.T - (ts + self.n * self.T) % self.T
                    elif action == 0:
                        delay = 0
                    elif action == -1:
                        delay = self.T * self.n / 2 / (1 - 1 / 12)  # average delay
                    reward = (
                        int(action != -1)
                        - self.alpha * delay
                        + self.gamma * self.get_value(index + 1, next_relative_ts)
                    )
                    if reward >= max_reward:
                        max_reward = reward
                        best_action = action

                self.value[index, ts] = max_reward
                self.policy[index, ts] = best_action


if __name__ == "__main__":
    # aligned_ts = np.load("../model/session_info_9.136.191.200_2023-11-28/good_863890_1,2,3,2.npy")
    iterator = TraceValueIterator(np.arange(0, 1000, 17))
    # iterator.fit()
    # print(iterator.policy[:20, :])
    # print(iterator.value[:20, :])
