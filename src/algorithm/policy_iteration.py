import os
import sys
import numpy as np
from itertools import product

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))

from src.env import TraceEnv


class TracePolicyIterator:
    def __init__(self, env):
        self.aligned_ts = env.aligned_ts
        self.delta_ts = np.diff(self.aligned_ts, append=0)
        self.T = env.slot_duration
        self.n = env.delay_threshold
        self.L = len(self.aligned_ts)

        # state = (frame_ptr, relative_ts)
        # relative_ts in [-nT, T), n is delay_threshold, T is slot_duration
        size = (self.L, self.T * (self.n + 1))
        self.policy = np.zeros(size, dtype=np.int8)  # pi(s)
        self.value = np.zeros(size, dtype=np.int32)  # V(s)

        # Init
        self.value[-1] = (self.policy[-1] != -1).astype(np.int32)

    def get_policy(self, index, ts):
        ts = int(ts)

        if ts >= self.T:
            return self.policy[index, ts % self.T]
        elif -self.n * self.T <= ts < self.T:
            return self.policy[index, ts]
        else:
            return -1

    def get_value(self, index, ts):
        ts = int(ts)

        if ts >= self.T:
            return self.value[index, ts % self.T]
        elif -self.n * self.T <= ts < self.T:
            return self.value[index, ts]
        elif index < self.L - 1:
            # delta = self.aligned_ts[index + 1] - self.aligned_ts[index]
            return self.get_value(index + 1, ts + self.delta_ts[index])
        else:
            return 0

    def __policy_evaluation(self):
        self.value[-1] = (self.policy[-1] != -1).astype(np.int32)
        for index in reversed(range(self.value.shape[0] - 1)):
            # delta = self.aligned_ts[index + 1] - self.aligned_ts[index]
            for ts in range(self.value.shape[1]):
                action = self.policy[index, ts]
                reward = int(action != -1)
                next_ts = ts + self.delta_ts[index] - (1 + action) * self.T
                self.value[index, ts] = reward + self.get_value(index + 1, next_ts)

    def __policy_improvement(self):
        old_policy = self.policy.copy()
        for index, ts in product(range(self.L), range(-self.n * self.T, self.T)):
            max_reward = 0
            best_action = -1
            for action in range(-1, self.n):
                # Restricted action space
                if action != -1 and action * self.T + ts < 0:
                    continue

                reward = int(action != -1)
                if index < self.L - 1:
                    # delta = self.aligned_ts[index + 1] - self.aligned_ts[index]
                    next_ts = ts + self.delta_ts[index] - (1 + action) * self.T
                    reward += self.get_value(index + 1, next_ts)
                if reward > max_reward:
                    max_reward = reward
                    best_action = action

            self.policy[index, ts] = best_action

        return np.count_nonzero(old_policy - self.policy)

    def fit(self):
        iter_counter = 0
        while True:
            print(f"Start iter #{iter_counter}")
            self.__policy_evaluation()
            improve_result = self.__policy_improvement()
            print(f"End iter #{iter_counter}, improved states: {improve_result}")
            if improve_result < 100:
                break

            iter_counter += 1


def get_policy_wrapper(policy, index, ts):
    ts = int(ts)

    if ts >= 17:
        return 1, policy[index, ts % 17]
    elif -34 <= ts < 17:
        return 0, policy[index, ts]
    else:
        return -1, -1


if __name__ == "__main__":
    PATH = "../data/session_info_9.136.191.200_2023-11-28/bad_876423_0,3,3,1.csv"
    POLICY_PATH = (
        "../model/session_info_9.136.191.200_2023-11-28/bad_876423_0,3,3,1.csv.npy"
    )
    env = TraceEnv(PATH, demo_trace=False)
    policy = np.load(POLICY_PATH)
    # print(policy_iterator.policy)

    # Run the policy over the env
    state = env.reset()
    terminal = False
    action_list = []
    while not terminal:
        old_state = state
        action_tuple = get_policy_wrapper(policy, state[0], state[1])
        action = action_tuple[1]
        action_list.append(action_tuple)
        state, reward, terminal, info = env.step(action)
        # print(f"State: {old_state}, Action: {action}, reward: {reward}, info: {info}")

    # Print the result
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist([x[0] for x in action_list if x[1] == -1], bins=range(-1, 4), align="left")
    plt.savefig("../image/action.png")
