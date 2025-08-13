import os
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))

from utils.data_loader import log_loader

ROOT = "../data"

if __name__ == "__main__":
    data = []
    for coeff in [0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        log = log_loader(f"../log/VI-alltear-buf2-normal-qoe{coeff}.log")
        avg_fps = np.mean([v[0] for k, v in log.items()])
        avg_delay = np.mean([v[1] for k, v in log.items()])
        data.append([coeff, avg_fps, avg_delay])

    plt.figure()
    for coeff, avg_fps, avg_delay in data:
        plt.scatter(avg_delay, avg_fps, label=f"alpha={coeff}")
    plt.grid()
    plt.xlabel("Average Delay")
    plt.ylabel("Average FPS")
    plt.legend()
    plt.show()
