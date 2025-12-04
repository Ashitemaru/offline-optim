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

def cdf_pair(x):
    y = np.arange(len(x)) / float(len(x))
    return x, y

if __name__ == "__main__":
    data = []
    for mode in ["optim", "VI", "vsync"]:
        log = log_loader(f"../log/{mode}-notear-buf2-normal-qoe0.log")
        log = {k: v for k, v in log.items() if v[0] > 0}
        fps = np.array([v[0] for k, v in log.items()])
        delay = np.array([v[1] for k, v in log.items()])
        data.append([mode, fps, delay])

        print(mode, np.mean(fps), np.mean(delay))

    plt.figure()
    for mode, fps, _ in data:
        x, y = cdf_pair(np.sort(fps))
        plt.plot(x, y, label=mode.capitalize())
    plt.grid()
    plt.xlabel("Average FPS")
    plt.ylabel("CDF")
    plt.legend()
    plt.savefig("./fps.png")

    plt.figure()
    for mode, _, delay in data:
        x, y = cdf_pair(np.sort(delay))
        plt.plot(x, y, label=mode.capitalize())
    plt.grid()
    plt.xlabel("Average Delay (ms)")
    plt.ylabel("CDF")
    plt.legend()
    plt.savefig("./delay.png")
