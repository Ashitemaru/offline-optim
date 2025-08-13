import os, sys

import load_data

from scipy.fft import fft, ifft, fftfreq
from scipy.signal import argrelextrema
from scipy.stats import kstest, shapiro

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.structural import UnobservedComponents
from bsts import bsts
from hmmlearn.hmm import GaussianHMM

from functools import partial
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.priors import const_prior

from bayesian_changepoint_detection.bayesian_models import online_changepoint_detection
import bayesian_changepoint_detection.online_likelihoods as online_ll
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
import bayesian_changepoint_detection.offline_likelihoods as offline_ll


def test_offline_change_point_detection(data):
    prior_function = partial(const_prior, p=1 / (len(data) + 1))
    Q, P, Pcp = offline_changepoint_detection(
        data, prior_function, offline_ll.StudentT(), truncate=-40
    )

    fig, ax = plt.subplots(2, figsize=[18, 16], sharex=True)
    ax[0].plot(data[:])
    ax[1].plot(np.exp(Pcp).sum(0))

    plt.show()


def test_online_change_point_detection(data):
    hazard_function = partial(constant_hazard, 250)
    R, maxes = online_changepoint_detection(
        data, hazard_function, online_ll.StudentT(alpha=0.1, beta=0.01, kappa=3, mu=2)
    )

    epsilon = 1e-7
    # fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
    # ax[0].plot(data)
    # sparsity = 5  # only plot every fifth data for faster display
    # density_matrix = -np.log(R[0:-1:sparsity, 0:-1:sparsity]+epsilon)
    # ax[1].pcolor(np.array(range(0, len(R[:,0]), sparsity)),
    #         np.array(range(0, len(R[:,0]), sparsity)),
    #         density_matrix,
    #         cmap=cm.Greys, vmin=0, vmax=density_matrix.max(),
    #             shading='auto')
    # Nw=15
    # ax[2].plot(R[Nw,Nw:-1])

    fig, ax = plt.subplots(2, figsize=[18, 16], sharex=True)
    ax[0].plot(data)
    Nw = 15
    ax[1].plot(R[Nw, Nw:-1])

    plt.show()


def test_hmm(data):
    # 创建时间索引
    date_rng = pd.date_range(start="1/1/2020", periods=len(data), freq="D")

    # 将NumPy数组转换为Pandas Series，并添加时间索引
    series = pd.Series(data, index=date_rng)

    # 拟合隐马尔可夫模型
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000)
    model.fit(series.values.reshape(-1, 1))

    # 预测隐含状态
    hidden_states = model.predict(series.values.reshape(-1, 1))

    future_steps = 600
    future_states = model.sample(future_steps)[1]

    # 根据未来的状态序列生成未来的观测值
    future_observations = []
    for state in future_states:
        mean = model.means_[state]
        covar = np.sqrt(model.covars_[state])
        future_observations.append(np.random.normal(mean, covar))

    # 创建未来的时间索引
    future_date_rng = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq="D"
    )

    # 将未来的观测值转换为Pandas Series，并添加时间索引
    future_series = pd.Series(future_observations, index=future_date_rng)

    # 绘制结果
    plt.figure(figsize=(24, 8))
    plt.plot(series, label="Original Series")
    for i in range(model.n_components):
        state = hidden_states == i
        plt.plot(series.index[state], series.values[state], ".", label=f"State {i}")
    plt.plot(future_series, label="Future Series", linestyle="--")
    plt.legend()
    plt.show()


def test_state_spaces(data):
    # 创建时间索引
    date_rng = pd.date_range(start="1/1/2020", periods=len(data), freq="D")

    # 将NumPy数组转换为Pandas Series，并添加时间索引
    series = pd.Series(data, index=date_rng)

    # 拟合状态空间模型
    model = UnobservedComponents(series, level="local level")
    result = model.fit()

    # 提取成分
    trend = result.level.smoothed
    seasonal = result.seasonal.smoothed
    residual = result.resid

    # 绘制分解结果
    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.plot(series, label="Original Series")
    plt.legend()
    plt.subplot(412)
    plt.plot(trend, label="Trend")
    plt.legend()
    plt.subplot(413)
    plt.plot(seasonal, label="Seasonal")
    plt.legend()
    plt.subplot(414)
    plt.plot(residual, label="Residual")
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_bsts(data):
    # 创建时间索引
    date_rng = pd.date_range(start="1/1/2020", periods=len(data), freq="D")

    # 将NumPy数组转换为Pandas Series，并添加时间索引
    series = pd.Series(data, index=date_rng)

    # 拟合BSTS模型
    model = bsts.BSTS(series, seasonality=None)
    model.fit()

    # 提取成分
    trend = model.get_trend()
    seasonal = model.get_seasonal()
    residual = model.get_residual()

    # 绘制分解结果
    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.plot(series, label="Original Series")
    plt.legend()
    plt.subplot(412)
    plt.plot(trend, label="Trend")
    plt.legend()
    plt.subplot(413)
    plt.plot(seasonal, label="Seasonal")
    plt.legend()
    plt.subplot(414)
    plt.plot(residual, label="Residual")
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_outliers(time_series):
    # 计算均值和标准差
    mean = np.mean(time_series)
    std = np.std(time_series)

    # 定义阈值
    threshold = 3  # 可以根据需要调整

    # 识别异常值
    outlier_idx = np.where(
        (time_series > mean + threshold * std) | (time_series < mean - threshold * std)
    )[0]
    print(mean, std, outlier_idx)

    # 绘制原始数据和异常值
    plt.figure(figsize=(10, 5))
    plt.plot(time_series, label="Original Data")
    plt.scatter(outlier_idx, time_series[outlier_idx], color="r")
    plt.legend()
    plt.show()


def test_fft(time_series):
    # 假设 time_series 是你的原始时间序列数据
    N = len(time_series)
    # fs = 1.0 / (time_series[1] - time_series[0])  # 假设时间戳是均匀的

    # 1. 傅里叶变换
    signal_fft = fft(time_series)

    # 2. 设置阈值
    threshold = np.max(np.abs(signal_fft)) * 0.05  # 例如，阈值设为最大值的10%

    # 3. 滤波器设计
    # 创建一个掩码，将异常值置为0，其他保持不变
    mask = np.abs(signal_fft) > threshold
    filtered_fft = signal_fft * mask

    # 4. 逆傅里叶变换
    filtered_signal = ifft(filtered_fft)

    # 5. 重构信号
    filtered_signal = np.real(filtered_signal)

    # 6. 可视化结果
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_series)
    plt.title("Original Signal with Outliers")
    plt.subplot(2, 1, 2)
    plt.plot(filtered_signal)
    plt.title("Signal after Outlier Removal")
    plt.tight_layout()
    plt.show()


def cal_data_periodicity(time_series):
    pacf_values = pacf(time_series, nlags=100)
    ordered_idx = np.argsort(-pacf_values)
    top5_idx = ordered_idx[1:6]
    top5_value = pacf_values[top5_idx]
    # plot_pacf(time_series, lags=100, alpha=0.05)
    # plt.title('Partial Autocorrelation Function')
    # plt.show()

    return top5_idx.tolist(), top5_value.tolist()


# 假设 time_series 是你的数据
# time_series = np.array([...])
def test_periodicity(time_series):
    # 计算 ACF 和 PACF
    acf_values = acf(time_series, nlags=100)  # nlags 是要计算的滞后数
    pacf_values = pacf(time_series, nlags=100)

    plt.subplot(311)
    plt.plot(time_series)

    # 绘制 ACF 图
    print("acf", acf_values)
    print("acf order", np.argsort(acf_values))
    plot_acf(time_series, lags=100, ax=plt.subplot(312), alpha=0.05)
    plt.title("Autocorrelation Function")

    # 绘制 PACF 图
    print("pacf", pacf_values)
    print("pacf order", np.argsort(pacf_values))
    plot_pacf(time_series, lags=100, ax=plt.subplot(313), alpha=0.05)
    plt.title("Partial Autocorrelation Function")

    plt.tight_layout()
    plt.show()

    # date_rng = pd.date_range(start='1/1/2020', periods=len(time_series), freq='M')
    # # 将NumPy数组转换为Pandas Series，并添加时间索引
    # series = pd.Series(time_series, index=date_rng)
    # result = seasonal_decompose(series, model='additive')
    # result.plot()
    # plt.show()


def test_acorr_ljungbox(time_series):

    # statistic, p_value = kstest(data, 'norm')
    # print(f"KS检验统计量: {statistic}, p值: {p_value}")

    # statistic, p_value = shapiro(time_series)
    # print(f"Shapiro-Wilk检验统计量: {statistic}, p值: {p_value}")

    # Ljung-Box检验
    result = acorr_ljungbox(time_series, lags=100, return_df=True)
    print(result)


# sample data
# 17.0,18.0,16.0,16.0,17.0,16.0,18.0,16.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,18.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,18.0,14.0,16.0,18.0,43.0,5.0,3.0,16.0,16.0,18.0,16.0,16.0,22.0,12.0,16.0,18.0,16.0,17.0,16.0,16.0,17.0,17.0,15.0,17.0,18.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,42.0,4.0,3.0,16.0,18.0,19.0,13.0,19.0,16.0,16.0,17.0,17.0,15.0,17.0,17.0,17.0,17.0,18.0,15.0,18.0,16.0,16.0,17.0,17.0,16.0,17.0,16.0,18.0,15.0,17.0,17.0,17.0,17.0,16.0,17.0,17.0,16.0,16.0,17.0,17.0,17.0,17.0,16.0,17.0,16.0,17.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,16.0,17.0,17.0,17.0,16.0,16.0,18.0,16.0,16.0,18.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,15.0,18.0,17.0,15.0,17.0,18.0,16.0,17.0,16.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,17.0,17.0,17.0,15.0,17.0,17.0,16.0,17.0,18.0,16.0,16.0,17.0,16.0,17.0,18.0,15.0,17.0,18.0,15.0,17.0,17.0,16.0,49.0,3.0,5.0,10.0,18.0,22.0,11.0,17.0,16.0,17.0,17.0,18.0,15.0,17.0,43.0,5.0,4.0,15.0,17.0,17.0,16.0,17.0,17.0,16.0,16.0,25.0,8.0,17.0,17.0,16.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,82.0,7.0,11.0,22.0,12.0,16.0,17.0,16.0,17.0,18.0,15.0,17.0,53.0,53.0,5.0,6.0,16.0,17.0,17.0,16.0,17.0,17.0,17.0,16.0,17.0,17.0,15.0,18.0,16.0,17.0,16.0,18.0,15.0,18.0,16.0,17.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,17.0,17.0,16.0,17.0,16.0,44.0,5.0,4.0,13.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,25.0,9.0,17.0,17.0,17.0,16.0,17.0,39.0,5.0,5.0,18.0,16.0,16.0,17.0,16.0,17.0,17.0,17.0,16.0,17.0,16.0,17.0,18.0,15.0,17.0,48.0,5.0,4.0,10.0,16.0,17.0,17.0,30.0,3.0,35.0,7.0,8.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,17.0,17.0,17.0,16.0,18.0,16.0,17.0,17.0,44.0,6.0,4.0,11.0,17.0,17.0,15.0,17.0,17.0,16.0,18.0,16.0,16.0,18.0,16.0,16.0,21.0,14.0,16.0,18.0,17.0,21.0,12.0,17.0,15.0,17.0,17.0,16.0,17.0,16.0,17.0,17.0,53.0,4.0,4.0,4.0,17.0,17.0,17.0,16.0,18.0,16.0,16.0,17.0,18.0,15.0,17.0,17.0,16.0,17.0,18.0,16.0,17.0,17.0,16.0,16.0,18.0,15.0,18.0,16.0,16.0,18.0,17.0,42.0,8.0,4.0,11.0,17.0,17.0,17.0,16.0,17.0,17.0,17.0,17.0,16.0,17.0,17.0,16.0,16.0,17.0,17.0,17.0,17.0,16.0,17.0,18.0,17.0,16.0,16.0,16.0,18.0,17.0,43.0,6.0,5.0,10.0,18.0,17.0,19.0,13.0,18.0,16.0,17.0,16.0,16.0,18.0,17.0,15.0,18.0,15.0,18.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,18.0,16.0,16.0,17.0,16.0,44.0,6.0,4.0,13.0,18.0,15.0,17.0,17.0,16.0,17.0,18.0,16.0,17.0,16.0,16.0,18.0,16.0,17.0,16.0,43.0,6.0,4.0,15.0,17.0,15.0,17.0,16.0,17.0,17.0,16.0,44.0,5.0,3.0,15.0,17.0,16.0,17.0,17.0,17.0,15.0,18.0,37.0,6.0,6.0,17.0,17.0,17.0,16.0,17.0,16.0,17.0,16.0,17.0,18.0,15.0,17.0,18.0,16.0,16.0,19.0,14.0,45.0,4.0,3.0,16.0,17.0,15.0,17.0,17.0,16.0,17.0,17.0,16.0,18.0,18.0,16.0,17.0,17.0,16.0,16.0,18.0,16.0,17.0,17.0,16.0,16.0,17.0,15.0,17.0,17.0,17.0,17.0,17.0,16.0,17.0,17.0,16.0,47.0,4.0,4.0,12.0,16.0,17.0,17.0,17.0,16.0,18.0,15.0,17.0,19.0,14.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,45.0,4.0,4.0,14.0,17.0,16.0,17.0,18.0,15.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,17.0,16.0,17.0,16.0


def test_cgs_interval(file_path):
    data, _ = load_data.load_formated_e2e_framerate_log_with_netinfo(
        file_path, start_idx=2400, len_limit=60 * 60 * 20
    )


def find_length(data):
    if len(data.shape) > 1:
        return 0
    data = data[: min(20000, len(data))]

    base = 3
    auto_corr = acf(data, nlags=120, fft=True)[base:]

    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max] < 3 or local_max[max_local_max] > 300:
            return 125
        return local_max[max_local_max] + base
    except:
        return 125


def count_consecutive_boolean(lst):
    consec = []
    for x, y in zip(lst, lst[1:]):
        if x and y:
            if len(consec) == 0:
                consec.append(1)
            else:
                consec[-1] += 1
        elif not x and y:
            consec.append(1)
    return consec


def plot_cdf(
    datas,
    xlabel,
    ylabel,
    labels=None,
    xlim=None,
    postfix="",
    output_dir="test_data/figures",
    x_logscale=False,
):
    fig = plt.figure(tight_layout=True)
    for idx, data in enumerate(datas):
        data = np.sort(data)
        p = 1.0 * np.arange(len(data)) / (len(data) - 1)
        if labels is not None:
            plt.plot(data, p, label=labels[idx])
        else:
            plt.plot(data, p)

    if xlim is not None:
        plt.xlim(xlim)
    if labels is not None:
        plt.legend()
    if x_logscale:
        plt.xscale("log")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

    plt.show()


class TimestampExtrapolator:
    def __init__(
        self, first_frame_sts, first_frame_recv_ts, mode=0
    ):  # 0 for naive mode, 1 for Kalman filter
        self.first_frame_sts = first_frame_sts
        self.first_frame_recv_ts = first_frame_recv_ts

        self.mode = mode

        if self.mode == 1:
            self.w = [1, 0]
            self.p = [[1, 0], [0, 1e5]]

    def update(self, frame_sts, frame_recv_ts):
        if self.mode == 0:
            pass
        elif self.mode == 1:
            recv_time_diff = frame_recv_ts - self.first_frame_recv_ts
            residual = (
                frame_sts
                - self.first_frame_sts
                - recv_time_diff * self.w[0]
                - self.w[1]
            )

            k = [0, 0]
            k[0] = self.p[0][0] * recv_time_diff + self.p[0][1]
            k[1] = self.p[1][0] * recv_time_diff + self.p[1][1]

            kLambda = 1
            tpt = kLambda + recv_time_diff * k[0] + k[1]
            k[0] /= tpt
            k[1] /= tpt

            self.w[0] = self.w[0] + k[0] * residual
            self.w[1] = self.w[1] + k[1] * residual

            p00 = (
                1
                / kLambda
                * (
                    self.p[0][0]
                    - (k[0] * recv_time_diff * self.p[0][0] + k[0] * self.p[1][0])
                )
            )
            p01 = (
                1
                / kLambda
                * (
                    self.p[0][1]
                    - (k[0] * recv_time_diff * self.p[0][1] + k[0] * self.p[1][1])
                )
            )
            self.p[1][0] = (
                1
                / kLambda
                * (
                    self.p[1][0]
                    - (k[1] * recv_time_diff * self.p[0][0] + k[1] * self.p[1][0])
                )
            )
            self.p[1][1] = (
                1
                / kLambda
                * (
                    self.p[1][1]
                    - (k[1] * recv_time_diff * self.p[0][1] + k[1] * self.p[1][1])
                )
            )
            self.p[0][0] = p00
            self.p[0][1] = p01
        else:
            raise NotImplementedError

    def reset(self, first_frame_sts, first_frame_recv_ts):
        if self.mode == 0:
            pass
        elif self.mode == 1:
            self.w = [1, 0]
            self.first_frame_sts = first_frame_sts
            self.first_frame_recv_ts = first_frame_recv_ts
        else:
            raise NotImplementedError

    def extrapolate_local_time(self, frame_sts):
        if self.mode == 0:
            return self.first_frame_recv_ts + (frame_sts - self.first_frame_sts)
        elif self.mode == 1:
            sts_diff = frame_sts - self.first_frame_sts
            return (
                self.first_frame_recv_ts + (sts_diff - self.w[1]) / self.w[0]
            )  # + 0.5
        else:
            raise NotImplementedError


def smooth_data(x, y):
    extrapolator = TimestampExtrapolator(x[0], y[0], mode=1)
    smoothed_y = np.zeros(x.size)
    smoothed_y[0] = y[0]
    for i in range(1, x.size):
        predicted_y = extrapolator.extrapolate_local_time(x[i])
        smoothed_y[i] = predicted_y
        extrapolator.update(x[i], y[i])

    return smoothed_y


def plot_smoothed_data(data):
    smoothed_y = smooth_data(data[:, 33], data[:, 5])
    # idx = 1015
    # print('\t'.join([str(int(item)) for item in smoothed_y[idx-5:idx+5]]))
    # print('\t'.join([str(int(item)) for item in data[idx-5:idx+5, 5]]))
    # print(idx, data[idx, 0], smoothed_y[idx], data[idx, 5])
    fig = plt.figure(tight_layout=True)

    plt.plot(smoothed_y - data[:, 5])
    # plt.plot(data, p, label=labels[idx])

    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    plt.grid()

    plt.show()


# 0 'render_index', 'frame_index', 'frame_type', 'size', 'loss_type',
# 5 'client_receive_ts', 'receive_and_unpack', 'decoder_outside_queue', 'decoder_insided_queue', 'decode', 'render_queue', 'display',
# 12 'proxy_recv_ts', 'proxy_recv_time', 'proxy_send_delay', 'send_time',
# 16 'net_time', 'proc_time', 'tot_time',
# 19 'basic_net_ts', 'ul_jitter', 'dl_jitter'
# 22 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,
# 27 vsync_diff,present_timer_offset,jitter_buf_size,server_optim_enabled,client_optim_enabled,client_vsync_enabled,
# 33 pts, ets, dts, sts, Mrts0ToRtsOffset, packet_lossed_perK
# 39 encoding_rate, cc_rate, smoothrate, width, height, sqoe, ori_sqoe, target_sqoe,
# 47 recomm_bitrate, actual_bitrate, scene_change, encoding_fps, satd, qp, mvx, mvy, intra_mb, inter_mb, cur_cgs_pause_cnt
# 58 client_vsync_ts, min_rtt, first_send_rtt,last_send_rtt,valid_rtt,ch_ack_delay,ch_send_delay, kernel_ack_time
if __name__ == "__main__":
    # data = np.array([
    #     17.0,17.0,17.0,17.0,15.0,17.0,17.0,16.0,16.0,17.0,16.0,17.0,18.0,16.0,17.0,16.0,17.0,16.0,16.0,17.0,47.0,4.0,4.0,12.0,17.0,17.0,17.0,17.0,16.0,17.0,16.0,18.0,16.0,16.0,18.0,17.0,15.0,17.0,17.0,16.0,17.0,16.0,16.0,18.0,16.0,16.0,18.0,17.0,16.0,17.0,17.0,44.0,6.0,3.0,13.0,17.0,17.0,17.0,16.0,17.0,16.0,16.0,18.0,16.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,18.0,43.0,6.0,4.0,12.0,18.0,17.0,16.0,18.0,16.0,16.0,17.0,16.0,16.0,17.0,17.0,16.0,17.0,18.0,15.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,18.0,15.0,17.0,17.0,17.0,17.0,17.0,17.0,44.0,5.0,3.0,13.0,17.0,18.0,15.0,17.0,17.0,15.0,18.0,16.0,17.0,17.0,18.0,15.0,17.0,16.0,18.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,16.0,50.0,8.0,4.0,4.0,17.0,17.0,17.0,17.0,17.0,16.0,16.0,17.0,17.0,16.0,18.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,15.0,43.0,5.0,4.0,16.0,17.0,16.0,17.0,16.0,17.0,16.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,16.0,18.0,16.0,17.0,16.0,18.0,15.0,18.0,16.0,16.0,17.0,18.0,46.0,6.0,3.0,10.0,19.0,16.0,15.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,18.0,16.0,17.0,17.0,16.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,44.0,5.0,3.0,15.0,18.0,16.0,17.0,17.0,16.0,17.0,15.0,17.0,17.0,17.0,17.0,17.0,16.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,16.0,17.0,17.0,17.0,16.0,17.0,16.0,18.0,16.0,16.0,17.0,18.0,43.0,6.0,3.0,14.0,17.0,18.0,16.0,17.0,15.0,16.0,18.0,16.0,17.0,17.0,17.0,16.0,16.0,18.0,16.0,17.0,20.0,13.0,16.0,18.0,15.0,18.0,18.0,15.0,17.0,17.0,17.0,16.0,18.0,15.0,43.0,6.0,3.0,15.0,17.0,17.0,15.0,18.0,16.0,17.0,18.0,15.0,17.0,17.0,15.0,18.0,17.0,17.0,16.0,17.0,16.0,16.0,17.0,16.0,17.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,16.0,18.0,16.0,45.0,5.0,4.0,13.0,17.0,17.0,15.0,17.0,18.0,16.0,17.0,17.0,16.0,17.0,17.0,15.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,44.0,3.0,4.0,17.0,16.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,17.0,16.0,18.0,16.0,16.0,17.0,16.0,17.0,17.0,17.0,16.0,17.0,18.0,15.0,18.0,15.0,17.0,18.0,15.0,17.0,44.0,5.0,3.0,15.0,17.0,16.0,18.0,16.0,16.0,18.0,16.0,16.0,16.0,18.0,16.0,17.0,16.0,16.0,17.0,17.0,16.0,18.0,17.0,16.0,17.0,16.0,16.0,17.0,17.0,17.0,16.0,17.0,17.0,43.0,4.0,3.0,16.0,17.0,17.0,17.0,17.0,15.0,18.0,17.0,16.0,17.0,18.0,17.0,16.0,16.0,16.0,18.0,17.0,16.0,17.0,16.0,16.0,18.0,17.0,16.0,17.0,16.0,18.0,16.0,16.0,17.0,45.0,6.0,4.0,11.0,17.0,17.0,16.0,18.0,16.0,16.0,17.0,17.0,15.0,18.0,16.0,16.0,17.0,17.0,16.0,18.0,17.0,17.0,16.0,18.0,15.0,17.0,17.0,16.0,17.0,16.0,17.0,16.0,18.0,45.0,4.0,4.0,12.0,17.0,17.0,17.0,16.0,17.0,16.0,18.0,16.0,17.0,17.0,16.0,16.0,17.0,18.0,15.0,18.0,16.0,16.0,18.0,16.0,16.0,18.0,16.0,17.0,17.0,17.0,43.0,5.0,3.0,15.0,16.0,18.0,16.0,16.0,17.0,17.0,17.0,17.0,16.0,17.0,16.0,17.0,16.0,17.0,17.0,16.0,18.0,16.0,16.0,18.0,15.0,17.0,17.0,16.0,17.0,17.0,16.0,17.0,18.0,16.0,16.0,18.0,16.0,17.0,16.0,16.0,44.0,5.0,3.0,15.0,18.0,17.0,16.0,16.0,16.0,17.0,17.0,16.0,17.0,18.0,15.0,18.0,17.0,16.0,17.0,17.0,17.0,16.0,18.0,15.0,17.0
    # ])

    # print(find_length(data))
    # test_fft(data)
    # test_outliers(data)
    # test_bsts(data)
    # test_state_spaces(data)
    # test_hmm(data)
    # test_cgs_interval('test_data/11.177.33.27_2024-04-14_2000625488_2,0,3,2,1,1,0,1,0.csv')

    # file_path = './test_data/good_2001130031_1,0,2,3.csv'
    # file_path = './test_data/good_2000669140_2,0,2,3.csv'
    # file_path = sys.argv[1]

    file_path = "./test_data/good_2000644489_1,0,2,3.csv"
    start_idx = 7600
    sim_data_len = 3000

    # file_path = './test_data/good_2000645761_1,0,2,3.csv'
    # start_idx = 122030
    # sim_data_len = 3000

    data, info = load_data.load_detailed_framerate_log(
        file_path, start_idx=start_idx, len_limit=sim_data_len
    )  # sim for 20min
    # plot_smoothed_data(data)

    # test_offline_change_point_detection(data[:, 21])
    # test_online_change_point_detection(data[:, 11])

    # print(data[:, 21] > 10, len(count_consecutive_boolean(data[:, 21] > 10)))
    # print(np.where(data[:, 21] == 0)[0])
    # print(data[:, 21])
    # print(count_consecutive_boolean(data[:, 21] > 5))
    # print(count_consecutive_boolean(data[:, 21] < 5))

    # jitter_intervals = count_consecutive_boolean(data[:, 21] <= 20)
    # print(len(jitter_intervals), count_consecutive_boolean(np.asarray(jitter_intervals)>30))
    # plot_cdf([jitter_intervals], xlabel='frame interval', ylabel='CDF')

    # test_periodicity(data[:, 21])
    test_periodicity(data[:, 16])
    # test_periodicity(data[:, 65])
    # test_periodicity(data[np.where(data[:,9] >= 0)[0], 9])
    # test_periodicity(data[:1000, 21])
    # test_acorr_ljungbox(data[:1000, 9])
    # test_acorr_ljungbox(data[:1000, 11])
    # test_acorr_ljungbox(data[:1000, 21])

    # from pyriodicity import Autoperiod
    # autoperiod = Autoperiod(data[:1000, 11])
    # print(autoperiod.fit())

    # test_fft(data[:1000, 9])

    # N = 1000
    # signal = data[:N, 9]
    # t = np.arange(N)

    # # # 生成示例数据
    # # np.random.seed(0)
    # # N = 500
    # # t = np.linspace(0, 10, N)
    # # # 模拟一个具有抖动的周期性信号
    # # signal = np.sin(2 * np.pi * 1.5 * t) + 0.5 * np.sin(2 * np.pi * 3.5 * t + np.random.normal(0, 0.5, N))

    # # 计算傅里叶变换
    # yf = fft(signal)
    # xf = fftfreq(N, t[1] - t[0])

    # # 绘制信号和频谱
    # fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    # ax[0].plot(t, signal)
    # ax[0].set_title('Time Domain Signal')

    # ax[1].stem(xf[:N // 2], np.abs(yf[:N // 2]))
    # ax[1].set_title('Frequency Domain (FFT)')
    # ax[1].set_xlabel('Frequency (Hz)')

    # plt.tight_layout()
    # plt.show()

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from tslearn.clustering import TimeSeriesKMeans
    # from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    # # 生成示例数据
    # np.random.seed(0)
    # n_ts, sz = 100, 50  # 100个时间序列，每个长度为50
    # X = np.zeros((n_ts, sz))

    # # 创建三种不同模式的时间序列
    # for i in range(n_ts):
    #     if i < n_ts // 3:
    #         X[i] = np.sin(np.linspace(0, 2 * np.pi, sz)) + np.random.normal(0, 0.1, sz)
    #     elif i < 2 * n_ts // 3:
    #         X[i] = np.cos(np.linspace(0, 2 * np.pi, sz)) + np.random.normal(0, 0.1, sz)
    #     else:
    #         X[i] = np.random.normal(0, 1, sz)

    # # 标准化数据
    # scaler = TimeSeriesScalerMeanVariance()
    # X_scaled = scaler.fit_transform(X)

    # # 使用K-means进行时序聚类
    # n_clusters = 3
    # model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10, random_state=0)
    # labels = model.fit_predict(X_scaled)

    # # 可视化聚类结果
    # plt.figure(figsize=(10, 6))
    # for i in range(n_clusters):
    #     plt.subplot(n_clusters, 1, i + 1)
    #     for x in X_scaled[labels == i]:
    #         plt.plot(x.ravel(), "k-", alpha=0.2)
    #     plt.plot(model.cluster_centers_[i].ravel(), "r-")
    #     plt.title(f"Cluster {i + 1}")
    # plt.tight_layout()
    # plt.show()
