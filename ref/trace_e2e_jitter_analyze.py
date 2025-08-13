import os, sys, shutil
import multiprocessing
import collections
import time

import load_data
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# from trace_simulate_v2 import cal_single_para_result
# from analyze_data import cal_data_periodicity

ANCHOR_FRAME_EXTRAPOLATOR_MODE = 4
START_IDX = 3600
SIM_DATA_LEN = 60 * 60 * 20
PRINT_LOG = True

jitter_amp_thrs = [5, 10, 20]
jitter_interval_thrs = [10, 20, 30]
pct_tile_nos = list(range(5, 100, 5))
# pct_tile_nos = [5, 25, 75, 95]


def cal_stats_between_percentile(data, pct1, pct2):
    tot_no = len(data)
    data = np.asarray(data)
    data = data[data.size // 100 * pct1 : data.size // 100 * pct2]
    if data.size == 0:
        return 0, 0, 0
    else:
        return [tot_no, np.mean(data), np.std(data)]


def count_consecutive_boolean(lst):
    consec = []
    for x, y in zip(lst, lst[1:]):
        if x and y:
            if len(consec) == 0:
                consec.append(2)
            else:
                consec[-1] += 1
        elif not x and y:
            consec.append(1)

    if len(consec) == 0:
        consec = [0]
    return consec


class Result:
    def __init__(self):
        self.res1 = []
        self.res2 = []
        self.res3 = []

    def update_result(self, result):
        log_path = result[0]
        cur_res = result[1]
        if cur_res is not None:
            self.res1.append(cur_res[0])
            self.res2.append(cur_res[1])
            self.res3.append(cur_res[2])


def plot_lines(
    datas,
    xlabel,
    ylabel,
    file_name,
    labels=None,
    xlim=None,
    postfix="",
    output_dir="test_data/figures",
):
    fig = plt.figure()
    for idx, data in enumerate(datas):
        if labels is not None:
            plt.plot(data, label=labels[idx])
        else:
            plt.plot(data)

    if xlim is not None:
        plt.xlim(xlim)
    if labels is not None:
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show()
    plt.grid()
    plt.savefig(xlabel + ".jpg", dpi=200)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, file_name + ".jpg"), dpi=400)

    plt.close(fig)


def rectify_anchor_vsync_ts(anchor_vsync_ts, cur_vsync_ts, frame_interval=16.667):
    slot_no = (cur_vsync_ts - anchor_vsync_ts) // frame_interval
    ts_diff = cur_vsync_ts - (np.ceil(frame_interval * slot_no) + anchor_vsync_ts)

    if ts_diff <= 5:
        new_anchor_ts = anchor_vsync_ts + ts_diff
    elif ts_diff >= frame_interval - 5:
        new_anchor_ts = anchor_vsync_ts - (np.ceil(frame_interval) - ts_diff)
    else:
        new_anchor_ts = anchor_vsync_ts

    return new_anchor_ts


def cal_next_vsync_ts(cur_ready_ts, anchor_vsync_ts, frame_interval):
    if cur_ready_ts < anchor_vsync_ts:
        return anchor_vsync_ts
    else:
        next_slot_no = (cur_ready_ts - anchor_vsync_ts) // frame_interval + 1
        return anchor_vsync_ts + np.ceil(next_slot_no * frame_interval)


class AnchorFrameExtrapolator:
    def __init__(self, anchor_frame_ts, anchor_frame_no, frame_interval, mode=0):
        # 0 for ewma, 1 for ewma with quick reset, 2 for Kalman filter
        # 3 for anchor ts ewma, 4 for window mode
        self.anchor_frame_ts = anchor_frame_ts
        self.anchor_frame_no = anchor_frame_no
        self.prev_frame_ts = anchor_frame_ts
        self.prev_frame_no = anchor_frame_no
        self.frame_interval = frame_interval

        self.init_frame_interval = frame_interval

        self.mode = mode

        if self.mode == 2:
            self.w = [1, 0]
            self.p = [[1, 0], [0, 1e3]]

        elif self.mode == 4:
            self.window_lth = 6
            self.window_duration = self.window_lth * self.init_frame_interval
            self.past_frame_pts = collections.deque(
                [anchor_frame_ts], maxlen=self.window_lth
            )
            self.past_frame_no = collections.deque(
                [anchor_frame_no], maxlen=self.window_lth
            )
            self.past_predicted_ts = collections.deque(
                [anchor_frame_ts], maxlen=self.window_lth
            )
            self.latest_update_pts = 0

    def update(self, frame_ts, frame_no):
        if frame_no <= self.prev_frame_no:
            return

        if self.mode == 0:
            cur_frame_interval = (frame_ts - self.prev_frame_ts) / (
                frame_no - self.prev_frame_no
            )
            if cur_frame_interval > 100:
                self.frame_interval = self.init_frame_interval
                self.reset(frame_ts, frame_no)
            else:
                self.frame_interval = (frame_ts - self.anchor_frame_ts) / (
                    frame_no - self.anchor_frame_no
                )

        elif self.mode == 1:
            cur_frame_interval = (frame_ts - self.prev_frame_ts) / (
                frame_no - self.prev_frame_no
            )
            if cur_frame_interval <= 2 * self.init_frame_interval:
                self.frame_interval = (
                    0.99 * self.frame_interval + 0.01 * cur_frame_interval
                )

            self.reset(frame_ts, frame_no)

        elif self.mode == 2:
            frame_interval_diff = (frame_ts - self.anchor_frame_ts) / (
                frame_no - self.anchor_frame_no
            )
            residual = self.frame_interval - frame_interval_diff * self.w[0] - self.w[1]

            k = [0, 0]
            k[0] = self.p[0][0] * frame_interval_diff + self.p[0][1]
            k[1] = self.p[1][0] * frame_interval_diff + self.p[1][1]

            kLambda = 1
            tpt = kLambda + frame_interval_diff * k[0] + k[1]
            k[0] /= tpt
            k[1] /= tpt

            self.w[0] = self.w[0] + k[0] * residual
            self.w[1] = self.w[1] + k[1] * residual

            p00 = (
                1
                / kLambda
                * (
                    self.p[0][0]
                    - (k[0] * frame_interval_diff * self.p[0][0] + k[0] * self.p[1][0])
                )
            )
            p01 = (
                1
                / kLambda
                * (
                    self.p[0][1]
                    - (k[0] * frame_interval_diff * self.p[0][1] + k[0] * self.p[1][1])
                )
            )
            self.p[1][0] = (
                1
                / kLambda
                * (
                    self.p[1][0]
                    - (k[1] * frame_interval_diff * self.p[0][0] + k[1] * self.p[1][0])
                )
            )
            self.p[1][1] = (
                1
                / kLambda
                * (
                    self.p[1][1]
                    - (k[1] * frame_interval_diff * self.p[0][1] + k[1] * self.p[1][1])
                )
            )
            self.p[0][0] = p00
            self.p[0][1] = p01

        elif self.mode == 3:
            expect_frame_ts = self.anchor_frame_ts + self.frame_interval * (
                frame_no - self.anchor_frame_no
            )
            ts_diff = frame_ts - expect_frame_ts
            self.anchor_frame_ts = self.anchor_frame_ts + np.ceil(ts_diff * 0.1)

        elif self.mode == 4:
            self.past_frame_pts.append(frame_ts)
            self.past_frame_no.append(frame_no)
            if len(self.past_frame_pts) < self.window_lth:
                return

            deviation = np.array(self.past_predicted_ts) - np.array(self.past_frame_pts)

            if np.any(deviation >= self.init_frame_interval * 3):
                self.reset(frame_ts, frame_no)
                self.latest_update_pts = frame_ts
                return

            if np.abs(deviation).mean() >= self.init_frame_interval:
                self.reset(frame_ts, frame_no)
                self.latest_update_pts = frame_ts
                return

            if not np.all(deviation > 0) and not np.all(deviation < 0):
                return

            if frame_ts - self.latest_update_pts <= 200:
                return

            base_anchor_frame_ts = self.past_frame_pts[0]
            base_anchor_frame_no = self.past_frame_no[0]
            ts_diff_sum = 0
            for i in range(self.window_lth):
                ts_diff_sum += (
                    self.past_frame_pts[i]
                    - base_anchor_frame_ts
                    - (self.past_frame_no[i] - base_anchor_frame_no)
                    * self.init_frame_interval
                )
            ts_diff_sum = ts_diff_sum // self.window_lth
            base_anchor_frame_ts += ts_diff_sum

            self.anchor_frame_ts = base_anchor_frame_ts
            self.anchor_frame_no = base_anchor_frame_no
            self.latest_update_pts = frame_ts

        else:
            raise NotImplementedError

        self.prev_frame_ts = frame_ts
        self.prev_frame_no = frame_no

    def reset(self, anchor_frame_ts, anchor_frame_no):
        self.anchor_frame_ts = anchor_frame_ts
        self.anchor_frame_no = anchor_frame_no
        if self.mode == 2:
            self.w = [1, 0]
        elif self.mode == 4:
            self.past_frame_pts = collections.deque(
                [anchor_frame_ts], maxlen=self.window_lth
            )
            self.past_frame_no = collections.deque(
                [anchor_frame_no], maxlen=self.window_lth
            )
            self.past_predicted_ts = collections.deque(
                [anchor_frame_ts], maxlen=self.window_lth
            )
        else:
            raise NotImplementedError

    def predict(self, frame_no):
        if self.mode == 0 or self.mode == 1:
            # print(self.anchor_frame_ts, self.frame_interval, (frame_no-self.anchor_frame_no),self.anchor_frame_ts + self.frame_interval * (frame_no-self.anchor_frame_no))
            # input()
            return self.anchor_frame_ts + self.frame_interval * (
                frame_no - self.anchor_frame_no
            )
        elif self.mode == 2:
            return self.anchor_frame_ts + (frame_no - self.anchor_frame_no) * (
                (self.frame_interval - self.w[1]) / self.w[0]
            )

        elif self.mode == 3:
            return self.anchor_frame_ts + self.init_frame_interval * (
                frame_no - self.anchor_frame_no
            )

        elif self.mode == 4:
            predicted_frame_ts = self.anchor_frame_ts + self.init_frame_interval * (
                frame_no - self.anchor_frame_no
            )
            self.past_predicted_ts.append(predicted_frame_ts)
            return predicted_frame_ts
        else:
            raise NotImplementedError

    def get_frame_interval(self):
        return self.frame_interval, self.anchor_frame_ts, self.anchor_frame_no


class SingleJitterPredictor:
    def __init__(self, window_lth=600, count_negative=False, count_flag=False, fps=60):
        self.window_lth = window_lth
        self.fps = fps

        self.past_jitter = np.zeros(self.window_lth)
        self.past_jitter_flag = np.zeros(self.window_lth)
        self.past_jitter_idx = 0

        self.near_past_jitter = np.zeros(self.fps)
        self.near_past_jitter_flag = np.zeros(self.fps)
        self.near_past_jitter_idx = 0

        self.jitter_thr = 999

        self.count_negative = count_negative
        self.count_flag = count_flag

    def update(self, jitter_amplitude, jitter_flag, jitter_thr=None):
        self.past_jitter[self.past_jitter_idx] = jitter_amplitude
        self.past_jitter_flag[self.past_jitter_idx] = jitter_flag
        self.past_jitter_idx = (self.past_jitter_idx + 1) % self.window_lth

        self.near_past_jitter[self.near_past_jitter_idx] = jitter_amplitude
        self.near_past_jitter_flag[self.near_past_jitter_idx] = jitter_flag
        self.near_past_jitter_idx = (self.near_past_jitter_idx + 1) % self.fps

        if jitter_thr is not None:
            self.jitter_thr = jitter_thr
        elif jitter_flag and jitter_amplitude < self.jitter_thr:
            self.jitter_thr = jitter_amplitude

    def predict_next_jitter_within_k_prob(self, predict_mode=1, k=30):
        if self.count_negative:
            single_frame_prob = (
                np.sum(self.past_jitter >= self.jitter_thr)
                + np.sum(self.past_jitter <= -self.jitter_thr)
            ) / self.window_lth
        elif self.count_flag:
            single_frame_prob = self.past_jitter_flag.sum() / self.window_lth
        else:
            single_frame_prob = (
                np.sum(self.past_jitter >= self.jitter_thr) / self.window_lth
            )

        if predict_mode > 0:
            #     total_prob = 1 - (1 - single_frame_prob) ** k
            #     return total_prob
            # elif predict_mode > 1 and predict_mode <= 9:
            return single_frame_prob
        else:
            raise NotImplementedError

    def get_jitter_thr(self):
        return self.jitter_thr

    def detect_change_point(self, k=30, detect_mode=1):
        if detect_mode == 1:
            all_past_jitter = self.past_jitter
            all_past_jitter_flag = self.past_jitter_flag
            all_near_past_jitter = self.near_past_jitter
            all_near_past_jitter_flag = self.near_past_jitter_flag

            if self.count_negative:
                single_frame_prob = (
                    np.sum(all_past_jitter >= self.jitter_thr)
                    + np.sum(all_past_jitter <= -self.jitter_thr)
                ) / self.window_lth
                near_single_frame_prob = (
                    np.sum(all_near_past_jitter >= self.jitter_thr)
                    + np.sum(all_near_past_jitter <= -self.jitter_thr)
                ) / self.window_lth
            elif self.count_flag:
                single_frame_prob = (
                    np.asarray(all_past_jitter_flag).sum() / self.window_lth
                )
                near_single_frame_prob = (
                    np.asarray(all_near_past_jitter_flag).sum() / self.window_lth
                )
            else:
                single_frame_prob = (
                    np.sum(all_past_jitter >= self.jitter_thr) / self.window_lth
                )
                near_single_frame_prob = (
                    np.sum(all_near_past_jitter >= self.jitter_thr) / self.window_lth
                )

            if np.abs(single_frame_prob - near_single_frame_prob) >= 1 / k:
                return True
            else:
                return False
        else:
            raise NotImplementedError

    def reset(self):
        self.past_jitter = np.zeros(self.window_lth)
        self.past_jitter_flag = np.zeros(self.window_lth)

        self.near_past_jitter = np.zeros(self.fps)
        self.near_past_jitter_flag = np.zeros(self.fps)


class E2EJitterPredictor:
    def __init__(
        self,
        min_rtt,
        avg_dec_time,
        avg_display_time,
        frame_interval=16.667,
        frame_stall_thr=50,
        small_ts_margin=2,
        window_lth=600,
        drop_frame_mode="lifo",
    ):
        self.min_rtt = min_rtt
        self.frame_interval = frame_interval
        self.frame_stall_thr = frame_stall_thr
        self.small_ts_margin = small_ts_margin
        self.avg_dec_time = avg_dec_time
        self.avg_display_time = avg_display_time
        self.drop_frame_mode = drop_frame_mode

        self.window_lth = window_lth
        self.past_dl_jitter = SingleJitterPredictor()
        self.past_render_jitter = SingleJitterPredictor(count_negative=False)
        self.past_server_jitter = SingleJitterPredictor()
        self.past_decoder_jitter = SingleJitterPredictor()
        self.past_display_jitter = SingleJitterPredictor()
        self.past_e2e_jitter = SingleJitterPredictor()
        self.past_network_problem = SingleJitterPredictor()
        self.past_near_vsync_jitter = SingleJitterPredictor(count_flag=True)

        self.frame_cnt = 0

        self.jitter_thr = 0

    def update(
        self,
        data,
        smoothed_frame_pts_diff,
        nearest_no_jitter_vsync_ts,
        actual_display_ts,
        idx,
        frame_jitter_flag,
    ):
        self.frame_cnt += 1

        if self.drop_frame_mode == "lifo":
            if idx == 0:
                return

            # idx -= 1
            while idx >= 0:
                idx -= 1
                if idx >= 0 and data[idx, 4] == 0 and data[idx, 5] != 0:
                    break

            if idx < 0:
                return

        cur_big_frame_flag = np.logical_and.reduce(
            (
                data[idx, 3] > data[idx, 40] * 1024 / 8 / 60 * 1.5,
                data[idx, 16] - self.min_rtt - data[idx, 20] - self.frame_interval > 0,
                data[idx, 3]
                / max(data[idx, 16] - self.min_rtt - data[idx, 20], 1)
                / 1024
                * 8
                * 1000
                > data[idx, 40] * 0.85,
            )
        )
        cur_stall_flag = data[idx, 21] >= self.frame_stall_thr
        cur_packet_loss_flag = np.logical_and(
            data[idx, 38] > 0, data[idx, 16] >= 2 * self.min_rtt
        )

        cur_dl_jitter = data[idx, 21]
        cur_dl_jitter_flag = np.logical_and(
            data[idx, 21] > self.small_ts_margin, data[idx, 21] < self.frame_stall_thr
        )

        # render related
        if self.drop_frame_mode == "lifo":
            cur_render_jitter = smoothed_frame_pts_diff[idx]
            cur_render_jitter_flag = smoothed_frame_pts_diff[idx] > self.small_ts_margin
            if idx < smoothed_frame_pts_diff.shape[0] - 1:
                cur_render_jitter_flag = np.logical_or(
                    cur_render_jitter_flag,
                    smoothed_frame_pts_diff[idx + 1] < -self.small_ts_margin,
                )
        elif self.drop_frame_mode == "fifo":
            cur_render_jitter = smoothed_frame_pts_diff[idx]
            cur_render_jitter_flag = smoothed_frame_pts_diff[idx] > self.small_ts_margin

        # server related
        cur_server_jitter = 0
        cur_server_jitter_flag = False
        if idx > 1:
            frame_cgs_render_interval = data[idx, 33] - data[idx - 1, 33]
            frame_proxy_recv_interval = data[idx, 12] - data[idx - 1, 12]
            cur_server_jitter = frame_proxy_recv_interval - frame_cgs_render_interval
            cur_server_jitter_flag = (
                frame_proxy_recv_interval - frame_cgs_render_interval
            ) > self.small_ts_margin

        # decoder related
        # decoder_jitter_flag = np.zeros(tot_frame_no)
        cur_decoder_jitter = data[idx, 9] - self.avg_dec_time
        cur_decoder_jitter_flag = (
            data[idx, 9] > self.avg_dec_time + self.small_ts_margin
        )
        # cur_decoder_jitter_flag = np.logical_or(data[idx, 9] > self.avg_dec_time + self.small_ts_margin, data[idx, 7:9].sum(-1) > 0)
        self.avg_dec_time = self.avg_dec_time * 0.99 + data[idx, 9] * 0.01

        cur_display_jitter = data[idx, 11] - self.avg_display_time
        cur_display_jitter_flag = (
            data[idx, 11] > self.avg_display_time + self.small_ts_margin
        )
        self.avg_display_time = self.avg_display_time * 0.99 + data[idx, 11] * 0.01

        # e2e_jitter related
        cur_e2e_jitter = (
            data[idx, 21]
            + smoothed_frame_pts_diff[idx]
            + cur_server_jitter
            + cur_decoder_jitter
        )  # + cur_display_jitter
        cur_e2e_jitter_flag = cur_e2e_jitter > self.small_ts_margin

        cur_near_vsync_jitter = actual_display_ts[idx] - nearest_no_jitter_vsync_ts[idx]
        near_vsync_jitter_flag = np.logical_and.reduce(
            (
                data[idx, 5:10].sum(-1) < nearest_no_jitter_vsync_ts[idx],
                actual_display_ts[idx] >= nearest_no_jitter_vsync_ts[idx],
                actual_display_ts[idx] - nearest_no_jitter_vsync_ts[idx]
                <= self.small_ts_margin,
            )
        )

        cur_network_problem_flag = np.logical_or.reduce(
            (
                cur_big_frame_flag,
                cur_stall_flag,
                cur_packet_loss_flag,
                cur_dl_jitter_flag,
            )
        )

        cur_jitter_thr = (
            nearest_no_jitter_vsync_ts[idx] - data[idx, 5:10].sum() + cur_e2e_jitter
        )
        if self.jitter_thr == 0:
            self.jitter_thr = cur_jitter_thr
        else:
            self.jitter_thr = self.jitter_thr * 0.99 + cur_jitter_thr * 0.01

        # jitter_thr = self.jitter_thr
        jitter_thr = cur_jitter_thr

        self.past_network_problem.update(
            cur_dl_jitter,
            np.logical_and.reduce(
                (
                    cur_network_problem_flag,
                    frame_jitter_flag,
                )
            ),
            jitter_thr,
        )
        self.past_dl_jitter.update(
            cur_dl_jitter,
            np.logical_and.reduce(
                (
                    cur_dl_jitter_flag,
                    frame_jitter_flag,
                    not cur_big_frame_flag,
                    not cur_stall_flag,
                    not cur_packet_loss_flag,
                )
            ),
            jitter_thr,
        )
        self.past_render_jitter.update(
            cur_render_jitter,
            np.logical_and.reduce(
                (
                    cur_render_jitter_flag,
                    frame_jitter_flag,
                    not cur_network_problem_flag,
                )
            ),
            jitter_thr,
        )
        self.past_server_jitter.update(
            cur_server_jitter,
            np.logical_and.reduce(
                (
                    cur_server_jitter_flag,
                    frame_jitter_flag,
                    not cur_network_problem_flag,
                    not cur_render_jitter_flag,
                )
            ),
            jitter_thr,
        )
        self.past_decoder_jitter.update(
            cur_decoder_jitter,
            np.logical_and.reduce(
                (
                    cur_decoder_jitter_flag,
                    frame_jitter_flag,
                    not cur_network_problem_flag,
                    not cur_render_jitter_flag,
                    not cur_server_jitter_flag,
                )
            ),
            jitter_thr,
        )
        self.past_display_jitter.update(
            cur_display_jitter,
            np.logical_and.reduce(
                (
                    cur_display_jitter_flag,
                    frame_jitter_flag,
                    not cur_network_problem_flag,
                    not cur_render_jitter_flag,
                    not cur_server_jitter_flag,
                    not cur_decoder_jitter_flag,
                )
            ),
            jitter_thr,
        )
        self.past_e2e_jitter.update(
            cur_e2e_jitter,
            # np.logical_and(cur_e2e_jitter_flag, frame_jitter_flag),
            np.logical_and.reduce(
                (
                    cur_e2e_jitter_flag,
                    frame_jitter_flag,
                    not cur_big_frame_flag,
                    not cur_stall_flag,
                    not cur_packet_loss_flag,
                )
            ),
            jitter_thr,
        )
        self.past_near_vsync_jitter.update(
            cur_near_vsync_jitter,
            # near_vsync_jitter_flag,
            np.logical_and.reduce(
                (
                    near_vsync_jitter_flag,
                    not cur_network_problem_flag,
                    not cur_render_jitter_flag,
                    not cur_server_jitter_flag,
                    not cur_decoder_jitter_flag,
                    not cur_display_jitter,
                )
            ),
            jitter_thr,
        )

    def predict_next_jitter_within_k_prob(self, predict_mode=1, k=30):
        if self.frame_cnt < self.window_lth:
            return 1, None, None

        dl_jitter_prob = self.past_dl_jitter.predict_next_jitter_within_k_prob(
            predict_mode, k
        )
        render_jitter_prob = self.past_render_jitter.predict_next_jitter_within_k_prob(
            predict_mode, k
        )
        server_jitter_prob = self.past_server_jitter.predict_next_jitter_within_k_prob(
            predict_mode, k
        )
        decoder_jitter_prob = (
            self.past_decoder_jitter.predict_next_jitter_within_k_prob(predict_mode, k)
        )
        display_jitter_prob = (
            self.past_display_jitter.predict_next_jitter_within_k_prob(predict_mode, k)
        )
        e2e_jitter_prob = self.past_e2e_jitter.predict_next_jitter_within_k_prob(
            predict_mode, k
        )
        network_problem_prob = (
            self.past_network_problem.predict_next_jitter_within_k_prob(predict_mode, k)
        )
        near_vsync_prob = self.past_near_vsync_jitter.predict_next_jitter_within_k_prob(
            predict_mode, k
        )

        details = [
            dl_jitter_prob,
            render_jitter_prob,
            server_jitter_prob,
            decoder_jitter_prob,
            display_jitter_prob,
            e2e_jitter_prob,
            network_problem_prob,
            near_vsync_prob,
        ]
        jitter_thrs = [
            self.past_dl_jitter.get_jitter_thr(),
            self.past_render_jitter.get_jitter_thr(),
            self.past_server_jitter.get_jitter_thr(),
            self.past_decoder_jitter.get_jitter_thr(),
            self.past_display_jitter.get_jitter_thr(),
            self.past_e2e_jitter.get_jitter_thr(),
            self.past_network_problem.get_jitter_thr(),
            self.past_near_vsync_jitter.get_jitter_thr(),
        ]

        # if network_problem_prob > 0:
        #     return 1, details, jitter_thrs

        predicted_prob = np.max(details)
        # if predict_mode != 2:
        #     predicted_prob = np.max(details)
        # else:
        #     predicted_prob = np.max(details[:-1])

        return predicted_prob, details, jitter_thrs

    def get_probability_thr(self, predict_mode=1, k=30):
        if predict_mode >= 0:
            #     return 0.5
            # elif predict_mode > 1:
            return 1 / k
        else:
            raise NotImplementedError

    def detect_change_point(self, k=30, detect_mode=1):
        results = [
            self.past_dl_jitter.detect_change_point(k, detect_mode),
            self.past_render_jitter.detect_change_point(k, detect_mode),
            self.past_server_jitter.detect_change_point(k, detect_mode),
            self.past_decoder_jitter.detect_change_point(k, detect_mode),
            self.past_display_jitter.detect_change_point(k, detect_mode),
            self.past_e2e_jitter.detect_change_point(k, detect_mode),
            self.past_network_problem.detect_change_point(k, detect_mode),
            self.past_near_vsync_jitter.detect_change_point(k, detect_mode),
        ]

        if np.any(np.asarray(results)):
            return True
        else:
            return False

    def reset(self):
        self.frame_cnt = 0
        self.past_dl_jitter.reset()
        self.past_render_jitter.reset()
        self.past_server_jitter.reset()
        self.past_decoder_jitter.reset()
        self.past_display_jitter.reset()
        self.past_e2e_jitter.reset()
        self.past_network_problem.reset()
        self.past_near_vsync_jitter.reset()


class E2EJitterPredictorV2:
    def __init__(self, frame_interval=16.667, window_lth=600):
        self.frame_interval = frame_interval
        self.window_lth = window_lth

        self.frame_cnt = 0
        self.jitter_thr = 0

    def update(self, data, idx):
        pass

    def predict_next_jitter_within_k_prob(self, predict_mode=1, k=30):
        pass

    def get_probability_thr(self, predict_mode=1, k=30):
        pass

    def detect_change_point(self, k=30, detect_mode=1):
        pass

    def reset(self):
        pass


def predict_render_time(
    cur_actual_render_time, prev_render_time, predictor_mode="oracle"
):
    if predictor_mode == "oracle":
        return cur_actual_render_time, cur_actual_render_time
    elif predictor_mode == "ewma":
        predicted_render_time = np.ceil(prev_render_time)
        prev_render_time = 0.99 * prev_render_time + 0.01 * cur_actual_render_time
        return predicted_render_time, prev_render_time
    elif predictor_mode == "fixed":
        return 1, 1
    else:
        raise NotImplementedError


#     0 'render_index', 'frame_index', 'frame_type', 'size', 'loss_type',
#     5 'client_receive_ts', 'receive_and_unpack', 'decoder_outside_queue', 'decoder_insided_queue', 'decode', 'render_queue', 'display',
#     12 'proxy_recv_ts', 'proxy_recv_time', 'proxy_send_delay', 'send_time',
#     16 'net_time', 'proc_time', 'tot_time',
#     19 'basic_net_ts', 'ul_jitter', 'dl_jitter'
#     22 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,
#     27 vsync_diff,present_timer_offset,jitter_buf_size,server_optim_enabled,client_optim_enabled,client_vsync_enabled,
#     33 pts, ets, dts, sts, Mrts0ToRtsOffset, packet_lossed_perK
#     39 encoding_rate, cc_rate, smoothrate, width, height, sqoe, ori_sqoe, target_sqoe,
#     47 recomm_bitrate, actual_bitrate, scene_change, encoding_fps, satd, qp, mvx, mvy, intra_mb, inter_mb, cur_cgs_pause_cnt
#     58 client_vsync_ts, min_rtt,
#     60 'cur_anchor_vsync_ts', 'decode_over_ts', 'client_invoke_ts', 'client_display_ts', 'predicted_render_time', 'predicted_decode_time', 'nearest_vsync_ts',
#     67 'nearest_display_slot', 'available_vsync_ts', 'actual_render_queue', 'client_render_queue', 'extra_display_ts',
#     72 'frame_buffer_cnt', 'invoke_present_ts', 'invoke_present_slot',
# 'actual_display_ts', 'actual_vsync_ts','actual_display_slot', 'cur_buf_size', 'buf_change_flag', 'failed_early_drop_frame', 'missed_early_drop_frame',
# 'predicted_quick_drop_probs', 'frame_e2e_jitter',
# 'dl_jitter_prob', 'render_jitter_prob', 'server_jitter_prob', 'decoder_jitter_prob', 'display_jitter_prob', 'e2e_jitter_prob', 'network_problem_prob', 'near_vsync_jitter_prob',
# 'dl_jitter_prob_thr', 'render_jitter_prob_thr', 'server_jitter_prob_thr', 'decoder_jitter_prob_thr', 'display_jitter_prob', 'e2e_jitter_prob_thr', 'network_problem_prob_thr', 'near_vsync_jitter_prob_thr', 'quick_drop_frame_cnt_hist',
# 'consecutive_frame_no', 'display_discarded_flag', 'algorithm_discard_flag', 'valid_frame_flag', 'original_valid_flag', 'bonus_fps_obtained', 'cur_bonus_fps_no', 'cur_valid_frame_no',
# 'dec_over_ts', 'dec_nearest_vsync_ts', 'dec_queued_frame_cnt', 'frame_queue_flag',
# 'original_pts', 'smoothed_pts', 'smoothed_frame_pts_diff', 'render_jitter_induced_queue', 'server_jitter_induced_queue',
# 'network_big_frame_induced_queue', 'network_i_frame_induced_queue',
# 'network_dl_jitter_induced_queue', 'network_stall_induced_queue', 'network_packet_loss_induced_queue',
# 'decoder_jitter_induced_queue', 'display_jitter_induced_queue', 'near_vsync_jitter_induced_queue',
# 'updated_frame_interval', 'anchor_frame_ts', 'anchor_frame_no'


class future_window_qos_predictor:
    def __init__(
        self,
        data,
        mode=1,
        past_window_size=300,
        predict_window_size=60,
        frame_interval=16.667,
    ):
        self.data = data
        self.mode = 1

        self.past_window_size = past_window_size
        self.predict_window_size = predict_window_size

        self.frame_interval = frame_interval

    def predict_by_momentum(self, idx):
        pass

    def predict(self, idx):
        pass

    def update(self):
        pass


def predict_framerate_by_window(
    file_path,
    enable_quick_drop=1,
    render_time_predictor="ewma",
    anchor_frame_extrapolator_mode=4,
    frame_interval=16.667,
    bonus_fps_no_thr=30,
    sim_data_len=60 * 60 * 20,
    start_idle_len=120,
    print_log=True,
):
    """
    Simulate the display process with a frame trace and a set of parameters.
    param display_mode: naive_vsync, simple_ctrl
    """
    df, _ = load_data.load_detailed_optimal_framerate_log(
        file_path, start_idx=0
    )  # sim for 20min

    if df is None:
        print("None data")
        return None, None

    data = df.to_numpy()

    # initialization: calculate the average decode and render time
    def cal_avg_client_time(samp_len):
        # valid_idx = (data[300:samp_len, 5:12].sum(-1) > data[299:samp_len-1, 5:12].sum(-1)) + 300
        valid_idx = np.where(data[:samp_len, 4] == 0)[0]
        if valid_idx.size == 0:
            return 999, 999, 999, 999
        avg_dec_time = np.mean(data[valid_idx, 9])
        avg_dec_total_time = np.mean(data[valid_idx, 6:10].sum(-1))
        avg_render_time = np.mean(data[valid_idx, 11])
        avg_proc_time = np.mean(data[valid_idx, 6:12].sum(-1))

        return avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time

    avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time = (
        cal_avg_client_time(start_idle_len)
    )

    df = df.iloc[start_idle_len:]
    data = data[start_idle_len:, :]

    values, counts = np.unique(data[:, 30], return_counts=True)
    idx = np.argmax(counts)
    server_optim_enabled = values[idx]

    values, counts = np.unique(data[:, 31], return_counts=True)
    idx = np.argmax(counts)
    client_optim_enabled = values[idx]

    values, counts = np.unique(data[:, 32], return_counts=True)
    idx = np.argmax(counts)
    client_vsync_enabled = values[idx]

    anchor_frame_extrapolator = AnchorFrameExtrapolator(
        data[0, 33], data[0, 1], frame_interval, anchor_frame_extrapolator_mode
    )

    tot_frame_no = data.shape[0]
    dec_over_ts = data[:, 5:10].sum(-1)
    # nearest_vsync_ts = np.zeros(tot_frame_no, dtype=np.int64)

    nearest_vsync_ts = df["nearest_vsync_ts"].to_numpy()
    display_discarded_flag = df["display_discarded_flag"].to_numpy()
    actual_display_ts = df["actual_display_ts"].to_numpy()
    buf_change_flag = df["buf_change_flag"].to_numpy()
    frame_jitter_flag = df["frame_queue_flag"].to_numpy()
    consecutive_frame_no = df["consecutive_frame_no"].to_numpy()

    nearest_no_jitter_vsync_ts = np.zeros(tot_frame_no, dtype=np.int64)
    cur_anchor_vsync_ts = np.zeros(tot_frame_no)

    predicted_decode_time = np.zeros(tot_frame_no)
    predicted_render_time = np.zeros(tot_frame_no)

    predicted_quick_drop_probs = np.zeros(tot_frame_no, dtype=np.float32)
    dl_jitter_prob = np.zeros(tot_frame_no, dtype=np.float32)
    render_jitter_prob = np.zeros(tot_frame_no, dtype=np.float32)
    server_jitter_prob = np.zeros(tot_frame_no, dtype=np.float32)
    decoder_jitter_prob = np.zeros(tot_frame_no, dtype=np.float32)
    display_jitter_prob = np.zeros(tot_frame_no, dtype=np.float32)
    e2e_jitter_prob = np.zeros(tot_frame_no, dtype=np.float32)
    network_problem_prob = np.zeros(tot_frame_no, dtype=np.float32)
    frame_e2e_jitter = np.zeros(tot_frame_no, dtype=np.float32)
    near_vsync_jitter_prob = np.zeros(tot_frame_no, dtype=np.float32)

    dl_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    render_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    server_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    decoder_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    display_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    e2e_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    network_problem_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    near_vsync_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)

    failed_early_drop_frame = np.zeros(tot_frame_no)
    missed_early_drop_frame = np.zeros(tot_frame_no)

    sim_st_idx = 0
    while True:
        if data[sim_st_idx, 4] == 0 and data[sim_st_idx, 5] != 0:
            break
        sim_st_idx += 1

    anchor_vsync_ts = data[sim_st_idx, 58]
    cur_anchor_vsync_ts[sim_st_idx] = anchor_vsync_ts
    prev_render_time = avg_render_time

    prev_vsync_ts = data[sim_st_idx, 58]
    nearest_no_jitter_vsync_ts[sim_st_idx] = nearest_vsync_ts[sim_st_idx]
    # nearest_vsync_ts[sim_st_idx] = cal_next_vsync_ts(dec_over_ts[sim_st_idx], data[sim_st_idx, 58], frame_interval)

    smoothed_frame_pts = np.zeros(tot_frame_no, dtype=np.int64)
    smoothed_frame_pts_diff = np.zeros(tot_frame_no, dtype=np.int64)
    updated_frame_interval = np.zeros(tot_frame_no)
    anchor_frame_ts = np.zeros(tot_frame_no)
    anchor_frame_no = np.zeros(tot_frame_no, dtype=np.int64)

    smoothed_frame_pts[sim_st_idx] = data[sim_st_idx, 33]
    anchor_frame_ts[sim_st_idx] = data[sim_st_idx, 33]
    anchor_frame_ts[sim_st_idx] = data[sim_st_idx, 1]

    min_rtt = np.min(data[:, 59])
    frame_stall_thr = 50
    small_ts_margin = 2
    jitter_history_lth = 120

    e2e_jitter_predictor = E2EJitterPredictor(
        min_rtt,
        avg_dec_time,
        avg_render_time,
        frame_interval,
        frame_stall_thr,
        small_ts_margin,
        window_lth=jitter_history_lth,
    )
    early_drop_prob_threshold = e2e_jitter_predictor.get_probability_thr(
        enable_quick_drop, bonus_fps_no_thr
    )

    # start the simulation
    # for idx in range(1, tot_frame_no):
    idx = sim_st_idx
    while True:
        idx += 1
        if idx >= tot_frame_no:
            break

        if data[idx, 4] != 0 or data[idx, 5] == 0:
            continue

        if data[idx, 58] != 0 and prev_vsync_ts != data[idx, 58]:
            anchor_vsync_ts = rectify_anchor_vsync_ts(anchor_vsync_ts, data[idx, 58])
            prev_vsync_ts = data[idx, 58]
        cur_anchor_vsync_ts[idx] = anchor_vsync_ts
        # nearest_vsync_ts[idx] = cal_next_vsync_ts(dec_over_ts[idx], anchor_vsync_ts, frame_interval)

        smoothed_frame_pts[idx] = anchor_frame_extrapolator.predict(data[idx, 1])
        smoothed_frame_pts_diff[idx] = data[idx, 33] - smoothed_frame_pts[idx]
        updated_frame_interval[idx], anchor_frame_ts[idx], anchor_frame_no[idx] = (
            anchor_frame_extrapolator.get_frame_interval()
        )
        anchor_frame_extrapolator.update(data[idx, 33], data[idx, 1])

        cur_server_jitter = 0
        if idx > 1:
            frame_cgs_render_interval = data[idx, 33] - data[idx - 1, 33]
            frame_proxy_recv_interval = data[idx, 12] - data[idx - 1, 12]
            cur_server_jitter = frame_proxy_recv_interval - frame_cgs_render_interval

        cur_e2e_jitter = (
            smoothed_frame_pts_diff[idx]
            + cur_server_jitter
            + data[idx, 21]
            + data[idx, 9]
            - avg_dec_time
        )
        nearest_no_jitter_vsync_ts[idx] = cal_next_vsync_ts(
            dec_over_ts[idx] - cur_e2e_jitter, anchor_vsync_ts, frame_interval
        )
        frame_e2e_jitter[idx] = cur_e2e_jitter

        cur_predicted_render_time, prev_render_time = predict_render_time(
            data[idx, 11], prev_render_time, predictor_mode=render_time_predictor
        )
        predicted_render_time[idx] = cur_predicted_render_time

        avg_dec_time = 0.99 * avg_dec_time + 0.01 * data[idx, 9]
        predicted_decode_time[idx] = avg_dec_time

        if enable_quick_drop >= 5 and e2e_jitter_predictor.detect_change_point(
            bonus_fps_no_thr
        ):
            e2e_jitter_predictor.reset()

        if enable_quick_drop > 0:
            predicted_prob, details, jitter_thrs = (
                e2e_jitter_predictor.predict_next_jitter_within_k_prob(
                    enable_quick_drop, bonus_fps_no_thr
                )
            )
            if details is not None:
                (
                    dl_jitter_prob[idx],
                    render_jitter_prob[idx],
                    server_jitter_prob[idx],
                    decoder_jitter_prob[idx],
                    display_jitter_prob[idx],
                    e2e_jitter_prob[idx],
                    network_problem_prob[idx],
                    near_vsync_jitter_prob[idx],
                ) = details
                (
                    dl_jitter_prob_thr[idx],
                    render_jitter_prob_thr[idx],
                    server_jitter_prob_thr[idx],
                    decoder_jitter_prob_thr[idx],
                    display_jitter_prob_thr[idx],
                    e2e_jitter_prob_thr[idx],
                    network_problem_prob_thr[idx],
                    near_vsync_jitter_prob_thr[idx],
                ) = jitter_thrs
            predicted_quick_drop_probs[idx] = predicted_prob
        if display_discarded_flag[idx] == 0 and enable_quick_drop > 0:
            e2e_jitter_predictor.update(
                data,
                smoothed_frame_pts_diff,
                nearest_no_jitter_vsync_ts,
                actual_display_ts,
                idx,
                frame_jitter_flag[idx],
            )
        end_time = time.time()

    df["smoothed_pts"] = smoothed_frame_pts
    df["smoothed_frame_pts_diff"] = smoothed_frame_pts_diff
    df["updated_frame_interval"] = updated_frame_interval
    df["anchor_frame_ts"] = anchor_frame_ts
    df["anchor_frame_no"] = anchor_frame_no

    df["predicted_quick_drop_probs"] = predicted_quick_drop_probs
    df["frame_e2e_jitter"] = frame_e2e_jitter
    df["dl_jitter_prob"] = dl_jitter_prob
    df["render_jitter_prob"] = render_jitter_prob
    df["server_jitter_prob"] = server_jitter_prob
    df["decoder_jitter_prob"] = decoder_jitter_prob
    df["display_jitter_prob"] = display_jitter_prob
    df["e2e_jitter_prob"] = e2e_jitter_prob
    df["network_problem_prob"] = network_problem_prob
    df["near_vsync_jitter_prob"] = near_vsync_jitter_prob

    df["dl_jitter_prob_thr"] = dl_jitter_prob_thr
    df["render_jitter_prob_thr"] = render_jitter_prob_thr
    df["server_jitter_prob_thr"] = server_jitter_prob_thr
    df["decoder_jitter_prob_thr"] = decoder_jitter_prob_thr
    # df['display_jitter_prob_thr'] = display_jitter_prob_thr
    df["e2e_jitter_prob_thr"] = e2e_jitter_prob_thr
    df["network_problem_prob_thr"] = network_problem_prob_thr
    df["near_vsync_jitter_prob_thr"] = near_vsync_jitter_prob_thr

    df["failed_early_drop_frame"] = failed_early_drop_frame
    df["missed_early_drop_frame"] = missed_early_drop_frame

    if print_log:
        save_path = file_path[:-4] + "_quickdrop%d_replay.csv" % (enable_quick_drop)
        df.to_csv(save_path, index=False)


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
# 58 client_vsync_ts, min_rtt, first_send_rtt,last_send_rtt,valid_rtt,ch_ack_delay,ch_send_delay
def analyze_frame_jitter_simplified(
    file_path,
    start_idx=3600,
    sim_data_len=60 * 60 * 20,
    anchor_frame_extrapolator_mode=4,
    frame_interval=16.667,
    print_log=False,
):
    """
    Simulate the display process with a frame trace and a set of parameters.
    param display_mode: naive_vsync, simple_ctrl
    """
    data, _ = load_data.load_detailed_framerate_log(
        file_path, start_idx=start_idx, len_limit=sim_data_len
    )  # sim for 20min

    if data is None:
        print("None data")
        return None, None

    if data.shape[0] < start_idx:
        print("trace too short, len:", data.shape[0])
        target_path = os.path.join(os.path.dirname(file_path), "small")
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)
        file_name = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(target_path, file_name))
        print("move file: %s to %s" % (file_path, os.path.join(target_path, file_name)))
        return None, None

    # only simulate 60FPS traces
    frame_render_interval = data[1:, 33] - data[:-1, 33]
    avg_render_interval = np.mean(frame_render_interval[frame_render_interval < 100])
    if np.abs(frame_interval - avg_render_interval) > 5:
        print(
            "wrong trace: %s avg_render_interval: %d" % (file_path, avg_render_interval)
        )
        if avg_render_interval > frame_interval + 5:
            target_path = os.path.join(os.path.dirname(file_path), "30fps")
            if not os.path.exists(target_path):
                os.makedirs(target_path, exist_ok=True)
            file_name = os.path.basename(file_path)
            shutil.move(file_path, os.path.join(target_path, file_name))
            print(
                "move file: %s to %s"
                % (file_path, os.path.join(target_path, file_name))
            )
        elif avg_render_interval < frame_interval - 5:
            target_path = os.path.join(os.path.dirname(file_path), "120fps")
            if not os.path.exists(target_path):
                os.makedirs(target_path, exist_ok=True)
            file_name = os.path.basename(file_path)
            shutil.move(file_path, os.path.join(target_path, file_name))
            print(
                "move file: %s to %s"
                % (file_path, os.path.join(target_path, file_name))
            )
        print()
        return None, None

    # initialization: calculate the average decode and render time
    def cal_avg_client_time(samp_len):
        # valid_idx = (data[300:samp_len, 5:12].sum(-1) > data[299:samp_len-1, 5:12].sum(-1)) + 300
        valid_idx = np.where(data[:samp_len, 4] == 0)[0]
        if valid_idx.size == 0:
            return 999, 999, 999, 999
        avg_dec_time = np.mean(data[valid_idx, 9])
        avg_dec_total_time = np.mean(data[valid_idx, 6:10].sum(-1))
        avg_render_time = np.mean(data[valid_idx, 11])
        avg_proc_time = np.mean(data[valid_idx, 6:12].sum(-1))

        return avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time

    avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time = (
        cal_avg_client_time(sim_data_len)
    )
    min_rtt = np.min(data[:, 59])

    values, counts = np.unique(data[:, 30], return_counts=True)
    idx = np.argmax(counts)
    server_optim_enabled = values[idx]

    values, counts = np.unique(data[:, 31], return_counts=True)
    idx = np.argmax(counts)
    client_optim_enabled = values[idx]

    values, counts = np.unique(data[:, 32], return_counts=True)
    idx = np.argmax(counts)
    client_vsync_enabled = values[idx]

    tot_frame_no = data.shape[0]
    dec_over_ts = data[:, 5:10].sum(-1)

    anchor_frame_extrapolator = AnchorFrameExtrapolator(
        data[0, 33], data[0, 1], frame_interval, anchor_frame_extrapolator_mode
    )
    smoothed_frame_pts = np.zeros(tot_frame_no, dtype=np.int64)
    smoothed_frame_pts_diff = np.zeros(tot_frame_no, dtype=np.int64)
    updated_frame_interval = np.zeros(tot_frame_no)
    anchor_frame_ts = np.zeros(tot_frame_no)
    anchor_frame_no = np.zeros(tot_frame_no, dtype=np.int64)

    # start the simulation
    idx = -1
    while True:
        idx += 1
        if idx >= tot_frame_no:
            break

        if data[idx, 4] != 0 or data[idx, 5] == 0:
            continue

        smoothed_frame_pts[idx] = anchor_frame_extrapolator.predict(data[idx, 1])
        smoothed_frame_pts_diff[idx] = data[idx, 33] - smoothed_frame_pts[idx]
        anchor_frame_extrapolator.update(data[idx, 33], data[idx, 1])
        updated_frame_interval[idx], anchor_frame_ts[idx], anchor_frame_no[idx] = (
            anchor_frame_extrapolator.get_frame_interval()
        )

    render_frame_interval = data[1:, 33] - data[:-1, 33]
    render_jitter = smoothed_frame_pts_diff
    dl_jitter = data[np.where(np.logical_and(data[:, 21] >= 0, data[:, 5] != 0))[0], 21]
    dec_jitter = (
        data[np.where(np.logical_and(data[:, 9] >= 0, data[:, 5] != 0))[0], 9]
        - avg_dec_time
    )
    display_jitter = (
        data[np.where(np.logical_and(data[:, 4] == 0, data[:, 5] != 0))[0], 11]
        - avg_render_time
    )

    # render_pacf_idx, render_pacf_values = cal_data_periodicity(render_frame_interval)
    # dl_pacf_idx, dl_pacf_values = cal_data_periodicity(dl_jitter)
    # dec_pacf_idx, dec_pacf_values = cal_data_periodicity(dec_jitter)
    # display_pacf_idx, display_pacf_values = cal_data_periodicity(display_jitter)

    # render_jitter
    render_jitter_frame_nos = []
    render_jitter_amps = []
    render_jitter_intervals = []
    render_jitter_consecs = []
    render_jitter_interval_stats = []
    render_jitter_consec_stats = []
    render_jitter_interval_no = []
    render_jitter_interval_same_max_nos = []
    render_jitter_consec_probs = []

    # dl jitter
    dl_jitter_frame_nos = []
    dl_jitter_amps = []
    dl_jitter_intervals = []
    dl_jitter_consecs = []
    dl_jitter_interval_stats = []
    dl_jitter_consec_stats = []
    dl_jitter_interval_no = []
    dl_jitter_interval_same_max_nos = []
    dl_jitter_consec_probs = []

    dl_jitter_type_nos = []

    # dec jitter
    dec_jitter_frame_nos = []
    dec_jitter_amps = []
    dec_jitter_intervals = []
    dec_jitter_consecs = []
    dec_jitter_interval_stats = []
    dec_jitter_consec_stats = []
    dec_jitter_interval_no = []
    dec_jitter_interval_same_max_nos = []
    dec_jitter_consec_probs = []

    # display jitter
    display_jitter_frame_nos = []
    display_jitter_amps = []
    display_jitter_intervals = []
    display_jitter_consecs = []
    display_jitter_interval_stats = []
    display_jitter_consec_stats = []
    display_jitter_interval_no = []
    display_jitter_interval_same_max_nos = []
    display_jitter_consec_probs = []

    for pct_tile_no in pct_tile_nos:
        render_jitter_amps.append(np.percentile(smoothed_frame_pts_diff, pct_tile_no))
        dl_jitter_amps.append(np.percentile(dl_jitter, pct_tile_no))
        dec_jitter_amps.append(np.percentile(dec_jitter, pct_tile_no))
        display_jitter_amps.append(np.percentile(display_jitter, pct_tile_no))

    # network related
    frame_stall_thr = 50

    new_data = data[np.where(np.logical_and(data[:, 21] >= 0, data[:, 5] != 0))[0], :]
    network_big_frame_flag = np.logical_and.reduce(
        (
            new_data[:, 3] > new_data[:, 40] * 1024 / 8 / 60 * 1.5,
            new_data[:, 16] - min_rtt - new_data[:, 20] - frame_interval > 0,
            new_data[:, 3]
            / np.maximum(
                new_data[:, 16] - min_rtt - new_data[:, 20], np.ones(tot_frame_no)
            )
            / 1024
            * 8
            * 1000
            > new_data[:, 40] * 0.85,
        )
    )[1:-1]
    network_stall_flag = (new_data[:, 21] >= frame_stall_thr)[1:-1]

    network_packet_loss_flag = np.logical_and(
        new_data[:, 38] > 0, new_data[:, 16] >= 2 * min_rtt
    )
    network_packet_loss_flag = np.logical_and.reduce(
        (
            network_packet_loss_flag[:-2],
            network_packet_loss_flag[1:-1],
            network_packet_loss_flag[2:],
        )
    )

    network_bitrate_change_flag = new_data[1:-1, 40] > new_data[:-2, 40]

    network_problem_flag = np.logical_or.reduce(
        (
            network_stall_flag,
            network_big_frame_flag,
            network_packet_loss_flag,
            network_bitrate_change_flag,
        )
    )

    window_max_loss_no = 3
    window_frame_no = 60 // window_max_loss_no
    window_tot_no = tot_frame_no // window_frame_no
    # print(tot_frame_no, min_rtt, avg_dec_time)
    for jitter_amp_thr in jitter_amp_thrs:
        render_jitter_flag = smoothed_frame_pts_diff > jitter_amp_thr
        render_jitter_frame_nos.append(np.sum(render_jitter_flag))
        render_jitter_consec_no = count_consecutive_boolean(render_jitter_flag)
        render_jitter_interval = count_consecutive_boolean(
            np.logical_not(render_jitter_flag)
        )
        render_jitter_consec_stats += cal_stats_between_percentile(
            render_jitter_consec_no, 25, 75
        )
        render_jitter_interval_stats += cal_stats_between_percentile(
            render_jitter_interval, 25, 75
        )

        for jitter_interval_thr in jitter_interval_thrs:
            render_jitter_interval_no.append(
                np.sum(np.asarray(render_jitter_interval) > jitter_interval_thr)
            )
            render_jitter_interval_same_max_no = count_consecutive_boolean(
                np.asarray(render_jitter_interval) > jitter_interval_thr
            )
            for pct_tile_no in pct_tile_nos:
                render_jitter_interval_same_max_nos.append(
                    np.percentile(render_jitter_interval_same_max_no, pct_tile_no)
                )

        render_jitter_consec_res = []
        render_jitter_interval_res = []
        for pct_tile_no in pct_tile_nos:
            render_jitter_consec_res.append(
                np.percentile(render_jitter_consec_no, pct_tile_no)
            )
            render_jitter_interval_res.append(
                np.percentile(render_jitter_interval, pct_tile_no)
            )
        render_jitter_consecs += render_jitter_consec_res
        render_jitter_intervals += render_jitter_interval_res

        dl_jitter_flag = dl_jitter > jitter_amp_thr
        dl_jitter_frame_nos.append(np.sum(dl_jitter_flag))
        dl_jitter_consec_no = count_consecutive_boolean(dl_jitter_flag)
        dl_jitter_interval = count_consecutive_boolean(np.logical_not(dl_jitter_flag))
        dl_jitter_consec_stats += cal_stats_between_percentile(
            dl_jitter_consec_no, 25, 75
        )
        dl_jitter_interval_stats += cal_stats_between_percentile(
            dl_jitter_interval, 25, 75
        )

        for jitter_interval_thr in jitter_interval_thrs:
            dl_jitter_interval_no.append(
                np.sum(np.asarray(dl_jitter_interval) > jitter_interval_thr)
            )
            dl_jitter_interval_same_max_no = count_consecutive_boolean(
                np.asarray(dl_jitter_interval) > jitter_interval_thr
            )
            for pct_tile_no in pct_tile_nos:
                dl_jitter_interval_same_max_nos.append(
                    np.percentile(dl_jitter_interval_same_max_no, pct_tile_no)
                )

        dl_jitter_consec_res = []
        dl_jitter_interval_res = []
        for pct_tile_no in pct_tile_nos:
            dl_jitter_consec_res.append(np.percentile(dl_jitter_consec_no, pct_tile_no))
            dl_jitter_interval_res.append(
                np.percentile(dl_jitter_interval, pct_tile_no)
            )
        dl_jitter_consecs += dl_jitter_consec_res
        dl_jitter_intervals += dl_jitter_interval_res

        # dl jitter types classification:
        # 1. big frame
        # 2. bitrate adjustment
        # 3. packet loss
        # 4. random jitter
        # 5. stall
        # network_stall_flag, network_big_frame_flag, network_packet_loss_flag, network_bitrate_change_flag

        dl_jitter_flag = dl_jitter_flag[1:-1]
        dl_jitter_type_no = [
            np.sum(dl_jitter_flag),
            np.sum(np.logical_and(dl_jitter_flag, network_stall_flag)),
            np.sum(np.logical_and(dl_jitter_flag, network_big_frame_flag)),
            np.sum(np.logical_and(dl_jitter_flag, network_packet_loss_flag)),
            np.sum(np.logical_and(dl_jitter_flag, network_bitrate_change_flag)),
            np.sum(
                np.logical_and(dl_jitter_flag, np.logical_not(network_problem_flag))
            ),
        ]
        dl_jitter_type_nos += dl_jitter_type_no

        dec_jitter_flag = dec_jitter > jitter_amp_thr
        dec_jitter_frame_nos.append(np.sum(dec_jitter_flag))
        dec_jitter_consec_no = count_consecutive_boolean(dec_jitter_flag)
        dec_jitter_interval = count_consecutive_boolean(np.logical_not(dec_jitter_flag))
        dec_jitter_consec_stats += cal_stats_between_percentile(
            dec_jitter_consec_no, 25, 75
        )
        dec_jitter_interval_stats += cal_stats_between_percentile(
            dec_jitter_interval, 25, 75
        )

        for jitter_interval_thr in jitter_interval_thrs:
            dec_jitter_interval_no.append(
                np.sum(np.asarray(dec_jitter_interval) > jitter_interval_thr)
            )
            dec_jitter_interval_same_max_no = count_consecutive_boolean(
                np.asarray(dec_jitter_interval) > jitter_interval_thr
            )
            for pct_tile_no in pct_tile_nos:
                dec_jitter_interval_same_max_nos.append(
                    np.percentile(dec_jitter_interval_same_max_no, pct_tile_no)
                )

        dec_jitter_consec_res = []
        dec_jitter_interval_res = []
        for pct_tile_no in pct_tile_nos:
            dec_jitter_consec_res.append(
                np.percentile(dec_jitter_consec_no, pct_tile_no)
            )
            dec_jitter_interval_res.append(
                np.percentile(dec_jitter_interval, pct_tile_no)
            )
        dec_jitter_consecs += dec_jitter_consec_res
        dec_jitter_intervals += dec_jitter_interval_res

        display_jitter_flag = display_jitter > jitter_amp_thr
        display_jitter_frame_nos.append(np.sum(display_jitter_flag))
        display_jitter_consec_no = count_consecutive_boolean(display_jitter_flag)
        display_jitter_interval = count_consecutive_boolean(
            np.logical_not(display_jitter_flag)
        )
        display_jitter_consec_stats += cal_stats_between_percentile(
            display_jitter_consec_no, 25, 75
        )
        display_jitter_interval_stats += cal_stats_between_percentile(
            display_jitter_interval, 25, 75
        )

        for jitter_interval_thr in jitter_interval_thrs:
            display_jitter_interval_no.append(
                np.sum(np.asarray(display_jitter_interval) > jitter_interval_thr)
            )
            display_jitter_interval_same_max_no = count_consecutive_boolean(
                np.asarray(display_jitter_interval) > jitter_interval_thr
            )
            for pct_tile_no in pct_tile_nos:
                display_jitter_interval_same_max_nos.append(
                    np.percentile(display_jitter_interval_same_max_no, pct_tile_no)
                )

        display_jitter_consec_res = []
        display_jitter_interval_res = []
        for pct_tile_no in pct_tile_nos:
            display_jitter_consec_res.append(
                np.percentile(display_jitter_consec_no, pct_tile_no)
            )
            display_jitter_interval_res.append(
                np.percentile(display_jitter_interval, pct_tile_no)
            )
        display_jitter_consecs += display_jitter_consec_res
        display_jitter_intervals += display_jitter_interval_res

        window_render_jitter_no = (
            (render_jitter[: render_jitter.size // window_frame_no * window_frame_no])
            .reshape(-1, window_frame_no)
            .sum(-1)
        )
        window_dl_jitter_no = (
            (
                dl_jitter[: dl_jitter.size // window_frame_no * window_frame_no]
                > jitter_amp_thr
            )
            .reshape(-1, window_frame_no)
            .sum(-1)
        )
        window_dec_jitter_no = (
            (
                dec_jitter[: dec_jitter.size // window_frame_no * window_frame_no]
                > jitter_amp_thr
            )
            .reshape(-1, window_frame_no)
            .sum(-1)
        )
        window_display_jitter_no = (
            (
                display_jitter[
                    : display_jitter.size // window_frame_no * window_frame_no
                ]
                > jitter_amp_thr
            )
            .reshape(-1, window_frame_no)
            .sum(-1)
        )

        render_jitter_fast_change = np.logical_and.reduce(
            (
                window_render_jitter_no[:-2] <= 0,
                window_render_jitter_no[1:-1] + window_render_jitter_no[2:]
                > window_max_loss_no,
            )
        )
        dl_jitter_fast_change = np.logical_and.reduce(
            (
                window_dl_jitter_no[:-2] <= 0,
                window_dl_jitter_no[1:-1] + window_dl_jitter_no[2:]
                > window_max_loss_no,
            )
        )
        dec_jitter_fast_change = np.logical_and.reduce(
            (
                window_dec_jitter_no[:-2] <= 0,
                window_dec_jitter_no[1:-1] + window_dec_jitter_no[2:]
                > window_max_loss_no,
            )
        )
        display_jitter_fast_change = np.logical_and.reduce(
            (
                window_display_jitter_no[:-2] <= 0,
                window_display_jitter_no[1:-1] + window_display_jitter_no[2:]
                > window_max_loss_no,
            )
        )

        render_jitter_fake_recover = np.logical_and.reduce(
            (
                window_render_jitter_no[:-2] > 1,
                window_render_jitter_no[1:-1] <= 0,
                window_render_jitter_no[2:] > 0,
            )
        )
        dl_jitter_fake_recover = np.logical_and.reduce(
            (
                window_dl_jitter_no[:-2] > 1,
                window_dl_jitter_no[1:-1] <= 0,
                window_dl_jitter_no[2:] > 0,
            )
        )
        dec_jitter_fake_recover = np.logical_and.reduce(
            (
                window_dec_jitter_no[:-2] > 1,
                window_dec_jitter_no[1:-1] <= 0,
                window_dec_jitter_no[2:] > 0,
            )
        )
        display_jitter_fake_recover = np.logical_and.reduce(
            (
                window_display_jitter_no[:-2] > 1,
                window_display_jitter_no[1:-1] <= 0,
                window_display_jitter_no[2:] > 0,
            )
        )

        # print('render_jitter', window_tot_no, np.sum(window_render_jitter_no[:-2] < 1), np.sum(window_render_jitter_no[:-2] + window_render_jitter_no[1:-1] + window_render_jitter_no[2:] > 3), np.sum(render_jitter_fast_change), np.sum(window_render_jitter_no[:-2] > 0), np.sum(render_jitter_fake_recover))
        # print('dl_jitter', window_tot_no, np.sum(window_dl_jitter_no[:-2] < 1), np.sum(window_dl_jitter_no[:-2] + window_dl_jitter_no[1:-1] + window_dl_jitter_no[2:] > 3), np.sum(dl_jitter_fast_change), np.sum(window_dl_jitter_no[:-2] > 0), np.sum(dl_jitter_fake_recover))
        # print('dec_jitter', window_tot_no, np.sum(window_dec_jitter_no[:-2] < 1), np.sum(window_dec_jitter_no[:-2] + window_dec_jitter_no[1:-1] + window_dec_jitter_no[2:] > 3), np.sum(dec_jitter_fast_change), np.sum(window_dec_jitter_no[:-2] > 0), np.sum(dec_jitter_fake_recover))
        render_no_jitter_window_num = np.sum(window_render_jitter_no[:-2] < 1)
        render_jitter_window_num = np.sum(
            window_render_jitter_no[:-2]
            + window_render_jitter_no[1:-1]
            + window_render_jitter_no[2:]
            > 3
        )
        render_jitter_fast_change_window_num = np.sum(render_jitter_fast_change)
        render_jitter_recover_window_num = np.sum(
            np.logical_and(
                window_display_jitter_no[:-2] > 1, window_display_jitter_no[1:-1] <= 0
            )
        )
        render_jitter_fake_recover_num = np.sum(render_jitter_fake_recover)

        dl_no_jitter_window_num = np.sum(window_dl_jitter_no[:-2] < 1)
        dl_jitter_window_num = np.sum(
            window_dl_jitter_no[:-2]
            + window_dl_jitter_no[1:-1]
            + window_dl_jitter_no[2:]
            > 3
        )
        dl_jitter_fast_change_window_num = np.sum(dl_jitter_fast_change)
        dl_jitter_recover_window_num = np.sum(
            np.logical_and(
                window_display_jitter_no[:-2] > 1, window_display_jitter_no[1:-1] <= 0
            )
        )
        dl_jitter_fake_recover_num = np.sum(dl_jitter_fake_recover)

        dec_no_jitter_window_num = np.sum(window_dec_jitter_no[:-2] < 1)
        dec_jitter_window_num = np.sum(
            window_dec_jitter_no[:-2]
            + window_dec_jitter_no[1:-1]
            + window_dec_jitter_no[2:]
            > 3
        )
        dec_jitter_fast_change_window_num = np.sum(dec_jitter_fast_change)
        dec_jitter_recover_window_num = np.sum(
            np.logical_and(
                window_display_jitter_no[:-2] > 1, window_display_jitter_no[1:-1] <= 0
            )
        )
        dec_jitter_fake_recover_num = np.sum(dec_jitter_fake_recover)

        display_no_jitter_window_num = np.sum(window_display_jitter_no[:-2] < 1)
        display_jitter_window_num = np.sum(
            window_display_jitter_no[:-2]
            + window_display_jitter_no[1:-1]
            + window_display_jitter_no[2:]
            > 3
        )
        display_jitter_fast_change_window_num = np.sum(display_jitter_fast_change)
        display_jitter_recover_window_num = np.sum(
            np.logical_and(
                window_display_jitter_no[:-2] > 1, window_display_jitter_no[1:-1] <= 0
            )
        )
        display_jitter_fake_recover_num = np.sum(display_jitter_fake_recover)

    result = (
        [server_optim_enabled, client_optim_enabled, client_vsync_enabled, tot_frame_no]
        + dl_jitter_type_nos
        + render_jitter_frame_nos
        + render_jitter_amps
        + render_jitter_consecs
        + render_jitter_intervals
        + render_jitter_consec_stats
        + render_jitter_interval_stats
        + render_jitter_interval_no
        + render_jitter_interval_same_max_nos
        + dl_jitter_frame_nos
        + dl_jitter_amps
        + dl_jitter_consecs
        + dl_jitter_intervals
        + dl_jitter_consec_stats
        + dl_jitter_interval_stats
        + dl_jitter_interval_no
        + dl_jitter_interval_same_max_nos
        + dec_jitter_frame_nos
        + dec_jitter_amps
        + dec_jitter_consecs
        + dec_jitter_intervals
        + dec_jitter_consec_stats
        + dec_jitter_interval_stats
        + dec_jitter_interval_no
        + dec_jitter_interval_same_max_nos
        + display_jitter_frame_nos
        + display_jitter_amps
        + display_jitter_consecs
        + display_jitter_intervals
        + display_jitter_consec_stats
        + display_jitter_interval_stats
        + display_jitter_interval_no
        + display_jitter_interval_same_max_nos
        + [window_tot_no]
        + [
            render_no_jitter_window_num,
            render_jitter_window_num,
            render_jitter_fast_change_window_num,
            render_jitter_recover_window_num,
            render_jitter_fake_recover_num,
        ]
        + [
            dl_no_jitter_window_num,
            dl_jitter_window_num,
            dl_jitter_fast_change_window_num,
            dl_jitter_recover_window_num,
            dl_jitter_fake_recover_num,
        ]
        + [
            dec_no_jitter_window_num,
            dec_jitter_window_num,
            dec_jitter_fast_change_window_num,
            dec_jitter_recover_window_num,
            dec_jitter_fake_recover_num,
        ]
        + [
            display_no_jitter_window_num,
            display_jitter_window_num,
            display_jitter_fast_change_window_num,
            display_jitter_recover_window_num,
            display_jitter_fake_recover_num,
        ]
    )
    # render_pacf_idx + render_pacf_values + dl_pacf_idx + dl_pacf_values + \
    # dec_pacf_idx + dec_pacf_values + display_pacf_idx + display_pacf_values
    # output_file = open(os.path.join('test_data', 'result_pacf_start%d_len%d.csv' %(start_idx, sim_data_len)), 'a')
    output_file = open(
        os.path.join(
            "test_data",
            "result_jitter_analysis_start%d_len%d.csv" % (START_IDX, SIM_DATA_LEN),
        ),
        "a",
    )
    output_file.write(
        file_path.replace(",", "_")
        + ", "
        + ", ".join([str(item) for item in result])
        + "\n"
    )
    output_file.close()
    print(
        file_path.replace(",", "_") + ", " + ", ".join([str(item) for item in result])
    )
    return file_path, result


def analyze_frame_jitter_detail(
    file_path,
    anchor_frame_extrapolator_mode=4,
    frame_interval=16.667,
    sim_data_len=60 * 60 * 20,
    start_idle_len=3600,
    print_log=False,
):
    """
    Simulate the display process with a frame trace and a set of parameters.
    param display_mode: naive_vsync, simple_ctrl
    """
    data, _ = load_data.load_detailed_framerate_log(
        file_path, start_idx=0, len_limit=sim_data_len + start_idle_len
    )  # sim for 20min

    if data is None:
        print("None data")
        return None, None

    if data.shape[0] < start_idle_len * 1:
        print("trace too short, len:", data.shape[0])
        target_path = os.path.join(os.path.dirname(file_path), "small")
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)
        file_name = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(target_path, file_name))
        print("move file: %s to %s" % (file_path, os.path.join(target_path, file_name)))
        return None, None

    # only simulate 60FPS traces
    frame_render_interval = data[1:, 33] - data[:-1, 33]
    avg_render_interval = np.mean(frame_render_interval[frame_render_interval < 100])
    if np.abs(frame_interval - avg_render_interval) > 5:
        print(
            "wrong trace: %s avg_render_interval: %d" % (file_path, avg_render_interval)
        )
        if avg_render_interval > frame_interval + 5:
            target_path = os.path.join(os.path.dirname(file_path), "30fps")
            if not os.path.exists(target_path):
                os.makedirs(target_path, exist_ok=True)
            file_name = os.path.basename(file_path)
            shutil.move(file_path, os.path.join(target_path, file_name))
            print(
                "move file: %s to %s"
                % (file_path, os.path.join(target_path, file_name))
            )
        elif avg_render_interval < frame_interval - 5:
            target_path = os.path.join(os.path.dirname(file_path), "120fps")
            if not os.path.exists(target_path):
                os.makedirs(target_path, exist_ok=True)
            file_name = os.path.basename(file_path)
            shutil.move(file_path, os.path.join(target_path, file_name))
            print(
                "move file: %s to %s"
                % (file_path, os.path.join(target_path, file_name))
            )
        print()
        return None, None

    # initialization: calculate the average decode and render time
    def cal_avg_client_time(samp_len):
        # valid_idx = (data[300:samp_len, 5:12].sum(-1) > data[299:samp_len-1, 5:12].sum(-1)) + 300
        valid_idx = np.where(data[:samp_len, 4] == 0)[0]
        if valid_idx.size == 0:
            return 999, 999, 999, 999
        avg_dec_time = np.mean(data[valid_idx, 9])
        avg_dec_total_time = np.mean(data[valid_idx, 6:10].sum(-1))
        avg_render_time = np.mean(data[valid_idx, 11])
        avg_proc_time = np.mean(data[valid_idx, 6:12].sum(-1))

        return avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time

    avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time = (
        cal_avg_client_time(start_idle_len)
    )
    if avg_render_time > 10:
        print("render time too large: %d" % avg_render_time)
        target_path = os.path.join(os.path.dirname(file_path), "render_problem")
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)
        file_name = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(target_path, file_name))
        print("move file: %s to %s" % (file_path, os.path.join(target_path, file_name)))
        return None, None

    data = data[start_idle_len:, :]

    values, counts = np.unique(data[:, 30], return_counts=True)
    idx = np.argmax(counts)
    server_optim_enabled = values[idx]

    values, counts = np.unique(data[:, 31], return_counts=True)
    idx = np.argmax(counts)
    client_optim_enabled = values[idx]

    values, counts = np.unique(data[:, 32], return_counts=True)
    idx = np.argmax(counts)
    client_vsync_enabled = values[idx]

    anchor_frame_extrapolator = AnchorFrameExtrapolator(
        data[0, 33], data[0, 1], frame_interval, anchor_frame_extrapolator_mode
    )

    tot_frame_no = data.shape[0]
    dec_over_ts = data[:, 5:10].sum(-1)
    nearest_vsync_ts = np.zeros(tot_frame_no, dtype=np.int64)
    cur_anchor_vsync_ts = np.zeros(tot_frame_no)
    dec_queued_frame_cnt = np.zeros(tot_frame_no, dtype=np.int64)
    invalid_frame_flag = np.zeros(tot_frame_no, dtype=np.int64)

    anchor_vsync_ts = data[0, 58]
    cur_anchor_vsync_ts[0] = anchor_vsync_ts
    prev_vsync_ts = data[0, 58]

    smoothed_frame_pts = np.zeros(tot_frame_no, dtype=np.int64)
    smoothed_frame_pts_diff = np.zeros(tot_frame_no, dtype=np.int64)
    updated_frame_interval = np.zeros(tot_frame_no)
    anchor_frame_ts = np.zeros(tot_frame_no)
    anchor_frame_no = np.zeros(tot_frame_no, dtype=np.int64)

    min_rtt = np.min(data[:, 59])
    frame_stall_thr = 50
    small_ts_margin = 2

    # start the simulation
    # for idx in range(1, tot_frame_no):
    idx = -1
    window_steps = [200, 300, 500, 1000]
    window_step_no = len(window_steps)
    window_start_ts = 0
    all_window_start_ts = [0] * window_step_no
    cur_window_valid_cnt = [0] * window_step_no
    hist_window_valid_cnt = [0] * window_step_no
    while True:
        idx += 1
        if idx >= tot_frame_no:
            break

        if data[idx, 4] != 0 or data[idx, 5] == 0:
            continue

        if prev_vsync_ts != data[idx, 58]:
            anchor_vsync_ts = rectify_anchor_vsync_ts(anchor_vsync_ts, data[idx, 58])
            prev_vsync_ts = data[idx, 58]
        cur_anchor_vsync_ts[idx] = anchor_vsync_ts
        nearest_vsync_ts[idx] = cal_next_vsync_ts(
            dec_over_ts[idx], anchor_vsync_ts, frame_interval
        )

        if window_start_ts == 0:
            window_start_ts = nearest_vsync_ts[idx]
        elif np.abs(nearest_vsync_ts[idx] - window_start_ts - 1000) <= 5:
            if cur_window_valid_cnt > 60:
                print(
                    "abs",
                    idx,
                    cur_window_valid_cnt,
                    window_start_ts,
                    nearest_vsync_ts[idx],
                    np.abs(nearest_vsync_ts[idx] - window_start_ts - 1000),
                )
            hist_window_valid_cnt.append(cur_window_valid_cnt)
            cur_window_valid_cnt = 0
            window_start_ts = nearest_vsync_ts[idx]
        elif nearest_vsync_ts[idx] - window_start_ts >= 1000:
            if cur_window_valid_cnt > 60:
                print(
                    1000,
                    idx,
                    cur_window_valid_cnt,
                    window_start_ts,
                    nearest_vsync_ts[idx],
                    np.abs(nearest_vsync_ts[idx] - window_start_ts - 1000),
                )
            hist_window_valid_cnt.append(cur_window_valid_cnt)
            cur_window_valid_cnt = 0
            window_start_ts = nearest_vsync_ts[idx]

        queued_frame_cnt = 1
        prev_idx = idx - 1
        while True:
            if prev_idx < 0:
                break

            if data[prev_idx, 4] == 0 and dec_over_ts[prev_idx] >= np.floor(
                nearest_vsync_ts[idx] - frame_interval
            ):  # small margin, since client doesn't respond that fast
                queued_frame_cnt += 1
            elif data[prev_idx, 4] == 0 and dec_over_ts[prev_idx] < np.floor(
                nearest_vsync_ts[idx] - frame_interval
            ):
                break

            prev_idx -= 1

        dec_queued_frame_cnt[idx] = queued_frame_cnt
        if queued_frame_cnt > 1:
            invalid_frame_flag[idx] = 1

        else:
            cur_window_valid_cnt += 1

            smoothed_frame_pts[idx] = anchor_frame_extrapolator.predict(data[idx, 1])
            smoothed_frame_pts_diff[idx] = data[idx, 33] - smoothed_frame_pts[idx]
            anchor_frame_extrapolator.update(data[idx, 33], data[idx, 1])
            updated_frame_interval[idx], anchor_frame_ts[idx], anchor_frame_no[idx] = (
                anchor_frame_extrapolator.get_frame_interval()
            )

        # if idx >= 5135 and idx <= 5197:
        #     print(idx, dec_over_ts[idx], nearest_vsync_ts[idx], dec_queued_frame_cnt[idx], invalid_frame_flag[idx], cur_window_valid_cnt)

    print(hist_window_valid_cnt)
    exit()
    # network related
    network_big_frame_flag = np.zeros(tot_frame_no)
    tmp_flag = np.logical_and.reduce(
        (
            data[:, 3] > data[:, 40] * 1024 / 8 / 60 * 1.5,
            data[:, 16] - min_rtt - data[:, 20] - frame_interval > 0,
            # data[:, 40] > 10000
            data[:, 3]
            / np.maximum(data[:, 16] - min_rtt - data[:, 20], np.ones(tot_frame_no))
            / 1024
            * 8
            * 1000
            > data[:, 40] * 0.85,
        )
    )
    network_big_frame_flag[1:] = tmp_flag[:-1]

    network_dl_jitter_flag = np.zeros(tot_frame_no)
    tmp_flag = np.logical_and.reduce(
        (data[:, 21] > small_ts_margin, data[:, 21] < frame_stall_thr)
    )
    network_dl_jitter_flag[1:] = tmp_flag[:-1]

    network_stall_flag = np.zeros(tot_frame_no)
    tmp_flag = data[:, 21] >= frame_stall_thr
    network_stall_flag[1:] = tmp_flag[:-1]

    network_packet_loss_flag = np.zeros(tot_frame_no)
    tmp_flag = np.logical_and(data[:, 38] > 0, data[:, 16] >= 2 * min_rtt)
    network_packet_loss_flag[1:] = tmp_flag[:-1]

    network_problem_flag = np.logical_or.reduce(
        (
            network_big_frame_flag,
            network_dl_jitter_flag,
            network_stall_flag,
            network_packet_loss_flag,
        )
    )

    # render related
    render_jitter_flag = smoothed_frame_pts_diff < -small_ts_margin
    render_jitter_flag[1:] = np.logical_or(
        render_jitter_flag[1:], smoothed_frame_pts_diff[:-1] > small_ts_margin
    )

    # server related
    server_jitter_flag = np.zeros(tot_frame_no)
    frame_cgs_render_interval = data[1:, 33] - data[:-1, 33]
    frame_proxy_recv_interval = data[1:, 12] - data[:-1, 12]
    server_jitter_flag[2:] = (frame_proxy_recv_interval - frame_cgs_render_interval)[
        :-1
    ] > small_ts_margin

    # decoder related
    decoder_jitter_flag = np.zeros(tot_frame_no)
    decoder_jitter_flag[1:] = np.logical_or(
        data[:, 9] > avg_dec_time + small_ts_margin, data[:, 7:9].sum(-1) > 0
    )[:-1]

    # network induced queue
    network_stall_induced_queue = np.logical_and.reduce(
        (dec_queued_frame_cnt > 1, network_stall_flag)
    )
    network_packet_loss_induced_queue = np.logical_and.reduce(
        (
            dec_queued_frame_cnt > 1,
            network_packet_loss_flag,
            np.logical_not(network_stall_flag),
        )
    )
    network_big_frame_induced_queue = np.logical_and.reduce(
        (
            dec_queued_frame_cnt > 1,
            network_big_frame_flag,
            np.logical_not(network_packet_loss_flag),
            np.logical_not(network_stall_flag),
        )
    )
    network_i_frame_induced_queue = np.logical_and(
        network_big_frame_induced_queue,
        np.logical_or(data[:, 2] == 2, data[:, 49] == 1),
    )

    network_dl_jitter_induced_queue = np.logical_and.reduce(
        (
            dec_queued_frame_cnt > 1,
            network_dl_jitter_flag,
            np.logical_not(network_big_frame_flag),
            np.logical_not(network_packet_loss_flag),
            np.logical_not(network_stall_flag),
        )
    )

    # render induced queue
    render_jitter_induced_queue = np.logical_and.reduce(
        (
            dec_queued_frame_cnt > 1,
            render_jitter_flag,
            np.logical_not(network_problem_flag),
        )
    )

    # server inside jitter induced queue
    server_jitter_induced_queue = np.logical_and.reduce(
        (
            dec_queued_frame_cnt > 1,
            server_jitter_flag,
            np.logical_not(network_problem_flag),
            np.logical_not(render_jitter_flag),
        )
    )

    # decoder induced queue
    decoder_jitter_induced_queue = np.logical_and.reduce(
        (
            dec_queued_frame_cnt > 1,
            decoder_jitter_flag,
            np.logical_not(network_problem_flag),
            np.logical_not(render_jitter_flag),
            np.logical_not(server_jitter_flag),
        )
    )

    if print_log:
        log_file = open(
            file_path[:-4]
            + "_anchorPredictor%d_jitter_analyze.csv"
            % (anchor_frame_extrapolator_mode),
            "w",
        )

        log_file.write(
            ",".join(
                [
                    "sim_index",
                    "render_index",
                    "frame_index",
                    "frame_type",
                    "size",
                    "loss_type",
                    "client_receive_ts",
                    "receive_and_unpack",
                    "decoder_outside_queue",
                    "decoder_insided_queue",
                    "decode",
                    "render_queue",
                    "display",
                    "proxy_recv_ts",
                    "proxy_recv_time",
                    "proxy_send_delay",
                    "send_time",
                    "net_time",
                    "proc_time",
                    "tot_time",
                    "basic_net_ts",
                    "ul_jitter",
                    "dl_jitter",
                    "expected_recv_ts",
                    "expected_proc_time",
                    "nearest_display_ts",
                    "expected_display_ts",
                    "actual_display_ts",
                    "vsync_diff",
                    "present_timer_offset",
                    "jitter_buf_size",
                    "server_optim_enabled",
                    "client_optim_enabled",
                    "client_vsync_enabled",
                    "pts",
                    "ets",
                    "dts",
                    "sts",
                    "Mrts0ToRtsOffset",
                    "packet_lossed_perK",
                    "encoding_rate",
                    "cc_rate",
                    "smoothrate",
                    "width",
                    "height",
                    "sqoe",
                    "ori_sqoe",
                    "target_sqoe",
                    "recomm_bitrate",
                    "actual_bitrate",
                    "scene_change",
                    "encoding_fps",
                    "satd",
                    "qp",
                    "mvx",
                    "mvy",
                    "intra_mb",
                    "inter_mb",
                    "cur_cgs_pause_cnt",
                    "client_vsync_ts",
                    "min_rtt",
                    "dec_over_ts",
                    "nearest_vsync_ts",
                    "dec_queued_frame_cnt",
                    "frame_queue_flag",
                    "original_pts",
                    "smoothed_pts",
                    "smoothed_frame_pts_diff",
                    "render_jitter_induced_queue",
                    "server_jitter_induced_queue",
                    "network_stall_induced_queue",
                    "network_packet_loss_induced_queue",
                    "network_big_frame_induced_queue",
                    "network_i_frame_induced_queue",
                    "network_dl_jitter_induced_queue",
                    "decoder_jitter_induced_queue",
                    "updated_frame_interval",
                    "anchor_frame_ts",
                    "anchor_frame_no",
                ]
            )
            + "\n"
        )
        for idx in range(tot_frame_no):
            log_file.write(
                ",".join(
                    str(item)
                    for item in [idx]
                    + data[idx, :-5].tolist()
                    + [
                        dec_over_ts[idx],
                        nearest_vsync_ts[idx],
                        dec_queued_frame_cnt[idx],
                        int(dec_queued_frame_cnt[idx] > 1),
                        data[idx, 33],
                        smoothed_frame_pts[idx],
                        smoothed_frame_pts_diff[idx],
                        int(render_jitter_induced_queue[idx]),
                        int(server_jitter_induced_queue[idx]),
                        int(network_stall_induced_queue[idx]),
                        int(network_packet_loss_induced_queue[idx]),
                        int(network_big_frame_induced_queue[idx]),
                        int(network_i_frame_induced_queue[idx]),
                        int(network_dl_jitter_induced_queue[idx]),
                        int(decoder_jitter_induced_queue[idx]),
                        updated_frame_interval[idx],
                        anchor_frame_ts[idx],
                        anchor_frame_no[idx],
                    ]
                )
                + "\n"
            )

    # ts_diff = np.abs(smoothed_frame_pts - data[:, 33])
    # result = [server_optim_enabled,client_optim_enabled,client_vsync_enabled, np.mean(ts_diff), np.max(ts_diff), np.min(ts_diff)]

    result = [
        server_optim_enabled,
        client_optim_enabled,
        client_vsync_enabled,
        tot_frame_no,
        np.sum(dec_queued_frame_cnt > 1),
        network_stall_induced_queue.sum(),
        network_packet_loss_induced_queue.sum(),
        network_big_frame_induced_queue.sum(),
        network_i_frame_induced_queue.sum(),
        network_dl_jitter_induced_queue.sum(),
        render_jitter_induced_queue.sum(),
        server_jitter_induced_queue.sum(),
        decoder_jitter_induced_queue.sum(),
    ]
    output_file = open(
        os.path.join(
            "test_data",
            "result_anchorPredictor%d_jitter_analyze.csv"
            % (ANCHOR_FRAME_EXTRAPOLATOR_MODE),
        ),
        "a",
    )
    output_file.write(
        file_path.replace(",", "_")
        + ", "
        + ", ".join([str(item) for item in result])
        + "\n"
    )
    output_file.close()

    print(
        file_path.replace(",", "_") + ", " + ", ".join([str(item) for item in result])
    )
    return file_path, result


def analyze_frame_jitter_v1(
    file_path,
    anchor_frame_extrapolator_mode=0,
    frame_interval=16.667,
    sim_data_len=60 * 60 * 20,
    start_idle_len=3600,
    print_log=False,
):
    """
    Simulate the display process with a frame trace and a set of parameters.
    param display_mode: naive_vsync, simple_ctrl
    """
    data, _ = load_data.load_detailed_framerate_log(
        file_path, start_idx=0, len_limit=sim_data_len + start_idle_len
    )  # sim for 20min

    if data is None:
        print("None data")
        return None, None

    if data.shape[0] < start_idle_len * 1:
        print("trace too short, len:", data.shape[0])
        target_path = os.path.join(os.path.dirname(file_path), "small")
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)
        file_name = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(target_path, file_name))
        print("move file: %s to %s" % (file_path, os.path.join(target_path, file_name)))
        return None, None

    # only simulate 60FPS traces
    frame_render_interval = data[1:, 33] - data[:-1, 33]
    avg_render_interval = np.mean(frame_render_interval[frame_render_interval < 100])
    if np.abs(frame_interval - avg_render_interval) > 5:
        print(
            "wrong trace: %s avg_render_interval: %d" % (file_path, avg_render_interval)
        )
        if avg_render_interval > frame_interval + 5:
            target_path = os.path.join(os.path.dirname(file_path), "30fps")
            if not os.path.exists(target_path):
                os.makedirs(target_path, exist_ok=True)
            file_name = os.path.basename(file_path)
            shutil.move(file_path, os.path.join(target_path, file_name))
            print(
                "move file: %s to %s"
                % (file_path, os.path.join(target_path, file_name))
            )
        elif avg_render_interval < frame_interval - 5:
            target_path = os.path.join(os.path.dirname(file_path), "120fps")
            if not os.path.exists(target_path):
                os.makedirs(target_path, exist_ok=True)
            file_name = os.path.basename(file_path)
            shutil.move(file_path, os.path.join(target_path, file_name))
            print(
                "move file: %s to %s"
                % (file_path, os.path.join(target_path, file_name))
            )
        print()
        return None, None

    # initialization: calculate the average decode and render time
    def cal_avg_client_time(samp_len):
        # valid_idx = (data[300:samp_len, 5:12].sum(-1) > data[299:samp_len-1, 5:12].sum(-1)) + 300
        valid_idx = np.where(data[:samp_len, 4] == 0)[0]
        if valid_idx.size == 0:
            return 999, 999, 999, 999
        avg_dec_time = np.mean(data[valid_idx, 9])
        avg_dec_total_time = np.mean(data[valid_idx, 6:10].sum(-1))
        avg_render_time = np.mean(data[valid_idx, 11])
        avg_proc_time = np.mean(data[valid_idx, 6:12].sum(-1))

        return avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time

    avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time = (
        cal_avg_client_time(start_idle_len)
    )
    if avg_render_time > 10:
        print("render time too large: %d" % avg_render_time)
        target_path = os.path.join(os.path.dirname(file_path), "render_problem")
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)
        file_name = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(target_path, file_name))
        print("move file: %s to %s" % (file_path, os.path.join(target_path, file_name)))
        return None, None

    data = data[start_idle_len:, :]

    values, counts = np.unique(data[:, 30], return_counts=True)
    idx = np.argmax(counts)
    server_optim_enabled = values[idx]

    values, counts = np.unique(data[:, 31], return_counts=True)
    idx = np.argmax(counts)
    client_optim_enabled = values[idx]

    values, counts = np.unique(data[:, 32], return_counts=True)
    idx = np.argmax(counts)
    client_vsync_enabled = values[idx]

    anchor_frame_extrapolator = AnchorFrameExtrapolator(
        data[0, 33], data[0, 1], frame_interval, anchor_frame_extrapolator_mode
    )

    tot_frame_no = data.shape[0]
    dec_over_ts = data[:, 5:10].sum(-1)
    nearest_vsync_ts = np.zeros(tot_frame_no, dtype=np.int64)
    cur_anchor_vsync_ts = np.zeros(tot_frame_no)
    dec_queued_frame_cnt = np.zeros(tot_frame_no, dtype=np.int64)

    anchor_vsync_ts = data[0, 58]
    cur_anchor_vsync_ts[0] = anchor_vsync_ts
    prev_vsync_ts = data[0, 58]

    smoothed_frame_pts = np.zeros(tot_frame_no, dtype=np.int64)
    smoothed_frame_pts_diff = np.zeros(tot_frame_no, dtype=np.int64)
    updated_frame_interval = np.zeros(tot_frame_no)
    anchor_frame_ts = np.zeros(tot_frame_no)
    anchor_frame_no = np.zeros(tot_frame_no, dtype=np.int64)

    min_rtt = np.min(data[:, 59])
    frame_stall_thr = 50
    small_ts_margin = 2

    jitter_predictor = E2EJitterPredictor(
        data, min_rtt, frame_interval, frame_stall_thr
    )
    # start the simulation
    # for idx in range(1, tot_frame_no):
    idx = 0
    while True:
        idx += 1
        if idx >= tot_frame_no:
            break

        if data[idx, 4] != 0 or data[idx, 5] == 0:
            continue

        if prev_vsync_ts != data[idx, 58]:
            anchor_vsync_ts = rectify_anchor_vsync_ts(anchor_vsync_ts, data[idx, 58])
            prev_vsync_ts = data[idx, 58]
        cur_anchor_vsync_ts[idx] = anchor_vsync_ts
        nearest_vsync_ts[idx] = cal_next_vsync_ts(
            dec_over_ts[idx], anchor_vsync_ts, frame_interval
        )

        queued_frame_cnt = 1
        prev_idx = idx - 1
        while True:
            if prev_idx < 0:
                break

            if (
                data[prev_idx, 4] == 0
                and dec_over_ts[idx] >= nearest_vsync_ts[idx] - frame_interval
            ):  # small margin, since client doesn't respond that fast
                queued_frame_cnt += 1
            elif (
                data[prev_idx, 4] == 0
                and dec_over_ts[idx] < nearest_vsync_ts[idx] - frame_interval
            ):
                break

            prev_idx -= 1

        # next_idx = idx + 1
        # while True:
        #     if next_idx >= tot_frame_no:
        #         break

        #     if data[next_idx, 4] == 0 and dec_over_ts[next_idx] < nearest_vsync_ts[idx]-5: # small margin, since client doesn't respond that fast
        #         queued_frame_cnt += 1
        #     elif data[next_idx, 4] == 0 and dec_over_ts[next_idx] >= nearest_vsync_ts[idx]-5:
        #         break

        #     next_idx += 1

        dec_queued_frame_cnt[idx] = queued_frame_cnt
        if queued_frame_cnt > 1:
            pass

        smoothed_frame_pts[idx] = anchor_frame_extrapolator.predict(data[idx, 1])
        smoothed_frame_pts_diff[idx] = data[idx, 33] - smoothed_frame_pts[idx]
        anchor_frame_extrapolator.update(data[idx, 33], data[idx, 1])
        updated_frame_interval[idx], anchor_frame_ts[idx], anchor_frame_no[idx] = (
            anchor_frame_extrapolator.get_frame_interval()
        )

        jitter_predictor.update(data, smoothed_frame_pts_diff, idx)

    # network related
    network_big_frame_flag = np.zeros(tot_frame_no)
    tmp_flag = np.logical_and.reduce(
        (
            data[:, 3] > data[:, 40] * 1024 / 8 / 60 * 1.5,
            data[:, 16] - min_rtt - data[:, 20] - frame_interval > 0,
            # data[:, 40] > 10000
            data[:, 3]
            / np.maximum(data[:, 16] - min_rtt - data[:, 20], np.ones(tot_frame_no))
            / 1024
            * 8
            * 1000
            > data[:, 40] * 0.85,
        )
    )
    network_big_frame_flag[1:] = tmp_flag[:-1]

    network_dl_jitter_flag = np.zeros(tot_frame_no)
    tmp_flag = np.logical_and.reduce(
        (data[:, 21] > small_ts_margin, data[:, 21] < frame_stall_thr)
    )
    network_dl_jitter_flag[1:] = tmp_flag[:-1]

    network_stall_flag = np.zeros(tot_frame_no)
    tmp_flag = data[:, 21] >= frame_stall_thr
    network_stall_flag[1:] = tmp_flag[:-1]

    network_packet_loss_flag = np.zeros(tot_frame_no)
    tmp_flag = np.logical_and(data[:, 38] > 0, data[:, 16] >= 2 * min_rtt)
    network_packet_loss_flag[1:] = tmp_flag[:-1]

    network_problem_flag = np.logical_or.reduce(
        (
            network_big_frame_flag,
            network_dl_jitter_flag,
            network_stall_flag,
            network_packet_loss_flag,
        )
    )

    # render related
    render_jitter_flag = smoothed_frame_pts_diff < -small_ts_margin
    render_jitter_flag[1:] = np.logical_or(
        render_jitter_flag[1:], smoothed_frame_pts_diff[:-1] > small_ts_margin
    )

    # server related
    server_jitter_flag = np.zeros(tot_frame_no)
    frame_cgs_render_interval = data[1:, 33] - data[:-1, 33]
    frame_proxy_recv_interval = data[1:, 12] - data[:-1, 12]
    server_jitter_flag[2:] = (frame_proxy_recv_interval - frame_cgs_render_interval)[
        :-1
    ] > small_ts_margin

    # decoder related
    decoder_jitter_flag = np.zeros(tot_frame_no)
    decoder_jitter_flag[1:] = np.logical_or(
        data[:, 9] > avg_dec_time + small_ts_margin, data[:, 7:9].sum(-1) > 0
    )[:-1]

    # network induced queue
    network_stall_induced_queue = np.logical_and.reduce(
        (dec_queued_frame_cnt > 1, network_stall_flag)
    )
    network_packet_loss_induced_queue = np.logical_and.reduce(
        (
            dec_queued_frame_cnt > 1,
            network_packet_loss_flag,
            np.logical_not(network_stall_flag),
        )
    )
    network_big_frame_induced_queue = np.logical_and.reduce(
        (
            dec_queued_frame_cnt > 1,
            network_big_frame_flag,
            np.logical_not(network_packet_loss_flag),
            np.logical_not(network_stall_flag),
        )
    )
    network_i_frame_induced_queue = np.logical_and(
        network_big_frame_induced_queue,
        np.logical_or(data[:, 2] == 2, data[:, 49] == 1),
    )

    network_dl_jitter_induced_queue = np.logical_and.reduce(
        (
            dec_queued_frame_cnt > 1,
            network_dl_jitter_flag,
            np.logical_not(network_big_frame_flag),
            np.logical_not(network_packet_loss_flag),
            np.logical_not(network_stall_flag),
        )
    )

    # render induced queue
    render_jitter_induced_queue = np.logical_and.reduce(
        (
            dec_queued_frame_cnt > 1,
            render_jitter_flag,
            np.logical_not(network_problem_flag),
        )
    )

    # server inside jitter induced queue
    server_jitter_induced_queue = np.logical_and.reduce(
        (
            dec_queued_frame_cnt > 1,
            server_jitter_flag,
            np.logical_not(network_problem_flag),
            np.logical_not(render_jitter_flag),
        )
    )

    # decoder induced queue
    decoder_jitter_induced_queue = np.logical_and.reduce(
        (
            dec_queued_frame_cnt > 1,
            decoder_jitter_flag,
            np.logical_not(network_problem_flag),
            np.logical_not(render_jitter_flag),
            np.logical_not(server_jitter_flag),
        )
    )

    if print_log:
        log_file = open(
            file_path[:-4]
            + "_anchorPredictor%d_jitter_analyze.csv"
            % (anchor_frame_extrapolator_mode),
            "w",
        )

        log_file.write(
            ",".join(
                [
                    "sim_index",
                    "render_index",
                    "frame_index",
                    "frame_type",
                    "size",
                    "loss_type",
                    "client_receive_ts",
                    "receive_and_unpack",
                    "decoder_outside_queue",
                    "decoder_insided_queue",
                    "decode",
                    "render_queue",
                    "display",
                    "proxy_recv_ts",
                    "proxy_recv_time",
                    "proxy_send_delay",
                    "send_time",
                    "net_time",
                    "proc_time",
                    "tot_time",
                    "basic_net_ts",
                    "ul_jitter",
                    "dl_jitter",
                    "expected_recv_ts",
                    "expected_proc_time",
                    "nearest_display_ts",
                    "expected_display_ts",
                    "actual_display_ts",
                    "vsync_diff",
                    "present_timer_offset",
                    "jitter_buf_size",
                    "server_optim_enabled",
                    "client_optim_enabled",
                    "client_vsync_enabled",
                    "pts",
                    "ets",
                    "dts",
                    "sts",
                    "Mrts0ToRtsOffset",
                    "packet_lossed_perK",
                    "encoding_rate",
                    "cc_rate",
                    "smoothrate",
                    "width",
                    "height",
                    "sqoe",
                    "ori_sqoe",
                    "target_sqoe",
                    "recomm_bitrate",
                    "actual_bitrate",
                    "scene_change",
                    "encoding_fps",
                    "satd",
                    "qp",
                    "mvx",
                    "mvy",
                    "intra_mb",
                    "inter_mb",
                    "cur_cgs_pause_cnt",
                    "client_vsync_ts",
                    "min_rtt",
                    "dec_over_ts",
                    "nearest_vsync_ts",
                    "dec_queued_frame_cnt",
                    "frame_queue_flag",
                    "original_pts",
                    "smoothed_pts",
                    "smoothed_frame_pts_diff",
                    "render_jitter_induced_queue",
                    "server_jitter_induced_queue",
                    "network_stall_induced_queue",
                    "network_packet_loss_induced_queue",
                    "network_big_frame_induced_queue",
                    "network_i_frame_induced_queue",
                    "network_dl_jitter_induced_queue",
                    "decoder_jitter_induced_queue",
                    "updated_frame_interval",
                    "anchor_frame_ts",
                    "anchor_frame_no",
                ]
            )
            + "\n"
        )
        for idx in range(tot_frame_no):
            log_file.write(
                ",".join(
                    str(item)
                    for item in [idx]
                    + data[idx, :-5].tolist()
                    + [
                        dec_over_ts[idx],
                        nearest_vsync_ts[idx],
                        dec_queued_frame_cnt[idx],
                        int(dec_queued_frame_cnt[idx] > 1),
                        data[idx, 33],
                        smoothed_frame_pts[idx],
                        smoothed_frame_pts_diff[idx],
                        int(render_jitter_induced_queue[idx]),
                        int(server_jitter_induced_queue[idx]),
                        int(network_stall_induced_queue[idx]),
                        int(network_packet_loss_induced_queue[idx]),
                        int(network_big_frame_induced_queue[idx]),
                        int(network_i_frame_induced_queue[idx]),
                        int(network_dl_jitter_induced_queue[idx]),
                        int(decoder_jitter_induced_queue[idx]),
                        updated_frame_interval[idx],
                        anchor_frame_ts[idx],
                        anchor_frame_no[idx],
                    ]
                )
                + "\n"
            )

    # ts_diff = np.abs(smoothed_frame_pts - data[:, 33])
    # result = [server_optim_enabled,client_optim_enabled,client_vsync_enabled, np.mean(ts_diff), np.max(ts_diff), np.min(ts_diff)]

    result = [
        server_optim_enabled,
        client_optim_enabled,
        client_vsync_enabled,
        tot_frame_no,
        np.sum(dec_queued_frame_cnt > 1),
        network_stall_induced_queue.sum(),
        network_packet_loss_induced_queue.sum(),
        network_big_frame_induced_queue.sum(),
        network_i_frame_induced_queue.sum(),
        network_dl_jitter_induced_queue.sum(),
        render_jitter_induced_queue.sum(),
        server_jitter_induced_queue.sum(),
        decoder_jitter_induced_queue.sum(),
    ]
    output_file = open(
        os.path.join(
            "test_data",
            "result_anchorPredictor%d_jitter_analyze.csv"
            % (ANCHOR_FRAME_EXTRAPOLATOR_MODE),
        ),
        "a",
    )
    output_file.write(
        file_path.replace(",", "_")
        + ", "
        + ", ".join([str(item) for item in result])
        + "\n"
    )
    output_file.close()

    print(
        file_path.replace(",", "_") + ", " + ", ".join([str(item) for item in result])
    )
    return file_path, result


def process_all_data_multithread(root_path, num_proc=16):

    # output_file = open(os.path.join('test_data', 'result_pacf_start%d_len%d.csv' %(START_IDX, SIM_DATA_LEN)), 'a')
    # output_file.write('file_name,server_optim_enabled,client_optim_enabled,client_vsync_enabled,tot_frame_no,render_pacf_lag1,render_pacf_lag2,render_pacf_lag3,render_pacf_lag4,render_pacf_lag5,render_pacf_values1,render_pacf_values2,render_pacf_values3,render_pacf_values4,render_pacf_values5,dl_pacf_lag1,dl_pacf_lag2,dl_pacf_lag3,dl_pacf_lag4,dl_pacf_lag5,dl_pacf_values1,dl_pacf_values2,dl_pacf_values3,dl_pacf_values4,dl_pacf_values5,dec_pacf_lag1,dec_pacf_lag2,dec_pacf_lag3,dec_pacf_lag4,dec_pacf_lag5,dec_pacf_values1,dec_pacf_values2,dec_pacf_values3,dec_pacf_values4,dec_pacf_values5,display_pacf_lag1,display_pacf_lag2,display_pacf_lag3,display_pacf_lag4,display_pacf_lag5,display_pacf_values1,display_pacf_values2,display_pacf_values3,display_pacf_values4,display_pacf_values5\n')

    output_file = open(
        os.path.join(
            "test_data",
            "result_jitter_analysis_start%d_len%d.csv" % (START_IDX, SIM_DATA_LEN),
        ),
        "a",
    )
    csv_header = "file_name,server_optim_enabled,client_optim_enabled,client_vsync_enabled,tot_frame_no"
    dl_jitter_type_nos = ",".join(
        [
            "dl_jitter_over%d_no,dl_jitter_over%d_stall,dl_jitter_over%d_bigframe,dl_jitter_over%d_loss,dl_jitter_over%d_bitrate,dl_jitter_over%d_random"
            % (amp, amp, amp, amp, amp, amp)
            for amp in jitter_amp_thrs
        ]
    )

    render_jitter_frame_nos = ",".join(
        ["render_jitter_over%d_frame_no" % amp for amp in jitter_amp_thrs]
    )
    dl_jitter_frame_nos = ",".join(
        ["dl_jitter_over%d_frame_no" % amp for amp in jitter_amp_thrs]
    )
    dec_jitter_frame_nos = ",".join(
        ["dec_jitter_over%d_frame_no" % amp for amp in jitter_amp_thrs]
    )
    display_jitter_frame_nos = ",".join(
        ["display_jitter_over%d_frame_no" % amp for amp in jitter_amp_thrs]
    )

    render_jitter_amps = ",".join(
        ["render_jitter_amp_pct%d" % pct_tile_no for pct_tile_no in pct_tile_nos]
    )
    dl_jitter_amps = ",".join(
        ["dl_jitter_amp_pct%d" % pct_tile_no for pct_tile_no in pct_tile_nos]
    )
    dec_jitter_amps = ",".join(
        ["dec_jitter_amp_pct%d" % pct_tile_no for pct_tile_no in pct_tile_nos]
    )
    display_jitter_amps = ",".join(
        ["display_jitter_amp_pct%d" % pct_tile_no for pct_tile_no in pct_tile_nos]
    )

    render_jitter_consec_stats = ",".join(
        [
            "render_jitter_over%d_consec_no,render_jitter_over%d_consec_mean,render_jitter_over%d_consec_std"
            % (amp, amp, amp)
            for amp in jitter_amp_thrs
        ]
    )
    dl_jitter_consec_stats = ",".join(
        [
            "dl_jitter_over%d_consec_no,dl_jitter_over%d_consec_mean,dl_jitter_over%d_consec_std"
            % (amp, amp, amp)
            for amp in jitter_amp_thrs
        ]
    )
    dec_jitter_consec_stats = ",".join(
        [
            "dec_jitter_over%d_consec_no,dec_jitter_over%d_consec_mean,dec_jitter_over%d_consec_std"
            % (amp, amp, amp)
            for amp in jitter_amp_thrs
        ]
    )
    display_jitter_consec_stats = ",".join(
        [
            "display_jitter_over%d_consec_no,display_jitter_over%d_consec_mean,display_jitter_over%d_consec_std"
            % (amp, amp, amp)
            for amp in jitter_amp_thrs
        ]
    )

    render_jitter_interval_stats = ",".join(
        [
            "render_jitter_over%d_interval_no,render_jitter_over%d_interval_mean,render_jitter_over%d_interval_std"
            % (amp, amp, amp)
            for amp in jitter_amp_thrs
        ]
    )
    dl_jitter_interval_stats = ",".join(
        [
            "dl_jitter_over%d_interval_no,dl_jitter_over%d_interval_mean,dl_jitter_over%d_interval_std"
            % (amp, amp, amp)
            for amp in jitter_amp_thrs
        ]
    )
    dec_jitter_interval_stats = ",".join(
        [
            "dec_jitter_over%d_interval_no,dec_jitter_over%d_interval_mean,dec_jitter_over%d_interval_std"
            % (amp, amp, amp)
            for amp in jitter_amp_thrs
        ]
    )
    display_jitter_interval_stats = ",".join(
        [
            "display_jitter_over%d_interval_no,display_jitter_over%d_interval_mean,display_jitter_over%d_interval_std"
            % (amp, amp, amp)
            for amp in jitter_amp_thrs
        ]
    )

    render_jitter_interval_no = ""
    render_jitter_interval_same_max_no = ""
    for jitter_amp_thr in jitter_amp_thrs:
        for jitter_interval_thr in jitter_interval_thrs:
            if len(render_jitter_interval_no) != 0:
                render_jitter_interval_no += ","
            render_jitter_interval_no += (
                "render_jitter_amp_over%d_interval_over%d_no"
                % (jitter_amp_thr, jitter_interval_thr)
            )
            for pct_tile_no in pct_tile_nos:
                if len(render_jitter_interval_same_max_no) != 0:
                    render_jitter_interval_same_max_no += ","
                render_jitter_interval_same_max_no += (
                    "render_jitter_amp_over%d_interval_over%d_same_no_pct%d"
                    % (jitter_amp_thr, jitter_interval_thr, pct_tile_no)
                )

    dl_jitter_interval_no = ""
    dl_jitter_interval_same_max_no = ""
    for jitter_amp_thr in jitter_amp_thrs:
        for jitter_interval_thr in jitter_interval_thrs:
            if len(dl_jitter_interval_no) != 0:
                dl_jitter_interval_no += ","
            dl_jitter_interval_no += "dl_jitter_amp_over%d_interval_over%d_no" % (
                jitter_amp_thr,
                jitter_interval_thr,
            )
            for pct_tile_no in pct_tile_nos:
                if len(dl_jitter_interval_same_max_no) != 0:
                    dl_jitter_interval_same_max_no += ","
                dl_jitter_interval_same_max_no += (
                    "dl_jitter_amp_over%d_interval_over%d_same_no_pct%d"
                    % (jitter_amp_thr, jitter_interval_thr, pct_tile_no)
                )

    dec_jitter_interval_no = ""
    dec_jitter_interval_same_max_no = ""
    for jitter_amp_thr in jitter_amp_thrs:
        for jitter_interval_thr in jitter_interval_thrs:
            if len(dec_jitter_interval_no) != 0:
                dec_jitter_interval_no += ","
            dec_jitter_interval_no += "dec_jitter_amp_over%d_interval_over%d_no" % (
                jitter_amp_thr,
                jitter_interval_thr,
            )
            for pct_tile_no in pct_tile_nos:
                if len(dec_jitter_interval_same_max_no) != 0:
                    dec_jitter_interval_same_max_no += ","
                dec_jitter_interval_same_max_no += (
                    "dec_jitter_amp_over%d_interval_over%d_same_no_pct%d"
                    % (jitter_amp_thr, jitter_interval_thr, pct_tile_no)
                )

    display_jitter_interval_no = ""
    display_jitter_interval_same_max_no = ""
    for jitter_amp_thr in jitter_amp_thrs:
        for jitter_interval_thr in jitter_interval_thrs:
            if len(display_jitter_interval_no) != 0:
                display_jitter_interval_no += ","
            display_jitter_interval_no += (
                "display_jitter_amp_over%d_interval_over%d_no"
                % (jitter_amp_thr, jitter_interval_thr)
            )
            for pct_tile_no in pct_tile_nos:
                if len(display_jitter_interval_same_max_no) != 0:
                    display_jitter_interval_same_max_no += ","
                display_jitter_interval_same_max_no += (
                    "display_jitter_amp_over%d_interval_over%d_same_no_pct%d"
                    % (jitter_amp_thr, jitter_interval_thr, pct_tile_no)
                )

    render_jitter_consecs = ""
    render_jitter_intervals = ""
    dl_jitter_consecs = ""
    dl_jitter_intervals = ""
    dec_jitter_consecs = ""
    dec_jitter_intervals = ""
    display_jitter_consecs = ""
    display_jitter_intervals = ""
    for jitter_amp_thr in jitter_amp_thrs:
        render_jitter_consecs = (
            render_jitter_consecs
            + ","
            + ",".join(
                [
                    "render_jitter_over%d_consec_pct%d" % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        render_jitter_intervals = (
            render_jitter_intervals
            + ","
            + ",".join(
                [
                    "render_jitter_over%d_interval_pct%d"
                    % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        dl_jitter_consecs = (
            dl_jitter_consecs
            + ","
            + ",".join(
                [
                    "dl_jitter_over%d_consec_pct%d" % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        dl_jitter_intervals = (
            dl_jitter_intervals
            + ","
            + ",".join(
                [
                    "dl_jitter_over%d_interval_pct%d" % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        dec_jitter_consecs = (
            dec_jitter_consecs
            + ","
            + ",".join(
                [
                    "dec_jitter_over%d_consec_pct%d" % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        dec_jitter_intervals = (
            dec_jitter_intervals
            + ","
            + ",".join(
                [
                    "dec_jitter_over%d_interval_pct%d" % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        display_jitter_consecs = (
            display_jitter_consecs
            + ","
            + ",".join(
                [
                    "display_jitter_over%d_consec_pct%d" % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        display_jitter_intervals = (
            display_jitter_intervals
            + ","
            + ",".join(
                [
                    "display_jitter_over%d_interval_pct%d"
                    % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
    render_jitter_consecs = render_jitter_consecs[1:]
    render_jitter_intervals = render_jitter_intervals[1:]
    dl_jitter_consecs = dl_jitter_consecs[1:]
    dl_jitter_intervals = dl_jitter_intervals[1:]
    dec_jitter_consecs = dec_jitter_consecs[1:]
    dec_jitter_intervals = dec_jitter_intervals[1:]
    display_jitter_consecs = display_jitter_consecs[1:]
    display_jitter_intervals = display_jitter_intervals[1:]

    csv_header = (
        csv_header
        + ","
        + dl_jitter_type_nos
        + ","
        + render_jitter_frame_nos
        + ","
        + render_jitter_amps
        + ","
        + render_jitter_consecs
        + ","
        + render_jitter_intervals
        + ","
        + render_jitter_consec_stats
        + ","
        + render_jitter_interval_stats
        + ","
        + render_jitter_interval_no
        + ","
        + render_jitter_interval_same_max_no
        + ","
        + dl_jitter_frame_nos
        + ","
        + dl_jitter_amps
        + ","
        + dl_jitter_consecs
        + ","
        + dl_jitter_intervals
        + ","
        + dl_jitter_consec_stats
        + ","
        + dl_jitter_interval_stats
        + ","
        + dl_jitter_interval_no
        + ","
        + dl_jitter_interval_same_max_no
        + ","
        + dec_jitter_frame_nos
        + ","
        + dec_jitter_amps
        + ","
        + dec_jitter_consecs
        + ","
        + dec_jitter_intervals
        + ","
        + dec_jitter_consec_stats
        + ","
        + dec_jitter_interval_stats
        + ","
        + dec_jitter_interval_no
        + ","
        + dec_jitter_interval_same_max_no
        + ","
        + display_jitter_frame_nos
        + ","
        + display_jitter_amps
        + ","
        + display_jitter_consecs
        + ","
        + display_jitter_intervals
        + ","
        + display_jitter_consec_stats
        + ","
        + display_jitter_interval_stats
        + ","
        + display_jitter_interval_no
        + ","
        + display_jitter_interval_same_max_no
        + ","
        + "window_tot_no,"
        + "render_no_jitter_window_num,render_jitter_window_num,render_jitter_fast_change_window_num,render_jitter_recover_window_num,render_jitter_fake_recover_num,"
        + "dl_no_jitter_window_num,dl_jitter_window_num,dl_jitter_fast_change_window_num,dl_jitter_recover_window_num,dl_jitter_fake_recover_num,"
        + "dec_no_jitter_window_num,dec_jitter_window_num,dec_jitter_fast_change_window_num,dec_jitter_recover_window_num,dec_jitter_fake_recover_num,"
        + "display_no_jitter_window_num,display_jitter_window_num,display_jitter_fast_change_window_num,display_jitter_recover_window_num,display_jitter_fake_recover_num\n"
    )

    output_file.write(csv_header)
    output_file.close()

    p = multiprocessing.Pool(processes=num_proc)
    result = Result()
    for data_folder in os.listdir(root_path):
        if not data_folder.startswith("2024-"):
            continue

        data_path = os.path.join(root_path, data_folder)

        for session_folder in os.listdir(data_path):
            if not session_folder.startswith("session_info"):
                continue

            session_path = os.path.join(data_path, session_folder)

            for log_name in os.listdir(session_path):
                if not log_name.endswith(".csv"):
                    continue
                if log_name.endswith("sim.csv"):
                    continue

                log_path = os.path.join(session_path, log_name)
                p.apply_async(
                    analyze_frame_jitter_simplified,
                    args=(
                        log_path,
                        START_IDX,
                        SIM_DATA_LEN,
                    ),
                    callback=result.update_result,
                )

    p.close()
    p.join()
    print(np.mean(result.res1), np.min(result.res1), np.max(result.res1))
    print(np.mean(result.res2), np.min(result.res2), np.max(result.res2))
    print(np.mean(result.res3), np.min(result.res3), np.max(result.res3))


def process_all_data(root_path):
    # output_file = open(os.path.join('test_data', 'result_pacf_start%d_len%d.csv' %(START_IDX, SIM_DATA_LEN)), 'a')
    # output_file.write('file_name,server_optim_enabled,client_optim_enabled,client_vsync_enabled,tot_frame_no,render_pacf_lag1,render_pacf_lag2,render_pacf_lag3,render_pacf_lag4,render_pacf_lag5,render_pacf_values1,render_pacf_values2,render_pacf_values3,render_pacf_values4,render_pacf_values5,dl_pacf_lag1,dl_pacf_lag2,dl_pacf_lag3,dl_pacf_lag4,dl_pacf_lag5,dl_pacf_values1,dl_pacf_values2,dl_pacf_values3,dl_pacf_values4,dl_pacf_values5,dec_pacf_lag1,dec_pacf_lag2,dec_pacf_lag3,dec_pacf_lag4,dec_pacf_lag5,dec_pacf_values1,dec_pacf_values2,dec_pacf_values3,dec_pacf_values4,dec_pacf_values5,display_pacf_lag1,display_pacf_lag2,display_pacf_lag3,display_pacf_lag4,display_pacf_lag5,display_pacf_values1,display_pacf_values2,display_pacf_values3,display_pacf_values4,display_pacf_values5\n')

    output_file = open(
        os.path.join(
            "test_data",
            "result_jitter_analysis_start%d_len%d.csv" % (START_IDX, SIM_DATA_LEN),
        ),
        "a",
    )
    csv_header = "file_name,server_optim_enabled,client_optim_enabled,client_vsync_enabled,tot_frame_no"

    render_jitter_amps = ",".join(
        ["render_jitter_amp_pct%d" % pct_tile_no for pct_tile_no in pct_tile_nos]
    )
    dl_jitter_amps = ",".join(
        ["dl_jitter_amp_pct%d" % pct_tile_no for pct_tile_no in pct_tile_nos]
    )
    dec_jitter_amps = ",".join(
        ["dec_jitter_amp_pct%d" % pct_tile_no for pct_tile_no in pct_tile_nos]
    )
    display_jitter_amps = ",".join(
        ["display_jitter_amp_pct%d" % pct_tile_no for pct_tile_no in pct_tile_nos]
    )

    render_jitter_consecs = ""
    render_jitter_intervals = ""
    dl_jitter_consecs = ""
    dl_jitter_intervals = ""
    dec_jitter_consecs = ""
    dec_jitter_intervals = ""
    display_jitter_consecs = ""
    display_jitter_intervals = ""
    for jitter_amp_thr in jitter_amp_thrs:
        render_jitter_consecs = (
            render_jitter_consecs
            + ","
            + ",".join(
                [
                    "render_jitter_over%d_consec_pct%d" % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        render_jitter_intervals = (
            render_jitter_intervals
            + ","
            + ",".join(
                [
                    "render_jitter_over%d_interval_pct%d"
                    % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        dl_jitter_consecs = (
            dl_jitter_consecs
            + ","
            + ",".join(
                [
                    "dl_jitter_over%d_consec_pct%d" % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        dl_jitter_intervals = (
            dl_jitter_intervals
            + ","
            + ",".join(
                [
                    "dl_jitter_over%d_interval_pct%d" % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        dec_jitter_consecs = (
            dec_jitter_consecs
            + ","
            + ",".join(
                [
                    "dec_jitter_over%d_consec_pct%d" % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        dec_jitter_intervals = (
            dec_jitter_intervals
            + ","
            + ",".join(
                [
                    "dec_jitter_over%d_interval_pct%d" % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        display_jitter_consecs = (
            display_jitter_consecs
            + ","
            + ",".join(
                [
                    "display_jitter_over%d_consec_pct%d" % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
        display_jitter_intervals = (
            display_jitter_intervals
            + ","
            + ",".join(
                [
                    "display_jitter_over%d_interval_pct%d"
                    % (jitter_amp_thr, pct_tile_no)
                    for pct_tile_no in pct_tile_nos
                ]
            )
        )
    render_jitter_consecs = render_jitter_consecs[1:]
    render_jitter_intervals = render_jitter_intervals[1:]
    dl_jitter_consecs = dl_jitter_consecs[1:]
    dl_jitter_intervals = dl_jitter_intervals[1:]
    dec_jitter_consecs = dec_jitter_consecs[1:]
    dec_jitter_intervals = dec_jitter_intervals[1:]
    display_jitter_consecs = display_jitter_consecs[1:]
    display_jitter_intervals = display_jitter_intervals[1:]

    csv_header = (
        csv_header
        + ","
        + render_jitter_amps
        + ","
        + render_jitter_consecs
        + ","
        + render_jitter_intervals
        + ","
        + dl_jitter_amps
        + ","
        + dl_jitter_consecs
        + ","
        + dl_jitter_intervals
        + ","
        + dec_jitter_amps
        + ","
        + dec_jitter_consecs
        + ","
        + dec_jitter_intervals
        + ","
        + display_jitter_amps
        + ","
        + display_jitter_consecs
        + ","
        + display_jitter_intervals
        + ","
        + "window_tot_no,"
        + "render_no_jitter_window_num,render_jitter_window_num,render_jitter_fast_change_window_num,render_jitter_recover_window_num,render_jitter_fake_recover_num,"
        + "dl_no_jitter_window_num,dl_jitter_window_num,dl_jitter_fast_change_window_num,dl_jitter_recover_window_num,dl_jitter_fake_recover_num,"
        + "dec_no_jitter_window_num,dec_jitter_window_num,dec_jitter_fast_change_window_num,dec_jitter_recover_window_num,dec_jitter_fake_recover_num,"
        + "display_no_jitter_window_num,display_jitter_window_num,display_jitter_fast_change_window_num,display_jitter_recover_window_num,display_jitter_fake_recover_num\n"
    )

    output_file.write(csv_header)
    output_file.close()

    for data_folder in os.listdir(root_path):
        if not data_folder.startswith("2024-"):
            continue

        data_path = os.path.join(root_path, data_folder)

        for session_folder in os.listdir(data_path):
            if not session_folder.startswith("session_info"):
                continue

            session_path = os.path.join(data_path, session_folder)

            for log_name in os.listdir(session_path):
                if not log_name.endswith(".csv"):
                    continue
                if log_name.endswith("sim.csv"):
                    continue

                log_path = os.path.join(session_path, log_name)
                file_path, cur_res = analyze_frame_jitter_simplified(
                    log_path, START_IDX, SIM_DATA_LEN
                )


def plot_multi_lines(
    datas, filename, xlabels, ylabels, output_folder="figures", title=None, flags=None
):
    subplot_no = len(datas)

    fig, axs = plt.subplots(
        subplot_no, 1, layout="constrained", figsize=(14, 3 * subplot_no)
    )

    for i in range(subplot_no):
        axs[i].scatter(range(datas[i].shape[0]), datas[i])
        axs[i].plot(datas[i])
        if flags is not None and flags[i] is not None:
            axs[i].scatter(np.where(flags[i]), datas[i][flags[i]], color="r")
        axs[i].set_xlabel(xlabels[i])
        axs[i].set_ylabel(ylabels[i])
        axs[i].grid(True)

    if title is not None:
        fig.suptitle(title)

    output_dir = os.path.join("test_data", output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, filename + ".jpg"), dpi=400)

    plt.close(fig)


def plot_jitter_analyze_result(log_path, start_idx=0, end_idx=-1):
    df = pd.read_csv(log_path)
    df = df.loc[df["display_discarded_flag"] == 0]

    render_jitter = df["smoothed_frame_pts_diff"]

    frame_type = df["frame_type"]
    frame_size_ratio = df["size"] / (df["cc_rate"] * 1024 / 8 / 60)

    dl_jitter = df["dl_jitter"]
    packet_lossed_perK = df["packet_lossed_perK"]

    decode = df["decode"]

    render_jitter_induced_queue = df["render_jitter_induced_queue"]
    network_big_frame_induced_queue = df["network_big_frame_induced_queue"]
    network_i_frame_induced_queue = df["network_i_frame_induced_queue"]
    network_dl_jitter_induced_queue = df["network_dl_jitter_induced_queue"]
    network_stall_induced_queue = df["network_stall_induced_queue"]
    network_packet_loss_induced_queue = df["network_packet_loss_induced_queue"]
    decoder_jitter_induced_queue = df["decoder_jitter_induced_queue"]

    frame_queue_flag = df["frame_queue_flag"]
    dec_queued_frame_cnt = df["dec_queued_frame_cnt"]
    cur_buf_size = df["cur_buf_size"]

    plot_multi_lines(
        [
            render_jitter.to_numpy()[start_idx:end_idx],
            # frame_type.to_numpy()[start_idx:end_idx],
            frame_size_ratio.to_numpy()[start_idx:end_idx],
            dl_jitter.to_numpy()[start_idx:end_idx],
            packet_lossed_perK.to_numpy()[start_idx:end_idx],
            decode.to_numpy()[start_idx:end_idx],
            cur_buf_size.to_numpy()[start_idx:end_idx],
        ],
        filename=os.path.basename(log_path)[:-4] + "_%d_%d" % (start_idx, end_idx),
        xlabels=[
            "render_jitter",
            "frame_size_ratio",
            "dl_jitter",
            "packet_lossed_perK",
            "decode",
            "cur_buf_size",
        ],
        ylabels=["ms", "ratio", "ms", "[pct]", "ms", "cnt"],
        output_folder="figures/jitter_analyze",
        title=os.path.basename(log_path)[:-36],
        flags=[
            render_jitter_induced_queue.to_numpy()[start_idx:end_idx] == 1,
            network_big_frame_induced_queue.to_numpy()[start_idx:end_idx] == 1,
            network_dl_jitter_induced_queue.to_numpy()[start_idx:end_idx] == 1,
            network_packet_loss_induced_queue.to_numpy()[start_idx:end_idx] == 1,
            decoder_jitter_induced_queue.to_numpy()[start_idx:end_idx] == 1,
            frame_queue_flag.to_numpy()[start_idx:end_idx] == 1,
        ],
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        for line in open(r"test_data/unit_test.txt").readlines():
            if len(line) <= 1:
                break
            input_path = os.path.join("test_data", line.strip())

    else:
        input_path = sys.argv[1]
        if os.path.isdir(input_path):
            process_all_data_multithread(sys.argv[1])
            # process_all_data(sys.argv[1])
        elif os.path.isfile(input_path):
            if input_path.endswith("sim.csv"):
                predict_frame_jitter(input_path)
            else:
                analyze_frame_jitter_simplified(input_path)
            # if input_path.endswith('jitter_analyze.csv') or input_path.endswith('sim.csv'):
            #     if len(sys.argv) == 4:
            #         plot_jitter_analyze_result(input_path, start_idx=int(sys.argv[2]), end_idx=int(sys.argv[3]))
            #     else:
            #         plot_jitter_analyze_result(input_path, start_idx=0, end_idx=-1)
            # else:
            #     analyze_frame_jitter(input_path, anchor_frame_extrapolator_mode=ANCHOR_FRAME_EXTRAPOLATOR_MODE, print_log=PRINT_LOG)
        else:
            pass
