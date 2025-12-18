import os, sys, shutil
import multiprocessing
import collections
import queue

import load_data
import numpy as np

from trace_e2e_jitter_analyze import (
    AnchorFrameExtrapolator,
    E2EJitterPredictor,
    E2EJitterPredictorV2,
)

# DISPLAY_MODE = 'naiveVsync'
DISPLAY_MODE = "simpleCtrl"
# DISPLAY_MODE = 'optimal'
MAX_BUF_SIZE = 2
FRAME_INTERVAL = 16.666667
ENABLE_PERIO_DROP = 2  # 1 for strict restriction, 2 for losse restriction
ENABLE_QUICK_DROP = 0  # 1 for simple probability-based method

if ENABLE_QUICK_DROP <= 3:
    JITTER_HISTORY_LTH = 600
elif ENABLE_QUICK_DROP == 4:
    JITTER_HISTORY_LTH = 300
elif ENABLE_QUICK_DROP == 5:
    JITTER_HISTORY_LTH = 120
elif ENABLE_QUICK_DROP == 6:
    JITTER_HISTORY_LTH = 60
elif ENABLE_QUICK_DROP == 7:
    JITTER_HISTORY_LTH = 120
elif ENABLE_QUICK_DROP == 8:
    JITTER_HISTORY_LTH = 600
elif ENABLE_QUICK_DROP == 9:
    JITTER_HISTORY_LTH = 120

BONUS_FPS_NO_THR = 30
# BONUS_FPS_NO_THR = 10
ANCHOR_FRAME_EXTRAPOLATOR_MODE = 4

# FPS_STATS_MODE = 'objective_fps'
# FPS_STATS_MODE = 'original_fps'
# FPS_STATS_MODE = 'noloss_fps'
# DROP_FRAME_MODE = 'fifo'
DROP_FRAME_MODE = "lifo"

# if DROP_FRAME_MODE == 'fifo':
#     RENDER_TIME_PREIDCTER='ewma'
# elif DROP_FRAME_MODE == 'lifo':
#     RENDER_TIME_PREIDCTER='fixed'

RENDER_TIME_PREIDCTER = "ewma"
# RENDER_TIME_PREIDCTER='fixed'
# RENDER_TIME_PREIDCTER='oracle'

# PRINT_LOG = True
PRINT_LOG = False
# PRINT_DEBUG_LOG = True
PRINT_DEBUG_LOG = False

BUFFER_OVERFLOW_THR = 0
OVERDUE_TS_THR = 0

MULTI_PARAMS = [
    # ============================================================
    # 1. 最优算法 (Optimal) - Oracle上界
    # ============================================================
    ["optimal", 2, "oracle", 2, 0, 30, "lifo", 4],
    
    # ============================================================
    # 2. 垂直同步算法 (naiveVsync) - Baseline
    # ============================================================
    ["naiveVsync", 2, "ewma", 1, 0, 30, "lifo", 4],
    
    # ============================================================
    # 3. simpleCtrl模式 - Grid Search
    # 固定参数: buffer=2, bonus_fps=30, drop_mode=lifo
    # 变化参数: predictor × perio_drop × quick_drop × anchor_mode = 3 × 3 × 10 × 5 = 450
    # ============================================================
]

# Grid Search: predictor × perio_drop × quick_drop × anchor_mode
_PREDICTORS = ["oracle", "ewma", "fixed"]
_PERIO_DROPS = [0, 1, 2]  # 0=disabled, 1=strict, 2=loose
_QUICK_DROPS = list(range(10))  # 0-9
_ANCHOR_MODES = list(range(5))  # 0-4 (0-4 different PTS prediction modes)

for predictor in _PREDICTORS:
    for perio_drop in _PERIO_DROPS:
        for quick_drop in _QUICK_DROPS:
            for anchor_mode in _ANCHOR_MODES:
                MULTI_PARAMS.append([
                    "simpleCtrl", 
                    2,              # buffer=2 (fixed)
                    predictor,      # oracle/ewma/fixed
                    perio_drop,     # 0/1/2
                    quick_drop,     # 0-9
                    30,             # bonus_fps=30 (fixed)
                    "lifo",         # drop_mode=lifo (fixed)
                    anchor_mode     # 0-4 (PTS prediction mode)
                ])

class Result:
    def __init__(self):
        self.res1 = []
        self.res2 = []
        self.res3 = []
        self.res4 = []

    def update_result(self, result):
        log_path = result[0]
        cur_res = result[1]
        if cur_res is not None:
            self.res1.append(cur_res[2])
            self.res2.append(cur_res[3])
            self.res3.append(cur_res[4])
            self.res4.append(cur_res[5])


def cal_next_vsync_ts(cur_ready_ts, anchor_vsync_ts, frame_interval):
    if cur_ready_ts < anchor_vsync_ts:
        return anchor_vsync_ts
    else:
        next_slot_no = (cur_ready_ts - anchor_vsync_ts) // frame_interval + 1
        return anchor_vsync_ts + np.ceil(next_slot_no * frame_interval)


def cal_frame_slot(frame_ts, anchor_ts, frame_interval):
    return (frame_ts - anchor_ts) // frame_interval + 1


def rectify_anchor_vsync_ts(anchor_vsync_ts, cur_vsync_ts, frame_interval=16.666667):
    slot_no = (cur_vsync_ts - anchor_vsync_ts) // frame_interval
    ts_diff = cur_vsync_ts - (np.ceil(frame_interval * slot_no) + anchor_vsync_ts)

    if ts_diff <= 3:
        new_anchor_ts = anchor_vsync_ts + ts_diff
    else:
        new_anchor_ts = anchor_vsync_ts - (np.ceil(frame_interval) - ts_diff)

    # new_anchor_ts = anchor_vsync_ts - (np.ceil(frame_interval) - ts_diff)

    # if ts_diff <= 5:
    #     new_anchor_ts = anchor_vsync_ts + ts_diff
    # elif ts_diff >= frame_interval - 5:
    #     new_anchor_ts = anchor_vsync_ts - (np.ceil(frame_interval) - ts_diff)
    # else:
    #     new_anchor_ts = anchor_vsync_ts

    return new_anchor_ts


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
def cal_single_para_result(
    file_path,
    display_mode="naive_vsync",
    max_buf_size=2,
    frame_interval=16.666667,
    render_time_predictor="oracle",
    enable_perio_drop=0,
    enable_quick_drop=0,
    drop_frame_mode="lifo",
    bonus_fps_no_thr=30,
    anchor_frame_extrapolator_mode=4,
    save_path="test_data",
    print_log=True,
    print_debug_log=False,
    sim_data_len=60 * 60 * 20,
    start_idle_len=3600,
):
    """
    Simulate the display process with a frame trace and a set of parameters.
    param display_mode: naive_vsync, simple_ctrl
    """
    # print(file_path[:-4] + '_%s_quickdrop%d_periodrop%d_maxbuf%d_renderTime_%s_%s_sim.csv' %(display_mode, enable_quick_drop, enable_perio_drop, max_buf_size, render_time_predictor), 'start simulation')

    data, info = load_data.load_detailed_framerate_log(
        file_path, start_idx=0, len_limit=sim_data_len + start_idle_len
    )  # sim for 20min
    data[:, 11] = np.maximum(data[:, 11], 0)

    if data is None:
        print("None data")
        return None, None, None

    if data.shape[0] < start_idle_len * 3:
        print("trace too short, len:", data.shape[0])
        target_path = os.path.join(os.path.dirname(file_path), "small")
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)
        file_name = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(target_path, file_name))
        print("move file: %s to %s" % (file_path, os.path.join(target_path, file_name)))
        return None, None, None

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
        return None, None, None

    # initialization: calculate the average decode and render time
    def cal_avg_client_ime(samp_len):
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
        cal_avg_client_ime(start_idle_len)
    )
    if avg_render_time > 10:
        print("render time too large: %d" % avg_render_time)
        target_path = os.path.join(os.path.dirname(file_path), "render_problem")
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)
        file_name = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(target_path, file_name))
        print("move file: %s to %s" % (file_path, os.path.join(target_path, file_name)))
        return None, None, None

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

    dec_over_ts = data[:, 5:10].sum(-1)

    anchor_frame_extrapolator = AnchorFrameExtrapolator(
        data[0, 33], data[0, 1], frame_interval, anchor_frame_extrapolator_mode
    )

    cur_buf_size = max_buf_size
    tot_frame_no = data.shape[0]
    nearest_vsync_ts = np.zeros(tot_frame_no, dtype=np.int64)
    nearest_no_jitter_vsync_ts = np.zeros(tot_frame_no, dtype=np.int64)
    available_vsync_ts = np.zeros(tot_frame_no, dtype=np.int64)
    nearest_display_slot = np.zeros(tot_frame_no, dtype=np.int64)
    invoke_present_ts = np.zeros(tot_frame_no, dtype=np.int64)
    invoke_present_slot = np.zeros(tot_frame_no, dtype=np.int64)
    actual_render_queue = np.zeros(tot_frame_no, dtype=np.int64)
    frame_buffer_cnt = np.zeros(tot_frame_no, dtype=np.int64)
    frame_backup_cnt = np.zeros(tot_frame_no, dtype=np.int64)
    actual_display_ts = np.zeros(tot_frame_no, dtype=np.int64)
    actual_vsync_ts = np.zeros(tot_frame_no, dtype=np.int64)
    extra_display_ts = np.zeros(tot_frame_no, dtype=np.int64)
    actual_display_slot = np.zeros(tot_frame_no, dtype=np.int64)
    display_discarded_flag = np.zeros(
        tot_frame_no, dtype=np.int64
    )  # 0 for display, 1 for discard
    valid_frame_flag = np.zeros(tot_frame_no, dtype=np.int64)
    frame_buf_size = np.zeros(tot_frame_no, dtype=np.int64)
    frame_buf_change_flag = np.zeros(tot_frame_no, dtype=np.int64)
    consecutive_frame_no = np.zeros(tot_frame_no, dtype=np.int64)
    prev_valid_frame_index = np.zeros(tot_frame_no, dtype=np.int64)
    consecutive_window_start_index = np.ones(tot_frame_no, dtype=np.int64) * -1
    jitter_window_start_index = np.ones(tot_frame_no, dtype=np.int64) * -1

    original_display_slot = np.zeros(tot_frame_no, dtype=np.int64)
    original_valid_flag = np.zeros(tot_frame_no, dtype=np.int64)
    predicted_render_time = np.zeros(tot_frame_no)
    predicted_decode_time = np.zeros(tot_frame_no)
    cur_anchor_vsync_ts = np.zeros(tot_frame_no)

    dec_nearest_vsync_ts = np.zeros(tot_frame_no, dtype=np.int64)
    dec_queued_frame_cnt = np.zeros(tot_frame_no, dtype=np.int64)
    smoothed_frame_pts = np.zeros(tot_frame_no, dtype=np.int64)
    smoothed_frame_pts_diff = np.zeros(tot_frame_no, dtype=np.int64)
    updated_frame_interval = np.zeros(tot_frame_no)
    anchor_frame_ts = np.zeros(tot_frame_no)
    anchor_frame_no = np.zeros(tot_frame_no, dtype=np.int64)
    frame_jitter_flag = np.zeros(tot_frame_no)
    bonus_fps_obtained = np.zeros(tot_frame_no)
    cur_bonus_fps_no = np.zeros(tot_frame_no)
    cur_valid_frame_no = np.zeros(tot_frame_no)

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

    # frame_jitter_thr = np.zeros(tot_frame_no, dtype=np.float32)
    dl_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    render_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    server_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    decoder_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    display_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    e2e_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    network_problem_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)
    near_vsync_jitter_prob_thr = np.zeros(tot_frame_no, dtype=np.float32)

    jitter_prob_updated = np.zeros(tot_frame_no)
    failed_early_drop_frame = np.zeros(tot_frame_no)
    missed_early_drop_frame = np.zeros(tot_frame_no)
    quick_drop_frame_cnt_hist = np.zeros(tot_frame_no)
    # frame_jitter_flag[1:] = np.logical_and(frame_buffer_cnt[:-1] == 1, frame_buffer_cnt[1:] > 1)

    sim_st_idx = 0
    while True:
        if data[sim_st_idx, 4] == 0 and data[sim_st_idx, 5] != 0:
            break
        display_discarded_flag[sim_st_idx] = 1

        sim_st_idx += 1
    # calculate the results for the first frame
    anchor_vsync_ts = (
        data[sim_st_idx, 58]
        if data[sim_st_idx, 58] != 0
        else dec_over_ts[sim_st_idx] + 4
    )
    cur_anchor_vsync_ts[sim_st_idx] = anchor_vsync_ts
    prev_render_time = avg_render_time

    nearest_vsync_ts[sim_st_idx] = cal_next_vsync_ts(
        dec_over_ts[sim_st_idx], data[sim_st_idx, 58], frame_interval
    )
    nearest_no_jitter_vsync_ts[sim_st_idx] = nearest_vsync_ts[sim_st_idx]
    dec_nearest_vsync_ts[sim_st_idx] = cal_next_vsync_ts(
        dec_over_ts[sim_st_idx], data[sim_st_idx, 58], frame_interval
    )
    nearest_display_slot[sim_st_idx] = 0
    available_vsync_ts[sim_st_idx] = data[sim_st_idx, 58]
    actual_display_ts[sim_st_idx] = (
        cal_next_vsync_ts(
            dec_over_ts[sim_st_idx] + data[sim_st_idx, 11],
            data[sim_st_idx, 58],
            frame_interval,
        )
        - 1
    )
    actual_vsync_ts[sim_st_idx] = actual_display_ts[sim_st_idx]
    actual_display_slot[sim_st_idx] = cal_frame_slot(
        actual_display_ts[sim_st_idx], anchor_vsync_ts, frame_interval
    )
    invoke_present_ts[sim_st_idx] = actual_vsync_ts[sim_st_idx] - data[sim_st_idx, 11]
    invoke_present_slot[sim_st_idx] = cal_frame_slot(
        invoke_present_ts[sim_st_idx], anchor_vsync_ts, frame_interval
    )
    actual_render_queue[sim_st_idx] = (
        actual_display_ts[sim_st_idx] - invoke_present_ts[sim_st_idx]
    )
    consecutive_frame_no[sim_st_idx] = 1
    consecutive_window_start_index[sim_st_idx] = sim_st_idx
    jitter_window_start_index[sim_st_idx] = sim_st_idx

    display_discarded_flag[sim_st_idx] = 0
    valid_frame_flag[sim_st_idx] = 1
    frame_buf_size[sim_st_idx] = cur_buf_size

    original_display_slot[sim_st_idx] = cal_frame_slot(
        dec_over_ts[sim_st_idx] + data[sim_st_idx, 10:12].sum(),
        anchor_vsync_ts,
        frame_interval,
    )
    original_valid_flag[sim_st_idx] = 1
    cur_valid_frame_no[sim_st_idx] = 1

    prev_valid_idx = sim_st_idx
    prev_vsync_ts = data[sim_st_idx, 58]

    # initlize some control parameters
    window_start_idx = sim_st_idx
    window_start_ts = data[sim_st_idx, 5]

    consecutive_window_start_idx = sim_st_idx
    consecutive_window_start_ts = data[sim_st_idx, 5]
    # window_update_interval_frame_cnt = 30
    window_update_interval_frame_cnt = bonus_fps_no_thr
    window_update_interval_ms = np.floor(
        window_update_interval_frame_cnt * frame_interval
    )

    quick_drop_interval_ms = 100
    quick_drop_frame_cnt = 6
    quick_drop_st_idx = 0
    quick_drop_st_ts = data[sim_st_idx, 5]
    need_check_quick_drop = False
    prev_quick_drop_flag = False

    quick_drop_frame_cnt_hist[idx] = quick_drop_frame_cnt

    hist_cnt = 60 * 5
    hist_jitter_queue = queue.Queue(maxsize=hist_cnt)

    jitter_thr = 0

    early_drop_frame_flag = False

    if print_debug_log:
        log_file = open(
            file_path[:-4]
            + "_%s_quickdrop%d_periodrop%d_maxbuf%d_renderTime_%s_debug.csv"
            % (
                display_mode,
                enable_quick_drop,
                enable_perio_drop,
                max_buf_size,
                render_time_predictor,
            ),
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
                    "cur_anchor_vsync_ts",
                    "decode_over_ts",
                    "predicted_render_time",
                    "nearest_vsync_ts",
                    "nearest_display_slot",
                    "available_vsync_ts",
                    "actual_render_queue",
                    "frame_buffer_cnt",
                    "invoke_present_ts",
                    "invoke_present_slot",
                    "actual_display_ts",
                    "actual_vsync_ts",
                    "actual_display_slot",
                    "cur_buf_size",
                    "buf_change_flag",
                    "consecutive_frame_no",
                    "display_discarded_flag",
                    "valid_frame_flag",
                    "original_valid_flag",
                    "dec_over_ts",
                    "dec_nearest_vsync_ts",
                    "dec_queued_frame_cnt",
                    "frame_jitter_flag",
                    "prev_valid_frame_index",
                    "consecutive_window_start_idx",
                    "jitter_window_start_index",
                ]
            )
            + "\n"
        )
        idx = sim_st_idx
        log_file.write(
            ",".join(
                str(item)
                for item in [idx]
                + data[idx, :-5].tolist()
                + [
                    cur_anchor_vsync_ts[idx],
                    dec_over_ts[idx],
                    predicted_render_time[idx],
                    nearest_vsync_ts[idx],
                    nearest_display_slot[idx],
                    available_vsync_ts[idx],
                    actual_render_queue[idx],
                    frame_buffer_cnt[idx],
                    invoke_present_ts[idx],
                    invoke_present_slot[idx],
                    actual_display_ts[idx],
                    actual_vsync_ts[idx],
                    actual_display_slot[idx],
                    frame_buf_size[idx],
                    frame_buf_change_flag[idx],
                    consecutive_frame_no[idx],
                    display_discarded_flag[idx],
                    valid_frame_flag[idx],
                    original_valid_flag[idx],
                    dec_over_ts[idx],
                    dec_nearest_vsync_ts[idx],
                    dec_queued_frame_cnt[idx],
                    int(frame_jitter_flag[idx]),
                    prev_valid_frame_index[idx],
                    consecutive_window_start_index[idx],
                    jitter_window_start_index[idx],
                ]
            )
            + "\n"
        )

    min_rtt = np.min(data[:, 59])
    frame_stall_thr = 50
    small_ts_margin = 2

    early_drop_prob_threshold = 0
    if display_mode != "naiveVsync" and enable_quick_drop > 0:
        if enable_quick_drop == 1:
            e2e_jitter_predictor = E2EJitterPredictor(
                min_rtt,
                avg_dec_time,
                avg_render_time,
                frame_interval,
                frame_stall_thr,
                small_ts_margin,
                window_lth=JITTER_HISTORY_LTH,
            )
        elif enable_quick_drop == 2:
            e2e_jitter_predictor = E2EJitterPredictorV2(
                min_rtt,
                frame_interval,
                frame_stall_thr,
                small_ts_margin,
                window_lth=JITTER_HISTORY_LTH,
            )

        early_drop_prob_threshold = e2e_jitter_predictor.get_probability_thr(
            enable_quick_drop, bonus_fps_no_thr
        )

    # start the simulation
    # for idx in range(1, tot_frame_no):
    idx = sim_st_idx
    sim_cnt = 0
    objective_bonus_fps_no = 0
    while True:
        sim_cnt += 1
        if sim_cnt > 9999999:
            print(
                file_path[:-4]
                + "_%s_quickdrop%d_periodrop%d_maxbuf%d_renderTime_%s_sim.csv"
                % (
                    display_mode,
                    enable_quick_drop,
                    enable_perio_drop,
                    max_buf_size,
                    render_time_predictor,
                ),
                "simulation failed",
            )
            return None, None, None

        if not early_drop_frame_flag:
            idx += 1
        if idx >= tot_frame_no:
            break
        tmp_idx = idx

        if data[idx, 4] != 0 or data[idx, 5] == 0:
            display_discarded_flag[idx] = 1
            continue

        if cur_anchor_vsync_ts[idx] == 0:
            if data[idx, 58] != 0 and prev_vsync_ts != data[idx, 58]:
                anchor_vsync_ts = rectify_anchor_vsync_ts(
                    anchor_vsync_ts, data[idx, 58], frame_interval=frame_interval
                )
        else:
            anchor_vsync_ts = cur_anchor_vsync_ts[idx]
        prev_vsync_ts = data[idx, 58]
        cur_anchor_vsync_ts[idx] = anchor_vsync_ts

        original_display_slot[idx] = cal_frame_slot(
            dec_over_ts[idx] + data[idx, 10:12].sum(), anchor_vsync_ts, frame_interval
        )
        if original_display_slot[idx] != original_display_slot[prev_valid_idx]:
            original_valid_flag[idx] = 1

        if smoothed_frame_pts[idx] == 0:
            smoothed_frame_pts[idx] = anchor_frame_extrapolator.predict(data[idx, 1])
            smoothed_frame_pts_diff[idx] = data[idx, 33] - smoothed_frame_pts[idx]
            updated_frame_interval[idx], anchor_frame_ts[idx], anchor_frame_no[idx] = (
                anchor_frame_extrapolator.get_frame_interval()
            )
            anchor_frame_extrapolator.update(data[idx, 33], data[idx, 1])

        nearest_vsync_ts[idx] = cal_next_vsync_ts(
            dec_over_ts[idx], anchor_vsync_ts, frame_interval
        )

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

        nearest_display_slot[idx] = cal_frame_slot(
            dec_over_ts[idx], anchor_vsync_ts, frame_interval
        )
        available_vsync_ts[idx] = max(
            [
                nearest_vsync_ts[idx],
                cal_next_vsync_ts(
                    actual_vsync_ts[prev_valid_idx] + 5, anchor_vsync_ts, frame_interval
                ),
                # np.ceil((actual_display_slot[prev_valid_idx]+1)*frame_interval)+anchor_vsync_ts
            ]
        )
        # available_vsync_ts[idx] = min(np.ceil(nearest_vsync_ts[idx] + (max_buf_size-1) * frame_interval), available_vsync_ts[idx])

        if data[idx, 4] == 0 and predicted_render_time[idx] == 0:
            cur_predicted_render_time, prev_render_time = predict_render_time(
                data[idx, 11], prev_render_time, predictor_mode=render_time_predictor
            )
            predicted_render_time[idx] = cur_predicted_render_time
        else:
            cur_predicted_render_time = predicted_render_time[idx]

        predicted_decode_time[idx] = avg_dec_time
        avg_dec_time = 0.99 * avg_dec_time + 0.01 * data[idx, 9]

        if (
            display_mode != "naiveVsync"
            and enable_quick_drop > 0
            and predicted_quick_drop_probs[idx] == 0
            and jitter_prob_updated[idx] == 0
        ):
            if enable_quick_drop >= 5 and e2e_jitter_predictor.detect_change_point(
                bonus_fps_no_thr
            ):
                e2e_jitter_predictor.reset()

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

        dec_nearest_vsync_ts[idx] = cal_next_vsync_ts(
            dec_over_ts[idx], anchor_vsync_ts, frame_interval
        )
        next_idx = idx + 1
        queued_frame_cnt = 1
        while True:
            if next_idx >= tot_frame_no:
                break

            if (
                data[next_idx, 4] == 0
                and dec_over_ts[next_idx] < dec_nearest_vsync_ts[idx] - OVERDUE_TS_THR
            ):  # small margin, since client doesn't respond that fast
                queued_frame_cnt += 1
            elif (
                data[next_idx, 4] == 0
                and dec_over_ts[next_idx] >= dec_nearest_vsync_ts[idx] - OVERDUE_TS_THR
            ):
                break

            next_idx += 1

        dec_queued_frame_cnt[idx] = queued_frame_cnt

        next_idx = idx + 1
        backup_frame_cnt = 0
        while True:
            if next_idx >= tot_frame_no:
                break

            if (
                data[next_idx, 4] == 0
                and dec_over_ts[next_idx]
                < available_vsync_ts[idx] - OVERDUE_TS_THR - cur_predicted_render_time
            ):  # small margin, since client doesn't respond that fast
                backup_frame_cnt += 1
            elif (
                data[next_idx, 4] == 0
                and dec_over_ts[next_idx]
                >= available_vsync_ts[idx] - OVERDUE_TS_THR - cur_predicted_render_time
            ):
                break

            next_idx += 1
        frame_backup_cnt[idx] = backup_frame_cnt

        no_frame_loss_flag = False
        if display_mode == "naiveVsync":
            prev_idx = prev_valid_idx
            queued_frame_cnt = 1
            while True:
                if prev_idx < 0:
                    break

                # if data[prev_idx, 4] == 0 and dec_over_ts[idx] < actual_display_ts[prev_idx] - OVERDUE_TS_THR: # small margin, since client doesn't respond that fast
                #     queued_frame_cnt += 1
                # elif data[prev_idx, 4] == 0 and dec_over_ts[idx] >= actual_display_ts[prev_idx] - OVERDUE_TS_THR:
                #    break

                # if data[prev_idx, 4] == 0 and dec_over_ts[idx] < actual_vsync_ts[prev_idx] - cur_predicted_render_time - OVERDUE_TS_THR: # small margin, since client doesn't respond that fast
                #     queued_frame_cnt += 1
                # elif data[prev_idx, 4] == 0 and dec_over_ts[idx] >= actual_vsync_ts[prev_idx] - cur_predicted_render_time - OVERDUE_TS_THR:
                #    break

                if (
                    data[prev_idx, 4] == 0
                    and queued_frame_cnt < max_buf_size
                    and dec_over_ts[idx] < actual_vsync_ts[prev_idx] - OVERDUE_TS_THR
                ):  # small margin, since client doesn't respond that fast
                    queued_frame_cnt += 1
                elif (
                    data[prev_idx, 4] == 0
                    and queued_frame_cnt >= max_buf_size
                    and max_buf_size == 2
                    and dec_over_ts[idx] < actual_display_ts[prev_idx] - OVERDUE_TS_THR
                ):
                    queued_frame_cnt += 1
                elif (
                    data[prev_idx, 4] == 0
                    and queued_frame_cnt >= max_buf_size
                    and max_buf_size > 2
                    and dec_over_ts[idx] < actual_vsync_ts[prev_idx] - OVERDUE_TS_THR
                ):
                    queued_frame_cnt += 1
                elif (
                    data[prev_idx, 4] == 0
                    and dec_over_ts[idx] >= actual_vsync_ts[prev_idx] - OVERDUE_TS_THR
                ):
                    break

                prev_idx -= 1

            if cur_buf_size == 1 and queued_frame_cnt > cur_buf_size:
                cur_buf_size = max_buf_size
                frame_buf_change_flag[idx] = 1

            elif cur_buf_size > 1 and queued_frame_cnt == 1:
                cur_buf_size = 1
                frame_buf_change_flag[idx] = 2

            frame_buffer_cnt[idx] = queued_frame_cnt
            frame_buf_size[idx] = cur_buf_size
            if queued_frame_cnt > cur_buf_size + BUFFER_OVERFLOW_THR:
                display_discarded_flag[idx] = 1

        elif display_mode == "simpleCtrl":
            if drop_frame_mode == "lifo":
                prev_idx = prev_valid_idx
                queued_frame_cnt = 1
                while True:
                    if prev_idx < 0:
                        break

                    # if data[prev_idx, 4] == 0 and dec_over_ts[idx] < actual_display_ts[prev_idx] - OVERDUE_TS_THR: # small margin, since client doesn't respond that fast
                    #     queued_frame_cnt += 1
                    # elif data[prev_idx, 4] == 0 and dec_over_ts[idx] >= actual_display_ts[prev_idx] - OVERDUE_TS_THR:
                    #     break

                    # if data[prev_idx, 4] == 0 and dec_over_ts[idx] < actual_vsync_ts[prev_idx] - cur_predicted_render_time - OVERDUE_TS_THR: # small margin, since client doesn't respond that fast
                    #     queued_frame_cnt += 1
                    # elif data[prev_idx, 4] == 0 and dec_over_ts[idx] >= actual_vsync_ts[prev_idx] - cur_predicted_render_time - OVERDUE_TS_THR:
                    #    break

                    if (
                        data[prev_idx, 4] == 0
                        and queued_frame_cnt < max_buf_size
                        and dec_over_ts[idx]
                        < actual_vsync_ts[prev_idx] - OVERDUE_TS_THR
                    ):  # small margin, since client doesn't respond that fast
                        queued_frame_cnt += 1
                    elif (
                        data[prev_idx, 4] == 0
                        and queued_frame_cnt >= max_buf_size
                        and dec_over_ts[idx]
                        < actual_display_ts[prev_idx] - OVERDUE_TS_THR
                    ):  # small margin, since client doesn't respond that fast
                        queued_frame_cnt += 1
                    elif (
                        data[prev_idx, 4] == 0
                        and dec_over_ts[idx]
                        >= actual_vsync_ts[prev_idx] - OVERDUE_TS_THR
                    ):
                        break

                    prev_idx -= 1

            elif drop_frame_mode == "fifo":
                next_idx = idx + 1
                queued_frame_cnt = 1
                while True:
                    if next_idx >= tot_frame_no:
                        break

                    if (
                        data[next_idx, 4] == 0
                        and dec_over_ts[next_idx] + cur_predicted_render_time
                        < available_vsync_ts[idx] - OVERDUE_TS_THR
                    ):  # small margin, since client doesn't respond that fast
                        queued_frame_cnt += 1
                    elif (
                        data[next_idx, 4] == 0
                        and dec_over_ts[next_idx] + cur_predicted_render_time
                        >= available_vsync_ts[idx] - OVERDUE_TS_THR
                    ):
                        break

                    next_idx += 1

            # def cal_frame_jitter(idx):
            #     if data[idx, 0] == data[idx-1, 0]+1:
            #         jitter_time = max(0, dec_over_ts[idx] - dec_over_ts[idx-1] - frame_interval)
            #     else:
            #         jitter_time = 0

            #     return jitter_time

            # cur_frame_jitter = cal_frame_jitter(idx)
            # if cur_frame_jitter > jitter_thr:
            #     cur_frame_jitter_flag = True

            if cur_buf_size == 1 and queued_frame_cnt > cur_buf_size:
                cur_buf_size = max_buf_size

                window_start_idx = idx
                window_start_ts = dec_over_ts[idx]

                need_check_quick_drop = True
                quick_drop_st_idx = idx
                quick_drop_st_ts = dec_over_ts[idx]

                frame_buf_change_flag[idx] = 1

            elif cur_buf_size > 1 and queued_frame_cnt == 1:
                cur_buf_size = 1

                frame_buf_change_flag[idx] = 2

            elif (
                enable_perio_drop == 1
                and cur_buf_size > 1
                and (
                    dec_over_ts[idx] - window_start_ts >= window_update_interval_ms
                    and idx - window_start_idx + 1 >= window_update_interval_frame_cnt
                    and prev_valid_idx - consecutive_window_start_idx + 1
                    >= window_update_interval_frame_cnt
                    and backup_frame_cnt > 0
                )
            ):
                cur_buf_size = 1

                frame_buf_change_flag[idx] = 4

                # consecutive_window_start_idx = prev_valid_idx

            elif (
                enable_perio_drop == 2
                and cur_buf_size > 1
                and (
                    prev_valid_idx - consecutive_window_start_idx + 1
                    >= window_update_interval_frame_cnt
                    and consecutive_frame_no[prev_valid_idx]
                    - consecutive_frame_no[consecutive_window_start_idx]
                    + 1
                    >= window_update_interval_frame_cnt
                    and backup_frame_cnt > 0
                )
            ):
                cur_buf_size = 1

                frame_buf_change_flag[idx] = 6

                # consecutive_window_start_idx = prev_valid_idx

            elif (
                enable_quick_drop > 0
                and cur_buf_size > 1
                and queued_frame_cnt > 1
                and need_check_quick_drop
                and (
                    dec_over_ts[idx] - quick_drop_st_ts >= quick_drop_interval_ms
                    or idx - quick_drop_st_idx >= quick_drop_frame_cnt
                )
            ):
                # need_check_quick_drop = False

                # print(idx, predicted_prob)
                if predicted_quick_drop_probs[idx] < early_drop_prob_threshold:
                    cur_buf_size = 1

                    frame_buf_change_flag[idx] = 3

                    prev_quick_drop_flag = True

            frame_buffer_cnt[idx] = queued_frame_cnt
            frame_buf_size[idx] = cur_buf_size
            if queued_frame_cnt > cur_buf_size + BUFFER_OVERFLOW_THR:
                display_discarded_flag[idx] = 1

        elif display_mode == "optimal":
            if drop_frame_mode == "lifo":
                prev_idx = prev_valid_idx
                queued_frame_cnt = 1
                while True:
                    if prev_idx < 0:
                        break

                    # if data[prev_idx, 4] == 0 and dec_over_ts[idx] < actual_vsync_ts[prev_idx] - cur_predicted_render_time - OVERDUE_TS_THR: # small margin, since client doesn't respond that fast
                    #     queued_frame_cnt += 1
                    # elif data[prev_idx, 4] == 0 and dec_over_ts[idx] >= actual_vsync_ts[prev_idx] - cur_predicted_render_time - OVERDUE_TS_THR:
                    #    break

                    # if data[prev_idx, 4] == 0 and queued_frame_cnt == 1 and dec_over_ts[idx] < actual_vsync_ts[prev_idx] - cur_predicted_render_time - OVERDUE_TS_THR: # small margin, since client doesn't respond that fast
                    #     queued_frame_cnt += 1
                    # elif data[prev_idx, 4] == 0 and queued_frame_cnt > 1 and dec_over_ts[idx] < actual_display_ts[prev_idx] - OVERDUE_TS_THR: # small margin, since client doesn't respond that fast
                    #     queued_frame_cnt += 1
                    # elif data[prev_idx, 4] == 0 and dec_over_ts[idx] >= actual_vsync_ts[prev_idx] - OVERDUE_TS_THR:
                    #     break

                    if (
                        data[prev_idx, 4] == 0
                        and queued_frame_cnt < max_buf_size
                        and dec_over_ts[idx]
                        < actual_vsync_ts[prev_idx] - data[idx, 11] - OVERDUE_TS_THR
                    ):  # small margin, since client doesn't respond that fast
                        queued_frame_cnt += 1
                    elif (
                        data[prev_idx, 4] == 0
                        and queued_frame_cnt >= max_buf_size
                        and dec_over_ts[idx]
                        < actual_display_ts[prev_idx] - OVERDUE_TS_THR
                    ):  # small margin, since client doesn't respond that fast
                        queued_frame_cnt += 1
                    elif (
                        data[prev_idx, 4] == 0
                        and dec_over_ts[idx]
                        >= actual_vsync_ts[prev_idx] - OVERDUE_TS_THR
                    ):
                        break

                    prev_idx -= 1

            elif drop_frame_mode == "fifo":
                next_idx = idx + 1
                queued_frame_cnt = 1
                while True:
                    if next_idx >= tot_frame_no:
                        break

                    if (
                        data[next_idx, 4] == 0
                        and dec_over_ts[next_idx] + cur_predicted_render_time
                        < available_vsync_ts[idx] - OVERDUE_TS_THR
                    ):  # small margin, since client doesn't respond that fast
                        queued_frame_cnt += 1
                    elif (
                        data[next_idx, 4] == 0
                        and dec_over_ts[next_idx] + cur_predicted_render_time
                        >= available_vsync_ts[idx] - OVERDUE_TS_THR
                    ):
                        break

                    next_idx += 1

            if early_drop_frame_flag and consecutive_window_start_index[idx] != -1:
                consecutive_window_start_idx = consecutive_window_start_index[idx]
            assert consecutive_window_start_idx <= idx, (
                "consecutive window start index small than current index, file_path: %s, idx: %d, window_idx: %d"
                % (file_path, idx, consecutive_window_start_idx)
            )

            if early_drop_frame_flag:
                assert queued_frame_cnt > 1, (
                    "early drop frame only happens when queued_frame_cnt > 1, file_path: %s, idx: %d, queued_frame_cnt: %d"
                    % (file_path, idx, queued_frame_cnt)
                )

                cur_buf_size = 1

                window_start_idx = idx
                window_start_ts = dec_over_ts[idx]

                frame_buf_change_flag[idx] = 5

                early_drop_frame_flag = False

            elif cur_buf_size == 1 and queued_frame_cnt > cur_buf_size:
                cur_buf_size = max_buf_size

                window_start_idx = idx
                window_start_ts = dec_over_ts[idx]

                frame_buf_change_flag[idx] = 1

            elif cur_buf_size > 1 and queued_frame_cnt == 1:
                cur_buf_size = 1

                window_start_idx = idx
                window_start_ts = dec_over_ts[idx]

                frame_buf_change_flag[idx] = 2

            elif (
                enable_perio_drop == 1
                and cur_buf_size > 1
                and (
                    dec_over_ts[idx] - window_start_ts >= window_update_interval_ms
                    and idx - window_start_idx >= window_update_interval_frame_cnt
                    and prev_valid_idx - consecutive_window_start_idx
                    >= window_update_interval_frame_cnt
                    and backup_frame_cnt > 0
                )
            ):
                cur_buf_size = 1

                frame_buf_change_flag[idx] = 4

                early_drop_frame_flag = True

                # consecutive_window_start_index[idx] = prev_valid_idx

            elif (
                enable_perio_drop == 2
                and cur_buf_size > 1
                and (
                    prev_valid_idx - consecutive_window_start_idx + 1
                    >= window_update_interval_frame_cnt
                    and consecutive_frame_no[prev_valid_idx]
                    - consecutive_frame_no[consecutive_window_start_idx]
                    + 1
                    >= window_update_interval_frame_cnt
                    and backup_frame_cnt > 0
                )
            ):
                cur_buf_size = 1

                frame_buf_change_flag[idx] = 6

                early_drop_frame_flag = True

                # consecutive_window_start_index[idx] = prev_valid_idx
            else:
                frame_buf_change_flag[idx] = 0

            frame_buffer_cnt[idx] = queued_frame_cnt
            frame_buf_size[idx] = cur_buf_size

            if queued_frame_cnt > cur_buf_size + BUFFER_OVERFLOW_THR:
                display_discarded_flag[idx] = 1
                invoke_present_ts[idx] = 0
                invoke_present_slot[idx] = 0
                actual_render_queue[idx] = 0
                actual_vsync_ts[idx] = 0
                actual_display_ts[idx] = 0
                actual_display_slot[idx] = 0
                valid_frame_flag[idx] = 0
                no_frame_loss_flag = False
            else:
                display_discarded_flag[idx] = 0

            if early_drop_frame_flag:
                prev_valid_frame_index[idx] = prev_valid_idx
                prev_valid_idx = prev_valid_frame_index[window_start_idx]
                idx = window_start_idx

        else:
            raise ValueError("Unknown display mode: %s" % display_mode)

        prev_valid_frame_index[tmp_idx] = prev_valid_idx
        jitter_window_start_index[tmp_idx] = window_start_idx

        if not early_drop_frame_flag and display_discarded_flag[idx] == 0:
            if frame_buf_size[idx] > 1 and frame_buf_size[prev_valid_idx] == 1:
                # if frame_buf_size[idx] > 1 and frame_buf_size[prev_valid_idx] == 1 and backup_frame_cnt > 0:
                frame_jitter_flag[idx] = 1
            else:
                frame_jitter_flag[idx] = 0

            if (
                display_mode != "naiveVsync"
                and enable_quick_drop > 0
                and jitter_prob_updated[idx] == 0
            ):
                e2e_jitter_predictor.update(
                    data,
                    smoothed_frame_pts_diff,
                    nearest_no_jitter_vsync_ts,
                    actual_display_ts,
                    idx,
                    frame_jitter_flag[idx],
                )
                jitter_prob_updated[idx] = 1

            if drop_frame_mode == "lifo":
                invoke_present_ts[idx] = max(
                    dec_over_ts[idx], actual_vsync_ts[prev_valid_idx]
                )
                # invoke_present_ts[idx] = max(dec_over_ts[idx], actual_vsync_ts[prev_valid_idx]+1)
            elif drop_frame_mode == "fifo":
                if (
                    dec_over_ts[idx] >= available_vsync_ts[idx] - frame_interval
                    or dec_over_ts[idx] >= actual_vsync_ts[prev_valid_idx]
                ):
                    invoke_present_ts[idx] = dec_over_ts[idx]
                else:
                    invoke_present_ts[idx] = min(
                        actual_vsync_ts[prev_valid_idx] + 1,
                        available_vsync_ts[idx] - frame_interval + 1,
                    )
                    invoke_present_ts[idx] = max(
                        invoke_present_ts[idx], actual_vsync_ts[prev_valid_idx]
                    )

            invoke_present_slot[idx] = cal_frame_slot(
                invoke_present_ts[idx], anchor_vsync_ts, frame_interval
            )
            if (
                invoke_present_slot[idx] == invoke_present_slot[prev_valid_idx]
                and invoke_present_ts[idx] != invoke_present_ts[prev_valid_idx]
            ):
                invoke_present_slot[idx] = invoke_present_slot[idx] + 1

            actual_display_ts[idx] = invoke_present_ts[idx] + data[idx, 11]
            actual_vsync_ts[idx] = cal_next_vsync_ts(
                actual_display_ts[idx], anchor_vsync_ts, frame_interval
            )
            actual_display_slot[idx] = cal_frame_slot(
                actual_vsync_ts[idx] - 1, anchor_vsync_ts, frame_interval
            )
            extra_display_ts[idx] = actual_vsync_ts[idx] - actual_display_ts[idx]

            # actual_render_queue[idx] = actual_vsync_ts[idx] - dec_over_ts[idx] - data[idx, 11]
            actual_render_queue[idx] = invoke_present_ts[idx] - dec_over_ts[idx]

            if (
                actual_display_slot[idx] == actual_display_slot[prev_valid_idx]
                and actual_display_ts[idx] != actual_display_ts[prev_valid_idx]
            ):
                actual_display_slot[idx] = actual_display_slot[idx] + 1
            valid_frame_flag[idx] = 1

            assert (
                actual_vsync_ts[idx] > actual_vsync_ts[prev_valid_idx]
            ), "latest frame displayed later, file_path: %s, frame_id: %d" % (
                file_path,
                idx,
            )
            assert invoke_present_ts[idx] >= actual_vsync_ts[prev_valid_idx] - 10, (
                "latest frame invoked later than previous frame vsync ts, file_path: %s, frame_id: %d"
                % (file_path, idx)
            )

            if idx > 1:
                assert (
                    actual_display_slot[idx] == 0 and valid_frame_flag[idx] == 0
                ) or (
                    actual_display_slot[idx] != 0 and valid_frame_flag[idx] == 1
                ), "discarded frame is not valid, file_path: %s, frame_id: %d" % (
                    file_path,
                    idx,
                )

            if actual_display_slot[idx] == actual_display_slot[prev_valid_idx] + 1:
                no_frame_loss_flag = True
                consecutive_frame_no[idx] = consecutive_frame_no[prev_valid_idx] + 1

            else:
                if consecutive_frame_no[prev_valid_idx] >= bonus_fps_no_thr:
                    objective_bonus_fps_no += 1

                consecutive_frame_no[idx] = 1
                consecutive_window_start_idx = idx
                consecutive_window_start_ts = data[idx, 5]
                window_start_idx = idx
                window_start_ts = dec_over_ts[idx]

            if enable_quick_drop > 0 and prev_quick_drop_flag:
                if consecutive_frame_no[idx] >= bonus_fps_no_thr:
                    prev_quick_drop_flag = False
                    quick_drop_interval_ms = max(100, quick_drop_interval_ms // 2)
                    quick_drop_frame_cnt = max(6, quick_drop_frame_cnt // 2)
                elif (
                    consecutive_frame_no[idx] == 1
                    and consecutive_frame_no[prev_valid_idx] < bonus_fps_no_thr * 0.9
                ):
                    prev_quick_drop_flag = False
                    quick_drop_interval_ms = min(quick_drop_interval_ms * 2, 400)
                    quick_drop_frame_cnt = min(quick_drop_frame_cnt * 2, 24)

            cur_valid_frame_no[idx] = cur_valid_frame_no[prev_valid_idx] + 1

            if display_discarded_flag[idx] == 0:
                prev_valid_idx = idx

        quick_drop_frame_cnt_hist[idx] = quick_drop_frame_cnt

        if not early_drop_frame_flag:
            consecutive_window_start_index[idx] = consecutive_window_start_idx

        if print_debug_log:
            log_file.write(
                ",".join(
                    str(item)
                    for item in [tmp_idx]
                    + data[tmp_idx, :-5].tolist()
                    + [
                        cur_anchor_vsync_ts[tmp_idx],
                        dec_over_ts[tmp_idx],
                        predicted_render_time[tmp_idx],
                        nearest_vsync_ts[tmp_idx],
                        nearest_display_slot[tmp_idx],
                        available_vsync_ts[tmp_idx],
                        actual_render_queue[tmp_idx],
                        frame_buffer_cnt[tmp_idx],
                        invoke_present_ts[tmp_idx],
                        invoke_present_slot[tmp_idx],
                        actual_display_ts[tmp_idx],
                        actual_vsync_ts[tmp_idx],
                        actual_display_slot[tmp_idx],
                        frame_buf_size[tmp_idx],
                        frame_buf_change_flag[tmp_idx],
                        consecutive_frame_no[tmp_idx],
                        display_discarded_flag[tmp_idx],
                        valid_frame_flag[tmp_idx],
                        original_valid_flag[tmp_idx],
                        dec_over_ts[tmp_idx],
                        dec_nearest_vsync_ts[tmp_idx],
                        dec_queued_frame_cnt[tmp_idx],
                        int(frame_jitter_flag[tmp_idx]),
                        prev_valid_frame_index[tmp_idx],
                        consecutive_window_start_index[tmp_idx],
                        jitter_window_start_index[tmp_idx],
                    ]
                )
                + "\n"
            )

    noloss_bonus_fps_no = 0

    quick_drop_total_cnt = 0
    quick_drop_failed_cnt = 0
    quick_drop_missed_cnt = 0
    prev_quick_drop_flag = False
    possible_quick_drop_chance_flag = False
    possible_quick_drop_chance_st_idx = 0
    if display_mode != "naiveVsync":
        prev_valid_idx = sim_st_idx
        for idx in range(sim_st_idx + 1, tot_frame_no):
            if data[idx, 4] != 0 or data[idx, 5] == 0:
                continue

            if display_mode == "simpleCtrl":
                if frame_buf_change_flag[idx] == 3:
                    quick_drop_total_cnt += 1
                    prev_quick_drop_flag = True

                if frame_buf_change_flag[idx] == 1:
                    possible_quick_drop_chance_flag = True
                    quick_drop_st_idx = idx

                if consecutive_frame_no[idx] > consecutive_frame_no[prev_valid_idx] + 1:
                    possible_quick_drop_chance_st_idx = idx

                if prev_quick_drop_flag:
                    if consecutive_frame_no[idx] >= bonus_fps_no_thr:
                        prev_quick_drop_flag = False
                    elif consecutive_frame_no[idx] == 1:
                        prev_quick_drop_flag = False
                        quick_drop_failed_cnt += 1
                        failed_early_drop_frame[idx] = 1

                if possible_quick_drop_chance_flag:
                    if (
                        consecutive_frame_no[idx]
                        - consecutive_frame_no[possible_quick_drop_chance_st_idx]
                        >= bonus_fps_no_thr - 1
                        and frame_backup_cnt[idx] > 0
                    ):  # and idx - quick_drop_st_idx >= quick_drop_frame_cnt:
                        possible_quick_drop_chance_flag = False
                        quick_drop_missed_cnt += 1
                        missed_early_drop_frame[idx] = 1
                    elif (
                        frame_buf_size[idx] == 1
                        or consecutive_frame_no[idx]
                        > consecutive_frame_no[prev_valid_idx] + 1
                    ):
                        possible_quick_drop_chance_flag = False
            elif display_mode == "optimal":
                if (
                    frame_buf_change_flag[idx] == 5
                    and predicted_quick_drop_probs[idx] >= early_drop_prob_threshold
                ):
                    missed_early_drop_frame[idx] = 1
                    quick_drop_missed_cnt += 1

                elif (
                    frame_buf_size[idx] > 1
                    and predicted_quick_drop_probs[idx] < early_drop_prob_threshold
                ):
                    failed_early_drop_frame[idx] = 1
                    quick_drop_failed_cnt += 1

            if (
                frame_buf_change_flag[idx] > 3
                and bonus_fps_obtained[prev_valid_idx] == 0
            ):
                noloss_bonus_fps_no += 1
                bonus_fps_obtained[prev_valid_idx] = 1

            if actual_display_slot[idx] == actual_display_slot[prev_valid_idx] + 1:
                bonus_fps_obtained[idx] = bonus_fps_obtained[prev_valid_idx]

            cur_bonus_fps_no[idx] = noloss_bonus_fps_no

            if display_discarded_flag[idx] == 0:
                prev_valid_idx = idx

    if drop_frame_mode == "lifo":
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
        render_jitter_flag = np.zeros(tot_frame_no)
        # render_jitter_flag[1:] = smoothed_frame_pts_diff[1:] < -small_ts_margin
        render_jitter_flag[1:] = np.logical_and(
            smoothed_frame_pts_diff[1:] < -small_ts_margin,
            dec_over_ts[1:] < available_vsync_ts[:-1],
        )
        render_jitter_flag[1:] = np.logical_or(
            render_jitter_flag[1:], smoothed_frame_pts_diff[:-1] > small_ts_margin
        )

        # server related
        server_jitter_flag = np.zeros(tot_frame_no)
        frame_cgs_render_interval = data[1:, 33] - data[:-1, 33]
        frame_proxy_recv_interval = data[1:, 12] - data[:-1, 12]
        server_jitter_flag[2:] = (
            frame_proxy_recv_interval - frame_cgs_render_interval
        )[:-1] > small_ts_margin

        # decoder related
        decoder_jitter_flag = np.zeros(tot_frame_no)
        decoder_jitter_flag[1:] = np.logical_or(
            data[:, 9] > avg_dec_time + small_ts_margin,
            data[:, 7:9].sum(-1) > small_ts_margin,
        )[:-1]

        # display related
        display_jitter_flag = data[:, 11] > predicted_render_time + small_ts_margin
        prev_display_jitter = np.zeros(tot_frame_no)
        prev_display_jitter[1:] = display_jitter_flag[:-1]

        near_vsync_jitter_flag = np.logical_and.reduce(
            (
                dec_over_ts < nearest_vsync_ts,
                actual_display_ts >= nearest_vsync_ts,
                actual_display_ts - nearest_vsync_ts <= small_ts_margin,
            )
        )
        prev_near_vsync_jitter = np.zeros(tot_frame_no)
        prev_near_vsync_jitter[1:] = near_vsync_jitter_flag[:-1]

    elif drop_frame_mode == "fifo":
        # network related
        network_big_frame_flag = np.logical_and.reduce(
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
        network_dl_jitter_flag = np.logical_and.reduce(
            (data[:, 21] > small_ts_margin, data[:, 21] < frame_stall_thr)
        )
        network_stall_flag = data[:, 21] >= frame_stall_thr
        network_packet_loss_flag = np.logical_and(
            data[:, 38] > 0, data[:, 16] >= 2 * min_rtt
        )
        network_problem_flag = np.logical_or.reduce(
            (
                network_big_frame_flag,
                network_dl_jitter_flag,
                network_stall_flag,
                network_packet_loss_flag,
            )
        )
        # render related
        render_jitter_flag = smoothed_frame_pts_diff > small_ts_margin
        render_jitter_flag[:-1] = np.logical_or(
            render_jitter_flag[:-1], smoothed_frame_pts_diff[1:] < -small_ts_margin
        )

        # server related
        server_jitter_flag = np.zeros(tot_frame_no)
        frame_cgs_render_interval = data[1:, 33] - data[:-1, 33]
        frame_proxy_recv_interval = data[1:, 12] - data[:-1, 12]
        server_jitter_flag[1:] = (
            np.abs(frame_proxy_recv_interval - frame_cgs_render_interval)
            > small_ts_margin
        )

        # decoder related
        # decoder_jitter_flag = np.zeros(tot_frame_no)
        decoder_jitter_flag = np.logical_or(
            data[:, 9] > avg_dec_time + small_ts_margin, data[:, 7:9].sum(-1) > 0
        )

        # display related
        display_jitter_flag = data[:, 11] > predicted_render_time + small_ts_margin
        prev_display_jitter = display_jitter_flag

        near_vsync_jitter_flag = np.logical_and.reduce(
            (
                dec_over_ts < nearest_vsync_ts,
                actual_display_ts >= nearest_vsync_ts,
                actual_display_ts - nearest_vsync_ts <= small_ts_margin,
            )
        )
        prev_near_vsync_jitter = near_vsync_jitter_flag

    jitter_amps = [
        smoothed_frame_pts_diff,
        data[:, 21],
        data[:, 9] - predicted_decode_time,
        data[:, 11] - predicted_render_time,
    ]
    all_jitter_amps = np.stack(jitter_amps, axis=1)
    biggest_jitter_idx = np.argsort(all_jitter_amps, -1)[:, -1]
    biggest_render_jitter_flag = biggest_jitter_idx == 0
    biggest_network_jitter_flag = biggest_jitter_idx == 1
    biggest_decode_jitter_flag = biggest_jitter_idx == 2
    biggest_display_jitter_flag = biggest_jitter_idx == 3

    biggest_render_jitter_queue_flag = np.logical_and(
        frame_jitter_flag, biggest_render_jitter_flag
    )
    biggest_network_jitter_queue_flag = np.logical_and(
        frame_jitter_flag, biggest_network_jitter_flag
    )
    biggest_decode_jitter_queue_flag = np.logical_and(
        frame_jitter_flag, biggest_decode_jitter_flag
    )
    biggest_display_jitter_queue_flag = np.logical_and(
        frame_jitter_flag, biggest_display_jitter_flag
    )

    # network induced queue
    network_stall_induced_queue = np.logical_and.reduce(
        (frame_jitter_flag, network_stall_flag)
    )
    network_packet_loss_induced_queue = np.logical_and.reduce(
        (
            frame_jitter_flag,
            network_packet_loss_flag,
            np.logical_not(network_stall_flag),
        )
    )
    network_big_frame_induced_queue = np.logical_and.reduce(
        (
            frame_jitter_flag,
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
            frame_jitter_flag,
            network_dl_jitter_flag,
            np.logical_not(network_big_frame_flag),
            np.logical_not(network_packet_loss_flag),
            np.logical_not(network_stall_flag),
        )
    )

    # render induced queue
    render_jitter_induced_queue = np.logical_and.reduce(
        (frame_jitter_flag, render_jitter_flag, np.logical_not(network_problem_flag))
    )

    # server inside jitter induced queue
    server_jitter_induced_queue = np.logical_and.reduce(
        (
            frame_jitter_flag,
            server_jitter_flag,
            np.logical_not(network_problem_flag),
            np.logical_not(render_jitter_flag),
        )
    )

    # decoder induced queue
    decoder_jitter_induced_queue = np.logical_and.reduce(
        (
            frame_jitter_flag,
            decoder_jitter_flag,
            np.logical_not(network_problem_flag),
            np.logical_not(render_jitter_flag),
            np.logical_not(server_jitter_flag),
        )
    )

    # display induced queue
    display_jitter_induced_queue = np.logical_and.reduce(
        (
            frame_jitter_flag,
            prev_display_jitter,
            np.logical_not(network_problem_flag),
            np.logical_not(render_jitter_flag),
            np.logical_not(server_jitter_flag),
            np.logical_not(decoder_jitter_flag),
        )
    )

    near_vsync_jitter_induced_queue = np.logical_and.reduce(
        (
            frame_jitter_flag,
            prev_near_vsync_jitter,
            np.logical_not(network_problem_flag),
            np.logical_not(render_jitter_flag),
            np.logical_not(server_jitter_flag),
            np.logical_not(decoder_jitter_flag),
            np.logical_not(prev_display_jitter),
        )
    )
    # idx=18433
    # print(actual_display_ts[idx], nearest_vsync_ts[idx], near_vsync_jitter_flag[idx], network_problem_flag[idx], render_jitter_flag[idx], server_jitter_flag[idx], decoder_jitter_flag[idx], prev_display_jitter[idx])
    # print(prev_near_vsync_jitter[idx+1])
    # exit()

    # find early drop opportunity
    # network induced queue
    plausible_early_drop_flag = frame_buf_change_flag == 5

    exploitable_network_stall_induced_queue = np.logical_and.reduce(
        (plausible_early_drop_flag, network_stall_flag)
    )
    exploitable_network_packet_loss_induced_queue = np.logical_and.reduce(
        (
            plausible_early_drop_flag,
            network_packet_loss_flag,
            np.logical_not(network_stall_flag),
        )
    )
    exploitable_network_big_frame_induced_queue = np.logical_and.reduce(
        (
            plausible_early_drop_flag,
            network_big_frame_flag,
            np.logical_not(network_packet_loss_flag),
            np.logical_not(network_stall_flag),
        )
    )
    exploitable_network_i_frame_induced_queue = np.logical_and(
        exploitable_network_big_frame_induced_queue,
        np.logical_or(data[:, 2] == 2, data[:, 49] == 1),
    )

    exploitable_network_dl_jitter_induced_queue = np.logical_and.reduce(
        (
            plausible_early_drop_flag,
            network_dl_jitter_flag,
            np.logical_not(network_big_frame_flag),
            np.logical_not(network_packet_loss_flag),
            np.logical_not(network_stall_flag),
        )
    )

    # render induced queue
    exploitable_render_jitter_induced_queue = np.logical_and.reduce(
        (
            plausible_early_drop_flag,
            render_jitter_flag,
            np.logical_not(network_problem_flag),
        )
    )

    exploitable_server_jitter_induced_queue = np.logical_and.reduce(
        (
            plausible_early_drop_flag,
            server_jitter_flag,
            np.logical_not(network_problem_flag),
            np.logical_not(render_jitter_flag),
        )
    )

    # decoder induced queue
    exploitable_decoder_jitter_induced_queue = np.logical_and.reduce(
        (
            plausible_early_drop_flag,
            decoder_jitter_flag,
            np.logical_not(network_problem_flag),
            np.logical_not(render_jitter_flag),
            np.logical_not(server_jitter_flag),
        )
    )

    # display induced queue
    exploitable_display_jitter_induced_queue = np.logical_and.reduce(
        (
            plausible_early_drop_flag,
            prev_display_jitter,
            np.logical_not(network_problem_flag),
            np.logical_not(render_jitter_flag),
            np.logical_not(server_jitter_flag),
            np.logical_not(decoder_jitter_flag),
        )
    )

    exploitable_near_vsync_jitter_induced_queue = np.logical_and.reduce(
        (
            plausible_early_drop_flag,
            prev_near_vsync_jitter,
            np.logical_not(network_problem_flag),
            np.logical_not(render_jitter_flag),
            np.logical_not(server_jitter_flag),
            np.logical_not(decoder_jitter_flag),
            np.logical_not(prev_display_jitter),
        )
    )

    tot_time = (data[-1, 5] - data[0, 5]) / 1000
    max_fps = data.shape[0] / tot_time
    min_valid_no = np.unique(np.ceil((dec_over_ts) / frame_interval)).size
    min_fps = min_valid_no / tot_time

    # bonus_fps_no = np.sum(consecutive_frame_no % bonus_fps_no_thr == 0)

    if client_vsync_enabled == 1:
        origin_valid_no = np.sum(data[:, 4] == 0)
    else:
        origin_valid_no = np.unique(original_display_slot).size

    data[-1, 11] = -1
    optimized_valid_no = np.sum(valid_frame_flag)

    origin_fps = origin_valid_no / tot_time
    optimized_fps = optimized_valid_no / tot_time
    optimized_objective_fps = (optimized_valid_no + objective_bonus_fps_no) / tot_time
    optimized_noloss_fps = (optimized_valid_no + noloss_bonus_fps_no) / tot_time

    origin_valid_idx = np.where(data[:, 4] == 0)[0]
    origin_render_queue = np.mean(data[origin_valid_idx, 10])

    optimized_valid_idx = np.where(display_discarded_flag == 0)
    optimized_render_queue = np.mean(actual_render_queue[optimized_valid_idx])
    extra_display_time = np.mean(extra_display_ts[optimized_valid_idx])

    quick_drop_fail_ratio = (
        (quick_drop_failed_cnt / quick_drop_total_cnt)
        if quick_drop_total_cnt > 0
        else 0
    )
    quick_drop_miss_ratio = (
        (
            quick_drop_missed_cnt
            / (quick_drop_missed_cnt + quick_drop_total_cnt - quick_drop_failed_cnt)
        )
        if (quick_drop_missed_cnt + quick_drop_total_cnt - quick_drop_failed_cnt) > 0
        else 0
    )

    # print(quick_drop_failed_cnt, quick_drop_total_cnt, quick_drop_fail_ratio)
    # print(quick_drop_missed_cnt, (quick_drop_missed_cnt+quick_drop_total_cnt-quick_drop_failed_cnt), quick_drop_miss_ratio)
    avg_dec_time, avg_dec_total_time, avg_render_time, avg_proc_time = (
        cal_avg_client_ime(tot_frame_no)
    )

    result = [
        server_optim_enabled,
        client_optim_enabled,
        client_vsync_enabled,
        max_fps,
        min_fps,
        origin_fps,
        optimized_fps,
        optimized_objective_fps,
        optimized_noloss_fps,
        origin_render_queue,
        optimized_render_queue,
        extra_display_time,
        optimized_render_queue + extra_display_time,
        avg_dec_time,
        avg_dec_total_time,
        avg_render_time,
        avg_proc_time,
        tot_frame_no,
        np.sum(frame_jitter_flag),
        network_big_frame_induced_queue.sum(),
        network_i_frame_induced_queue.sum(),
        network_dl_jitter_induced_queue.sum(),
        network_stall_induced_queue.sum(),
        network_packet_loss_induced_queue.sum(),
        render_jitter_induced_queue.sum(),
        server_jitter_induced_queue.sum(),
        decoder_jitter_induced_queue.sum(),
        display_jitter_induced_queue.sum(),
        near_vsync_jitter_induced_queue.sum(),
        np.sum(data[:, 16] > 100),
        np.sum(data[:, 11] > 12),
        np.sum((data[1:, 33] - data[:-1, 33]) > frame_interval * 1.5),
        quick_drop_failed_cnt,
        quick_drop_fail_ratio,
        quick_drop_missed_cnt,
        quick_drop_miss_ratio,
        quick_drop_total_cnt,
        exploitable_network_big_frame_induced_queue.sum(),
        exploitable_network_i_frame_induced_queue.sum(),
        exploitable_network_dl_jitter_induced_queue.sum(),
        exploitable_network_stall_induced_queue.sum(),
        exploitable_network_packet_loss_induced_queue.sum(),
        exploitable_render_jitter_induced_queue.sum(),
        exploitable_server_jitter_induced_queue.sum(),
        exploitable_decoder_jitter_induced_queue.sum(),
        exploitable_display_jitter_induced_queue.sum(),
        exploitable_near_vsync_jitter_induced_queue.sum(),
        biggest_render_jitter_queue_flag.sum(),
        biggest_network_jitter_queue_flag.sum(),
        biggest_decode_jitter_queue_flag.sum(),
        biggest_display_jitter_queue_flag.sum(),
    ]

    # print(file_path + ',\t' + ',\t'.join([f"{item = }" for item in result]))
    print(file_path + ",\t" + ",\t".join(["%.3f" % item for item in result]))
    if print_log:
        log_file = open(
            file_path[:-4]
            + "_%s_quickdrop%d_periodrop%d_maxbuf%d_renderTime_%s_%s_sim.csv"
            % (
                display_mode,
                enable_quick_drop,
                enable_perio_drop,
                max_buf_size,
                render_time_predictor,
                drop_frame_mode,
            ),
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
                    "cur_anchor_vsync_ts",
                    "decode_over_ts",
                    "client_invoke_ts",
                    "client_display_ts",
                    "predicted_render_time",
                    "predicted_decode_time",
                    "nearest_vsync_ts",
                    "nearest_display_slot",
                    "available_vsync_ts",
                    "actual_render_queue",
                    "client_render_queue",
                    "extra_display_ts",
                    "frame_buffer_cnt",
                    "invoke_present_ts",
                    "invoke_present_slot",
                    "actual_display_ts",
                    "actual_vsync_ts",
                    "actual_display_slot",
                    "cur_buf_size",
                    "buf_change_flag",
                    "failed_early_drop_frame",
                    "missed_early_drop_frame",
                    "predicted_quick_drop_probs",
                    "frame_e2e_jitter",
                    "dl_jitter_prob",
                    "render_jitter_prob",
                    "server_jitter_prob",
                    "decoder_jitter_prob",
                    "display_jitter_prob",
                    "e2e_jitter_prob",
                    "network_problem_prob",
                    "near_vsync_jitter_prob",
                    "dl_jitter_prob_thr",
                    "render_jitter_prob_thr",
                    "server_jitter_prob_thr",
                    "decoder_jitter_prob_thr",
                    "display_jitter_prob",
                    "e2e_jitter_prob_thr",
                    "network_problem_prob_thr",
                    "near_vsync_jitter_prob_thr",
                    "quick_drop_frame_cnt_hist",
                    "consecutive_frame_no",
                    "display_discarded_flag",
                    "algorithm_discard_flag",
                    "valid_frame_flag",
                    "original_valid_flag",
                    "bonus_fps_obtained",
                    "cur_bonus_fps_no",
                    "cur_valid_frame_no",
                    "dec_over_ts",
                    "dec_nearest_vsync_ts",
                    "dec_queued_frame_cnt",
                    "frame_queue_flag",
                    "original_pts",
                    "smoothed_pts",
                    "smoothed_frame_pts_diff",
                    "render_jitter_induced_queue",
                    "server_jitter_induced_queue",
                    "network_big_frame_induced_queue",
                    "network_i_frame_induced_queue",
                    "network_dl_jitter_induced_queue",
                    "network_stall_induced_queue",
                    "network_packet_loss_induced_queue",
                    "decoder_jitter_induced_queue",
                    "display_jitter_induced_queue",
                    "near_vsync_jitter_induced_queue",
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
                        cur_anchor_vsync_ts[idx],
                        dec_over_ts[idx],
                        dec_over_ts[idx] + data[idx, 10],
                        dec_over_ts[idx] + data[idx, 10] + data[idx, 11],
                        predicted_render_time[idx],
                        predicted_decode_time[idx],
                        nearest_vsync_ts[idx],
                        nearest_display_slot[idx],
                        available_vsync_ts[idx],
                        actual_render_queue[idx],
                        data[idx, 10],
                        extra_display_ts[idx],
                        frame_buffer_cnt[idx],
                        invoke_present_ts[idx],
                        invoke_present_slot[idx],
                        actual_display_ts[idx],
                        actual_vsync_ts[idx],
                        actual_display_slot[idx],
                        frame_buf_size[idx],
                        frame_buf_change_flag[idx],
                        failed_early_drop_frame[idx],
                        missed_early_drop_frame[idx],
                        predicted_quick_drop_probs[idx],
                        frame_e2e_jitter[idx],
                        dl_jitter_prob[idx],
                        render_jitter_prob[idx],
                        server_jitter_prob[idx],
                        decoder_jitter_prob[idx],
                        display_jitter_prob[idx],
                        e2e_jitter_prob[idx],
                        network_problem_prob[idx],
                        near_vsync_jitter_prob[idx],
                        dl_jitter_prob_thr[idx],
                        render_jitter_prob_thr[idx],
                        server_jitter_prob_thr[idx],
                        decoder_jitter_prob_thr[idx],
                        display_jitter_prob[idx],
                        e2e_jitter_prob_thr[idx],
                        network_problem_prob_thr[idx],
                        near_vsync_jitter_prob_thr[idx],
                        quick_drop_frame_cnt_hist[idx],
                        consecutive_frame_no[idx],
                        display_discarded_flag[idx],
                        display_discarded_flag[idx] - min(1, data[idx, 4] == 4),
                        valid_frame_flag[idx],
                        original_valid_flag[idx],
                        bonus_fps_obtained[idx],
                        cur_bonus_fps_no[idx],
                        cur_valid_frame_no[idx],
                        dec_over_ts[idx],
                        dec_nearest_vsync_ts[idx],
                        dec_queued_frame_cnt[idx],
                        int(frame_jitter_flag[idx]),
                        data[idx, 33],
                        smoothed_frame_pts[idx],
                        smoothed_frame_pts_diff[idx],
                        int(render_jitter_induced_queue[idx]),
                        int(server_jitter_induced_queue[idx]),
                        int(network_big_frame_induced_queue[idx]),
                        int(network_i_frame_induced_queue[idx]),
                        int(network_dl_jitter_induced_queue[idx]),
                        int(network_stall_induced_queue[idx]),
                        int(network_packet_loss_induced_queue[idx]),
                        int(decoder_jitter_induced_queue[idx]),
                        int(display_jitter_induced_queue[idx]),
                        int(near_vsync_jitter_induced_queue[idx]),
                        updated_frame_interval[idx],
                        anchor_frame_ts[idx],
                        anchor_frame_no[idx],
                    ]
                )
                + "\n"
            )

    output_file = open(
        os.path.join(
            save_path,
            "result-%s-periodrop%d_quickdrop%d_maxbuf%d_bonusfps%d_%s.csv"
            % (
                display_mode,
                enable_perio_drop,
                enable_quick_drop,
                max_buf_size,
                bonus_fps_no_thr,
                drop_frame_mode,
            ),
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

    return (
        file_path,
        result,
        [
            data,
            dec_over_ts,
            nearest_vsync_ts,
            nearest_display_slot,
            available_vsync_ts,
            actual_render_queue,
            frame_buffer_cnt,
            invoke_present_ts,
            invoke_present_slot,
            actual_display_ts,
            actual_vsync_ts,
            actual_display_slot,
            frame_buf_size,
            frame_buf_change_flag,
            display_discarded_flag,
            valid_frame_flag,
            original_valid_flag,
            dec_nearest_vsync_ts,
            dec_queued_frame_cnt,
            smoothed_frame_pts,
            smoothed_frame_pts_diff,
            render_jitter_induced_queue,
            network_big_frame_induced_queue,
            network_i_frame_induced_queue,
            network_dl_jitter_induced_queue,
            network_stall_induced_queue,
            network_packet_loss_induced_queue,
            decoder_jitter_induced_queue,
            display_jitter_induced_queue,
            optimized_objective_fps,
            optimized_noloss_fps,
            frame_jitter_flag,
        ],
    )


def cal_mulit_para_result(file_path, print_log=False, save_path="test_data"):
    params = MULTI_PARAMS

    param_no = len(params)
    details = []
    results = []
    for (
        display_mode,
        max_buf_size,
        render_time_predictor,
        enable_perio_drop,
        enable_quick_drop,
        bonus_fps_no_thr,
        drop_frame_mode,
        anchor_frame_extrapolator_mode,
    ) in params:
        file_path, result, detail = cal_single_para_result(
            file_path,
            display_mode=display_mode,
            max_buf_size=max_buf_size,
            render_time_predictor=render_time_predictor,
            enable_perio_drop=enable_perio_drop,
            enable_quick_drop=enable_quick_drop,
            drop_frame_mode=drop_frame_mode,
            bonus_fps_no_thr=bonus_fps_no_thr,
            anchor_frame_extrapolator_mode=anchor_frame_extrapolator_mode,
            print_log=print_log,
        )

        if file_path is None:
            return None, None, None
        details.append(detail)
        results.append(result)

    cur_result = results[0][:3] + results[0][9:13] + results[0][3:6] + [results[0][7]]
    for i in range(param_no):
        cur_result += [results[i][6], results[i][8]]

    output_file = open(os.path.join(save_path, "result-multi_param.csv"), "a")
    output_file.write(
        file_path.replace(",", "_")
        + ", "
        + ", ".join([str(item) for item in cur_result])
        + "\n"
    )
    output_file.close()

    if print_log:
        tot_frame_no = details[0][0].shape[0]
        log_file = open(file_path[:-4] + "_multi_param_sim.csv", "w")
        header = [
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
            "decode_over_ts",
            "original_valid_flag",
            "dec_nearest_vsync_ts",
            "dec_queued_frame_cnt",
            "frame_queue_flag",
            "original_pts",
            "smoothed_pts",
            "smoothed_frame_pts_diff",
            "render_jitter_induced_queue",
            "network_big_frame_induced_queue",
            "network_i_frame_induced_queue",
            "network_dl_jitter_induced_queue",
            "network_stall_induced_queue",
            "network_packet_loss_induced_queue",
            "decoder_jitter_induced_queue",
            "display_jitter_induced_queue",
        ]
        for i in range(param_no):
            header += [
                "frame_buffer_cnt_%d" % i,
                "cur_buf_size_%d" % i,
                "buf_change_flag_%d" % i,
                "actual_render_queue_%d" % i,
                "actual_display_ts_%d" % i,
                "actual_vsync_ts_%d" % i,
                "actual_display_slot_%d" % i,
                "display_discarded_flag_%d" % i,
                "valid_frame_flag_%d" % i,
            ]
        log_file.write(",".join(header) + "\n")

        for idx in range(tot_frame_no):
            values = (
                [idx]
                + details[0][0][idx, :-5].tolist()
                + [
                    details[0][1][idx],
                    details[0][16][idx],
                    details[0][17][idx],
                    details[0][18][idx],
                    int(details[0][18][idx] > 1),
                    details[0][0][idx, 33],
                    details[0][19][idx],
                    details[0][20][idx],
                    int(details[0][21][idx]),
                    int(details[0][22][idx]),
                    int(details[0][23][idx]),
                    int(details[0][24][idx]),
                    int(details[0][25][idx]),
                    int(details[0][26][idx]),
                    int(details[0][27][idx]),
                    int(details[0][28][idx]),
                ]
            )
            for i in range(param_no):
                values += [
                    details[i][6][idx],
                    details[i][12][idx],
                    details[i][13][idx],
                    details[i][5][idx],
                    details[i][9][idx],
                    details[i][10][idx],
                    details[i][11][idx],
                    details[i][14][idx],
                    details[i][15][idx],
                ]
            log_file.write(",".join(str(item) for item in values) + "\n")

    return file_path, cur_result, details


def process_all_data(root_path):
    res1 = []
    res2 = []
    res3 = []
    res4 = []
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

                log_path = os.path.join(session_path, log_name)
                # _, cur_res, _ = cal_single_para_result(log_path, DISPLAY_MODE, MAX_BUF_SIZE, FRAME_INTERVAL,
                #                                             RENDER_TIME_PREIDCTER, ENABLE_PERIO_DROP, ENABLE_QUICK_DROP,
                #                                             DROP_FRAME_MODE, BONUS_FPS_NO_THR)
                _, cur_res, _ = cal_mulit_para_result(log_path)
                if cur_res is not None:
                    res1.append(cur_res[2])
                    res2.append(cur_res[3])
                    res3.append(cur_res[4])
                    res4.append(cur_res[5])

    print(np.mean(res1), np.min(res1), np.max(res1))
    print(np.mean(res2), np.min(res2), np.max(res2))
    print(np.mean(res3), np.min(res3), np.max(res3))
    print(np.mean(res4), np.min(res4), np.max(res4))


def process_all_data_multithread(
    root_path, multi_param=False, num_proc=32, save_path="test_data"
):
    p = multiprocessing.Pool(processes=num_proc)
    result = Result()
    # 'naiveVsync', MAX_BUF_SIZE, RENDER_TIME_PREIDCTER, 1, 1, BONUS_FPS_NO_THR
    if multi_param:
        for (
            display_mode,
            max_buf_size,
            render_predictor,
            enable_perio_drop,
            enable_quick_drop,
            bonus_fps_no_thr,
            drop_frame_mode,
            anchor_frame_extrapolator_mode,
        ) in MULTI_PARAMS:
            output_file = open(
                os.path.join(
                    save_path,
                    "result-%s-periodrop%d_quickdrop%d_maxbuf%d_bonusfps%d_%s_anchor%d.csv"
                    % (
                        display_mode,
                        enable_perio_drop,
                        enable_quick_drop,
                        max_buf_size,
                        bonus_fps_no_thr,
                        drop_frame_mode,
                        anchor_frame_extrapolator_mode,
                    ),
                ),
                "a",
            )
            postfix = "-%s-periodrop%d_quickdrop%d_bonusfps%d_%s" % (
                display_mode,
                enable_perio_drop,
                enable_quick_drop,
                bonus_fps_no_thr,
                drop_frame_mode,
            )
            output_file.write(
                "file_name,server_optim_enabled,client_optim_enabled,client_vsync_enabled,max_fps,min_fps,origin_fps,optimized_fps%s,optimized_objective_fps%s,optimized_noloss_fps%s,origin_render_queue,optimized_render_queue%s,extra_display_ts%s,optimized_total_render_queue%s,avg_dec_time,avg_dec_tot_time,avg_render_time,avg_proc_time,tot_frame_no,tot_queue_cnt%s,network_big_frame_induced_queue%s,network_i_frame_induced_queue%s,network_dl_jitter_induced_queue%s,network_stall_induced_queue%s,network_packet_loss_induced_queue%s,render_jitter_induced_queue%s,server_jitter_induced_queue%s,decoder_jitter_induced_queue%s,display_jitter_induced_queue%s,near_vsync_jitter_induced_queue%s,netts_over100_cnt,displayts_over12_cnt,large_renderinterval_cnt,quick_drop_failed_cnt%s,quick_drop_fail_ratio%s,quick_drop_missed_cnt%s,quick_drop_miss_ratio%s,quick_drop_total_cnt%s,exploitable_network_big_frame_induced_queue%s,exploitable_network_i_frame_induced_queue%s,exploitable_network_dl_jitter_induced_queue%s,exploitable_network_stall_induced_queue%s,exploitable_network_packet_loss_induced_queue%s,exploitable_render_jitter_induced_queue%s,exploitable_server_jitter_induced_queue%s,exploitable_decoder_jitter_induced_queue%s,exploitable_display_jitter_induced_queue%s,exploitable_near_vsync_jitter_induced_queue%s,biggest_render_jitter_queue%s,biggest_network_jitter_queue%s,biggest_decode_jitter_queue%s,biggest_display_jitter_queue%s\n"
                % (
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                    postfix,
                )
            )
            output_file.close()
    else:
        output_file = open(
            os.path.join(
                save_path,
                "result-%s-periodrop%d_quickdrop%d_maxbuf%d_bonusfps%d_%s.csv"
                % (
                    DISPLAY_MODE,
                    ENABLE_PERIO_DROP,
                    ENABLE_QUICK_DROP,
                    MAX_BUF_SIZE,
                    BONUS_FPS_NO_THR,
                    DROP_FRAME_MODE,
                ),
            ),
            "a",
        )
        postfix = "-%s-periodrop%d_quickdrop%d_bonusfps%d_%s" % (
            DISPLAY_MODE,
            ENABLE_PERIO_DROP,
            ENABLE_QUICK_DROP,
            BONUS_FPS_NO_THR,
            DROP_FRAME_MODE,
        )
        output_file.write(
            "file_name,server_optim_enabled,client_optim_enabled,client_vsync_enabled,max_fps,min_fps,origin_fps,optimized_fps%s,optimized_objective_fps%s,optimized_noloss_fps%s,origin_render_queue,optimized_render_queue%s,extra_display_ts%s,optimized_total_render_queue%s,avg_dec_time,avg_dec_tot_time,avg_render_time,avg_proc_time,tot_frame_no,tot_queue_cnt%s,network_big_frame_induced_queue%s,network_i_frame_induced_queue%s,network_dl_jitter_induced_queue%s,network_stall_induced_queue%s,network_packet_loss_induced_queue%s,render_jitter_induced_queue%s,server_jitter_induced_queue%s,decoder_jitter_induced_queue%s,display_jitter_induced_queue%s,near_vsync_jitter_induced_queue%s,netts_over100_cnt,displayts_over12_cnt,large_renderinterval_cnt,quick_drop_failed_cnt%s,quick_drop_fail_ratio%s,quick_drop_missed_cnt%s,quick_drop_miss_ratio%s,quick_drop_total_cnt%s,exploitable_network_big_frame_induced_queue%s,exploitable_network_i_frame_induced_queue%s,exploitable_network_dl_jitter_induced_queue%s,exploitable_network_stall_induced_queue%s,exploitable_network_packet_loss_induced_queue%s,exploitable_render_jitter_induced_queue%s,exploitable_server_jitter_induced_queue%s,exploitable_decoder_jitter_induced_queue%s,exploitable_display_jitter_induced_queue%s,exploitable_near_vsync_jitter_induced_queue%s,biggest_render_jitter_queue%s,biggest_network_jitter_queue%s,biggest_decode_jitter_queue%s,biggest_display_jitter_queue%s\n"
            % (
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
            )
        )
        output_file.close()

    if multi_param:
        output_file = open(os.path.join(save_path, "result-multi_param.csv"), "a")
        header = "file_name,server_optim_enabled,client_optim_enabled,client_vsync_enabled,avg_dec_time,avg_dec_tot_time,avg_render_time,avg_proc_time,max_fps,min_fps,origin_fps,origin_render_queue"
        param_no = len(MULTI_PARAMS)
        for i in range(param_no):
            header += ",valid_fps_%d,render_queue_%d" % (i + 1, i + 1)
        output_file.write(header + "\n")
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
                if multi_param:
                    p.apply_async(
                        cal_mulit_para_result,
                        args=(log_path,),
                        callback=result.update_result,
                    )
                else:
                    p.apply_async(
                        cal_single_para_result,
                        args=(
                            log_path,
                            DISPLAY_MODE,
                            MAX_BUF_SIZE,
                            FRAME_INTERVAL,
                            RENDER_TIME_PREIDCTER,
                            ENABLE_PERIO_DROP,
                            ENABLE_QUICK_DROP,
                            DROP_FRAME_MODE,
                            BONUS_FPS_NO_THR,
                            ANCHOR_FRAME_EXTRAPOLATOR_MODE,
                        ),
                        callback=result.update_result,
                    )

    p.close()
    p.join()
    print(np.mean(result.res1), np.min(result.res1), np.max(result.res1))
    print(np.mean(result.res2), np.min(result.res2), np.max(result.res2))
    print(np.mean(result.res3), np.min(result.res3), np.max(result.res3))
    print(np.mean(result.res4), np.min(result.res4), np.max(result.res4))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        p = multiprocessing.Pool(processes=16)
        target_path = "/mydata/clwwwu/frame_log/sample_gain3.5"
        output_file = open(
            os.path.join(
                target_path,
                "result-%s-periodrop%d_quickdrop%d_maxbuf%d_bonusfps%d_%s.csv"
                % (
                    DISPLAY_MODE,
                    ENABLE_PERIO_DROP,
                    ENABLE_QUICK_DROP,
                    MAX_BUF_SIZE,
                    BONUS_FPS_NO_THR,
                    DROP_FRAME_MODE,
                ),
            ),
            "a",
        )
        postfix = "-%s-periodrop%d_quickdrop%d_bonusfps%d_%s" % (
            DISPLAY_MODE,
            ENABLE_PERIO_DROP,
            ENABLE_QUICK_DROP,
            BONUS_FPS_NO_THR,
            DROP_FRAME_MODE,
        )
        output_file.write(
            "file_name,server_optim_enabled,client_optim_enabled,client_vsync_enabled,max_fps,min_fps,origin_fps,optimized_fps%s,origin_render_queue,optimized_render_queue%s,extra_display_ts%s,optimized_total_render_queue%s,avg_dec_time,avg_dec_tot_time,avg_render_time,avg_proc_time,tot_frame_no,tot_queue_cnt%s,network_big_frame_induced_queue%s,network_i_frame_induced_queue%s,network_dl_jitter_induced_queue%s,network_stall_induced_queue%s,network_packet_loss_induced_queue%s,render_jitter_induced_queue%s,server_jitter_induced_queue%s,decoder_jitter_induced_queue%s,display_jitter_induced_queue%s,near_vsync_jitter_induced_queue%s,bonus_fps%s,netts_over100_cnt,displayts_over12_cnt,large_renderinterval_cnt,quick_drop_failed_cnt%s,quick_drop_fail_ratio%s,quick_drop_missed_cnt%s,quick_drop_miss_ratio%s,quick_drop_total_cnt%s,exploitable_network_big_frame_induced_queue%s,exploitable_network_i_frame_induced_queue%s,exploitable_network_dl_jitter_induced_queue%s,exploitable_network_stall_induced_queue%s,exploitable_network_packet_loss_induced_queue%s,exploitable_render_jitter_induced_queue%s,exploitable_server_jitter_induced_queue%s,exploitable_decoder_jitter_induced_queue%s,exploitable_display_jitter_induced_queue%s,exploitable_near_vsync_jitter_induced_queue%s,biggest_render_jitter_queue%s,biggest_network_jitter_queue%s,biggest_decode_jitter_queue%s,biggest_display_jitter_queue%s\n"
            % (
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
                postfix,
            )
        )
        output_file.close()

        for line in open(os.path.join(target_path, "unit_test.txt")).readlines():
            if len(line) <= 1:
                break
            input_path = os.path.join(target_path, line.strip())
            # print(input_path)
            p.apply_async(
                cal_single_para_result,
                args=(
                    input_path,
                    DISPLAY_MODE,
                    MAX_BUF_SIZE,
                    FRAME_INTERVAL,
                    RENDER_TIME_PREIDCTER,
                    ENABLE_PERIO_DROP,
                    ENABLE_QUICK_DROP,
                    DROP_FRAME_MODE,
                    BONUS_FPS_NO_THR,
                    ANCHOR_FRAME_EXTRAPOLATOR_MODE,
                    target_path,
                    True,
                ),
            )
            # cal_mulit_para_result(input_path, print_log=PRINT_LOG)
            # cal_single_para_result(input_path, display_mode=DISPLAY_MODE, max_buf_size=MAX_BUF_SIZE, render_time_predictor=RENDER_TIME_PREIDCTER, enable_perio_drop=ENABLE_PERIO_DROP,
            #                        enable_quick_drop=ENABLE_QUICK_DROP, bonus_fps_no_thr=BONUS_FPS_NO_THR, save_path=target_path, print_log=PRINT_LOG)

        p.close()
        p.join()

    else:
        input_path = sys.argv[1]
        if os.path.isdir(input_path):

            # process_all_data(sys.argv[1])
            if len(sys.argv) == 3:
                process_all_data_multithread(sys.argv[1], int(sys.argv[2]))
            else:
                process_all_data_multithread(sys.argv[1])
        elif os.path.isfile(input_path):
            cal_single_para_result(
                input_path,
                display_mode=DISPLAY_MODE,
                max_buf_size=MAX_BUF_SIZE,
                frame_interval=FRAME_INTERVAL,
                render_time_predictor=RENDER_TIME_PREIDCTER,
                enable_perio_drop=ENABLE_PERIO_DROP,
                enable_quick_drop=ENABLE_QUICK_DROP,
                drop_frame_mode=DROP_FRAME_MODE,
                bonus_fps_no_thr=BONUS_FPS_NO_THR,
                print_log=PRINT_LOG,
                print_debug_log=PRINT_DEBUG_LOG,
            )
            # cal_mulit_para_result(input_path, print_log=PRINT_LOG)
        else:
            pass
