import json
import pandas as pd
import numpy as np


def baseline_loader(path="../log/baseline.log"):
    result = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "(None, None)":
                continue

            left = line.find("'")
            right = line.rfind("'")
            file_name = line[left + 1 : right].replace("../../data/", "")

            left = line.find("[")
            right = line.rfind("]")
            values = json.loads(line[left : right + 1])

            result[file_name] = [values[2], values[7], values[10]]

    return result


def log_loader(path):
    result = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            result[line[0]] = [float(x) for x in line[1:]]

    return result


def trace_loader(path, start_idx=300, filter_incomp=True, len_limit=-5):
    df = pd.read_csv(path)
    data = df.iloc[
        start_idx:,
        [
            0,
            1,
            3,
            4,
            2,
            24,
            25,
            26,
            27,
            28,
            39,
            29,
            23,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
        ],
    ]

    # Data format
    # 0  "render_index", "frame_index", "frame_type", "size", "loss_type",
    # 5  "receive_timestamp", "receive_and_unpack",
    # 7  "decoder_outside_queue", "decoder_insided_queue", "decode", "render_queue", "display",
    # 12 "capture_timestamp", "send_ts", "net_ts", "proc_ts", "tot_ts",
    # 17 "basic_frame_ts", "ul_jitter", "dl_jitter",
    # 20 "expected_recv_ts", "expected_proc_time", "nearest_display_ts",
    # 23 "expected_display_ts", "actual_display_ts", "vsync_diff",
    # 26 "jitter_buf_size", "server_optim_enabled", "client_optim_enabled",

    data = data.to_numpy()
    if filter_incomp is True and np.all(data[:2000, 7:9].sum(-1) > 0):
        return None

    if filter_incomp is True:
        data = data[np.where(data[:, 4] == 0)[0], :]

    try:
        if np.any(data < -9999):
            return None
    except:
        print(path)

    if len_limit > 0:
        len_limit = min(len_limit, data.shape[0] - 5)

    return data[:len_limit, :]


# 0 frame_render_id,frame_index,loss_type,frame_type,size,
# 5 encoding_rate,cc_rate,smoothrate,width,height,sqoe,ori_sqoe,target_sqoe,recomm_bitrate,actual_bitrate,scene_change,encoding_fps,satd,qp,mvx,mvy,intra_mb,inter_mb,
# 23 proxy_recv_ts,client_recv_ts, unpack,sdk_outside_queue,sdk_inside_queue,decode,render,
# 30 send_time,net_time,proc_time,tot_time,basic_net_time,ul_jitter,dl_jitter,
# 37 cur_mouse_ev_cnt,cur_keyboard_ev_cnt,render_cache,cur_cgs_pause_cnt,pts,ets,dts,
# 44 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,
# 49 vsync_diff,jitter_buf_size,server_optim_enabled,client_optim_enabled,client_vsync_enabled
# 54 Mrts0ToRtsOffset, packet_lossed_perK
def formated_e2e_trace_loader(
    data_path, start_idx=300, filter_incomp=True, len_limit=-5
):
    df = pd.read_csv(data_path)  # , header=None)

    data = df.iloc[
        start_idx:,
        [
            0,
            1,
            3,
            4,
            2,
            24,
            25,
            26,
            27,
            28,
            39,
            29,
            23,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            41,
            42,
            43,
            54,
            55,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
        ],
    ]
    #     0 'render_index', 'frame_index', 'frame_type', 'size', 'loss_type',
    #     5 'receive_timestamp', 'receive_and_unpack',
    #     7 'decoder_outside_queue', 'decoder_insided_queue', 'decode', 'render_queue', 'display',
    #     12 'capture_timestamp', 'send_ts', 'net_ts', 'proc_ts', 'tot_ts',
    #     17 'basic_net_ts', 'ul_jitter', 'dl_jitter'
    #     20 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,vsync_diff
    #     26 jitter_buf_size,server_optim_enabled,client_optim_enabled
    #     29 pts, ets, dts, Mrts0ToRtsOffset, packet_lossed_perK
    data = data.to_numpy()
    if filter_incomp is True and np.all(data[:2000, 7:9].sum(-1) > 0):
        return None, None

    if filter_incomp is True:
        data = data[np.where(data[:, 4] == 0)[0], :]

    if np.any(data < -9999):
        return None, None

    if len_limit > 0:
        len_limit = min(len_limit, data.shape[0] - 5)

    return data[:len_limit, :], [
        df.columns.to_list()[idx]
        for idx in [
            0,
            1,
            3,
            4,
            2,
            24,
            25,
            26,
            27,
            28,
            39,
            29,
            23,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            41,
            42,
            43,
            54,
            55,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
        ]
    ]


if __name__ == "__main__":
    pass
