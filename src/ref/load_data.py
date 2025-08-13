import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

DEVICE_TPYE = ["Unknown", "Desktop", "Laptop", "Phone", "Pad", "STB", "TV"]

SYSTEM_TYPE = ["Windows", "iOS", "MacOS", "AndroidPhone", "AndroidTV"]

NETWORK_TYPE = ["Unknown", "Mobile", "ETH", "WiFi", "Other"]


# frame_render_id,frame_index,loss_type,frame_type,size,
# encoding_rate,cc_rate,smoothrate,width,height,sqoe,ori_sqoe,target_sqoe,recomm_bitrate,actual_bitrate,scene_change,encoding_fps,satd,qp,mvx,mvy,intra_mb,inter_mb,
# 23 proxy_recv_ts,client_recv_ts, unpack,sdk_outside_queue,sdk_inside_queue,decode,render,
# 30 send_time,net_time,proc_time,tot_time,basic_net_time,ul_jitter,dl_jitter,
# cur_mouse_ev_cnt,cur_keyboard_ev_cnt
def load_formated_e2e_log(data_path, start_idx=300, filter_incomp=True, len_limit=-5):
    """
    csv format
    """
    df = pd.read_csv(data_path)  # , header=None)
    data_summary = data_path[-11:-4].split(",")
    device_type = int(data_summary[0])
    system_type = int(data_summary[1])
    network_type = int(data_summary[2])
    subnet_type = int(data_summary[3])

    if system_type == 0:
        data = df.iloc[
            start_idx:,
            [0, 1, 3, 4, 2, 24, 25, 26, 28, 27, 29, 23, 30, 31, 32, 33, 34, 35, 36],
        ]
        data.insert(8, "placeholder", 0)
    else:
        data = df.iloc[
            start_idx:,
            [0, 1, 3, 4, 2, 24, 25, 26, 27, 28, 29, 23, 30, 31, 32, 33, 34, 35, 36],
        ]  # 7, 8, 9, 10, 11, 12, 6, 13, 14, 15, 16, 17, 18, 19]]
        data.insert(10, "placeholder", 0)

    # data.set_axis(['render_index', 'frame_index', 'frame_type', 'size', 'loss_type',
    #     'receive_timestamp', 'receive_and_unpack',
    #     'decoder_outside_queue', 'decoder_insided_queue', 'decode', 'render_queue', 'display',
    #     'capture_timestamp', 'send_ts', 'net_ts', 'proc_ts', 'tot_ts',
    #     'basic_frame_ts', 'ul_jitter', 'dl_jitter'], axis=1, inplace=True)

    # data.insert(20, 'expected_recv_ts', 0)
    # data.insert(21, 'recv_ts_shift', 0)
    # data.insert(22, 'expected_proc_time', 0)
    # data.insert(23, 'jitter_buf_size', 0)
    # data.insert(24, 'framerate_optim_enabled', 0)

    data = data.to_numpy()
    if filter_incomp is True and np.all(data[:2000, 7:9].sum(-1) > 0):
        return None, None

    if filter_incomp is True:
        data = data[np.where(data[:, 4] == 0)[0], :]

    if np.any(data < -9999):
        return None, None

    if len_limit > 0:
        len_limit = min(len_limit, data.shape[0] - 5)

    return data[:len_limit, :], (device_type, system_type, network_type, subnet_type)


# frame_render_id,frame_index, cgs_frame_id, loss_type,frame_type,size,
# encoding_rate,cc_rate,smoothrate,width,height,sqoe,ori_sqoe,target_sqoe,recomm_bitrate,actual_bitrate,scene_change,encoding_fps,satd,qp,mvx,mvy,intra_mb,inter_mb,
# 24 proxy_recv_ts,client_recv_ts, unpack,sdk_outside_queue,sdk_inside_queue,decode, render_cache, render,
# 32 send_time,net_time,proc_time,tot_time,basic_net_time,ul_jitter,dl_jitter,
# 39 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,vsync_diff,jitter_buf_size,server_optim_enabled,client_optim_enabled


# 0 frame_render_id,frame_index,loss_type,frame_type,size,
# encoding_rate,cc_rate,smoothrate,width,height,sqoe,ori_sqoe,target_sqoe,recomm_bitrate,actual_bitrate,scene_change,encoding_fps,satd,qp,mvx,mvy,intra_mb,inter_mb,
# 23 proxy_recv_ts,client_recv_ts, unpack,sdk_outside_queue,sdk_inside_queue,decode,render,
# 30 send_time,net_time,proc_time,tot_time,basic_net_time,ul_jitter,dl_jitter,
# 37 cur_mouse_ev_cnt,cur_keyboard_ev_cnt,render_cache,cur_cgs_pause_cnt,pts,ets,dts,
# 44 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,
# 49 vsync_diff,jitter_buf_size,server_optim_enabled,client_optim_enabled,client_vsync_enabled
def load_formated_e2e_framerate_log(
    data_path, start_idx=300, filter_incomp=True, len_limit=-5
):
    """
    csv format
    """
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
        ],
    ]
    # data.set_axis(['render_index', 'frame_index', 'frame_type', 'size', 'loss_type',
    #     5 'receive_timestamp', 'receive_and_unpack',
    #     7 'decoder_outside_queue', 'decoder_insided_queue', 'decode', 'render_queue', 'display',
    #     12 'capture_timestamp', 'send_ts', 'net_ts', 'proc_ts', 'tot_ts',
    #     17 'basic_frame_ts', 'ul_jitter', 'dl_jitter'], axis=1, inplace=True)
    #     20 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,vsync_diff
    #     26 jitter_buf_size,server_optim_enabled,client_optim_enabled
    data = data.to_numpy()
    if filter_incomp is True and np.all(data[:2000, 7:9].sum(-1) > 0):
        return None, None

    if filter_incomp is True:
        data = data[np.where(data[:, 4] == 0)[0], :]

    if np.any(data < -9999):
        return None, None

    if len_limit > 0:
        len_limit = min(len_limit, data.shape[0] - 5)

    return data[:len_limit, :], None
