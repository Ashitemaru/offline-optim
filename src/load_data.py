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


# 0 frame_render_id,frame_index,loss_type,frame_type,size,
# 5 encoding_rate,cc_rate,smoothrate,width,height,sqoe,ori_sqoe,target_sqoe,recomm_bitrate,actual_bitrate,scene_change,encoding_fps,satd,qp,mvx,mvy,intra_mb,inter_mb,
# 23 proxy_recv_ts,client_recv_ts, unpack,sdk_outside_queue,sdk_inside_queue,decode,render,
# 30 send_time,net_time,proc_time,tot_time,basic_net_time,ul_jitter,dl_jitter,
# 37 cur_mouse_ev_cnt,cur_keyboard_ev_cnt,render_cache,cur_cgs_pause_cnt,pts,ets,dts,
# 44 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,
# 49 vsync_diff,jitter_buf_size,server_optim_enabled,client_optim_enabled,client_vsync_enabled
# 54 Mrts0ToRtsOffset, packet_lossed_perK
def load_formated_e2e_framerate_log_with_netinfo(
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
            40,
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
    #     34 encoding_rate, cc_rate, smoothrate, width, height, sqoe, ori_sqoe, target_sqoe,
    #     42 recomm_bitrate, actual_bitrate, scene_change, encoding_fps,
    #     46 satd, qp, mvx, mvy, intra_mb, inter_mb, cur_cgs_pause_cnt
    data = data.to_numpy()
    if filter_incomp is True and np.all(data[:2000, 7:9].sum(-1) > 0):
        return None, None

    # if filter_incomp is True:
    #     idx = np.where(data[:, 4]==0)[0]
    #     data = data[idx, :]

    if len_limit > 0:
        len_limit = min(len_limit, data.shape[0] - 5)

    return data[:len_limit, :], None


def load_formated_e2e_framerate_log_with_netinfo_and_flag(
    data_path, start_idx=0, filter_incomp=True, len_limit=0
):
    """
    csv format
    """
    df = pd.read_csv(data_path)  # , header=None)

    data = df.iloc[start_idx:]
    #     0 'render_index', 'frame_index', 'frame_type', 'size', 'loss_type',
    #     5 'receive_timestamp', 'receive_and_unpack',
    #     7 'decoder_outside_queue', 'decoder_insided_queue', 'decode', 'render_queue', 'display',
    #     12 'capture_timestamp', 'send_ts', 'net_ts', 'proc_ts', 'tot_ts',
    #     17 'basic_net_ts', 'ul_jitter', 'dl_jitter'
    #     20 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,vsync_diff
    #     26 jitter_buf_size,server_optim_enabled,client_optim_enabled
    #     29 pts, ets, dts, Mrts0ToRtsOffset, packet_lossed_perK
    #     34 encoding_rate, cc_rate, smoothrate, width, height, sqoe, ori_sqoe, target_sqoe,
    #     42 recomm_bitrate, actual_bitrate, scene_change, encoding_fps,
    #     46 satd, qp, mvx, mvy, intra_mb, inter_mb
    #     52 delay_time,valid_frame_flag
    data = data.to_numpy()
    if filter_incomp is True and np.all(data[:2000, 7:9].sum(-1) > 0):
        return None, None

    if filter_incomp is True:
        data = data[np.where(data[:, 4] == 0)[0], :]

    # if np.any(data < -9999):
    #     return None, None

    if len_limit > 0:
        len_limit = min(len_limit, data.shape[0] - 5)
        data = data[:len_limit, :]

    del df

    return data, None


# 0 cgs_frame_id,frame_index,loss_type,frame_type,size,
# 5 encoding_rate,cc_rate,smoothrate,width,height,sqoe,ori_sqoe,target_sqoe,
# 13 recomm_bitrate,actual_bitrate,scene_change,encoding_fps,satd,qp,mvx,mvy,intra_mb,inter_mb,
# 23 proxy_recv_ts,client_recv_ts, unpack,sdk_outside_queue,sdk_inside_queue,decode,render,
# 30 send_time,net_time,proc_time,tot_time,basic_net_time,ul_jitter,dl_jitter,
# 37 cur_mouse_ev_cnt,cur_keyboard_ev_cnt,render_cache,cur_cgs_pause_cnt,pts,ets,dts,
# 44 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,
# 49 vsync_diff,present_timer_offset,jitter_buf_size,server_optim_enabled,client_optim_enabled,client_vsync_enabled
# 55 Mrts0ToRtsOffset, packet_lossed_perK
# 57 jit_flag,send_type,send_chs,kernel_ack_time,mctp_ack_time,echo_ack_time
# 63 basic_frame_id,vsync_max_buffer,client_vsync_ts,cgs_send_ts,
# 67 frame_recv_time,frame_send_delay,ch_delay_on_rcv
# 70 min_rtt,ngx_buf_size,mirror_fly_size,recv_rate,next_to_pause_cnt,ch_index,net_type,sub_net_type,
# 78 frame_stat_ch,first_send_rtt,last_send_rtt,valid_rtt,ch_ack_delay,ch_send_delay


def load_detailed_framerate_log(
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
            67,
            68,
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
            53,
            54,
            41,
            42,
            43,
            66,
            55,
            56,
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
            40,
            65,
            70,
            79,
            80,
            81,
            82,
            83,
            60,
        ],
    ]
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
    #     58 client_vsync_ts, min_rtt, first_send_rtt,last_send_rtt,valid_rtt,ch_ack_delay,ch_send_delay
    data = data.to_numpy()

    sorted_idx = np.argsort(data[:, 0])
    data = data[sorted_idx]

    if filter_incomp is True:
        data = data[np.where(data[:, 5] != 0)[0], :]

    if len_limit > 0:
        len_limit = min(len_limit, data.shape[0] - 5)

    return data[:len_limit, :], None


def load_detailed_optimal_framerate_log(data_path, start_idx=300):
    """
    csv format
    """
    df = pd.read_csv(data_path)  # , header=None)

    data = df.iloc[start_idx:, 1:]
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
    # 'cur_anchor_vsync_ts', 'decode_over_ts', 'client_invoke_ts', 'client_display_ts', 'predicted_render_time', 'predicted_decode_time', 'nearest_vsync_ts',
    # 'nearest_display_slot', 'available_vsync_ts', 'actual_render_queue', 'client_render_queue', 'extra_display_ts',
    # 'frame_buffer_cnt', 'invoke_present_ts', 'invoke_present_slot',
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

    # data = data.to_numpy()
    # sorted_idx = np.argsort(data[:, 0])
    # data = data[sorted_idx]

    return data, None
