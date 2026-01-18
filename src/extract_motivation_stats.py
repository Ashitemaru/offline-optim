"""
从云游戏trace中提取motivation分析所需的统计数据：
1. 帧的输出间隔分布
2. 网络传输耗时分布
3. 解码时长分布

同时解析网络环境（4G/5G/WiFi等）
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

# 设备/系统/网络类型定义（与load_data.py一致）
DEVICE_TYPE = ["Unknown", "Desktop", "Laptop", "Phone", "Pad", "STB", "TV"]
SYSTEM_TYPE = ["Windows", "iOS", "MacOS", "AndroidPhone", "AndroidTV"]
NETWORK_TYPE = ["Unknown", "Mobile", "ETH", "WiFi", "Other"]

# 子网类型定义（推测）
# 对于Mobile: 1=4G, 2=5G, 3=Other
# 对于WiFi: 1=2.4G, 2=5G, 3=Other
SUBNET_TYPE_MOBILE = ["Unknown", "4G", "5G", "Other"]
SUBNET_TYPE_WIFI = ["Unknown", "2.4G", "5G", "Other"]


def parse_network_info_from_filename(file_path: str) -> dict:
    """
    从文件名中解析网络信息
    文件名格式: xxx_device,system,network,subnet.csv
    例如: good_2000622580_2,0,3,1.csv
    """
    base_name = os.path.basename(file_path)
    
    try:
        # 提取最后的 device,system,network,subnet 部分
        parts = base_name.replace('.csv', '').split('_')
        info_str = parts[-1]  # "2,0,3,1"
        info_parts = info_str.split(',')
        
        if len(info_parts) >= 4:
            device_type = int(info_parts[0])
            system_type = int(info_parts[1])
            network_type = int(info_parts[2])
            subnet_type = int(info_parts[3])
            
            # 解析网络类型字符串
            network_str = NETWORK_TYPE[network_type] if network_type < len(NETWORK_TYPE) else "Unknown"
            
            # 解析子网类型字符串
            if network_type == 1:  # Mobile
                subnet_str = SUBNET_TYPE_MOBILE[subnet_type] if subnet_type < len(SUBNET_TYPE_MOBILE) else "Unknown"
            elif network_type == 3:  # WiFi
                subnet_str = SUBNET_TYPE_WIFI[subnet_type] if subnet_type < len(SUBNET_TYPE_WIFI) else "Unknown"
            else:
                subnet_str = str(subnet_type)
            
            return {
                'device_type': device_type,
                'device_str': DEVICE_TYPE[device_type] if device_type < len(DEVICE_TYPE) else "Unknown",
                'system_type': system_type,
                'system_str': SYSTEM_TYPE[system_type] if system_type < len(SYSTEM_TYPE) else "Unknown",
                'network_type': network_type,
                'network_str': network_str,
                'subnet_type': subnet_type,
                'subnet_str': subnet_str,
                'full_network': f"{network_str}_{subnet_str}"
            }
    except (ValueError, IndexError) as e:
        pass
    
    return {
        'device_type': 0, 'device_str': "Unknown",
        'system_type': 0, 'system_str': "Unknown",
        'network_type': 0, 'network_str': "Unknown",
        'subnet_type': 0, 'subnet_str': "Unknown",
        'full_network': "Unknown"
    }


def load_trace_data(file_path: str, start_idx: int = 300, len_limit: int = -5) -> tuple:
    """
    加载trace数据，返回numpy数组和网络信息
    
    列索引（来自load_data.load_detailed_framerate_log）:
    0 'render_index', 'frame_index', 'frame_type', 'size', 'loss_type',
    5 'client_receive_ts', 'receive_and_unpack', 'decoder_outside_queue', 'decoder_insided_queue', 'decode', 'render_queue', 'display',
    12 'proxy_recv_ts', 'proxy_recv_time', 'proxy_send_delay', 'send_time',
    16 'net_time', 'proc_time', 'tot_time',
    19 'basic_net_ts', 'ul_jitter', 'dl_jitter'
    22 expected_recv_ts,expected_proc_time,nearest_display_ts,expected_display_ts,actual_display_ts,
    27 vsync_diff,present_timer_offset,jitter_buf_size,server_optim_enabled,client_optim_enabled,client_vsync_enabled,
    33 pts, ets, dts, sts, Mrts0ToRtsOffset, packet_lossed_perK
    39 encoding_rate, cc_rate, smoothrate, width, height, sqoe, ori_sqoe, target_sqoe,
    47 recomm_bitrate, actual_bitrate, scene_change, encoding_fps, satd, qp, mvx, mvy, intra_mb, inter_mb, cur_cgs_pause_cnt
    58 client_vsync_ts, min_rtt, first_send_rtt,last_send_rtt,valid_rtt,ch_ack_delay,ch_send_delay
    """
    try:
        df = pd.read_csv(file_path)
        
        # 选择需要的列（与load_detailed_framerate_log一致）
        col_indices = [
            0, 1, 3, 4, 2,  # render_index, frame_index, frame_type, size, loss_type
            24, 25, 26, 27, 28, 39, 29,  # client_recv_ts, unpack, sdk_outside, sdk_inside, decode, render_cache, render
            23, 67, 68, 30,  # proxy_recv_ts, proxy_recv_time, proxy_send_delay, send_time
            31, 32, 33, 34, 35, 36,  # net_time, proc_time, tot_time, basic_net_ts, ul_jitter, dl_jitter
            44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,  # expected_recv_ts到client_vsync_enabled
            41, 42, 43, 66, 55, 56,  # pts, ets, dts, sts, Mrts0ToRtsOffset, packet_lossed_perK
            5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 40,  # encoding相关
            65, 70, 79, 80, 81, 82, 83, 60,  # client_vsync_ts, min_rtt, rtt相关
        ]
        
        # 检查列数是否足够
        if df.shape[1] <= max(col_indices):
            # 使用简化版本
            data = df.iloc[start_idx:].to_numpy()
        else:
            data = df.iloc[start_idx:, col_indices].to_numpy()
        
        # 按render_index排序
        sorted_idx = np.argsort(data[:, 0])
        data = data[sorted_idx]
        
        # 过滤无效帧
        data = data[np.where(data[:, 5] != 0)[0], :]
        
        if len_limit > 0:
            len_limit = min(len_limit, data.shape[0] - 5)
            data = data[:len_limit, :]
        
        network_info = parse_network_info_from_filename(file_path)
        
        return data, network_info
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def extract_frame_stats(data: np.ndarray) -> dict:
    """
    从trace数据中提取帧统计信息
    
    返回：
    - frame_intervals: 帧输出间隔（基于pts）
    - network_times: 网络传输耗时
    - decode_times: 解码时长
    """
    if data is None or data.shape[0] < 10:
        return None
    
    # 列索引（基于load_detailed_framerate_log的输出格式）
    COL_LOSS_TYPE = 4
    COL_DECODE = 9
    COL_NET_TIME = 16
    COL_PTS = 33
    
    # 只分析有效帧（loss_type == 0）
    valid_mask = data[:, COL_LOSS_TYPE] == 0
    valid_data = data[valid_mask]
    
    if valid_data.shape[0] < 10:
        return None
    
    # 1. 帧输出间隔（相邻帧的pts差值）
    pts_values = valid_data[:, COL_PTS]
    frame_intervals = np.diff(pts_values)
    # 过滤异常值（<0或>500ms）
    frame_intervals = frame_intervals[(frame_intervals > 0) & (frame_intervals < 500)]
    
    # 2. 网络传输耗时
    network_times = valid_data[:, COL_NET_TIME]
    # 过滤异常值
    network_times = network_times[(network_times > 0) & (network_times < 1000)]
    
    # 3. 解码时长
    decode_times = valid_data[:, COL_DECODE]
    # 过滤异常值
    decode_times = decode_times[(decode_times >= 0) & (decode_times < 100)]
    
    return {
        'frame_intervals': frame_intervals,
        'network_times': network_times,
        'decode_times': decode_times,
        'num_frames': valid_data.shape[0]
    }


def process_all_traces(root_path: str, output_path: str = None, len_limit: int = 72000) -> dict:
    """
    处理所有trace文件，提取统计数据
    
    参数：
    - root_path: 数据根目录（包含2024-xx-xx子目录）
    - output_path: 输出pickle文件路径
    - len_limit: 每个trace的最大帧数（默认72000，约20分钟@60fps）
    
    返回：
    - 按网络类型分组的统计数据
    """
    results = {
        'all': {
            'frame_intervals': [],
            'network_times': [],
            'decode_times': [],
            'file_count': 0,
            'total_frames': 0
        }
    }
    
    # 按网络类型分组
    network_groups = ['Mobile_4G', 'Mobile_5G', 'WiFi_2.4G', 'WiFi_5G', 'ETH', 'Unknown']
    for group in network_groups:
        results[group] = {
            'frame_intervals': [],
            'network_times': [],
            'decode_times': [],
            'file_count': 0,
            'total_frames': 0
        }
    
    # 收集所有csv文件
    all_files = []
    for data_folder in os.listdir(root_path):
        if not data_folder.startswith("2024-"):
            continue
        
        data_path = os.path.join(root_path, data_folder)
        if not os.path.isdir(data_path):
            continue
        
        for session_folder in os.listdir(data_path):
            session_path = os.path.join(data_path, session_folder)
            if not os.path.isdir(session_path):
                continue
            
            for file_name in os.listdir(session_path):
                if file_name.endswith('.csv') and file_name.startswith('good_'):
                    all_files.append(os.path.join(session_path, file_name))
    
    print(f"Found {len(all_files)} trace files")
    
    # 处理每个文件
    for file_path in tqdm(all_files, desc="Processing traces"):
        data, network_info = load_trace_data(file_path, start_idx=300, len_limit=len_limit)
        
        if data is None:
            continue
        
        stats = extract_frame_stats(data)
        if stats is None:
            continue
        
        # 确定网络类型分组
        full_network = network_info['full_network']
        if full_network in results:
            group = full_network
        elif network_info['network_str'] == 'ETH':
            group = 'ETH'
        else:
            group = 'Unknown'
        
        # 添加到对应分组
        results[group]['frame_intervals'].extend(stats['frame_intervals'].tolist())
        results[group]['network_times'].extend(stats['network_times'].tolist())
        results[group]['decode_times'].extend(stats['decode_times'].tolist())
        results[group]['file_count'] += 1
        results[group]['total_frames'] += stats['num_frames']
        
        # 添加到总体统计
        results['all']['frame_intervals'].extend(stats['frame_intervals'].tolist())
        results['all']['network_times'].extend(stats['network_times'].tolist())
        results['all']['decode_times'].extend(stats['decode_times'].tolist())
        results['all']['file_count'] += 1
        results['all']['total_frames'] += stats['num_frames']
    
    # 转换为numpy数组
    for group in results:
        results[group]['frame_intervals'] = np.array(results[group]['frame_intervals'])
        results[group]['network_times'] = np.array(results[group]['network_times'])
        results[group]['decode_times'] = np.array(results[group]['decode_times'])
    
    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {output_path}")
    
    return results


def print_stats_summary(results: dict):
    """打印统计摘要"""
    print("\n" + "=" * 60)
    print("Cloud Gaming Trace Statistics Summary")
    print("=" * 60)
    
    for group, data in results.items():
        if data['file_count'] == 0:
            continue
        
        print(f"\n--- {group} ---")
        print(f"  Files: {data['file_count']}, Total Frames: {data['total_frames']}")
        
        if len(data['frame_intervals']) > 0:
            fi = data['frame_intervals']
            print(f"  Frame Interval: mean={np.mean(fi):.2f}ms, "
                  f"std={np.std(fi):.2f}ms, "
                  f"P50={np.percentile(fi, 50):.2f}ms, "
                  f"P95={np.percentile(fi, 95):.2f}ms")
        
        if len(data['network_times']) > 0:
            nt = data['network_times']
            print(f"  Network Time: mean={np.mean(nt):.2f}ms, "
                  f"std={np.std(nt):.2f}ms, "
                  f"P50={np.percentile(nt, 50):.2f}ms, "
                  f"P95={np.percentile(nt, 95):.2f}ms")
        
        if len(data['decode_times']) > 0:
            dt = data['decode_times']
            print(f"  Decode Time: mean={np.mean(dt):.2f}ms, "
                  f"std={np.std(dt):.2f}ms, "
                  f"P50={np.percentile(dt, 50):.2f}ms, "
                  f"P95={np.percentile(dt, 95):.2f}ms")
    
    print("\n" + "=" * 60)


def process_single_trace(file_path: str) -> dict:
    """
    处理单个trace文件，用于测试
    """
    data, network_info = load_trace_data(file_path, start_idx=300, len_limit=72000)
    
    if data is None:
        print(f"Failed to load {file_path}")
        return None
    
    print(f"Loaded {data.shape[0]} frames from {file_path}")
    print(f"Network Info: {network_info}")
    
    stats = extract_frame_stats(data)
    if stats is None:
        print("Failed to extract stats")
        return None
    
    print(f"\nFrame Intervals: {len(stats['frame_intervals'])} samples")
    if len(stats['frame_intervals']) > 0:
        print(f"  Mean: {np.mean(stats['frame_intervals']):.2f}ms")
        print(f"  Std: {np.std(stats['frame_intervals']):.2f}ms")
        print(f"  P50: {np.percentile(stats['frame_intervals'], 50):.2f}ms")
        print(f"  P95: {np.percentile(stats['frame_intervals'], 95):.2f}ms")
    
    print(f"\nNetwork Times: {len(stats['network_times'])} samples")
    if len(stats['network_times']) > 0:
        print(f"  Mean: {np.mean(stats['network_times']):.2f}ms")
        print(f"  Std: {np.std(stats['network_times']):.2f}ms")
        print(f"  P50: {np.percentile(stats['network_times'], 50):.2f}ms")
        print(f"  P95: {np.percentile(stats['network_times'], 95):.2f}ms")
    
    print(f"\nDecode Times: {len(stats['decode_times'])} samples")
    if len(stats['decode_times']) > 0:
        print(f"  Mean: {np.mean(stats['decode_times']):.2f}ms")
        print(f"  Std: {np.std(stats['decode_times']):.2f}ms")
        print(f"  P50: {np.percentile(stats['decode_times'], 50):.2f}ms")
        print(f"  P95: {np.percentile(stats['decode_times'], 95):.2f}ms")
    
    return {
        'data': data,
        'network_info': network_info,
        'stats': stats
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 处理单个文件
        file_path = sys.argv[1]
        process_single_trace(file_path)
    else:
        # 默认处理所有数据
        # 修改为你的数据根目录
        data_root = "E:/pure_framerate_data"
        output_path = "./output/motivation_stats.pkl"
        
        if os.path.exists(data_root):
            results = process_all_traces(data_root, output_path)
            print_stats_summary(results)
        else:
            print(f"Data root not found: {data_root}")
            print("Usage: python extract_motivation_stats.py [trace_file.csv]")
            print("  Or modify data_root in the script to process all traces")
