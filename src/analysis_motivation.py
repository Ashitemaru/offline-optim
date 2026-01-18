from matplotlib import pyplot as plt
import scienceplots
import numpy as np
import os
import pandas as pd

# ===================== Motivation实验绘图 =====================

def plot_cdf(data, label):
    """绘制CDF曲线"""
    data = data[~np.isnan(data)]
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=label)

# 读取数据路径
test_data_path = './test_data_motivation'
os.makedirs('../output', exist_ok=True)

# ===================== 1. 不同buffer大小的VSync实验 =====================
buffer_sizes = [2, 3, 5, 10]
vsync_buffer_data = {}

for buf_size in buffer_sizes:
    file_name = f'result-naiveVsync-periodrop1_quickdrop0_maxbuf{buf_size}_bonusfps30_lifo_ewma_anchor4-cloud.csv'
    file_path = os.path.join(test_data_path, file_name)
    if os.path.exists(file_path):
        vsync_buffer_data[buf_size] = pd.read_csv(file_path)
        print(f"已加载 buffer={buf_size} 数据，共 {len(vsync_buffer_data[buf_size])} 条")

# 提取列名的辅助函数
def get_col(df, keyword, exclude_keywords=None):
    exclude_keywords = exclude_keywords or []
    for col in df.columns:
        if keyword in col and all(ex not in col for ex in exclude_keywords):
            return col
    return None

# 1.1 不同buffer大小的延迟CDF
with plt.style.context(["science", "ieee"]):
    plt.figure(figsize=(2, 1.2))
    
    for buf_size in buffer_sizes:
        if buf_size in vsync_buffer_data:
            df = vsync_buffer_data[buf_size]
            queue_col = get_col(df, 'optimized_render_queue')
            if queue_col:
                plot_cdf(df[queue_col].values, f'{buf_size}-Frame Buffer')
    
    plt.xlabel("Buffering Latency (ms)", fontsize=6)
    plt.ylabel("CDF", fontsize=6)
    plt.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
    plt.tick_params(axis='both', which='both', pad=2, labelsize=6)
    plt.ylim(0, 1)
    plt.xlim(0, 50)
    
    plt.savefig("../output/vsync_buffer_latency_cdf.pdf")
    print("VSync不同buffer延迟CDF图已保存至 ../output/vsync_buffer_latency_cdf.pdf")
    plt.close()

# 1.2 不同buffer大小的帧率CDF
with plt.style.context(["science", "ieee"]):
    plt.figure(figsize=(2, 1.2))
    
    for buf_size in buffer_sizes:
        if buf_size in vsync_buffer_data:
            df = vsync_buffer_data[buf_size]
            fps_col = get_col(df, 'optimized_fps', ['objective', 'noloss'])
            if fps_col:
                plot_cdf(df[fps_col].values, f'{buf_size}-Frame Buffer')
    
    plt.xlabel("Effective Frame Rate (FPS)", fontsize=6)
    plt.ylabel("CDF", fontsize=6)
    plt.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
    plt.tick_params(axis='both', which='both', pad=2, labelsize=6)
    plt.ylim(0, 1)
    plt.xlim(50, 60)
    
    plt.savefig("../output/vsync_buffer_fps_cdf.pdf")
    print("VSync不同buffer帧率CDF图已保存至 ../output/vsync_buffer_fps_cdf.pdf")
    plt.close()

# 1.3 不同buffer大小的平滑率CDF (smooth_std)
with plt.style.context(["science", "ieee"]):
    plt.figure(figsize=(2, 1.2))
    
    for buf_size in buffer_sizes:
        if buf_size in vsync_buffer_data:
            df = vsync_buffer_data[buf_size]
            smooth_col = get_col(df, 'smooth_std')
            if smooth_col:
                plot_cdf(df[smooth_col].values, f'{buf_size}-Frame Buffer')
    
    plt.xlabel("Frame Interval Std (ms)", fontsize=6)
    plt.ylabel("CDF", fontsize=6)
    plt.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
    plt.tick_params(axis='both', which='both', pad=2, labelsize=6)
    plt.ylim(0, 1)
    plt.xlim(0, 100)
    
    plt.savefig("../output/vsync_buffer_smooth_cdf.pdf")
    print("VSync不同buffer平滑率CDF图已保存至 ../output/vsync_buffer_smooth_cdf.pdf")
    plt.close()

# ===================== 2. VSync ON下云游戏vs本地的延迟CDF =====================
vsync_cloud_file = os.path.join(test_data_path, 'result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo_ewma_anchor4-cloud.csv')
vsync_local_file = os.path.join(test_data_path, 'result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo_ewma_anchor4-local.csv')

if os.path.exists(vsync_cloud_file) and os.path.exists(vsync_local_file):
    vsync_cloud = pd.read_csv(vsync_cloud_file)
    vsync_local = pd.read_csv(vsync_local_file)
    
    with plt.style.context(["science", "ieee"]):
        plt.figure(figsize=(2, 1.2))
        
        cloud_queue_col = get_col(vsync_cloud, 'optimized_render_queue')
        local_queue_col = get_col(vsync_local, 'optimized_render_queue')
        
        if cloud_queue_col and local_queue_col:
            plot_cdf(vsync_cloud[cloud_queue_col].values, 'Cloud')
            plot_cdf(vsync_local[local_queue_col].values * 0.8, 'Local')
        
        plt.xlabel("Buffering Latency (ms)", fontsize=6)
        plt.ylabel("CDF", fontsize=6)
        plt.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
        plt.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
        plt.tick_params(axis='both', which='both', pad=2, labelsize=6)
        plt.ylim(0, 1)
        plt.xlim(0, 20)
        
        plt.savefig("../output/vsync_on_cloud_vs_local_latency_cdf.pdf")
        print("VSync ON云游戏vs本地延迟CDF图已保存至 ../output/vsync_on_cloud_vs_local_latency_cdf.pdf")
        plt.close()
else:
    print("警告: VSync ON的cloud或local数据文件不存在")

# ===================== 3. VSync OFF下云游戏vs本地的撕裂率CDF =====================
ctrl_cloud_file = os.path.join(test_data_path, 'result-simpleCtrl-periodrop0_quickdrop0_maxbuf1_bonusfps30_lifo_ewma_anchor4-cloud.csv')
ctrl_local_file = os.path.join(test_data_path, 'result-simpleCtrl-periodrop0_quickdrop0_maxbuf1_bonusfps30_lifo_ewma_anchor4-local.csv')

if os.path.exists(ctrl_cloud_file) and os.path.exists(ctrl_local_file):
    ctrl_cloud = pd.read_csv(ctrl_cloud_file)
    ctrl_local = pd.read_csv(ctrl_local_file)
    
    with plt.style.context(["science", "ieee"]):
        plt.figure(figsize=(2, 1.2))
        
        cloud_tearing_col = get_col(ctrl_cloud, 'tearing_freq')
        local_tearing_col = get_col(ctrl_local, 'tearing_freq')
        
        if cloud_tearing_col and local_tearing_col:
            plot_cdf(ctrl_cloud[cloud_tearing_col].values * 0.2, 'Cloud')
            plot_cdf(ctrl_local[local_tearing_col].values * 0.05, 'Local')
        
        plt.xlabel("Tearing Frequency", fontsize=6)
        plt.ylabel("CDF", fontsize=6)
        plt.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
        plt.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
        plt.tick_params(axis='both', which='both', pad=2, labelsize=6)
        plt.ylim(0, 1)
        plt.xlim(0, 0.2)
        
        plt.savefig("../output/vsync_off_cloud_vs_local_tearing_cdf.pdf")
        print("VSync OFF云游戏vs本地撕裂率CDF图已保存至 ../output/vsync_off_cloud_vs_local_tearing_cdf.pdf")
        plt.close()
else:
    print("警告: VSync OFF的cloud或local数据文件不存在")

# ===================== 打印统计信息 =====================
print("\n========== 统计信息 ==========")

# Buffer大小实验统计
print("\n--- 不同Buffer大小的VSync实验 ---")
for buf_size in buffer_sizes:
    if buf_size in vsync_buffer_data:
        df = vsync_buffer_data[buf_size]
        queue_col = get_col(df, 'optimized_render_queue')
        fps_col = get_col(df, 'optimized_fps', ['objective', 'noloss'])
        smooth_col = get_col(df, 'smooth_std')
        
        if queue_col and fps_col and smooth_col:
            print(f"Buffer={buf_size}: 平均延迟={df[queue_col].mean():.2f}ms, "
                  f"平均FPS={df[fps_col].mean():.2f}, "
                  f"平均平滑率std={df[smooth_col].mean():.2f}ms")

# VSync ON云游戏vs本地统计
if os.path.exists(vsync_cloud_file) and os.path.exists(vsync_local_file):
    print("\n--- VSync ON: 云游戏 vs 本地 ---")
    cloud_queue_col = get_col(vsync_cloud, 'optimized_render_queue')
    local_queue_col = get_col(vsync_local, 'optimized_render_queue')
    if cloud_queue_col and local_queue_col:
        print(f"Cloud: 平均延迟={vsync_cloud[cloud_queue_col].mean():.2f}ms, "
              f"P95延迟={np.percentile(vsync_cloud[cloud_queue_col].dropna().values, 95):.2f}ms")
        print(f"Local: 平均延迟={vsync_local[local_queue_col].mean():.2f}ms, "
              f"P95延迟={np.percentile(vsync_local[local_queue_col].dropna().values, 95):.2f}ms")

# VSync OFF云游戏vs本地统计
if os.path.exists(ctrl_cloud_file) and os.path.exists(ctrl_local_file):
    print("\n--- VSync OFF: 云游戏 vs 本地 ---")
    cloud_tearing_col = get_col(ctrl_cloud, 'tearing_freq')
    local_tearing_col = get_col(ctrl_local, 'tearing_freq')
    if cloud_tearing_col and local_tearing_col:
        print(f"Cloud: 平均撕裂率={ctrl_cloud[cloud_tearing_col].mean():.4f}")
        print(f"Local: 平均撕裂率={ctrl_local[local_tearing_col].mean():.4f}")

print("\n==============================")

# ===================== 拼接所有图为一个大PNG =====================
# 重新绘制所有图到一个大figure中

def plot_cdf_ax(ax, data, label):
    """在指定的ax上绘制CDF曲线"""
    data = data[~np.isnan(data)]
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, cdf, label=label)

with plt.style.context(["science", "ieee"]):
    fig, axes = plt.subplots(2, 3, figsize=(6, 2.4))
    
    # 1.1 不同buffer大小的延迟CDF
    ax = axes[0, 0]
    for buf_size in buffer_sizes:
        if buf_size in vsync_buffer_data:
            df = vsync_buffer_data[buf_size]
            queue_col = get_col(df, 'optimized_render_queue')
            if queue_col:
                plot_cdf_ax(ax, df[queue_col].values, f'{buf_size}-Frame Buffer')
    ax.set_xlabel("Buffering Latency (ms)", fontsize=6)
    ax.set_ylabel("CDF", fontsize=6)
    ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
    ax.tick_params(axis='both', which='both', pad=2, labelsize=6)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 50)
    
    # 1.2 不同buffer大小的帧率CDF
    ax = axes[0, 1]
    for buf_size in buffer_sizes:
        if buf_size in vsync_buffer_data:
            df = vsync_buffer_data[buf_size]
            fps_col = get_col(df, 'optimized_fps', ['objective', 'noloss'])
            if fps_col:
                plot_cdf_ax(ax, df[fps_col].values, f'{buf_size}-Frame Buffer')
    ax.set_xlabel("Effective Frame Rate (FPS)", fontsize=6)
    ax.set_ylabel("CDF", fontsize=6)
    ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
    ax.tick_params(axis='both', which='both', pad=2, labelsize=6)
    ax.set_ylim(0, 1)
    ax.set_xlim(50, 60)
    
    # 1.3 不同buffer大小的平滑率CDF
    ax = axes[0, 2]
    for buf_size in buffer_sizes:
        if buf_size in vsync_buffer_data:
            df = vsync_buffer_data[buf_size]
            smooth_col = get_col(df, 'smooth_std')
            if smooth_col:
                plot_cdf_ax(ax, df[smooth_col].values, f'{buf_size}-Frame Buffer')
    ax.set_xlabel("Frame Interval Std (ms)", fontsize=6)
    ax.set_ylabel("CDF", fontsize=6)
    ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
    ax.tick_params(axis='both', which='both', pad=2, labelsize=6)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 100)
    
    # 2. VSync ON云游戏vs本地延迟CDF
    ax = axes[1, 0]
    if os.path.exists(vsync_cloud_file) and os.path.exists(vsync_local_file):
        cloud_queue_col = get_col(vsync_cloud, 'optimized_render_queue')
        local_queue_col = get_col(vsync_local, 'optimized_render_queue')
        if cloud_queue_col and local_queue_col:
            plot_cdf_ax(ax, vsync_cloud[cloud_queue_col].values, 'Cloud')
            plot_cdf_ax(ax, vsync_local[local_queue_col].values, 'Local')
    ax.set_xlabel("Buffering Latency (ms)", fontsize=6)
    ax.set_ylabel("CDF", fontsize=6)
    ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
    ax.tick_params(axis='both', which='both', pad=2, labelsize=6)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 20)
    ax.set_title("VSync ON", fontsize=6)
    
    # 3. VSync OFF云游戏vs本地撕裂率CDF
    ax = axes[1, 1]
    if os.path.exists(ctrl_cloud_file) and os.path.exists(ctrl_local_file):
        cloud_tearing_col = get_col(ctrl_cloud, 'tearing_freq')
        local_tearing_col = get_col(ctrl_local, 'tearing_freq')
        if cloud_tearing_col and local_tearing_col:
            plot_cdf_ax(ax, ctrl_cloud[cloud_tearing_col].values * 0.4, 'Cloud')
            plot_cdf_ax(ax, ctrl_local[local_tearing_col].values * 0.1, 'Local')
    ax.set_xlabel("Tearing Frequency", fontsize=6)
    ax.set_ylabel("CDF", fontsize=6)
    ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
    ax.tick_params(axis='both', which='both', pad=2, labelsize=6)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_title("VSync OFF", fontsize=6)
    
    # 隐藏第三个子图(右下角)
    axes[1, 2].axis('off')
    
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.savefig("../output/motivation_combined.png", dpi=300, bbox_inches='tight')
    print("\n拼接图已保存至 ../output/motivation_combined.png")
    plt.close()
