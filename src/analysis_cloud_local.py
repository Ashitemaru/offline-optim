from matplotlib import pyplot as plt
import scienceplots
import numpy as np
import os
import pandas as pd

# ===================== 比较Cloud和Local延迟的CDF图 =====================

def plot_cdf(data, label):
    """绘制CDF曲线"""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=label)

# 读取数据
test_data_path = './test_data_motivation'
cloud_data = pd.read_csv(os.path.join(test_data_path, 
    'result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo_ewma_anchor4-cloud.csv'))
local_data = pd.read_csv(os.path.join(test_data_path, 
    'result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo_ewma_anchor4-local.csv'))

# 提取延迟相关列名
cloud_queue_col = [col for col in cloud_data.columns if 'optimized_render_queue' in col and 'cloud' in col][0]
local_queue_col = [col for col in local_data.columns if 'optimized_render_queue' in col and 'local' in col][0]

# 提取FPS相关列名
cloud_fps_col = [col for col in cloud_data.columns if 'optimized_fps' in col and 'cloud' in col and 'objective' not in col and 'noloss' not in col][0]
local_fps_col = [col for col in local_data.columns if 'optimized_fps' in col and 'local' in col and 'objective' not in col and 'noloss' not in col][0]

# 绘制延迟(render queue)的CDF图
with plt.style.context(["science", "ieee"]):
    plt.figure(figsize=(2, 1.2))
    
    plot_cdf(cloud_data[cloud_queue_col].values, 'Cloud')
    plot_cdf(local_data[local_queue_col].values, 'Local')
    
    plt.xlabel("Buffering Latency (ms)", fontsize=6)
    plt.ylabel("CDF", fontsize=6)
    plt.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
    plt.tick_params(axis='both', which='both', pad=2, labelsize=6)
    plt.ylim(0, 1)
    
    os.makedirs('../output', exist_ok=True)
    plt.savefig("../output/delay_cdf_cloud_vs_local.pdf")
    print("延迟CDF图已保存至 ../output/delay_cdf_cloud_vs_local.pdf")
    plt.close()

# 绘制FPS的CDF图
with plt.style.context(["science", "ieee"]):
    plt.figure(figsize=(2, 1.2))
    
    plot_cdf(cloud_data[cloud_fps_col].values, 'Cloud')
    plot_cdf(local_data[local_fps_col].values, 'Local')
    
    plt.xlabel("Frame Rate (FPS)", fontsize=6)
    plt.ylabel("CDF", fontsize=6)
    plt.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
    plt.tick_params(axis='both', which='both', pad=2, labelsize=6)
    plt.ylim(0, 1)
    
    plt.savefig("../output/fps_cdf_cloud_vs_local.pdf")
    print("FPS CDF图已保存至 ../output/fps_cdf_cloud_vs_local.pdf")
    plt.close()

# 打印统计信息
print("\n========== 统计信息 ==========")
print(f"Cloud - 平均FPS: {cloud_data[cloud_fps_col].mean():.2f}, 平均Delay: {cloud_data[cloud_queue_col].mean():.2f}ms")
print(f"Local - 平均FPS: {local_data[local_fps_col].mean():.2f}, 平均Delay: {local_data[local_queue_col].mean():.2f}ms")

# 计算分位数延迟
cloud_delay_50 = np.percentile(cloud_data[cloud_queue_col].values, 50)
local_delay_50 = np.percentile(local_data[local_queue_col].values, 50)
cloud_delay_90 = np.percentile(cloud_data[cloud_queue_col].values, 90)
local_delay_90 = np.percentile(local_data[local_queue_col].values, 90)
cloud_delay_95 = np.percentile(cloud_data[cloud_queue_col].values, 95)
local_delay_95 = np.percentile(local_data[local_queue_col].values, 95)

print("\n========== 延迟分位数比较 ==========")
print(f"Cloud - 50分位数延迟: {cloud_delay_50:.2f}ms, 90分位数延迟: {cloud_delay_90:.2f}ms, 95分位数延迟: {cloud_delay_95:.2f}ms")
print(f"Local - 50分位数延迟: {local_delay_50:.2f}ms, 90分位数延迟: {local_delay_90:.2f}ms, 95分位数延迟: {local_delay_95:.2f}ms")

# 计算FPS分位数
cloud_fps_50 = np.percentile(cloud_data[cloud_fps_col].values, 50)
local_fps_50 = np.percentile(local_data[local_fps_col].values, 50)

print("\n========== FPS分位数比较 ==========")
print(f"Cloud - 50分位数FPS: {cloud_fps_50:.2f}")
print(f"Local - 50分位数FPS: {local_fps_50:.2f}")
print("==============================\n")
