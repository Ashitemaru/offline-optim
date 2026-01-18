from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import scienceplots
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

# ===================== 比较最优simpleCtrl与vsync和oracle的CDF图 =====================

def plot_cdf(data, label):
    """绘制CDF曲线"""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=label)

# 读取数据
test_data_path = './test_data_ablation'
best_simple_ctrl = pd.read_csv(os.path.join(test_data_path, 
    'result-simpleCtrl-periodrop1_quickdrop1_maxbuf2_bonusfps30_lifo_ewma_anchor4.csv'))
vsync_data = pd.read_csv(os.path.join(test_data_path, 
    'result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo_ewma_anchor4.csv'))
oracle_data = pd.read_csv(os.path.join(test_data_path, 
    'result-optimal-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo_oracle_anchor4.csv'))

# 提取列名
simple_ctrl_fps_col = [col for col in best_simple_ctrl.columns if 'optimized_fps' in col and 'simpleCtrl' in col and 'objective' not in col and 'noloss' not in col][0]
simple_ctrl_queue_col = [col for col in best_simple_ctrl.columns if 'optimized_render_queue' in col and 'simpleCtrl' in col][0]
simple_ctrl_extra_ts_col = [col for col in best_simple_ctrl.columns if 'extra_display_ts' in col and 'simpleCtrl' in col][0]

vsync_fps_col = [col for col in vsync_data.columns if 'optimized_fps' in col and 'naiveVsync' in col and 'objective' not in col and 'noloss' not in col][0]
vsync_queue_col = [col for col in vsync_data.columns if 'optimized_render_queue' in col and 'naiveVsync' in col][0]
vsync_extra_ts_col = [col for col in vsync_data.columns if 'extra_display_ts' in col and 'naiveVsync' in col][0]

oracle_fps_col = [col for col in oracle_data.columns if 'optimized_fps' in col and 'optimal' in col and 'objective' not in col and 'noloss' not in col][0]
oracle_queue_col = [col for col in oracle_data.columns if 'optimized_render_queue' in col and 'optimal' in col][0]
oracle_extra_ts_col = [col for col in oracle_data.columns if 'extra_display_ts' in col and 'optimal' in col][0]

# 绘制FPS的CDF图
with plt.style.context(["science", "ieee"]):
    plt.figure(figsize=(2, 1.2))
    
    plot_cdf(best_simple_ctrl[simple_ctrl_fps_col].values, 'PASync')
    plot_cdf(vsync_data[vsync_fps_col].values, 'VSync')
    plot_cdf(oracle_data[oracle_fps_col].values, 'Oracle')
    
    plt.xlabel("Frame Rate (FPS)", fontsize=6)
    plt.ylabel("CDF", fontsize=6)
    plt.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
    plt.tick_params(axis='both', which='both', pad=2, labelsize=6)
    plt.ylim(0, 1)
    plt.xlim(40, 60)
    
    os.makedirs('../output', exist_ok=True)
    plt.savefig("../output/fps_cdf_comparison.pdf")
    print("FPS CDF图已保存至 ../output/fps_cdf_comparison.pdf")
    plt.close()

# 绘制延迟(render queue)的CDF图
with plt.style.context(["science", "ieee"]):
    plt.figure(figsize=(2, 1.2))
    
    plot_cdf(best_simple_ctrl[simple_ctrl_queue_col].values, 'PASync')
    plot_cdf(vsync_data[vsync_queue_col].values, 'VSync')
    plot_cdf(oracle_data[oracle_queue_col].values, 'Oracle')
    
    plt.xlabel("Buffering Latency (ms)", fontsize=6)
    plt.ylabel("CDF", fontsize=6)
    plt.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
    plt.tick_params(axis='both', which='both', pad=2, labelsize=6)
    plt.ylim(0, 1)
    plt.xlim(0, 20)
    
    plt.savefig("../output/delay_cdf_comparison.pdf")
    print("延迟CDF图已保存至 ../output/delay_cdf_comparison.pdf")
    plt.close()

# 打印统计信息
print("\n========== 统计信息 ==========")
print(f"SimpleCtrl (Best) - 平均FPS: {best_simple_ctrl[simple_ctrl_fps_col].mean():.2f}, 平均Delay: {best_simple_ctrl[simple_ctrl_queue_col].mean():.2f}ms")
print(f"VSync - 平均FPS: {vsync_data[vsync_fps_col].mean():.2f}, 平均Delay: {vsync_data[vsync_queue_col].mean():.2f}ms")
print(f"Oracle - 平均FPS: {oracle_data[oracle_fps_col].mean():.2f}, 平均Delay: {oracle_data[oracle_queue_col].mean():.2f}ms")

# 计算95分位数延迟
vsync_delay_95 = np.percentile(vsync_data[vsync_queue_col].values, 90)
simple_ctrl_delay_95 = np.percentile(best_simple_ctrl[simple_ctrl_queue_col].values, 90)
oracle_delay_95 = np.percentile(oracle_data[oracle_queue_col].values, 90)

print("\n========== 95分位数延迟优化幅度 (相对VSync) ==========")
print(f"VSync 95分位数延迟: {vsync_delay_95:.2f}ms")
print(f"SimpleCtrl 95分位数延迟: {simple_ctrl_delay_95:.2f}ms, 优化幅度: {(vsync_delay_95 - simple_ctrl_delay_95) / vsync_delay_95 * 100:.2f}%")
print(f"Oracle 95分位数延迟: {oracle_delay_95:.2f}ms, 优化幅度: {(vsync_delay_95 - oracle_delay_95) / vsync_delay_95 * 100:.2f}%")
print("==============================\n")