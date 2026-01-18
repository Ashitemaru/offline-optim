import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import os

output_dir = r"d:\Ashitemaru\CodingFolder\Projects\NOSSDAV\drawio"
os.makedirs(output_dir, exist_ok=True)

# Common configuration from analysis.py
figsize = (2, 1.2)
label_fontsize = 6
tick_fontsize = 6
legend_fontsize = 6
grid_alpha = 0.3
grid_linestyle = '--'
grid_linewidth = 0.5
legend_params = {
    'frameon': True,
    'labelspacing': 0.1,
    'columnspacing': 0.1,
    'handletextpad': 0.1,
    'fontsize': legend_fontsize
}

# Data for Figure 7: Mechanism Effectiveness
# Baseline: VSync
# Periodic: simpleCtrl + periodrop2
# Full: simpleCtrl + periodrop2 + quickdrop1 (EWMA)
mechanisms = ['VSync', 'Periodic', 'PASync']
avg_latencies = [9.31, 5.72, 5.29]
# Note: PASync latency is 5.29ms

with plt.style.context(["science", "ieee"]):
    fig7, ax7 = plt.subplots(figsize=figsize)
    bars = ax7.bar(mechanisms, avg_latencies, color=['#A9A9A9', '#87CEFA', '#CD5C5C'], width=0.6, alpha=0.9)

    # Add text labels
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=tick_fontsize)

    ax7.set_ylabel('Avg. Latency (ms)', fontsize=label_fontsize)
    ax7.set_xlabel('Mechanism', fontsize=label_fontsize)
    # ax7.set_title('Impact of Drop Mechanisms', fontsize=label_fontsize) # Titles are often omitted in paper figures
    ax7.grid(axis='y', linestyle=grid_linestyle, alpha=grid_alpha, linewidth=grid_linewidth)
    ax7.set_ylim(0, 12.5)
    
    ax7.tick_params(axis='both', which='both', pad=2, labelsize=tick_fontsize)
    # plt.tight_layout()
    # ax7.minorticks_off()
    plt.savefig(os.path.join(output_dir, 'ablation_mechanism.pdf'))
    print("Generated ablation_mechanism.pdf")
    plt.close()

# Data for Figure 8: Predictor Robustness
# Comparing Fixed, EWMA (Adaptive), Oracle Predictor
# Metrics: Latency (lower better), FPS (higher better)
predictors = ['Fixed', 'Adaptive', 'Oracle'] # Shortened 'Adaptive (EWMA)' and 'Ground Truth Pred.' for space
latencies = [5.32, 5.29, 5.29] # Average Latency
fps_values = [53.33, 53.46, 53.43] # Effective FPS

with plt.style.context(["science", "ieee"]):
    # Just use a dual-axis chart or side-by-side bars because scatter with 3 points is sparse
    fig8, ax8 = plt.subplots(figsize=figsize)
    x = np.arange(len(predictors))
    width = 0.35

    rects1 = ax8.bar(x - width/2, latencies, width, label='Latency', color='#CD5C5C', alpha=0.9)
    ax8_right = ax8.twinx()
    rects2 = ax8_right.bar(x + width/2, fps_values, width, label='Frame Rate', color='#2E8B57', alpha=0.9)

    ax8.set_ylabel('Avg. Latency (ms)', fontsize=label_fontsize)
    ax8_right.set_ylabel('Frame Rate (FPS)', fontsize=label_fontsize)
    ax8.set_xlabel('Predictor Type', fontsize=label_fontsize)
    
    ax8.set_xticks(x)
    ax8.set_xticklabels(predictors, fontsize=tick_fontsize)
    
    ax8.tick_params(axis='y', which='both', labelsize=tick_fontsize, pad=2)
    ax8_right.tick_params(axis='y', which='both', labelsize=tick_fontsize, pad=2)

    # legend
    lines_1, labels_1 = ax8.get_legend_handles_labels()
    lines_2, labels_2 = ax8_right.get_legend_handles_labels()
    
    # Fundamental Fix:
    # 1. Attach legend to ax8_right (the top-most axis) to avoid being covered by bars on ax8_right.
    # 2. Set zorder high explicitly.
    leg = ax8_right.legend(lines_1 + lines_2, labels_1 + labels_2, 
                         loc='upper center', 
                         ncol=2,
                         framealpha=1.0, # Ensure opacity
                         **legend_params)
    leg.set_zorder(100)

    # Revert ylims to tight bounds (No empty space workarounds)
    ax8.set_ylim(5.0, 5.5) 
    ax8_right.set_ylim(53.0, 53.6)
    
    # Adjust layout manually (No tight_layout)
    # right=0.82 ensures right Y-label is not clipped ("missing piece")
    # top/bottom left default
    plt.subplots_adjust(right=0.82)
    
    ax8.grid(axis='y', linestyle=grid_linestyle, alpha=grid_alpha, linewidth=grid_linewidth)
    plt.savefig(os.path.join(output_dir, 'ablation_predictor.pdf'))
    print("Generated ablation_predictor.pdf")
    plt.close()
