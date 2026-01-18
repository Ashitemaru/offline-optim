import pandas as pd
import numpy as np
import os

test_data_path = './test_data_ablation'

def get_col_by_pattern(df, pattern):
    cols = [col for col in df.columns if pattern in col]
    if not cols:
        return None
    return cols[0]

def analyze_file(filepath, label):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    
    # Identify key columns
    fps_col = get_col_by_pattern(df, 'optimized_fps')
    queue_col = get_col_by_pattern(df, 'optimized_render_queue')
    
    if not fps_col or not queue_col:
        print(f"Columns not found in {filepath}")
        return None
        
    stats = {
        'Label': label,
        'Latency_Mean': df[queue_col].mean(),
        'Latency_Std': df[queue_col].std(),
        'Latency_P95': df[queue_col].quantile(0.95),
        'Latency_P99': df[queue_col].quantile(0.99),
        'FPS_Mean': df[fps_col].mean(),
        'FPS_Std': df[fps_col].std(),
        'FPS_P01': df[fps_col].quantile(0.01), # 1% low FPS
        'FPS_P05': df[fps_col].quantile(0.05),
    }
    return stats

def generate_latex_table(stats_list, caption, label):
    print(f"\n% Table for {caption}")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{" + caption + "}")
    print(r"\label{" + label + "}")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{l|cccc|cccc}")
    print(r"\hline")
    print(r"\textbf{Method} & \multicolumn{4}{c|}{\textbf{Latency (ms)}} & \multicolumn{4}{c}{\textbf{Frame Rate (FPS)}} \\")
    print(r" & Avg & Std & P95 & P99 & Avg & Std & P1 & P5 \\")
    print(r"\hline")
    
    for s in stats_list:
        print(f"{s['Label']} & {s['Latency_Mean']:.2f} & {s['Latency_Std']:.2f} & {s['Latency_P95']:.2f} & {s['Latency_P99']:.2f} & "
              f"{s['FPS_Mean']:.2f} & {s['FPS_Std']:.2f} & {s['FPS_P01']:.2f} & {s['FPS_P05']:.2f} \\\\")
        
    print(r"\hline")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{table}")

# === Figure 7: Mechanism Effectiveness ===
files_fig7 = [
    ('VSync', 'result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo_ewma_anchor4.csv'),
    ('Periodic', 'result-simpleCtrl-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo_ewma_anchor4.csv'),
    ('Predictive', 'result-simpleCtrl-periodrop0_quickdrop1_maxbuf2_bonusfps30_lifo_ewma_anchor4.csv'),
    ('PASync', 'result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo_ewma_anchor4.csv')
]

stats_fig7 = []
for label, filename in files_fig7:
    stats = analyze_file(os.path.join(test_data_path, filename), label)
    if stats:
        stats_fig7.append(stats)

if stats_fig7:
    generate_latex_table(stats_fig7, "Impact of Drop Mechanisms", "tab:mech_effectiveness")

# === Figure 8: Predictor Robustness ===
files_fig8 = [
    ('Fixed', 'result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo_fixed_anchor4.csv'),
    ('Adaptive', 'result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo_ewma_anchor4.csv'),
    ('Oracle', 'result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo_oracle_anchor4.csv')
]

stats_fig8 = []
for label, filename in files_fig8:
    stats = analyze_file(os.path.join(test_data_path, filename), label)
    if stats:
        stats_fig8.append(stats)

if stats_fig8:
    generate_latex_table(stats_fig8, "Predictor Robustness", "tab:pred_robustness")
