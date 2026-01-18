import pandas as pd
import numpy as np
import os

test_data_path = './test_data'

files = [
    ('VSync', 'result-naiveVsync-periodrop1_quickdrop0_maxbuf2_bonusfps30_lifo_ewma_anchor4.csv'),
    ('PASync', 'result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo_ewma_anchor4.csv'),
    ('Oracle', 'result-optimal-periodrop2_quickdrop0_maxbuf2_bonusfps30_lifo_oracle_anchor4.csv')
]

def get_col_by_pattern(df, pattern):
    cols = [col for col in df.columns if pattern in col]
    if not cols:
        return None
    return cols[0]

for label, filename in files:
    filepath = os.path.join(test_data_path, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
    
    df = pd.read_csv(filepath)
    queue_col = get_col_by_pattern(df, 'optimized_render_queue')
    
    if queue_col:
        p90 = df[queue_col].quantile(0.90)
        p95 = df[queue_col].quantile(0.95)
        mean = df[queue_col].mean()
        print(f"{label} - Mean: {mean:.2f}, P90: {p90:.2f}, P95: {p95:.2f}")