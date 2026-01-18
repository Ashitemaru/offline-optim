import pandas as pd
import numpy as np
import os

test_data_path = './test_data'

files = [
    ('Fixed', 'result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo_fixed_anchor4.csv'),
    ('Adaptive', 'result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo_ewma_anchor4.csv'),
    ('Oracle', 'result-simpleCtrl-periodrop2_quickdrop1_maxbuf2_bonusfps30_lifo_oracle_anchor4.csv')
]

def get_col_by_pattern(df, pattern):
    cols = [col for col in df.columns if pattern in col]
    if not cols:
        return None
    return cols[0]

print(f"{'Method':<10} | {'Type':<6} | {'Lat Avg':<8} | {'Lat P95':<8} | {'Lat P99':<8} | {'FPS Avg':<8} | {'Count':<5}")
print("-" * 80)

for label, filename in files:
    filepath = os.path.join(test_data_path, filename)
    if not os.path.exists(filepath):
        continue
    
    df = pd.read_csv(filepath)
    fps_col = get_col_by_pattern(df, 'optimized_fps')
    queue_col = get_col_by_pattern(df, 'optimized_render_queue')
    
    # Filter by network type
    # Assuming 'file_name' is the first column
    df['is_bad'] = df['file_name'].astype(str).str.contains('bad')
    
    # Analyze Bad
    bad_df = df[df['is_bad']]
    if not bad_df.empty:
        stats = {
            'lat_avg': bad_df[queue_col].mean(),
            'lat_p95': bad_df[queue_col].quantile(0.95),
            'lat_p99': bad_df[queue_col].quantile(0.99),
            'fps_avg': bad_df[fps_col].mean(),
            'count': len(bad_df)
        }
        print(f"{label:<10} | {'Bad':<6} | {stats['lat_avg']:<8.2f} | {stats['lat_p95']:<8.2f} | {stats['lat_p99']:<8.2f} | {stats['fps_avg']:<8.2f} | {stats['count']:<5}")

    # Analyze Good
    good_df = df[~df['is_bad']]
    if not good_df.empty:
        stats = {
            'lat_avg': good_df[queue_col].mean(),
            'lat_p95': good_df[queue_col].quantile(0.95),
            'lat_p99': good_df[queue_col].quantile(0.99),
            'fps_avg': good_df[fps_col].mean(),
            'count': len(good_df)
        }
        print(f"{label:<10} | {'Good':<6} | {stats['lat_avg']:<8.2f} | {stats['lat_p95']:<8.2f} | {stats['lat_p99']:<8.2f} | {stats['fps_avg']:<8.2f} | {stats['count']:<5}")
    print("-" * 80)