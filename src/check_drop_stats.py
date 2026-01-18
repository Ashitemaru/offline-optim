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

print(f"{'Method':<15} | {'Failed Drops':<12} | {'Missed Drops':<12} | {'Total Drops':<12} | {'Fail Rate':<10}")
print("-" * 70)

for label, filename in files:
    filepath = os.path.join(test_data_path, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
    
    df = pd.read_csv(filepath)
    
    # Identify key columns for drop statistics
    # Note: These column names are based on inspection of simulator_v2.py or previous csv outputs
    # Pattern likely contains 'quick_drop'
    failed_col = get_col_by_pattern(df, 'quick_drop_failed_cnt')
    missed_col = get_col_by_pattern(df, 'quick_drop_missed_cnt')
    total_col = get_col_by_pattern(df, 'quick_drop_total_cnt')
    
    if failed_col and missed_col and total_col:
        failed = df[failed_col].sum() # Summing across all traces to get total count
        missed = df[missed_col].sum()
        total = df[total_col].sum()
        
        # Calculate rates based on average per trace or just raw sums? 
        # Usually average per trace is better for magnitude, but sum shows total events.
        # Let's show average per trace to be consistent with other metrics
        
        avg_failed = df[failed_col].mean()
        avg_missed = df[missed_col].mean()
        avg_total = df[total_col].mean()
        
        fail_rate = (avg_failed / avg_total * 100) if avg_total > 0 else 0
        
        print(f"{label:<15} | {avg_failed:<12.2f} | {avg_missed:<12.2f} | {avg_total:<12.2f} | {fail_rate:<9.2f}%")
    else:
        print(f"{label}: Columns not found")
