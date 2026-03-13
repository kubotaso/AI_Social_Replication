#!/usr/bin/env python3
"""
Check: does log(labor_inc/hours) differ from log_hourly_wage?
And: what is the SE of regression for different wage constructions?
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')

# Compare wage measures
df['my_hourly_wage'] = df['labor_inc'] / df['hours']
df['my_log_wage'] = np.log(df['my_hourly_wage'])

# Filter to valid observations
valid = (df['hours'] > 0) & (df['labor_inc'] > 0) & df['log_hourly_wage'].notna() & np.isfinite(df['my_log_wage'])
df_v = df[valid].copy()

print(f"Observations with valid wages: {len(df_v)}")
print(f"\nlog_hourly_wage vs log(labor_inc/hours):")
print(f"  Correlation: {df_v['log_hourly_wage'].corr(df_v['my_log_wage']):.6f}")
diff = df_v['log_hourly_wage'] - df_v['my_log_wage']
print(f"  Mean difference: {diff.mean():.6f}")
print(f"  Std of difference: {diff.std():.6f}")
print(f"  Max abs difference: {diff.abs().max():.6f}")
print(f"  Exact matches: {(diff.abs() < 1e-10).sum()}")
print(f"  Close matches (<0.001): {(diff.abs() < 0.001).sum()}")

# Check if hourly_wage = wages/hours or labor_inc/hours
df_v['from_wages'] = np.log(df_v['wages'] / df_v['hours'])
diff2 = df_v['log_hourly_wage'] - df_v['from_wages']
print(f"\nlog_hourly_wage vs log(wages/hours):")
print(f"  Correlation: {df_v['log_hourly_wage'].corr(df_v['from_wages']):.6f}")
print(f"  Mean difference: {diff2.mean():.6f}")
print(f"  Std of difference: {diff2.std():.6f}")
print(f"  Close matches (<0.001): {(diff2.abs() < 0.001).sum()}")

# Check if wages != labor_inc
diff3 = df_v['wages'] - df_v['labor_inc']
print(f"\nwages vs labor_inc:")
print(f"  Exact matches: {(diff3 == 0).sum()}")
print(f"  Non-matches: {(diff3 != 0).sum()}")
print(f"  Mean diff: {diff3.mean():.2f}")

# What is 'wages' vs 'labor_inc'?
print(f"\nwages stats: mean={df_v['wages'].mean():.1f}, std={df_v['wages'].std():.1f}")
print(f"labor_inc stats: mean={df_v['labor_inc'].mean():.1f}, std={df_v['labor_inc'].std():.1f}")
print(f"hourly_wage stats: mean={df_v['hourly_wage'].mean():.2f}")
print(f"hours stats: mean={df_v['hours'].mean():.1f}")

# Check what the data construction script does
print(f"\nlog_hourly_wage: mean={df_v['log_hourly_wage'].mean():.4f}, std={df_v['log_hourly_wage'].std():.4f}")
print(f"my_log_wage:     mean={df_v['my_log_wage'].mean():.4f}, std={df_v['my_log_wage'].std():.4f}")

# Check: hourly_wage column vs wages/hours vs labor_inc/hours
df_v['hw_check'] = df_v['wages'] / df_v['hours']
diff4 = df_v['hourly_wage'] - df_v['hw_check']
print(f"\nhourly_wage vs wages/hours:")
print(f"  Close matches (<0.01): {(diff4.abs() < 0.01).sum()}")
print(f"  Non-matches: {(diff4.abs() >= 0.01).sum()}")

df_v['hw_check2'] = df_v['labor_inc'] / df_v['hours']
diff5 = df_v['hourly_wage'] - df_v['hw_check2']
print(f"\nhourly_wage vs labor_inc/hours:")
print(f"  Close matches (<0.01): {(diff5.abs() < 0.01).sum()}")
print(f"  Non-matches: {(diff5.abs() >= 0.01).sum()}")
