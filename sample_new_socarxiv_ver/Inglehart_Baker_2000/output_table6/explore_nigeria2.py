"""
Nigeria Wave 3: try weighted approaches to get 87%
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")

wvs = pd.read_csv(wvs_path, low_memory=False)

nga3 = wvs[(wvs['S002VS'] == 3) & (wvs['S003'] == 566)].copy()
print(f"Nigeria Wave 3 rows: {len(nga3)}")
print(f"S017 range: {nga3['S017'].min()} to {nga3['S017'].max()}")
print(f"S017 non-null: {nga3['S017'].notna().sum()}")

# Show all weight columns
weight_cols = [c for c in nga3.columns if 'weight' in c.lower() or c.startswith('S017') or c.startswith('W')]
print(f"Weight-like cols: {weight_cols[:10]}")

# Look for all columns with 'S0' prefix that might be weight
s_cols = [c for c in nga3.columns if c.startswith('S0') or c.startswith('S1')]
print(f"S0/S1 cols: {s_cols}")

# Look for actual weight variable
for col in wvs.columns:
    if 'WEIGHT' in col.upper() or 'WGHT' in col.upper():
        print(f"  Found weight col: {col}")

# Try all weight columns in WVS
all_cols = list(wvs.columns)
print("\nAll WVS columns:")
for c in all_cols[:50]:
    print(f"  {c}")

# Nigeria standard approach
valid = nga3[nga3['F028'].isin([1,2,3,4,5,6,7,8])]
monthly = nga3[nga3['F028'].isin([1,2,3])]

# Try S017 weighted
w = valid['S017'].fillna(1.0)
mw = (valid['F028'].isin([1,2,3]) * w).sum()
tw = w.sum()
print(f"\nS017 weighted (all valid): {mw:.1f}/{tw:.1f} = {mw/tw*100:.1f}%")

# Maybe exclude value 7 from valid denominator?
valid_no7 = nga3[nga3['F028'].isin([1,2,3,4,5,6,8])]
m_no7 = nga3[nga3['F028'].isin([1,2,3])]
print(f"Excluding value 7 from denom: {len(m_no7)}/{len(valid_no7)} = {len(m_no7)/len(valid_no7)*100:.1f}%")

# Check what values are in Nigeria
print(f"\nNigeria F028 value counts:\n{nga3['F028'].value_counts().sort_index()}")
print("Note: Nigeria has 131 respondents with value 7 (less often)")
print("If value 7 is miscoded and should be missing, we get:")
# Without the 131 respondents with value 7
valid_7excl = nga3[nga3['F028'].isin([1,2,3,4,5,6,8])]
m = nga3[nga3['F028'].isin([1,2,3])]
print(f"  {len(m)}/{len(valid_7excl)} = {len(m)/len(valid_7excl)*100:.1f}%")
