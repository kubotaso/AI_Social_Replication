"""
Try to get South Korea 1981 = 29% (currently getting 27%)
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")

wvs = pd.read_csv(wvs_path, low_memory=False)

kor1 = wvs[(wvs['S002VS'] == 1) & (wvs['S003'] == 410)].copy()
print(f"Korea Wave 1 rows: {len(kor1)}")
print(f"Years: {sorted(kor1['S020'].unique())}")

print("\nF028 distribution:")
print(kor1['F028'].value_counts().sort_index())

# Currently: including -2 in denominator
# Standard approach (with -2 in denom):
valid_vals = [1, 2, 3, 4, 5, 6, 7, 8, -2]
valid = kor1[kor1['F028'].isin(valid_vals)]
monthly = kor1[kor1['F028'].isin([1, 2, 3])]
print(f"\nWith -2 in denom: {len(monthly)}/{len(valid)} = {len(monthly)/len(valid)*100:.1f}%")

# Without -2 in denominator
valid_no2 = kor1[kor1['F028'].isin([1, 2, 3, 4, 5, 6, 7, 8])]
m_no2 = kor1[kor1['F028'].isin([1, 2, 3])]
print(f"Without -2 in denom: {len(m_no2)}/{len(valid_no2)} = {len(m_no2)/len(valid_no2)*100:.1f}%")

# What about just the -4 and -2 codes?
neg_cols = kor1[kor1['F028'] < 0]
print(f"\nNegative F028 codes: {neg_cols['F028'].value_counts().sort_index()}")

# How many respondents total?
print(f"\nTotal rows: {len(kor1)}")
print(f"F028 NaN count: {kor1['F028'].isna().sum()}")

# To get 29% from ~1511 total (just the monthly out of valid)
# 29% * N = monthly
# Let's compute what N would give 29%
m = len(monthly)
target = 0.29
N_needed = m / target
print(f"\nMonthly count: {m}")
print(f"To get 29%, need denominator of: {N_needed:.1f}")
print(f"Current denominator (with -2): {len(valid)}")
print(f"Without -2: {len(valid_no2)}")

# Maybe -4 code should NOT be excluded?
all_f028 = kor1['F028'].dropna()
print(f"\nAll F028 unique values: {sorted(all_f028.unique())}")

# Check using F028 == -4 in denominator
neg4_vals = [1,2,3,4,5,6,7,8,-2,-4]
valid4 = kor1[kor1['F028'].isin(neg4_vals)]
print(f"With -2 and -4 in denom: {len(monthly)}/{len(valid4)} = {len(monthly)/len(valid4)*100:.1f}%")

# What about completely different denominator - all non-null?
all_valid = kor1[kor1['F028'].notna()]
print(f"All non-null F028: {len(monthly)}/{len(all_valid)} = {len(monthly)/len(all_valid)*100:.1f}%")

# The answer must be: to get 29%, use smaller denominator
# Try: exclude -2 from BOTH numerator and denominator (i.e. treat as if 7pt scale?)
valid_7pt = kor1[kor1['F028'].isin([1,2,3,4,5,6,7])]
monthly_7pt = kor1[kor1['F028'].isin([1,2,3])]  # same
print(f"7pt scale: {len(monthly_7pt)}/{len(valid_7pt)} = {len(monthly_7pt)/len(valid_7pt)*100:.1f}%")

# Different F028 value sets
print("\n--- Trying all reasonable value combinations ---")
for denom_vals in [[1,2,3,4,5,6,7], [1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8,-2], [1,2,3,4,6,7,8]]:
    v = kor1[kor1['F028'].isin(denom_vals)]
    m = kor1[kor1['F028'].isin([1,2,3])]
    if len(v) > 0:
        print(f"denom={denom_vals}: {len(m)}/{len(v)} = {len(m)/len(v)*100:.1f}%")
