"""
Try to get Nigeria 1995-1998 = 87% (currently getting 89%)
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")

wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'S020', 'F028', 'S017', 'COUNTRY_ALPHA'],
                  low_memory=False)

nga3 = wvs[(wvs['S002VS'] == 3) & (wvs['S003'] == 566)].copy()
print(f"Nigeria Wave 3 rows: {len(nga3)}")
print(f"Years: {sorted(nga3['S020'].unique())}")
print(f"F028:\n{nga3['F028'].value_counts().sort_index()}")

# Standard unweighted
valid = nga3[nga3['F028'].isin([1,2,3,4,5,6,7,8])]
monthly = nga3[nga3['F028'].isin([1,2,3])]
print(f"\nUnweighted: {len(monthly)}/{len(valid)} = {len(monthly)/len(valid)*100:.1f}%")

# Weighted
w = valid['S017'].fillna(1.0)
mw = (valid['F028'].isin([1,2,3]) * w).sum()
tw = w.sum()
print(f"Weighted S017: {mw:.1f}/{tw:.1f} = {mw/tw*100:.1f}%")

# What if Nigeria uses weights for wave 3?
# Target: 87%
# We need: X/N = 0.87
# Currently: M/N = 0.89
# So we need to reduce the numerator or increase the denominator
# Currently M = len(monthly), N = len(valid)
print(f"\nMonthly: {len(monthly)}, Valid: {len(valid)}")
print(f"Target: 87% = {0.87*len(valid):.1f} monthly")

# Include -2 in denominator
neg2_denom = nga3[nga3['F028'].isin([1,2,3,4,5,6,7,8,-2])]
m_neg2 = nga3[nga3['F028'].isin([1,2,3])]
print(f"\nWith -2 in denom: {len(m_neg2)}/{len(neg2_denom)} = {len(m_neg2)/len(neg2_denom)*100:.1f}%")

# Check F028 values more carefully
print(f"\nAll F028 values: {sorted(nga3['F028'].dropna().unique())}")
print(f"Total rows: {len(nga3)}, NaN: {nga3['F028'].isna().sum()}")
