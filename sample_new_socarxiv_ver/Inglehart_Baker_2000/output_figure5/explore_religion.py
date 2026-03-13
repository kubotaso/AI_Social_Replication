#!/usr/bin/env python3
"""Explore religion variable F034 for the 6 target countries."""
import pandas as pd
import numpy as np

df = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                 usecols=['S002VS','S003','S024','COUNTRY_ALPHA','F034','F025','F025_WVS'],
                 low_memory=False)
df = df[df['S002VS'].isin([2,3])]

countries = ['DEU','CHE','NLD','IND','NGA','USA']
sub = df[df['COUNTRY_ALPHA'].isin(countries)]

print("=== F034 (religious denomination) ===")
for c in countries:
    csub = sub[sub['COUNTRY_ALPHA']==c]
    print(f'\n{c} (n={len(csub)}, S024 values: {sorted(csub["S024"].unique())}):')
    f034 = pd.to_numeric(csub['F034'], errors='coerce')
    f034_valid = f034[f034 >= 0]
    vc = f034_valid.value_counts().sort_index()
    for val, cnt in vc.items():
        print(f"  F034={int(val)}: {cnt}")

print("\n\n=== F025 (religious denomination - legacy) ===")
for c in countries:
    csub = sub[sub['COUNTRY_ALPHA']==c]
    print(f'\n{c}:')
    f025 = pd.to_numeric(csub['F025'], errors='coerce')
    f025_valid = f025[f025 >= 0]
    vc = f025_valid.value_counts().sort_index()
    for val, cnt in vc.items():
        print(f"  F025={int(val)}: {cnt}")

# Check West Germany specifically (S024=276001)
print("\n\n=== West Germany (S024=276001) F034 ===")
wg = df[(df['COUNTRY_ALPHA']=='DEU') & (df['S024']==276001)]
print(f"n={len(wg)}")
f034 = pd.to_numeric(wg['F034'], errors='coerce')
f034_valid = f034[f034 >= 0]
vc = f034_valid.value_counts().sort_index()
for val, cnt in vc.items():
    print(f"  F034={int(val)}: {cnt}")

# Also check East Germany
print("\n=== East Germany (S024=276002) F034 ===")
eg = df[(df['COUNTRY_ALPHA']=='DEU') & (df['S024']==276002)]
print(f"n={len(eg)}")
f034 = pd.to_numeric(eg['F034'], errors='coerce')
f034_valid = f034[f034 >= 0]
vc = f034_valid.value_counts().sort_index()
for val, cnt in vc.items():
    print(f"  F034={int(val)}: {cnt}")
