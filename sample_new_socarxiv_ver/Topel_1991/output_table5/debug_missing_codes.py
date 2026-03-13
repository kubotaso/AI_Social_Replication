"""Check what 3-digit codes aren't being mapped"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')
df = df[df['tenure_topel'] >= 1].copy()
df = df.dropna(subset=['log_hourly_wage']).copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

# All unique occ_1digit values > 9
codes_3d = sorted(df[df['occ_1digit'] > 9]['occ_1digit'].unique())

# Check which Census code ranges are covered
mapped_ranges = [
    (1, 195, 'Professional'),
    (201, 245, 'Managers'),
    (260, 285, 'Sales'),
    (301, 395, 'Clerical'),
    (401, 580, 'Craftsmen'),
    (601, 695, 'Operatives'),
    (701, 785, 'Laborers/Transport'),
    (801, 824, 'Farmers'),
    (900, 965, 'Service'),
]

unmapped = []
for code in codes_3d:
    found = False
    for lo, hi, name in mapped_ranges:
        if lo <= code <= hi:
            found = True
            break
    if not found:
        n = len(df[df['occ_1digit'] == code])
        unmapped.append((code, n))

print("Unmapped 3-digit codes:")
for code, n in sorted(unmapped):
    print(f"  {code}: {n} rows")

# Also check Census code ranges 286-300 (between Sales and Clerical)
# and 581-600 (between Craftsmen and Operatives)
# and 696-700, 786-800, 825-899, 966-999
gap_ranges = [
    (246, 259, 'Between Managers and Sales'),
    (286, 300, 'Between Sales and Clerical'),
    (396, 400, 'Between Clerical and Craftsmen'),
    (581, 600, 'Between Craftsmen and Operatives'),
    (696, 700, 'Between Operatives and Transport'),
    (786, 800, 'Between Laborers and Farmers'),
    (825, 899, 'Between Farmers and Service'),
    (966, 999, 'After Service'),
]

for lo, hi, name in gap_ranges:
    codes_in_gap = [c for c in codes_3d if lo <= c <= hi]
    if codes_in_gap:
        total = sum(len(df[df['occ_1digit'] == c]) for c in codes_in_gap)
        print(f"\nGap {name} ({lo}-{hi}): {len(codes_in_gap)} codes, {total} rows")
        for c in codes_in_gap:
            n = len(df[df['occ_1digit'] == c])
            print(f"  {c}: {n}")

# Count total unmapped
total_unmapped = sum(n for _, n in unmapped)
print(f"\nTotal unmapped rows: {total_unmapped}")

# What if these go to BC? How many are union?
unmapped_codes = [c for c, _ in unmapped]
um_data = df[df['occ_1digit'].isin(unmapped_codes)]
print(f"Unmapped union_member distribution:")
print(um_data['union_member'].value_counts(dropna=False))

# What if we expand Sales to 260-300? And Clerical stays 301-395?
# Check how many codes 286-300 exist
codes_286_300 = [c for c in codes_3d if 286 <= c <= 300]
n_286_300 = sum(len(df[df['occ_1digit'] == c]) for c in codes_286_300)
print(f"\nCodes 286-300: {len(codes_286_300)} codes, {n_286_300} rows")

# Actually, looking at Census 1970 codes more carefully:
# 260-285: Sales Workers
# 301-395: Clerical Workers
# But Census actually has:
# 260-285: Sales
# 290-395: Clerical
# Let's check 286-289
codes_286_289 = [c for c in codes_3d if 286 <= c <= 289]
print(f"Codes 286-289: {codes_286_289}")
