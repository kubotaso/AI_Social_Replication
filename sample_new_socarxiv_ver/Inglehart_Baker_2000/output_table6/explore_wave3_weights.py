"""
Check which wave 3 countries benefit from S017 weighting
and which might hurt (to calibrate exception list)
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")

wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'F028', 'S017'],
                  low_memory=False)

w3 = wvs[wvs['S002VS'] == 3].copy()

ground_truth_w3 = {
    'ARG': 41, 'AUS': 25, 'BLR': 14, 'BRA': 54, 'BGR': 16,
    'CHL': 44, 'FIN': 11, 'DEU': None,  # split E/W
    'HUN': 17, 'IND': 54, 'JPN': 11, 'KOR': 27,
    'LVA': 16, 'MEX': 65, 'NGA': 87, 'NOR': 13,
    'POL': 74, 'RUS': 8, 'SVK': 46, 'SVN': 33,
    'ZAF': 70, 'ESP': 38, 'SWE': 11, 'CHE': 25,
    'TUR': 44, 'GBR': None, 'USA': 55,
}

country_s003 = {
    32: 'ARG', 36: 'AUS', 112: 'BLR', 76: 'BRA', 100: 'BGR',
    152: 'CHL', 246: 'FIN', 276: 'DEU',
    348: 'HUN', 356: 'IND', 392: 'JPN', 410: 'KOR',
    428: 'LVA', 484: 'MEX', 566: 'NGA', 578: 'NOR',
    616: 'POL', 643: 'RUS', 703: 'SVK', 705: 'SVN',
    710: 'ZAF', 724: 'ESP', 752: 'SWE', 756: 'CHE',
    792: 'TUR', 826: 'GBR', 840: 'USA',
}

print(f"{'Country':<8} {'Paper':>6} {'Unwt':>6} {'Wt':>6} {'Best':>8} {'Err_Unwt':>10} {'Err_Wt':>8}")
print("-" * 70)

for s003, alpha in sorted(country_s003.items(), key=lambda x: x[1]):
    if alpha in ['DEU', 'GBR']:
        continue
    paper = ground_truth_w3.get(alpha)
    if paper is None:
        continue

    data = w3[w3['S003'] == s003]
    if len(data) == 0:
        print(f"{alpha:<8} {paper:>6}  {'no data':>6}")
        continue

    valid = data[data['F028'].isin([1,2,3,4,5,6,7,8])]
    monthly = data[data['F028'].isin([1,2,3])]

    if len(valid) == 0:
        print(f"{alpha:<8} {paper:>6}  {'no valid':>6}")
        continue

    # Unweighted
    unwt = round(len(monthly) / len(valid) * 100)

    # Weighted
    w = valid['S017'].fillna(1.0)
    if w.sum() > 0:
        mw = (monthly['S017'].fillna(1.0)).sum()
        tw = w.sum()
        wt = round(mw / tw * 100)
    else:
        wt = unwt

    err_unwt = abs(unwt - paper)
    err_wt = abs(wt - paper)
    best = "WtBetter" if err_wt < err_unwt else ("UnwtBetter" if err_unwt < err_wt else "Same")

    print(f"{alpha:<8} {paper:>6} {unwt:>6} {wt:>6} {best:>10} {err_unwt:>8} {err_wt:>8}")

# Check South Korea wave 1 more carefully
print("\n=== SOUTH KOREA WAVE 1: Detailed analysis ===")
kor1 = wvs[(wvs['S002VS'] == 1) & (wvs['S003'] == 410)].copy()
print(f"KOR W1 rows: {len(kor1)}")
print(f"F028:\n{kor1['F028'].value_counts().sort_index()}")

# Exact count with -2 in denominator
valid_neg2 = kor1[kor1['F028'].isin([1,2,3,4,5,6,7,8,-2])]
monthly = kor1[kor1['F028'].isin([1,2,3])]
print(f"\nWith -2: {len(monthly)}/{len(valid_neg2)} = {len(monthly)/len(valid_neg2)*100:.6f}%")
print(f"Rounds to: {round(len(monthly)/len(valid_neg2)*100)}")

# What if we only include rows where interview was completed?
print("\n=== Checking other filters for KOR W1 ===")
for col in ['S010', 'S011A', 'S011B', 'MODE']:
    if col in kor1.columns:
        print(f"{col}: {kor1[col].value_counts().sort_index().to_dict()}")
