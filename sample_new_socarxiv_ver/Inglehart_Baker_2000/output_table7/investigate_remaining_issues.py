"""
Investigate remaining wrong cells to find fixes for attempt 14.
Focus on:
1. India 1990: WVS gives 37%, paper=44 - explore variable alternatives
2. East Germany 1995: WVS gives 9%, paper=6 - find the right subset
3. South Africa 1981: WVS gives 53%, paper=50 - find the right approach
4. India 1995: WVS gives 54%, paper=56 - confirm unfixable
"""
import pandas as pd
import numpy as np
import math

wvs_path = "data/WVS_Time_Series_1981-2022_csv_v5_0.csv"

# Load relevant columns
wvs = pd.read_csv(wvs_path, low_memory=False,
                  usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'S020', 'G006', 'S017'])

print("=== INDIA 1990 (WVS wave 2) ===")
ind_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'IND') & (wvs['S002VS'] == 2)]
print(f"Total rows: {len(ind_w2)}")
print(f"F063 distribution:\n{ind_w2['F063'].value_counts().sort_index()}")
valid = ind_w2[(ind_w2['F063'] >= 1) & (ind_w2['F063'] <= 10)]
print(f"Valid F063 N={len(valid)}")
if len(valid) > 0:
    pct = (valid['F063'] == 10).mean() * 100
    w = ind_w2.loc[valid.index, 'S017']
    valid_w = w[w > 0]
    if len(valid_w) == len(w):
        pct_w = ((valid['F063'] == 10) * w).sum() / w.sum() * 100
    else:
        pct_w = pct
    print(f"Unweighted %10: {pct:.4f}% (n10={(valid['F063']==10).sum()}, n_valid={len(valid)})")
    print(f"Weighted %10: {pct_w:.4f}%")
    print(f"floor={math.floor(pct)}, std_round={math.floor(pct+0.5)}, ceil={math.ceil(pct)}")
    print(f"Paper=44: how many 10s needed? {int(0.44 * len(valid))} vs actual {(valid['F063']==10).sum()}")

# Check year distribution for India wave 2
print(f"\nIndia wave 2 year distribution:\n{ind_w2['S020'].value_counts().sort_index()}")

print("\n=== INDIA 1995 (WVS wave 3) ===")
ind_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'IND') & (wvs['S002VS'] == 3)]
print(f"Total rows: {len(ind_w3)}")
valid = ind_w3[(ind_w3['F063'] >= 1) & (ind_w3['F063'] <= 10)]
print(f"Valid F063 N={len(valid)}")
if len(valid) > 0:
    pct = (valid['F063'] == 10).mean() * 100
    print(f"Unweighted %10: {pct:.4f}% -> floor={math.floor(pct)}, std={math.floor(pct+0.5)}, ceil={math.ceil(pct)}")
    print(f"Paper=56: need {int(0.56 * len(valid))} 10s, have {(valid['F063']==10).sum()}")
    print(f"diff from paper: {pct - 56:.4f}%")
# Check year distribution
print(f"\nIndia wave 3 year distribution:\n{ind_w3['S020'].value_counts().sort_index()}")

print("\n=== EAST GERMANY 1995 (WVS wave 3) ===")
deu_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'DEU') & (wvs['S002VS'] == 3)]
print(f"Total DEU w3: {len(deu_w3)}")
print(f"G006 distribution:\n{deu_w3['G006'].value_counts().sort_index()}")

for g006_vals, label in [
    ([2, 3], 'G006=[2,3] (standard East)'),
    ([2], 'G006=2 only'),
    ([3], 'G006=3 only'),
    ([2, 3, 5], 'G006=[2,3,5]'),
]:
    sub = deu_w3[deu_w3['G006'].isin(g006_vals)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)]
    if len(valid) > 0:
        pct = (valid['F063'] == 10).mean() * 100
        w = sub.loc[valid.index, 'S017']
        pct_w = ((valid['F063'] == 10) * w).sum() / w.sum() * 100 if w.gt(0).all() else pct
        print(f"  {label}: N_valid={len(valid)}, unweighted={pct:.4f}% -> {math.floor(pct+0.5)}, weighted={pct_w:.4f}% -> {math.floor(pct_w+0.5)}")

# F063 distribution for East Germany
sub_east = deu_w3[deu_w3['G006'].isin([2,3])]
valid_east = sub_east[(sub_east['F063'] >= 1) & (sub_east['F063'] <= 10)]
print(f"\nEast Germany F063 distribution:\n{valid_east['F063'].value_counts().sort_index()}")
print(f"n10={(valid_east['F063']==10).sum()}, N_valid={len(valid_east)}")
print(f"For paper=6: need N_valid * 0.06 = {len(valid_east) * 0.06:.1f} 10s, have {(valid_east['F063']==10).sum()}")

# S020 year breakdown
print(f"\nDEU w3 year distribution:\n{deu_w3['S020'].value_counts().sort_index()}")

print("\n=== SOUTH AFRICA 1981 (WVS wave 1) ===")
zaf_w1 = wvs[(wvs['COUNTRY_ALPHA'] == 'ZAF') & (wvs['S002VS'] == 1)]
print(f"Total rows: {len(zaf_w1)}")
valid = zaf_w1[(zaf_w1['F063'] >= 1) & (zaf_w1['F063'] <= 10)]
print(f"Valid F063 N={len(valid)}")
if len(valid) > 0:
    pct = (valid['F063'] == 10).mean() * 100
    w = zaf_w1.loc[valid.index, 'S017']
    pct_w = ((valid['F063'] == 10) * w).sum() / w.sum() * 100 if w.gt(0).all() else pct
    print(f"Unweighted %10: {pct:.4f}% -> round={math.floor(pct+0.5)}, floor={math.floor(pct)}, ceil={math.ceil(pct)}")
    print(f"Weighted %10: {pct_w:.4f}% -> round={math.floor(pct_w+0.5)}")
    print(f"\nF063 distribution:\n{valid['F063'].value_counts().sort_index()}")
    print(f"\nPaper=50, diff={pct - 50:.4f}%")
    n10 = (valid['F063'] == 10).sum()
    ntotal = len(valid)
    print(f"n10={n10}, ntotal={ntotal}, exact %={n10*100/ntotal:.4f}%")
    # Need n10/N = 50% -> N = n10/0.5
    print(f"For %=50: need {int(n10/0.5)} total records, have {ntotal}")
    print(f"\nS017 (weight) stats: min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}")

# Check if S020 year matters
print(f"\nZAF w1 year distribution:\n{zaf_w1['S020'].value_counts().sort_index()}")

print("\n=== WVS WAVE 2 INDIA: Check if different valid range matters ===")
ind_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'IND') & (wvs['S002VS'] == 2)]
for min_v, max_v in [(1,10), (0,10), (1,9)]:
    v = ind_w2[(ind_w2['F063'] >= min_v) & (ind_w2['F063'] <= max_v)]
    if len(v) > 0:
        pct = (v['F063'] == 10).mean() * 100 if max_v == 10 else 0
        if max_v == 10:
            p10 = (v['F063'] == 10).mean() * 100
        else:
            p10 = (ind_w2[(ind_w2['F063'] >= min_v) & (ind_w2['F063'] <= max_v)]['F063'] == 10).mean() * 100
        print(f"  range [{min_v},{max_v}]: N={len(v)}, %10 of range = n/a, need different calc")

    # How many = 10?
    n10 = (ind_w2['F063'] == 10).sum()
    total_range = len(ind_w2[(ind_w2['F063'] >= min_v) & (ind_w2['F063'] <= max_v)])
    if total_range > 0:
        pct = n10 * 100 / total_range
        print(f"  range [{min_v},{max_v}]: N_denom={total_range}, %10={pct:.4f}% -> round={math.floor(pct+0.5)}")
