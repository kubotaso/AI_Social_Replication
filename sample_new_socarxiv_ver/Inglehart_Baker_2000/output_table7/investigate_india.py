"""
Deep investigation of India 1990 to understand why WVS v5.0 gives 37% vs paper's 44%.
Try various subsetting approaches.
"""
import pandas as pd
import numpy as np
import math

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv', low_memory=False,
                  usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'S020', 'G006', 'S017', 'S006'])

def std_round(x):
    return math.floor(x + 0.5)

ind_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'IND') & (wvs['S002VS'] == 2)].copy()
print(f"India wave 2 total rows: {len(ind_w2)}")
print(f"Year distribution: {ind_w2['S020'].value_counts().sort_index()}")
print(f"S017 distribution: {ind_w2['S017'].value_counts().sort_index()}")

valid = ind_w2[(ind_w2['F063'] >= 1) & (ind_w2['F063'] <= 10)]
print(f"\nValid F063: {len(valid)}")
print(f"F063=10 count: {(valid['F063']==10).sum()}")
print(f"%10 = {(valid['F063']==10).mean()*100:.4f}%")
print(f"\nFor paper=44%: need {int(0.44 * len(valid))} 10s out of {len(valid)}")
print(f"But we have {(valid['F063']==10).sum()} 10s")
print(f"Deficit: {int(0.44 * len(valid)) - (valid['F063']==10).sum()} 10s")

# Check if any S006 (respondent number) range matters
if 'S006' in ind_w2.columns:
    print(f"\nS006 min={ind_w2['S006'].min()}, max={ind_w2['S006'].max()}")
    for n in [500, 1000, 1500, 2000, 2474]:
        sub = valid.nsmallest(n, 'S006') if 'S006' in valid.columns else valid.head(n)
        pct = (sub['F063'] == 10).mean() * 100
        print(f"  First {n} by S006: %10={pct:.4f}% -> {std_round(pct)}")

print("\n=== CHECK INDIA 1990 DETAILED ===")
print(f"S020 values: {ind_w2['S020'].unique()}")
print(f"F063 all values: {ind_w2['F063'].value_counts().sort_index()}")

# Check India 1995 similarly
ind_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'IND') & (wvs['S002VS'] == 3)].copy()
valid3 = ind_w3[(ind_w3['F063'] >= 1) & (ind_w3['F063'] <= 10)]
print(f"\n=== INDIA 1995 ===")
print(f"Total: {len(ind_w3)}, Valid: {len(valid3)}")
pct3 = (valid3['F063'] == 10).mean() * 100
print(f"%10 = {pct3:.4f}% -> std_round={std_round(pct3)}, ceil={math.ceil(pct3)}, paper=56")
print(f"n10={( valid3['F063']==10).sum()}, need {int(0.56*len(valid3))} for 56%")
print(f"S020: {ind_w3['S020'].unique()}")

for yr in ind_w3['S020'].unique():
    sub = ind_w3[ind_w3['S020'] == yr]
    v = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)]
    if len(v) > 0:
        pct = (v['F063'] == 10).mean() * 100
        print(f"  Year {yr}: N={len(v)}, %10={pct:.4f}% -> {std_round(pct)}")
