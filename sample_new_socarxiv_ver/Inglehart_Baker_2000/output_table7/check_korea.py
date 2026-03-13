"""
Check Korea F063 availability in WVS waves 1 and 2.
"""
import pandas as pd
import math

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv', low_memory=False,
                  usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'S020', 'S017'])

for wave, paper_val in [(1, 29), (2, 39)]:
    sub = wvs[(wvs['COUNTRY_ALPHA'] == 'KOR') & (wvs['S002VS'] == wave)]
    print(f"Korea wave {wave}: N={len(sub)}")
    print(f"F063 dist: {sub['F063'].value_counts().sort_index().to_dict()}")
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)]
    print(f"Valid F063: {len(valid)} (paper={paper_val})")
    if len(valid) > 0:
        pct = (valid['F063'] == 10).mean() * 100
        print(f"Korea wave {wave} %10 = {pct:.4f}% -> round={math.floor(pct+0.5)}")
    print()
