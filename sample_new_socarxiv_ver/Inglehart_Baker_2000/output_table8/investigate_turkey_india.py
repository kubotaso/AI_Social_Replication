"""Investigate Turkey and India deeper - these have large discrepancies"""
import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','S003','COUNTRY_ALPHA','F001','S020','S017','S018','X048WVS'])

# Turkey W3: paper says 50%, we get 49.5 (uw) or 47.4 (S017)
print("=== TURKEY W3 ===")
tur = wvs[(wvs['COUNTRY_ALPHA']=='TUR') & (wvs['S002VS']==3)]
print(f"N={len(tur)}, year={tur['S020'].unique()}")
f = tur['F001']
print(f"F001 dist: {dict(f.value_counts().sort_index())}")
valid = f[f>0]
print(f"Valid N={len(valid)}")
print(f"Often%: {100*(valid==1).mean():.3f}")
# Try using round(x, 0) - Python's banker's rounding
print(f"round(49.46)={round(49.46)}, round(49.5)={round(49.5)}, round(50.5)={round(50.5)}")
# Maybe int(x + 0.5)?
print(f"int(49.46+0.5)={int(49.46+0.5)}, int(49.5+0.5)={int(49.5+0.5)}")

# Maybe a different wave version? Check S002 (not S002VS)
wvs2 = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                    usecols=['S002','S002VS','S003','COUNTRY_ALPHA','F001','S020','S017'])
tur_all = wvs2[wvs2['COUNTRY_ALPHA']=='TUR']
print(f"\nTurkey all waves: S002={sorted(tur_all['S002'].unique())}, S002VS={sorted(tur_all['S002VS'].unique())}")
for w in sorted(tur_all['S002VS'].unique()):
    sub = tur_all[tur_all['S002VS']==w]
    v = sub['F001'][sub['F001']>0]
    if len(v) > 0:
        print(f"  W{w}: n={len(v)}, Often%={100*(v==1).mean():.1f}")

print("\n=== INDIA ===")
ind_all = wvs2[wvs2['COUNTRY_ALPHA']=='IND']
print(f"India all waves: S002={sorted(ind_all['S002'].unique())}, S002VS={sorted(ind_all['S002VS'].unique())}")
for w in sorted(ind_all['S002VS'].unique()):
    sub = ind_all[ind_all['S002VS']==w]
    v = sub['F001'][sub['F001']>0]
    if len(v) > 0:
        print(f"  W{w}: n={len(v)}, Often%={100*(v==1).mean():.1f}")

# India: paper says 28% for W2 and 23% for W3
# Our data: 33.7% for W2, 25.2% for W3
# This is a HUGE gap. Maybe the data version is different
# Or maybe they used a subset? Check if there's literate-only or urban-only subsample
wvs3 = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                    usecols=['S002VS','COUNTRY_ALPHA','F001','X025','X001','X003'],
                    low_memory=False)
ind_w2 = wvs3[(wvs3['COUNTRY_ALPHA']=='IND') & (wvs3['S002VS']==2)]
print(f"\nIndia W2 education (X025) dist: {dict(ind_w2['X025'].value_counts().sort_index())}")
print(f"India W2 sex (X001): {dict(ind_w2['X001'].value_counts().sort_index())}")

# What if we filter for literate respondents only?
# X025: education level. For India, 1=inadequately completed elementary, etc.
# Let's try excluding certain education levels
for ed_thresh in [1,2,3]:
    sub = ind_w2[(ind_w2['X025']>=ed_thresh) & (ind_w2['F001']>0)]
    if len(sub)>0:
        pct = 100*(sub['F001']==1).mean()
        print(f"  X025>={ed_thresh}: n={len(sub)}, Often%={pct:.1f}")
