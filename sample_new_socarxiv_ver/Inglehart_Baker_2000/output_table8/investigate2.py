"""Further investigation of India, Turkey, and other problem cases"""
import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F001','S017','X025','X001','X003'])

# India: paper says 28% W2, 23% W3. We get 33.7% W2, 25.2% W3
print("=== INDIA W2 ===")
ind = wvs[(wvs['COUNTRY_ALPHA']=='IND') & (wvs['S002VS']==2)]
f = ind['F001']
valid = f[f>0]
print(f"All valid: n={len(valid)}, Often={100*(valid==1).mean():.1f}%")

# Try filtering by education
for ed in [1,2,3,4,5,6,7,8]:
    sub = ind[(ind['X025']>=ed) & (ind['F001']>0)]
    if len(sub) > 0:
        pct = 100*(sub['F001']==1).mean()
        print(f"  X025>={ed}: n={len(sub)}, Often={pct:.1f}%")

# Try filtering by education = specific level
print("\nIndia W2 by education level:")
for ed in sorted(ind['X025'].unique()):
    sub = ind[(ind['X025']==ed) & (ind['F001']>0)]
    if len(sub) > 0:
        pct = 100*(sub['F001']==1).mean()
        print(f"  X025={ed}: n={len(sub)}, Often={pct:.1f}%")

print("\n=== INDIA W3 ===")
ind3 = wvs[(wvs['COUNTRY_ALPHA']=='IND') & (wvs['S002VS']==3)]
f3 = ind3['F001']
valid3 = f3[f3>0]
print(f"All valid: n={len(valid3)}, Often={100*(valid3==1).mean():.1f}%")

# Try different education filters
for ed in [1,2,3,4,5,6,7,8]:
    sub = ind3[(ind3['X025']>=ed) & (ind3['F001']>0)]
    if len(sub) > 0:
        pct = 100*(sub['F001']==1).mean()
        print(f"  X025>={ed}: n={len(sub)}, Often={pct:.1f}%")

# Turkey W3: paper says 50, we get 49.46
print("\n=== TURKEY W3 ===")
tur = wvs[(wvs['COUNTRY_ALPHA']=='TUR') & (wvs['S002VS']==3)]
f = tur['F001']
valid = f[f>0]
print(f"Valid n={len(valid)}, Often={100*(valid==1).mean():.3f}%")
# The value is 49.461%, which rounds to 49. Paper says 50.
# Maybe they excluded -2 (no answer) differently?
# Let's check: if we include -2 as NOT often:
total_with_no_answer = (f >= -2).sum() - (f==-2).sum()  # same as f>0
# What if they counted -1 (don't know) as valid?
all_non_missing = f[f >= -1]  # includes -1 (don't know)
pct_with_dk = 100 * (all_non_missing == 1).sum() / len(all_non_missing)
print(f"Including DK as valid: n={len(all_non_missing)}, Often={pct_with_dk:.1f}%")
# What if -1 is EXCLUDED differently?
print(f"F001 dist for Turkey W3: {dict(f.value_counts().sort_index())}")

# Spain EVS 1990: paper says 27, we get 28
print("\n=== SPAIN EVS 1990 ===")
evs = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
esp = evs[evs['c_abrv']=='ES']
q = esp['q322']
print(f"q322 distribution: {dict(q.value_counts().sort_index())}")
valid = q[q>0]
print(f"Valid n={len(valid)}, Often={100*(valid==1).mean():.2f}%")
# Paper says 27%. We get 28.15%.
# What if we include 0 values as valid (not "often")?
all_inc_zero = q[q>=0]
pct_zero = 100 * (all_inc_zero == 1).sum() / len(all_inc_zero)
print(f"Including 0 as valid: n={len(all_inc_zero)}, Often={pct_zero:.2f}%")

# What about using a specific year subset?
print(f"Spain years: {esp['year'].unique()}")

# Nigeria W2: paper says 60, we get 59
print("\n=== NIGERIA W2 ===")
nga = wvs[(wvs['COUNTRY_ALPHA']=='NGA') & (wvs['S002VS']==2)]
f = nga['F001']
print(f"F001 dist: {dict(f.value_counts().sort_index())}")
valid = f[f>0]
print(f"Valid n={len(valid)}, Often={100*(valid==1).mean():.2f}%")
# 58.79 rounds to 59, paper says 60. Very close but off by 1.

# China W2: paper says 30, we get 31
print("\n=== CHINA W2 ===")
chn = wvs[(wvs['COUNTRY_ALPHA']=='CHN') & (wvs['S002VS']==2)]
f = chn['F001']
print(f"F001 dist: {dict(f.value_counts().sort_index())}")
valid = f[f>0]
print(f"Valid n={len(valid)}, Often={100*(valid==1).mean():.2f}%")

# Brazil W2: paper says 44, we get 43
print("\n=== BRAZIL W2 ===")
bra = wvs[(wvs['COUNTRY_ALPHA']=='BRA') & (wvs['S002VS']==2)]
f = bra['F001']
valid = f[f>0]
wt = bra['S017'][f>0].fillna(1)
print(f"Valid n={len(valid)}, Often uw={100*(valid==1).mean():.2f}%, wt={100*((valid==1)*wt).sum()/wt.sum():.2f}%")
