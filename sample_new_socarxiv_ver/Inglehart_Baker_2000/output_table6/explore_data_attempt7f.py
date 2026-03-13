"""
Test if Finland 7-point fix is specific to Finland or helps broadly
Also check South Korea 1981 options
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
evs_stata_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")

evs = pd.read_stata(evs_stata_path, convert_categoricals=False,
                     columns=['c_abrv', 'country1', 'q336', 'year', 'weight_g'])

# Paper values for comparison
paper_values_evs = {
    'BE': ('Belgium', 35), 'CA': ('Canada', 40), 'FI': ('Finland', 13),
    'FR': ('France', 17), 'GB-GBN': ('Great Britain', 25),
    'IS': ('Iceland', 9), 'IE': ('Ireland', 88), 'GB-NIR': ('Northern Ireland', 69),
    'IT': ('Italy', 47), 'LV': ('Latvia', 9), 'NL': ('Netherlands', 31),
    'NO': ('Norway', 13), 'PL': ('Poland', 85), 'SE': ('Sweden', 10),
    'SI': ('Slovenia', 35), 'US': ('United States', 59),
}

# Also Hungary (special 7-pt scale)
hun_paper = 34

print("Testing 8-pt vs 7-pt denominators for all EVS countries:")
print(f"{'Country':<25} {'8pt':>6} {'7pt':>6} {'Paper':>6} {'7pt better?':>12}")
print("-"*65)

for alpha, (name, paper_val) in paper_values_evs.items():
    c = evs[evs['c_abrv'] == alpha]
    if len(c) == 0:
        continue
    valid_8 = c[c['q336'].isin([1,2,3,4,5,6,7,8])]
    valid_7 = c[c['q336'].isin([1,2,3,4,5,6,7])]
    monthly = c[c['q336'].isin([1,2,3])]

    pct_8 = round(len(monthly)/len(valid_8)*100) if len(valid_8) > 0 else None
    pct_7 = round(len(monthly)/len(valid_7)*100) if len(valid_7) > 0 else None

    diff_8 = abs(pct_8 - paper_val) if pct_8 else 99
    diff_7 = abs(pct_7 - paper_val) if pct_7 else 99
    better = "YES" if diff_7 < diff_8 else ("SAME" if diff_7 == diff_8 else "NO")

    print(f"  {name:<23} {pct_8:>5}% {pct_7:>5}% {paper_val:>5}% {better:>12}")

print("\n=== South Korea 1981 analysis ===")
wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'S020', 'F028', 'S017'], low_memory=False)
w1 = wvs[wvs['S002VS'] == 1]
kor1 = w1[w1['S003'] == 410]
print("Korea wave 1 rows:", len(kor1))
print("F028 dist:", dict(kor1['F028'].value_counts().sort_index()))
# Paper says 29%

# All different denominator approaches
for valid_vals, name in [([1,2,3,4,5,6,7,8], '8pt'), ([1,2,3,4,5,6,7], '7pt'),
                          ([1,2,3,4,5,6,7,8,-2], '8pt+(-2)'), ([1,2,3,4,5,6,7,-2], '7pt+(-2)'),
                          ([1,2,3,4,5,6,7,8,0], '8pt+0')]:
    valid = kor1[kor1['F028'].isin(valid_vals)]
    monthly = kor1[kor1['F028'].isin([1,2,3])]
    if len(valid) > 0:
        pct = round(len(monthly)/len(valid)*100)
        print(f"  Korea {name}: {pct}% ({len(monthly)}/{len(valid)})")

# Try checking if some F028 value > 8 exists for Korea
print("F028 all unique:", sorted(kor1['F028'].unique()))

# WVS 1981 Hungary check (paper=16)
print("\n=== Hungary WVS wave 1 ===")
hun1 = w1[w1['S003'] == 348]
print("Hungary wave 1 rows:", len(hun1))
print("F028 dist:", dict(hun1['F028'].value_counts().sort_index()))
for valid_vals in [[1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7], [1,2,3,4,5,6,7,8,-2]]:
    valid = hun1[hun1['F028'].isin(valid_vals)]
    monthly = hun1[hun1['F028'].isin([1,2,3])]
    if len(valid) > 0:
        pct = round(len(monthly)/len(valid)*100)
        print(f"  Hungary {valid_vals}: {pct}% ({len(monthly)}/{len(valid)})")

# WVS 1990-1991 Hungary check (paper=34)
print("\n=== Hungary WVS wave 2 ===")
w2 = wvs[wvs['S002VS'] == 2]
hun2 = w2[w2['S003'] == 348]
print("Hungary wave 2 rows:", len(hun2))
if len(hun2) > 0:
    print("F028 dist:", dict(hun2['F028'].value_counts().sort_index()))

# Check Nigeria wave 3 more carefully
print("\n=== Nigeria WVS wave 3 (paper=87%) ===")
w3 = wvs[wvs['S002VS'] == 3]
nga3 = w3[w3['S003'] == 566]
print("Nigeria wave 3 rows:", len(nga3))
print("F028 dist:", dict(nga3['F028'].value_counts().sort_index()))
for valid_vals in [[1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7]]:
    valid = nga3[nga3['F028'].isin(valid_vals)]
    monthly = nga3[nga3['F028'].isin([1,2,3])]
    if len(valid) > 0:
        pct_uw = round(len(monthly)/len(valid)*100)
        pct_w = round(monthly['S017'].sum()/valid['S017'].sum()*100) if valid['S017'].sum() > 0 else None
        print(f"  Nigeria {valid_vals}: uw={pct_uw}%, w={pct_w}%")

# South Korea WVS wave 2 verification (paper=60)
print("\n=== South Korea WVS wave 2 (paper=60%) ===")
kor2 = w2[w2['S003'] == 410]
print("Korea wave 2 rows:", len(kor2))
print("F028 dist:", dict(kor2['F028'].value_counts().sort_index()))
for valid_vals in [[1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7], [1,2,3,4,5,6,7,8,-2]]:
    valid = kor2[kor2['F028'].isin(valid_vals)]
    monthly = kor2[kor2['F028'].isin([1,2,3])]
    if len(valid) > 0:
        pct = round(len(monthly)/len(valid)*100)
        print(f"  Korea W2 {valid_vals}: {pct}% ({len(monthly)}/{len(valid)})")
