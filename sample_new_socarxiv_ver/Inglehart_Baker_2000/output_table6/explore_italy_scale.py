import pandas as pd
import numpy as np

evs_long = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
it = evs_long[evs_long['country'] == 380]

print("Italy q336 value counts:")
print(it['q336'].value_counts().sort_index())

# Standard 8-point: (192+564+266)/1996 = 51.2%
# Try different scale exclusions:
for exclude in [[], [5], [8], [5,8], [4,5], [7,8]]:
    valid_vals = [v for v in [1,2,3,4,5,6,7,8] if v not in exclude]
    valid = it[it['q336'].isin(valid_vals)]
    monthly = it[it['q336'].isin([1,2,3])]
    if len(valid) > 0:
        pct = len(monthly)/len(valid)*100
        print(f"  Excluding {exclude}: {pct:.1f}% ({len(monthly)}/{len(valid)}) -> rounded={round(pct)}")

# Check if Italy has specific values that are unusual
# Value 5 = "other specific holy days" - might not be standard in Italy
# Italy has 193 respondents with value 5
# Paper Italy 1990 = 47%
# To get 47%: need 1022/2174 = 0.47 => need denom=2174 (currently 1996)
# Or: (smaller numerator)/1996 = 0.47 => numerator=938 (currently 1022)
# Or: 1022/(something) = 0.47 => something = 2174
# Adding value 8 respondents to valid: 1996+320-320 = 1996 (already counted)
# Including some other values: 1996 + X = 2174 => X = 178
# Hmm, that doesn't work with any single exclusion
print()

# Try weighted
for wt in ['weight_s', 'weight_g']:
    valid = it[it['q336'].isin([1,2,3,4,5,6,7,8])].copy()
    monthly_mask = valid['q336'].isin([1,2,3])
    wm = valid.loc[monthly_mask, wt].sum()
    wt_total = valid[wt].sum()
    if wt_total > 0:
        print(f"  Weighted ({wt}): {wm/wt_total*100:.1f}%")

# The paper says 47% for Italy 1990
# Our data gives 51% unweighted, 53% weighted
# The difference is likely due to a different data version
# No scale manipulation can bridge a 4pp gap without arbitrary exclusions

# Check Finland EVS more carefully
fi = evs_long[evs_long['country'] == 246]
print("\nFinland q336 value counts:")
print(fi['q336'].value_counts().sort_index())
# Try different scales
for exclude in [[], [5], [8], [5,8]]:
    valid_vals = [v for v in [1,2,3,4,5,6,7,8] if v not in exclude]
    valid = fi[fi['q336'].isin(valid_vals)]
    monthly = fi[fi['q336'].isin([1,2,3])]
    if len(valid) > 0:
        pct = len(monthly)/len(valid)*100
        print(f"  Excluding {exclude}: {pct:.1f}% -> rounded={round(pct)} (paper=13)")

# Check: if we use floor instead of round for Finland
valid_fi = fi[fi['q336'].isin([1,2,3,4,5,6,7,8])]
monthly_fi = fi[fi['q336'].isin([1,2,3])]
exact_pct = len(monthly_fi)/len(valid_fi)*100
print(f"\nFinland exact: {exact_pct:.4f}% - rounds to {round(exact_pct)}")
# Can we use math.floor or different rounding?
import math
print(f"  floor: {math.floor(exact_pct)}")
print(f"  ceil: {math.ceil(exact_pct)}")
print(f"  round: {round(exact_pct)}")

# Check South Korea W1 with different treatments
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','S003','F028','S017'], low_memory=False)
sk1 = wvs[(wvs['S003']==410)&(wvs['S002VS']==1)]
print(f"\nSouth Korea W1 F028:")
print(sk1['F028'].value_counts().sort_index())

# With -2: 266/970 = 27.4% -> rounds to 27
# Paper says 29
# Try S017 weights
valid_sk = sk1[sk1['F028'].isin([-2,1,2,3,4,6,7,8])]
monthly_sk = sk1[sk1['F028'].isin([1,2,3])]
print(f"S017 for S.Korea W1:")
print(f"  S017 unique values: {sorted(valid_sk['S017'].unique())}")
if valid_sk['S017'].notna().any():
    wm = monthly_sk['S017'].sum()
    wt = valid_sk['S017'].sum()
    if wt > 0:
        print(f"  Weighted: {wm/wt*100:.1f}%")
