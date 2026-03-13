#!/usr/bin/env python3
import pandas as pd
import numpy as np

# For the WVS data, let's check F034 coding more carefully
# In WVS Time Series, F034 is defined as:
# "Do you belong to a religious denomination? [IF YES] Which one?"
# Standard WVS coding: negative = missing, 0 = not applicable
# The actual denomination codes vary

# For the EVS 1990, F034 coding is:
# 1 = Roman Catholic
# 2 = Protestant (includes Reformed/Lutheran/etc.)
# 3 = Other

# Let me verify by checking the WVS CHE (Switzerland) data where we have F025
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F025','F034'],
                   low_memory=False)
wvs = wvs[wvs['S002VS'].isin([2,3])]

# Switzerland: known ~42% Catholic, ~33% Protestant
che = wvs[wvs['COUNTRY_ALPHA']=='CHE']
f025 = pd.to_numeric(che['F025'], errors='coerce')
f034 = pd.to_numeric(che['F034'], errors='coerce')

# Focus on F025=1 (Catholic) and F025=2 (Protestant)
# and see what F034 values they have
print("=== CHE: F025=1 (Catholic) → F034 ===")
cath = che[f025==1]
cath_f034 = pd.to_numeric(cath['F034'], errors='coerce')
print(cath_f034[cath_f034>=0].value_counts().sort_index())

print("\n=== CHE: F025=2 (Protestant) → F034 ===")
prot = che[f025==2]
prot_f034 = pd.to_numeric(prot['F034'], errors='coerce')
print(prot_f034[prot_f034>=0].value_counts().sort_index())

# Also check USA
usa = wvs[wvs['COUNTRY_ALPHA']=='USA']
f025_usa = pd.to_numeric(usa['F025'], errors='coerce')
f034_usa = pd.to_numeric(usa['F034'], errors='coerce')

print("\n=== USA: F025=1 (Catholic) → F034 ===")
cath_usa = usa[f025_usa==1]
print(pd.to_numeric(cath_usa['F034'], errors='coerce').pipe(lambda x: x[x>=0]).value_counts().sort_index())

print("\n=== USA: F025=2 (Protestant) → F034 ===")
prot_usa = usa[f025_usa==2]
print(pd.to_numeric(prot_usa['F034'], errors='coerce').pipe(lambda x: x[x>=0]).value_counts().sort_index())

# Summary: What proportion of F025=1 goes to F034=1 vs F034=2?
print("\n\n=== SUMMARY ===")
for c in ['CHE', 'DEU', 'USA']:
    csub = wvs[wvs['COUNTRY_ALPHA']==c]
    f025_c = pd.to_numeric(csub['F025'], errors='coerce')
    f034_c = pd.to_numeric(csub['F034'], errors='coerce')

    for denom, denom_name in [(1, 'Catholic'), (2, 'Protestant')]:
        mask = (f025_c == denom) & (f034_c >= 0)
        vals = f034_c[mask].value_counts().sort_index()
        total = mask.sum()
        if total > 0:
            print(f"{c} F025={denom}({denom_name}): F034 → {dict(vals)} (n={total})")
