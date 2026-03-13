#!/usr/bin/env python3
import pandas as pd
import numpy as np

# F034 in WVS Time Series codebook:
# The variable description should tell us the coding
# Let's try to figure out F034 coding by looking at known distributions

# For USA (WVS): we know ~55% Protestant, ~25% Catholic
# F025 USA: 0=293(None), 1=382(Cath), 2=545(Prot), 9=238
# F034 USA: 1=1234, 2=251, 3=17
# So F034=1 is the majority → Protestant in USA; F034=2 is second → Catholic

# For Germany (WVS):
# F025: 0=961(None), 1=397(Cath), 2=600(Prot)
# F034: 1=917, 2=720, 3=330
# F025=0(None) → mostly F034=2(491) and F034=3(302)
# F025=1(Cath) → F034=1(303), F034=2(78)
# F025=2(Prot) → F034=1(441), F034=2(130)
# Hmm, both Catholic and Protestant map to F034=1!

# Actually wait - let me re-examine the cross-tab
# DEU: F034=1 includes F025=1(303 Cath) + F025=2(441 Prot) = 744 religious
# F034=2 includes F025=0(491 None) = non-religious
# F034=3 includes F025=0(302 None) = also non-religious
# So F034 for Germany: 1=Religious (any), 2=Non-religious, 3=Other/unaffiliated

# Actually F034 might not be denomination at all!
# Let me check the WVS codebook for F034

# F034 is "How often do you attend religious services?" in some waves
# or it could be something else. Let me check the actual variable description

# Let me look at F034 values more carefully across countries
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F034','F025','F028'],
                   low_memory=False)
wvs = wvs[wvs['S002VS'].isin([2,3])]

# Nigeria has clear Muslim population
# F025 NGA: 1=723(Cath), 2=1102(Prot), 3=176(Orth), 5=769(Muslim)
# F034 NGA: 1=2732, 2=150, 3=33
# F034=1 includes everyone!
# So F034 is not denomination-specific in WVS

# Let me use F025 for WVS countries and F034 for EVS countries where F025 is not available
# But EVS doesn't have F025!

# For EVS, let me check F028 which might be denomination
evs = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)

print("=== EVS F028 (religious belonging) ===")
for c in ['NLD', 'DEU', 'CHE', 'USA']:
    csub = evs[evs['COUNTRY_ALPHA']==c]
    if len(csub) > 0 and 'F028' in evs.columns:
        f028 = pd.to_numeric(csub['F028'], errors='coerce')
        f028_valid = f028[f028>=0]
        print(f'\n{c} F028:')
        for val, cnt in f028_valid.value_counts().sort_index().items():
            print(f"  {int(val)}: {cnt}")

print("\n\n=== EVS F034 cross-tab with F028 for NLD ===")
nld = evs[evs['COUNTRY_ALPHA']=='NLD']
if 'F028' in evs.columns and 'F034' in evs.columns:
    f028 = pd.to_numeric(nld['F028'], errors='coerce')
    f034 = pd.to_numeric(nld['F034'], errors='coerce')
    ct = pd.crosstab(f028[f028>=0], f034[f034>=0])
    print(ct)

# Check Switzerland EVS
print("\n=== CHE EVS F034 ===")
che_evs = evs[evs['COUNTRY_ALPHA']=='CHE']
if len(che_evs) > 0:
    f034 = pd.to_numeric(che_evs['F034'], errors='coerce')
    f034_valid = f034[f034>=0]
    print(f034_valid.value_counts().sort_index())
    if 'F028' in evs.columns:
        f028 = pd.to_numeric(che_evs['F028'], errors='coerce')
        ct = pd.crosstab(f028[f028>=0], f034[f034>=0])
        print('\nCHE F028 vs F034:')
        print(ct)

# Check DEU EVS F034
print("\n=== DEU EVS F034 ===")
deu_evs = evs[evs['COUNTRY_ALPHA']=='DEU']
if len(deu_evs) > 0:
    f034 = pd.to_numeric(deu_evs['F034'], errors='coerce')
    f034_valid = f034[f034>=0]
    print(f034_valid.value_counts().sort_index())
    if 'F028' in evs.columns:
        f028 = pd.to_numeric(deu_evs['F028'], errors='coerce')
        ct = pd.crosstab(f028[f028>=0], f034[f034>=0])
        print('\nDEU F028 vs F034:')
        print(ct)
