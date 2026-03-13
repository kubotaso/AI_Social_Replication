#!/usr/bin/env python3
import pandas as pd
import numpy as np

# The EVS data we have might be a processed version. Let me look at what the
# original EVS 1990 creation script did

# Let me also check if the WVS Time Series has NLD in any wave
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F025','F025_WVS'],
                   low_memory=False)

nld = wvs[wvs['COUNTRY_ALPHA']=='NLD']
print('NLD in WVS:')
for w in sorted(nld['S002VS'].unique()):
    wsub = nld[nld['S002VS']==w]
    print(f"  Wave {w}: n={len(wsub)}")

# Check if NLD has F025 in any wave
print('\nNLD F025 in all waves:')
nld_f025 = pd.to_numeric(nld['F025'], errors='coerce')
nld_f025_valid = nld_f025[nld_f025>=0]
if len(nld_f025_valid) > 0:
    print(nld_f025_valid.value_counts().sort_index())
else:
    print("No valid F025 for NLD in WVS")

# Check F025_WVS
print('\nNLD F025_WVS in all waves:')
nld_f025w = pd.to_numeric(nld['F025_WVS'], errors='coerce')
nld_f025w_valid = nld_f025w[nld_f025w>=0]
if len(nld_f025w_valid) > 0:
    print(nld_f025w_valid.value_counts().sort_index())
else:
    print("No valid F025_WVS for NLD in WVS")

# For the paper (Inglehart & Baker 2000), they used WVS waves 1-3 + EVS 1990
# Netherlands was in WVS wave 2 (1990) which is the same as EVS 1990
# The EVS 1990 data may have been used for NLD

# Let me check if NLD is in wave 2 of WVS
nld_w2 = nld[nld['S002VS']==2]
print(f'\nNLD wave 2: n={len(nld_w2)}')
if len(nld_w2) > 0:
    f025 = pd.to_numeric(nld_w2['F025'], errors='coerce')
    f025_valid = f025[f025>=0]
    print('F025:', f025_valid.value_counts().sort_index() if len(f025_valid)>0 else 'no valid')
