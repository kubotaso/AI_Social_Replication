#!/usr/bin/env python3
import pandas as pd

evs = pd.read_csv('data/EVS_1990_wvs_format.csv')
print('EVS columns:', list(evs.columns))
print()

# Check DEU in EVS for East/West split
deu_evs = evs[evs['COUNTRY_ALPHA']=='DEU']
print(f'DEU in EVS: n={len(deu_evs)}')
for col in ['S003', 'S024', 'S001']:
    if col in deu_evs.columns:
        print(f'  {col}: {sorted(deu_evs[col].unique())}')

# Check WVS for S024 column
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','S020','S003','S024'], low_memory=False)
deu_wvs = wvs[(wvs['COUNTRY_ALPHA']=='DEU') & (wvs['S002VS']==3)]
print(f'\nWVS DEU wave 3: S024 values: {sorted(deu_wvs["S024"].unique())}')
print(f'  n by S024:')
print(deu_wvs.groupby('S024').size())

# Check EVS for S024
if 'S024' in evs.columns:
    print(f'\nEVS DEU S024: {sorted(deu_evs["S024"].unique())}')
    print(deu_evs.groupby('S024').size())

# Also - the figure shows East Germany 90->97 and West Germany 81->97
# We need to split Germany into East/West
# In WVS: S024=276001 might be West, 276002 East (or similar)
# In EVS 1990: they might have separate codes

# Check if 900 codes for East/West Germany
print('\nLooking for East Germany codes in WVS and EVS...')
# S003=276 is Germany, but East Germany was 278 in some codings
deu_s3 = wvs[wvs['S003'].isin([276, 278])]
print(f'S003=276 or 278 in WVS waves 1-3:')
for w in [1,2,3]:
    sub = deu_s3[deu_s3['S002VS']==w]
    if len(sub) > 0:
        print(f'  Wave {w}: S003={sorted(sub.S003.unique())}, S024={sorted(sub.S024.unique())}, n={len(sub)}')
