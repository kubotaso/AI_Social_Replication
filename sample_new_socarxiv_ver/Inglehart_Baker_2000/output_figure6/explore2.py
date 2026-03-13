#!/usr/bin/env python3
import pandas as pd

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','S020','S003'], low_memory=False)
w1 = wvs[wvs['S002VS']==1]
print('WVS Wave 1 countries:')
for ca in sorted(w1['COUNTRY_ALPHA'].unique()):
    years = sorted(w1[w1['COUNTRY_ALPHA']==ca]['S020'].unique())
    print(f'  {ca}: {[int(y) for y in years]}')

print('\nWVS Wave 2 countries:')
w2 = wvs[wvs['S002VS']==2]
for ca in sorted(w2['COUNTRY_ALPHA'].unique()):
    years = sorted(w2[w2['COUNTRY_ALPHA']==ca]['S020'].unique())
    print(f'  {ca}: {[int(y) for y in years]}')

# Check if DEU has East/West split
print('\nDEU S003/S024 details:')
deu = wvs[wvs['COUNTRY_ALPHA']=='DEU']
for w in [1,2,3]:
    sub = deu[deu['S002VS']==w]
    if len(sub) > 0:
        print(f'  Wave {w}: S003={sorted(sub.S003.unique())}, n={len(sub)}')

# Check S024 for East/West
if 'S024' in wvs.columns:
    deu_w3 = deu[deu['S002VS']==3]
    print(f'  DEU wave 3 S024: {sorted(deu_w3["S024"].unique())}')

# Also check EVS for Germany
evs = pd.read_csv('data/EVS_1990_wvs_format.csv')
if 'DEU' in evs['COUNTRY_ALPHA'].values:
    deu_evs = evs[evs['COUNTRY_ALPHA']=='DEU']
    print(f'\nEVS DEU: n={len(deu_evs)}, years={sorted(deu_evs.S020.unique())}')
    if 'S003' in deu_evs.columns:
        print(f'  S003: {sorted(deu_evs.S003.unique())}')
    if 'S024' in deu_evs.columns:
        print(f'  S024: {sorted(deu_evs.S024.unique())}')
