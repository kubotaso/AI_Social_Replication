#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

df = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                 usecols=['S002VS','S003','S024','COUNTRY_ALPHA'],
                 low_memory=False)
df = df[df['S002VS'].isin([2,3])]
deu = df[df['COUNTRY_ALPHA']=='DEU']
print('DEU S024:', sorted(deu['S024'].unique()))
print('DEU S003:', sorted(deu['S003'].unique()))
nld = df[df['COUNTRY_ALPHA']=='NLD']
print('NLD S024:', sorted(nld['S024'].unique()))
print('NLD rows:', len(nld))

# Check EVS
evs_cols = pd.read_csv('data/EVS_1990_wvs_format.csv', nrows=0).columns.tolist()
print('\nEVS columns with F:', [c for c in evs_cols if c.startswith('F')])
evs = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)
nld_evs = evs[evs['COUNTRY_ALPHA']=='NLD']
print('NLD EVS rows:', len(nld_evs))
deu_evs = evs[evs['COUNTRY_ALPHA']=='DEU']
print('DEU EVS rows:', len(deu_evs))
if 'S024' in evs.columns:
    print('DEU EVS S024:', sorted(deu_evs['S024'].unique()))

# Check F025 in EVS for NLD
if 'F025' in evs.columns and len(nld_evs) > 0:
    f025 = pd.to_numeric(nld_evs['F025'], errors='coerce')
    f025_valid = f025[f025 >= 0]
    print('\nNLD EVS F025:')
    for val, cnt in f025_valid.value_counts().sort_index().items():
        print(f"  F025={int(val)}: {cnt}")

# Check F034 in EVS
if 'F034' in evs.columns:
    print('\nF034 available in EVS')
    for c in ['DEU','NLD','CHE','IND','NGA','USA']:
        csub = evs[evs['COUNTRY_ALPHA']==c]
        if len(csub) > 0:
            f034 = pd.to_numeric(csub['F034'], errors='coerce')
            f034_valid = f034[f034 >= 0]
            print(f'{c}: {f034_valid.value_counts().sort_index().to_dict()}')
else:
    print('\nF034 NOT in EVS')
