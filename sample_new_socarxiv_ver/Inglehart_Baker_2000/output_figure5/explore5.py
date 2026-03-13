#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Check EVS for F025 in NLD and also check if EVS has X048WVS
evs = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)
print('EVS columns:', sorted(evs.columns.tolist()))

# Check NLD F025
nld = evs[evs['COUNTRY_ALPHA']=='NLD']
if 'F025' in evs.columns:
    f025 = pd.to_numeric(nld['F025'], errors='coerce')
    f025_valid = f025[f025>=0]
    print('\nNLD EVS F025:')
    for val, cnt in f025_valid.value_counts().sort_index().items():
        print(f"  {int(val)}: {cnt}")

# NLD F034
if 'F034' in evs.columns:
    nld_f034 = pd.to_numeric(nld['F034'], errors='coerce')
    nld_f034_valid = nld_f034[nld_f034>=0]
    print('\nNLD EVS F034:')
    for val, cnt in nld_f034_valid.value_counts().sort_index().items():
        print(f"  {int(val)}: {cnt}")

# Cross-tab F025 vs F034 for NLD
if 'F025' in evs.columns and 'F034' in evs.columns:
    f025 = pd.to_numeric(nld['F025'], errors='coerce')
    f034 = pd.to_numeric(nld['F034'], errors='coerce')
    ct = pd.crosstab(f025[f025>=0], f034[f034>=0])
    print('\nNLD EVS F025 vs F034:')
    print(ct)

# Check what factor items are available in EVS for NLD
factor_items = ['GOD_IMP', 'AUTONOMY', 'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
for item in factor_items:
    if item in evs.columns:
        nld_item = pd.to_numeric(nld[item], errors='coerce')
        valid = nld_item[nld_item>=0]
        print(f'{item}: {len(valid)} valid values')
    else:
        print(f'{item}: NOT in EVS')

# Check X048WVS in WVS
import csv
with open('data/WVS_Time_Series_1981-2022_csv_v5_0.csv', 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]
print('\nX048WVS in WVS:', 'X048WVS' in header)

# Check if F025 is in WVS
print('F025 in WVS:', 'F025' in header)
print('F025_WVS in WVS:', 'F025_WVS' in header)
