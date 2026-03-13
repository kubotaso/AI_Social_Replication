#!/usr/bin/env python3
"""Explore EVS 1990 NLD data for factor items and religion coding."""
import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from shared_factor_analysis import FACTOR_ITEMS, recode_factor_items

evs = pd.read_csv(os.path.join(BASE_DIR, 'data/EVS_1990_wvs_format.csv'))
print('EVS 1990 all columns:', list(evs.columns))
print()

nld_evs = evs[evs['COUNTRY_ALPHA'] == 'NLD'].copy()
print(f'NLD EVS 1990: {len(nld_evs)} rows')

# Build GOD_IMP
if 'A006' in nld_evs.columns:
    nld_evs['GOD_IMP'] = nld_evs['A006']
    a6 = pd.to_numeric(nld_evs['GOD_IMP'], errors='coerce')
    print(f'GOD_IMP from A006: dist={a6[a6>=0].value_counts().sort_index().to_dict()}')
elif 'F063' in nld_evs.columns:
    nld_evs['GOD_IMP'] = nld_evs['F063']
    f63 = pd.to_numeric(nld_evs['GOD_IMP'], errors='coerce')
    print(f'GOD_IMP from F063: dist={f63[f63>=0].value_counts().sort_index().to_dict()}')

# Build AUTONOMY
for v in ['A042', 'A034', 'A029']:
    if v in nld_evs.columns:
        nld_evs[v] = pd.to_numeric(nld_evs[v], errors='coerce')
        nld_evs[v] = nld_evs[v].where(nld_evs[v] >= 0, np.nan)

if all(v in nld_evs.columns for v in ['A042', 'A034', 'A029']):
    nld_evs['AUTONOMY'] = nld_evs['A042'] + nld_evs['A034'] - nld_evs['A029']
    print(f'AUTONOMY: {nld_evs["AUTONOMY"].describe()}')

print()
print('Factor items available in EVS NLD:', [f for f in FACTOR_ITEMS if f in nld_evs.columns])
print('Factor items MISSING in EVS NLD:', [f for f in FACTOR_ITEMS if f not in nld_evs.columns])

print()
print('F034 distribution (denomination):')
if 'F034' in nld_evs.columns:
    f034 = pd.to_numeric(nld_evs['F034'], errors='coerce')
    print(f034[f034 >= 0].value_counts().sort_index().to_dict())
    # Try grouping by F034
    nld_evs['F034_num'] = f034
    nld_rec = recode_factor_items(nld_evs)
    for f034_code, label in [(1, 'F034=1'), (2, 'F034=2'), (3, 'F034=3')]:
        g = nld_rec[nld_rec['F034_num'] == f034_code]
        if len(g) > 0:
            means = g[FACTOR_ITEMS].mean()
            print(f'{label} (n={len(g)}): {means.to_dict()}')

print()
print('F028 distribution (belongs to religious group):')
if 'F028' in nld_evs.columns:
    f028 = pd.to_numeric(nld_evs['F028'], errors='coerce')
    print(f028[f028 >= 0].value_counts().sort_index().to_dict())
