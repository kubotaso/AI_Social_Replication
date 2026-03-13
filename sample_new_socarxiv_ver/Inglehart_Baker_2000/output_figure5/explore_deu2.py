#!/usr/bin/env python3
"""Explore EVS 1990 DEU data for West/East split and factor items."""
import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from shared_factor_analysis import FACTOR_ITEMS, recode_factor_items

evs = pd.read_csv(os.path.join(BASE_DIR, 'data/EVS_1990_wvs_format.csv'))
deu = evs[evs['COUNTRY_ALPHA'] == 'DEU'].copy()
print(f'DEU in EVS 1990: {len(deu)} rows')

# Check for any regional/split variable
print('DEU columns:', list(deu.columns))
print()

# Build factor items for DEU
if 'A006' in deu.columns: deu['GOD_IMP'] = deu['A006']
elif 'F063' in deu.columns: deu['GOD_IMP'] = deu['F063']
for v in ['A042', 'A034', 'A029']:
    if v in deu.columns:
        deu[v] = pd.to_numeric(deu[v], errors='coerce')
        deu[v] = deu[v].where(deu[v] >= 0, np.nan)
if all(v in deu.columns for v in ['A042', 'A034', 'A029']):
    deu['AUTONOMY'] = deu['A042'] + deu['A034'] - deu['A029']

deu_rec = recode_factor_items(deu)
deu_rec['F034'] = pd.to_numeric(deu['F034'], errors='coerce')

print('Factor items available:', [f for f in FACTOR_ITEMS if f in deu_rec.columns])
print()

# Factor item means by F034 group
deu_overall_mean = deu_rec[FACTOR_ITEMS].mean()
print(f'DEU overall: {deu_overall_mean.round(3).to_dict()}')
print()

for f034_val, label in [(1, 'F034=1 (Protestant)'), (2, 'F034=2 (Catholic)')]:
    g = deu_rec[deu_rec['F034'] == f034_val]
    if len(g) > 0:
        gmean = g[FACTOR_ITEMS].mean()
        print(f'{label} (n={len(g)}):')
        print(f'  {gmean.round(3).to_dict()}')
        print(f'  GOD_IMP={gmean["GOD_IMP"]:.2f}, AUTONOMY={gmean["AUTONOMY"]:.3f}')
        print()
