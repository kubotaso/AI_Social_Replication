#!/usr/bin/env python3
"""Explore EVS 1990 DEU data for religion coding."""
import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from shared_factor_analysis import FACTOR_ITEMS, recode_factor_items

evs = pd.read_csv(os.path.join(BASE_DIR, 'data/EVS_1990_wvs_format.csv'))
print('EVS 1990 countries:', evs['COUNTRY_ALPHA'].value_counts().sort_index().to_dict())
print()

# Check Germany (DEU or DEU_W/DEU_E?)
for code in ['DEU', 'DEU_W', 'DEU_E', 'WGR', 'FRG', 'GER']:
    if code in evs['COUNTRY_ALPHA'].values:
        subset = evs[evs['COUNTRY_ALPHA'] == code]
        print(f'{code}: {len(subset)} rows')
        for col in ['F034', 'F028', 'F025']:
            if col in subset.columns:
                vals = pd.to_numeric(subset[col], errors='coerce')
                vals_clean = vals[vals >= 0]
                print(f'  {col}: {vals_clean.value_counts().sort_index().to_dict()}')

# Check what country codes are in EVS that might be West Germany
print('\nAll EVS country codes:', sorted(evs['COUNTRY_ALPHA'].unique().tolist()))
