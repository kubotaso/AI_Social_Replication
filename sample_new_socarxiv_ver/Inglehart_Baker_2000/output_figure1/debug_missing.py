#!/usr/bin/env python3
"""Debug missing countries PAK and GHA."""
import pandas as pd
import csv
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

FACTOR_ITEMS = ['A006', 'A042', 'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']

with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

cols = ['S002VS', 'COUNTRY_ALPHA', 'S020'] + FACTOR_ITEMS
available = [c for c in cols if c in header]
df = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
df = df[df['S002VS'].isin([2, 3])]

for code in ['PAK', 'GHA']:
    sub = df[df['COUNTRY_ALPHA'] == code]
    print(f"\n{code}: {len(sub)} rows in waves 2-3")
    if len(sub) > 0:
        print(f"  Waves: {sorted(sub['S002VS'].unique())}")
        print(f"  Years: {sorted(sub['S020'].unique())}")
        for item in FACTOR_ITEMS:
            if item in sub.columns:
                valid = sub[item].apply(lambda x: pd.to_numeric(x, errors='coerce'))
                valid = valid[valid >= 0]
                print(f"  {item}: {valid.notna().sum()} valid values out of {len(sub)}")
