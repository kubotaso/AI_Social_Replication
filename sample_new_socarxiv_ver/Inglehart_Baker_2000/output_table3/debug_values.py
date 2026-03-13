#!/usr/bin/env python3
"""Debug script to check value distributions of key variables."""
import pandas as pd
import numpy as np
import csv

DATA_PATH = 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'
with open(DATA_PATH, 'r') as f:
    header = [h.strip('"') for h in next(csv.reader(f))]

usecols = ['S002VS','COUNTRY_ALPHA','A032','A035','D022','D058','E019','E015',
           'B002','B003','F125','A124_02','A124_06','A124_07','D018']
avail = [c for c in usecols if c in header]
df = pd.read_csv(DATA_PATH, usecols=avail, low_memory=False)
df = df[df['S002VS'].isin([2,3])]

for c in avail:
    if c not in ['S002VS','COUNTRY_ALPHA']:
        vc = df[c].value_counts().head(10)
        print(f'\n{c}:')
        for v, cnt in vc.items():
            print(f'  {v}: {cnt}')
