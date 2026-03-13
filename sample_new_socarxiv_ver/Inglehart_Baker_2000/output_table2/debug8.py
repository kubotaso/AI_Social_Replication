#!/usr/bin/env python3
"""Comprehensive variable mapping check for Table 2."""
import pandas as pd
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

# In the WVS Time Series V5:
# F028 has range 1-8, with NGA=1.90 (frequent), SWE=6.43 (infrequent)
# This MUST be church attendance, not "believe in heaven"
#
# For "believe in heaven": likely F049, F051, or in the F-series
# Let's check F049 (believe in life after death), F050 (comfort from religion)
# F051 (believe in hell), F052 (believe in heaven?)
# Actually, in WVS the "Do you believe in: Heaven?" is variable F048 or F049

check_vars = ['F022', 'F028', 'F028B', 'F029', 'F031', 'F032', 'F033',
              'F040', 'F041', 'F042', 'F043', 'F044', 'F045', 'F046', 'F047',
              'F048', 'F049', 'F050', 'F051', 'F052', 'F053', 'F054', 'F055',
              'F057', 'F059', 'F060', 'F062', 'F064', 'F065']

with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

avail = ['S002VS', 'COUNTRY_ALPHA'] + [v for v in check_vars if v in header]
df = pd.read_csv(DATA_PATH, usecols=avail, low_memory=False)
df = df[df['S002VS'].isin([2, 3])]

print("=== Religion-related variables (waves 2-3) ===")
for v in check_vars:
    if v in df.columns:
        vals = pd.to_numeric(df[v], errors='coerce')
        pos = vals[vals >= 0].dropna()
        if len(pos) > 5000:
            nga = df[df['COUNTRY_ALPHA'] == 'NGA']
            swe = df[df['COUNTRY_ALPHA'] == 'SWE']
            usa = df[df['COUNTRY_ALPHA'] == 'USA']
            chn = df[df['COUNTRY_ALPHA'] == 'CHN']
            nga_v = pd.to_numeric(nga[v], errors='coerce').pipe(lambda x: x[x>=0]).dropna()
            swe_v = pd.to_numeric(swe[v], errors='coerce').pipe(lambda x: x[x>=0]).dropna()
            usa_v = pd.to_numeric(usa[v], errors='coerce').pipe(lambda x: x[x>=0]).dropna()
            chn_v = pd.to_numeric(chn[v], errors='coerce').pipe(lambda x: x[x>=0]).dropna()

            nga_m = f"{nga_v.mean():.2f}" if len(nga_v) > 0 else "N/A"
            swe_m = f"{swe_v.mean():.2f}" if len(swe_v) > 0 else "N/A"
            usa_m = f"{usa_v.mean():.2f}" if len(usa_v) > 0 else "N/A"
            chn_m = f"{chn_v.mean():.2f}" if len(chn_v) > 0 else "N/A"

            print(f"{v}: range={pos.min():.0f}-{pos.max():.0f}, N={len(pos)}")
            print(f"  NGA={nga_m}, SWE={swe_m}, USA={usa_m}, CHN={chn_m}")
            print(f"  values: {sorted(pos.unique())[:10]}")
