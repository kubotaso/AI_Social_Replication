#!/usr/bin/env python3
"""Debug Ghana availability."""
import pandas as pd
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

df = pd.read_csv(DATA_PATH, usecols=['S002VS', 'COUNTRY_ALPHA', 'S020'], low_memory=False)

# Check Ghana across all waves
gha = df[df['COUNTRY_ALPHA'] == 'GHA']
print(f"Ghana total rows: {len(gha)}")
for wave in sorted(gha['S002VS'].unique()):
    sub = gha[gha['S002VS'] == wave]
    years = sorted(sub['S020'].unique())
    print(f"  Wave {wave}: {len(sub)} rows, years {years}")

# The paper uses 1990-1998 data. Ghana may be in wave 4 (1999-2004)
# But the paper was published in 2000, so it likely uses data up to ~1998
# Let's check wave 4
gha4 = gha[gha['S002VS'] == 4]
if len(gha4) > 0:
    print(f"\nGhana wave 4 years: {sorted(gha4['S020'].unique())}")
