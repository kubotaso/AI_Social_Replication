#!/usr/bin/env python3
"""Check G007 and B008 variable availability."""
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_factor_analysis import load_combined_data, clean_missing, get_latest_per_country

df = load_combined_data(waves_wvs=[2, 3], include_evs=True)
df = get_latest_per_country(df)

# Check G007 sub-items
print('=== G007 sub-items ===')
g007_cols = sorted([c for c in df.columns if c.startswith('G007')])
for col in g007_cols:
    vals = pd.to_numeric(df[col], errors='coerce')
    pos = vals[vals >= 0].dropna()
    if len(pos) > 500:
        print(f'{col}: N={len(pos)}, values={sorted(pos.unique())[:10]}')

# Check B008
print('\n=== B008 ===')
if 'B008' in df.columns:
    vals = pd.to_numeric(df['B008'], errors='coerce')
    pos = vals[vals >= 0].dropna()
    print(f'B008: N={len(pos)}, values={sorted(pos.unique())[:10]}')
else:
    print('B008 not in columns')

# Check F024
print('\n=== F024 (good and evil) ===')
if 'F024' in df.columns:
    vals = pd.to_numeric(df['F024'], errors='coerce')
    pos = vals[vals >= 0].dropna()
    print(f'F024: N={len(pos)}, values={sorted(pos.unique())[:10]}')

# Check A003 (importance of leisure time -> might be used for 'make parents proud')
# Actually A003 = "Important in life: leisure time" (1-4)
# "Make parents proud" - in WVS wave 3, this is A003 in some codebooks, or it might be in D054 area
# Let me check A062 area or others
print('\n=== Checking for make parents proud ===')
# In WVS, "One of my main goals in life has been to make my parents proud"
# This is typically D054 in some versions, or A003 in others
# A029 = Good manners (children quality)
# Let me check D054
if 'D054' in df.columns:
    vals = pd.to_numeric(df['D054'], errors='coerce')
    pos = vals[vals >= 0].dropna()
    print(f'D054: N={len(pos)}, values={sorted(pos.unique())[:10]}')

# Check A062, A063, A064, A065 area
for v in ['A062', 'A063', 'A064', 'A065', 'A066', 'A067', 'A068', 'A069', 'A070']:
    if v in df.columns:
        vals = pd.to_numeric(df[v], errors='coerce')
        pos = vals[vals >= 0].dropna()
        if len(pos) > 0:
            print(f'{v}: N={len(pos)}, values={sorted(pos.unique())[:10]}')

# Check E023 (discuss politics)
print('\n=== E023 (interest in politics) ===')
if 'E023' in df.columns:
    vals = pd.to_numeric(df['E023'], errors='coerce')
    pos = vals[vals >= 0].dropna()
    print(f'E023: N={len(pos)}, values={sorted(pos.unique())[:10]}')

# E033 (left-right)
print('\n=== E033 ===')
if 'E033' in df.columns:
    vals = pd.to_numeric(df['E033'], errors='coerce')
    pos = vals[vals >= 0].dropna()
    print(f'E033: N={len(pos)}, values={sorted(pos.unique())[:10]}')
