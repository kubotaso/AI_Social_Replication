#!/usr/bin/env python3
"""Check A006 coding in WVS - is 1=very important or 10=very important?"""
import pandas as pd
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

# Read A006 for a few countries in waves 2-3
df = pd.read_csv(DATA_PATH, usecols=['S002VS', 'COUNTRY_ALPHA', 'A006'], low_memory=False)
df = df[df['S002VS'].isin([2, 3])]
df['A006'] = pd.to_numeric(df['A006'], errors='coerce')
df = df[df['A006'] >= 0]

# Check Nigeria (highly religious) vs Sweden (secular)
for c in ['NGA', 'SWE', 'JPN', 'USA', 'IND', 'BRA']:
    sub = df[df['COUNTRY_ALPHA'] == c]['A006']
    if len(sub) > 0:
        print(f"{c}: mean={sub.mean():.2f}, median={sub.median():.1f}, mode={sub.mode().values[0]:.0f}, "
              f"pct_1={100*(sub==1).mean():.1f}%, pct_10={100*(sub==10).mean():.1f}%")

# In WVS: A006 = "How important is God in your life?"
# 1 = Not at all important
# 10 = Very important
# So HIGHER = more important = more traditional
# But Nigeria shows mean=1.11 which contradicts this if Nigeria is religious...
# Wait - maybe the WVS coding changed. Let me check the raw distribution

print("\n=== Nigeria A006 distribution ===")
nga = df[df['COUNTRY_ALPHA'] == 'NGA']['A006']
print(nga.value_counts().sort_index())

print("\n=== Sweden A006 distribution ===")
swe = df[df['COUNTRY_ALPHA'] == 'SWE']['A006']
print(swe.value_counts().sort_index())

print("\n=== USA A006 distribution ===")
usa = df[df['COUNTRY_ALPHA'] == 'USA']['A006']
print(usa.value_counts().sort_index())
