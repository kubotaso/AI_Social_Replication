#!/usr/bin/env python3
"""Check D055-D060 variable meanings via country-level patterns."""
import pandas as pd
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

df = pd.read_csv(DATA_PATH, usecols=['S002VS', 'COUNTRY_ALPHA',
                                      'D054', 'D055', 'D056', 'D057', 'D058', 'D059', 'D060'],
                 low_memory=False)
df = df[df['S002VS'].isin([2, 3])]
latest = df.groupby('COUNTRY_ALPHA')['S002VS'].max().reset_index()
latest.columns = ['COUNTRY_ALPHA', 'lw']
df = df.merge(latest, on='COUNTRY_ALPHA')
df = df[df['S002VS'] == df['lw']].drop('lw', axis=1)

for col in ['D054', 'D055', 'D056', 'D057', 'D058', 'D059', 'D060']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].where(df[col] >= 0, np.nan)

# Check country means to infer variable meaning
print("=== D054-D060 country means ===")
print(f"{'Country':<8}", end='')
for col in ['D054', 'D055', 'D056', 'D057', 'D058', 'D059', 'D060']:
    print(f"  {col:>6}", end='')
print()

for c in ['NGA', 'SWE', 'JPN', 'USA', 'IND', 'BRA', 'CHN', 'POL', 'DEU']:
    sub = df[df['COUNTRY_ALPHA'] == c]
    print(f"{c:<8}", end='')
    for col in ['D054', 'D055', 'D056', 'D057', 'D058', 'D059', 'D060']:
        if col in sub.columns:
            val = sub[col].mean()
            if pd.notna(val):
                print(f"  {val:>6.2f}", end='')
            else:
                print(f"     NA", end='')
        else:
            print(f"     NA", end='')
    print()

# D054: "If a woman earns more money than her husband..."
# 1=Agree strongly, 2=Agree, 3=Disagree, 4=Strongly disagree (or similar)
# NGA should be low (agree = traditional)

# D058: "Parents' duty to do best for children, even at expense of their own well-being"
# This might be what the paper describes
# If 1=Agree strongly, traditional countries should be low (more agree)

# Also check: does D055 correspond to "make parents proud"?
# In WVS: D055 might be "A working mother can establish just as warm and secure..."
# D056: "Being a housewife is just as fulfilling..."
# D057: "Having a job is the best way for a woman to be independent"
# D058: "Both husband and wife should contribute to household income"
# D059: "Parents' duty to do best for their children even at the expense..."
# D060: "One must always love and respect their parents"

# Wait, D060 might be "love and respect parents" which we already have as A025
# And D059 might be "parents duty"

# Let me also check if "make parents proud" might be a specific A-series question
# In WVS, A003 = "Important in life: Leisure time" (1-4)
# The "make parents proud" life goal might not be in WVS Time Series at all

# Check G007_01 - use WAVE 2 ONLY (don't filter by latest)
print("\n=== G007_01 using wave 2 only (not latest per country) ===")
df2 = pd.read_csv(DATA_PATH, usecols=['S002VS', 'COUNTRY_ALPHA', 'G007_01'], low_memory=False)
df2 = df2[df2['S002VS'] == 2]  # Wave 2 only
df2['G007_01'] = pd.to_numeric(df2['G007_01'], errors='coerce')
df2['G007_01'] = df2['G007_01'].where(df2['G007_01'] >= 0, np.nan)
g_means = df2.groupby('COUNTRY_ALPHA')['G007_01'].mean().dropna()
print(f"Countries: {len(g_means)}")
print(g_means.sort_values())

# Similarly check F024 in wave 2
print("\n=== F024 using all waves (2 and 3) ===")
df3 = pd.read_csv(DATA_PATH, usecols=['S002VS', 'COUNTRY_ALPHA', 'F024'], low_memory=False)
df3 = df3[df3['S002VS'].isin([2, 3])]
df3['F024'] = pd.to_numeric(df3['F024'], errors='coerce')
df3['F024'] = df3['F024'].where(df3['F024'] >= 0, np.nan)
f_means = df3.groupby('COUNTRY_ALPHA')['F024'].mean().dropna()
print(f"Countries: {len(f_means)}")
if len(f_means) > 0:
    print(f_means.sort_values())
