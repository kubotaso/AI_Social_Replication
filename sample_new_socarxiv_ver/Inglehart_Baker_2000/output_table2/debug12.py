#!/usr/bin/env python3
"""Search for proxy variables for missing Table 2 items."""
import pandas as pd
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

# Item 1: "Make parents proud" (corr=.81)
# This is about making parents proud as a life goal
# Possible proxies:
# - A029: "Good manners" as child quality (0/1)
# - A030-A042: Other child qualities
# - D054-D060: Family role attitudes
# Actually, in WVS/EVS the question is:
# "One of my main goals in life has been to make my parents proud"
# This is typically measured as D054 in some codebooks, but D054 in WVS TS V5
# is about "woman earns more causes problems"

# Let me check A029 (good manners) as proxy
# Also D058, D059, D060 (family attitudes)
# And A025 (love parents) which we already use

# Item 2: "Parents' duty to do best for children even at own expense" (corr=.60)
# This might be variable D055 or D058 in WVS
# D058: "Parents responsibility to do best for children" type question
# Check D055-D060

# Item 3: "Stricter limits on selling foreign goods" (corr=.63)
# G007_01 has very limited data. Check how many countries
check_cols = ['S002VS', 'COUNTRY_ALPHA', 'G007_01', 'F024', 'D055', 'D056', 'D057',
              'D058', 'D059', 'D060', 'D061', 'D062', 'A029', 'A030', 'A032',
              'A034', 'A035', 'A038', 'A039', 'A040', 'A041']

avail = ['S002VS', 'COUNTRY_ALPHA'] + [v for v in check_cols[2:] if v in header]
df = pd.read_csv(DATA_PATH, usecols=avail, low_memory=False)
df = df[df['S002VS'].isin([2, 3])]

# Get latest per country
latest = df.groupby('COUNTRY_ALPHA')['S002VS'].max().reset_index()
latest.columns = ['COUNTRY_ALPHA', 'lw']
df = df.merge(latest, on='COUNTRY_ALPHA')
df = df[df['S002VS'] == df['lw']].drop('lw', axis=1)

for v in avail[2:]:
    vals = pd.to_numeric(df[v], errors='coerce')
    pos = vals[vals >= 0].dropna()
    n_countries = df[vals >= 0].dropna(subset=[v])['COUNTRY_ALPHA'].nunique() if len(pos) > 0 else 0
    if len(pos) > 0:
        print(f"{v}: N={len(pos)}, Countries={n_countries}, values={sorted(pos.unique())[:8]}")

# Specifically check G007_01 country list
print("\n=== G007_01 countries ===")
if 'G007_01' in df.columns:
    g_valid = pd.to_numeric(df['G007_01'], errors='coerce')
    g_pos = g_valid[g_valid >= 0].dropna()
    g_countries = df.loc[g_pos.index, 'COUNTRY_ALPHA'].unique()
    print(f"Countries with G007_01 data: {sorted(g_countries)}")

# Check F024 countries
print("\n=== F024 countries ===")
if 'F024' in df.columns:
    f_valid = pd.to_numeric(df['F024'], errors='coerce')
    f_pos = f_valid[f_valid >= 0].dropna()
    f_countries = df.loc[f_pos.index, 'COUNTRY_ALPHA'].unique()
    print(f"Countries with F024 data: {sorted(f_countries)}")
    print(f"Number: {len(f_countries)}")
