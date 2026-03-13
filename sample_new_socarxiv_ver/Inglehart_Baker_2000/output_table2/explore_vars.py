#!/usr/bin/env python3
"""Explore variable distributions for Table 2 missing items."""
import pandas as pd
import csv
import numpy as np

DATA_PATH = 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'
with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

# Load subset to explore
needed = ['S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'E114', 'E115', 'E116', 'E117', 'E118',
          'G007_01', 'G007_02', 'G007_03', 'G007_04', 'G007_05', 'G007_06', 'G007_07',
          'G001', 'G002', 'G003', 'G005', 'A043B', 'A044', 'A045', 'A046', 'A047', 'A048',
          'C001', 'C002', 'C004', 'F051', 'F063']
available = [c for c in needed if c in header]
df = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
df = df[df['S002VS'].isin([2, 3])]
for col in available:
    if col not in ['S003', 'COUNTRY_ALPHA', 'S002VS', 'S020']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].where(df[col] >= 0, np.nan)

print('E114 value counts (sample):')
print(df['E114'].value_counts().head(10))

print()
print('E115 value counts:')
print(df['E115'].value_counts().head(10))

print()
print('E116 value counts:')
print(df['E116'].value_counts().head(10))

print()
print('G007_01 value counts:')
print(df['G007_01'].value_counts().head(10))

print()
print('G007_02 value counts:')
print(df['G007_02'].value_counts().head(10))

print()
# Wave 2 G007
w2 = df[df['S002VS'] == 2]
cnt1 = w2.groupby('COUNTRY_ALPHA')['G007_01'].count()
cnt2 = w2.groupby('COUNTRY_ALPHA')['G007_02'].count()
print('Wave 2 G007_01 non-null countries:', cnt1[cnt1 > 0].index.tolist())
print('Wave 2 G007_02 non-null countries:', cnt2[cnt2 > 0].index.tolist())

print()
print('G001 value counts (goals in life):')
print(df['G001'].value_counts().head(10))

print()
print('G002 value counts:')
print(df['G002'].value_counts().head(10))

print()
print('A043B value counts:')
print(df['A043B'].value_counts().head(10))

print()
print('A044 value counts:')
print(df['A044'].value_counts().head(10))

print()
print('C001 value counts:')
print(df['C001'].value_counts().head(10))

print()
print('C002 value counts:')
print(df['C002'].value_counts().head(10))

# Also look at waves 2 only
print()
print('=== Wave 2 only ===')
w2_countries = df[df['S002VS'] == 2]['COUNTRY_ALPHA'].unique()
print(f'Wave 2 countries: {sorted(w2_countries)}')
print(f'Wave 2 count: {len(w2_countries)}')

# Check if G002 in wave 2 might be 'make parents proud'
if 'G002' in df.columns:
    g2_w2 = df[df['S002VS'] == 2]['G002']
    g2_w3 = df[df['S002VS'] == 3]['G002']
    print(f'G002 wave 2 non-null: {g2_w2.notna().sum()}')
    print(f'G002 wave 3 non-null: {g2_w3.notna().sum()}')
    print(f'G002 value range: {df["G002"].min()} to {df["G002"].max()}')
