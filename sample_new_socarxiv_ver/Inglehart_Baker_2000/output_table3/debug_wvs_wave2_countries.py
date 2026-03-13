#!/usr/bin/env python3
"""Check if WVS wave 2 has the EVS-only countries with additional items."""
import pandas as pd
import csv

DATA_PATH = 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'

# Check wave 2 countries
df = pd.read_csv(DATA_PATH, usecols=['S002VS', 'COUNTRY_ALPHA', 'D059', 'C006'], low_memory=False)
df = df[df['S002VS'] == 2]

for c in df.columns:
    if c not in ['S002VS', 'COUNTRY_ALPHA']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].where(df[c] >= 0, pd.NA)

w2_countries = sorted(df['COUNTRY_ALPHA'].unique())
print(f'Wave 2 countries: {len(w2_countries)}')
print(f'Countries: {w2_countries}')

# Check D059 availability in wave 2
d059_valid = df[df['D059'].notna()]
d059_countries = sorted(d059_valid['COUNTRY_ALPHA'].unique())
print(f'\nWave 2 countries with D059: {len(d059_countries)}')
print(f'Countries: {d059_countries}')

# Check C006 availability in wave 2
c006_valid = df[df['C006'].notna()]
c006_countries = sorted(c006_valid['COUNTRY_ALPHA'].unique())
print(f'\nWave 2 countries with C006: {len(c006_countries)}')
print(f'Countries: {c006_countries}')

# EVS-only countries
evs_only = ['AUT', 'BEL', 'CAN', 'DNK', 'FRA', 'IRL', 'ISL', 'ITA', 'MLT', 'NIR', 'NLD', 'PRT']
for c in evs_only:
    in_w2 = c in w2_countries
    has_d059 = c in d059_countries
    has_c006 = c in c006_countries
    print(f'  {c}: in_wave2={in_w2}, has_D059={has_d059}, has_C006={has_c006}')

# Also check wave 1
df1 = pd.read_csv(DATA_PATH, usecols=['S002VS', 'COUNTRY_ALPHA'], low_memory=False)
df1 = df1[df1['S002VS'] == 1]
w1_countries = sorted(df1['COUNTRY_ALPHA'].unique())
print(f'\nWave 1 countries: {len(w1_countries)}')
print(f'Countries: {w1_countries}')
