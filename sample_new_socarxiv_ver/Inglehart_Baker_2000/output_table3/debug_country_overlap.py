#!/usr/bin/env python3
"""Check country overlap between WVS and EVS datasets."""
import pandas as pd
import csv

DATA_PATH = 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'
df = pd.read_csv(DATA_PATH, usecols=['S002VS', 'COUNTRY_ALPHA'], low_memory=False)
df = df[df['S002VS'].isin([2,3])]
wvs_countries = sorted(df['COUNTRY_ALPHA'].unique())
print(f'WVS countries (waves 2-3): {len(wvs_countries)}')

evs = pd.read_csv('data/EVS_1990_wvs_format.csv')
evs_countries = sorted(evs['COUNTRY_ALPHA'].unique())
print(f'EVS countries: {len(evs_countries)}')

only_evs = sorted(set(evs_countries) - set(wvs_countries))
print(f'Only in EVS (not WVS): {only_evs}')
print(f'Count only-EVS: {len(only_evs)}')

only_wvs = sorted(set(wvs_countries) - set(evs_countries))
print(f'Only in WVS: {only_wvs}')
print(f'Count only-WVS: {len(only_wvs)}')

both = sorted(set(wvs_countries) & set(evs_countries))
print(f'In both: {both}')
print(f'Count both: {len(both)}')

all_c = sorted(set(wvs_countries) | set(evs_countries))
print(f'Total union: {len(all_c)}')

# Check which WVS countries have D059 data
df2 = pd.read_csv(DATA_PATH, usecols=['S002VS', 'COUNTRY_ALPHA', 'D059'], low_memory=False)
df2 = df2[df2['S002VS'].isin([2,3])]
df2['D059'] = pd.to_numeric(df2['D059'], errors='coerce')
df2 = df2[df2['D059'] >= 0]
d059_countries = sorted(df2['COUNTRY_ALPHA'].unique())
print(f'\nCountries with D059 data: {len(d059_countries)}')

# Which EVS-only countries lack D059?
evs_only_no_d059 = sorted(set(only_evs) - set(d059_countries))
print(f'EVS-only countries lacking D059: {evs_only_no_d059}')

# Check "latest per country" logic - do we get EVS data overridden by WVS?
print(f'\nCountries in both that would use WVS (later wave):')
for c in both:
    wvs_years = df[df['COUNTRY_ALPHA']==c]['S002VS'].unique()
    evs_years = evs[evs['COUNTRY_ALPHA']==c]['S020'].unique()
    print(f'  {c}: WVS waves={sorted(wvs_years)}, EVS years={sorted(evs_years)}')
