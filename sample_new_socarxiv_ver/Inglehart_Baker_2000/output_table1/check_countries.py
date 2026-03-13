#!/usr/bin/env python3
"""Check country lists to match the paper's 65 societies."""
import pandas as pd, csv
with open('data/WVS_Time_Series_1981-2022_csv_v5_0.csv', 'r') as f:
    header = [h.strip('"') for h in next(csv.reader(f))]
cols = ['S002VS', 'COUNTRY_ALPHA', 'S020', 'S003']
avail = [c for c in cols if c in header]
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv', usecols=avail, low_memory=False)
wvs = wvs[wvs['S002VS'].isin([2, 3])]
wvs_countries = sorted(wvs['COUNTRY_ALPHA'].unique())
evs = pd.read_csv('data/EVS_1990_wvs_format.csv')
evs_countries = sorted(evs['COUNTRY_ALPHA'].unique())
all_countries = sorted(set(wvs_countries) | set(evs_countries))
all_countries = [c for c in all_countries if c != 'MNE']
print(f"Total: {len(all_countries)}")
print(f"WVS: {wvs_countries}")
print(f"EVS: {evs_countries}")
print(f"Combined: {all_countries}")

# Check S003 numeric codes for Germany
de_rows = wvs[wvs['COUNTRY_ALPHA'] == 'DEU']
if 'S003' in de_rows.columns:
    print(f"\nDEU S003 codes: {sorted(de_rows['S003'].unique())}")
    for code in sorted(de_rows['S003'].unique()):
        n = len(de_rows[de_rows['S003'] == code])
        print(f"  S003={code}: N={n}")
# Check if MLT and ALB exist
for c in ['MLT', 'ALB', 'BGD', 'PAK', 'SRB', 'SCG']:
    found = c in all_countries
    print(f"  {c}: {'yes' if found else 'no'}")
