"""Explore available data for Table 8 replication - check for 1981 European data"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Check EVS 1990 Stata file
print("=" * 60)
print("EVS ZA4460 Stata file exploration")
print("=" * 60)
evs = pd.read_stata(os.path.join(base, 'data', 'ZA4460_v3-0-0.dta'), convert_categoricals=False)
print(f"Shape: {evs.shape}")
print(f"Columns (first 40): {list(evs.columns[:40])}")
print()

# Check for wave/year columns
for col in evs.columns:
    if any(x in col.lower() for x in ['wave', 'year', 'study', 'date']):
        vals = evs[col].unique()
        print(f"  {col}: unique values = {sorted(vals[:20])}")

print()
print("Unique c_abrv:", sorted(evs['c_abrv'].unique()) if 'c_abrv' in evs.columns else 'N/A')

# Check the EVS 1990 WVS format CSV
print()
print("=" * 60)
print("EVS_1990_wvs_format.csv exploration")
print("=" * 60)
evs_csv = pd.read_csv(os.path.join(base, 'data', 'EVS_1990_wvs_format.csv'))
print(f"Shape: {evs_csv.shape}")
print(f"Columns: {list(evs_csv.columns[:30])}")
print()
# Check if there's wave info
for col in evs_csv.columns:
    if any(x in col.lower() for x in ['wave', 'year', 'study', 's002', 's020']):
        vals = evs_csv[col].unique()
        print(f"  {col}: unique values = {sorted(vals[:20])}")

# Check S002VS and S020 in EVS CSV
if 'S002VS' in evs_csv.columns:
    print(f"\nS002VS values: {sorted(evs_csv['S002VS'].unique())}")
if 'S020' in evs_csv.columns:
    print(f"S020 values: {sorted(evs_csv['S020'].unique())}")
if 'S003' in evs_csv.columns:
    print(f"S003 values: {sorted(evs_csv['S003'].unique())}")
if 'COUNTRY_ALPHA' in evs_csv.columns:
    print(f"COUNTRY_ALPHA values: {sorted(evs_csv['COUNTRY_ALPHA'].unique())}")

# Check WVS Time Series for Wave 1 countries with S002EVS
print()
print("=" * 60)
print("WVS Time Series - checking S002EVS and wave markers")
print("=" * 60)
wvs_cols = pd.read_csv(os.path.join(base, 'data', 'WVS_Time_Series_1981-2022_csv_v5_0.csv'), nrows=0)
print(f"Total columns: {len(wvs_cols.columns)}")
evs_related = [c for c in wvs_cols.columns if 'evs' in c.lower() or 'S002' in c]
print(f"EVS/S002 related columns: {evs_related}")

# Load Wave 1 and 2 to check for European countries
wvs = pd.read_csv(os.path.join(base, 'data', 'WVS_Time_Series_1981-2022_csv_v5_0.csv'),
                   usecols=['S002VS', 'S002EVS', 'S003', 'COUNTRY_ALPHA', 'F001', 'S020', 'S017'])

# Check S002EVS
print(f"\nS002EVS unique values: {sorted(wvs['S002EVS'].dropna().unique())}")

# Wave 1 and 2 details
for wave in [1, 2]:
    w = wvs[wvs['S002VS'] == wave]
    print(f"\nWave {wave} (S002VS={wave}):")
    for alpha in sorted(w['COUNTRY_ALPHA'].unique()):
        sub = w[w['COUNTRY_ALPHA'] == alpha]
        f001_valid = (sub['F001'] > 0).sum()
        evs_vals = sub['S002EVS'].unique()
        print(f"  {alpha}: n={len(sub)}, F001_valid={f001_valid}, year={sub['S020'].iloc[0]}, S002EVS={evs_vals}")
