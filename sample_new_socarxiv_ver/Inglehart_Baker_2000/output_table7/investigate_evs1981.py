"""
Check if WVS Time Series v5.0 includes EVS 1981 data for European countries.
Check the full extent of wave 1 (S002VS=1) data.
"""
import pandas as pd
import numpy as np

wvs_path = "data/WVS_Time_Series_1981-2022_csv_v5_0.csv"

# Read ALL wave 1 data with all relevant columns
wvs = pd.read_csv(wvs_path,
                   usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'A006', 'S020', 'G006', 'S017', 'S003'],
                   low_memory=False)

w1 = wvs[wvs['S002VS'] == 1]
print(f"Wave 1 total rows: {len(w1)}")
print(f"Wave 1 countries: {sorted(w1['COUNTRY_ALPHA'].unique())}")
print(f"Wave 1 S020 (year) range: {sorted(w1['S020'].unique())}")
print()

# Check F063 availability for each wave 1 country
print("Wave 1 country details:")
for country in sorted(w1['COUNTRY_ALPHA'].unique()):
    sub = w1[w1['COUNTRY_ALPHA'] == country]
    f063_valid = sub[sub['F063'].between(1, 10)]
    years = sorted(sub['S020'].unique())
    print(f"  {country}: N_total={len(sub)}, N_f063_valid={len(f063_valid)}, years={years}")

print()

# Specifically check European countries that should have 1981 data:
# BEL, CAN, FRA, DEU(W), GBR, ISL, IRL, NIR, ITA, NLD, NOR, ESP, SWE, USA, KOR
target_countries = ['BEL', 'CAN', 'FRA', 'DEU', 'GBR', 'ISL', 'IRL', 'NIR', 'ITA', 'NLD', 'NOR', 'ESP', 'SWE', 'USA', 'KOR']
print("Missing wave 1 countries check:")
for c in target_countries:
    sub = w1[w1['COUNTRY_ALPHA'] == c]
    if len(sub) == 0:
        print(f"  {c}: NOT IN WAVE 1 of WVS Time Series v5.0")
    else:
        f063_valid = sub[sub['F063'].between(1, 10)]
        print(f"  {c}: {len(sub)} rows, F063 valid: {len(f063_valid)}")

print()
# Check S003 (country code) for any European countries not showing in COUNTRY_ALPHA
print("All wave 1 S003 (country code) values:")
print(w1['S003'].value_counts().sort_index())

print()
# Try checking if any data exists with different wave coding
print("All unique S002VS values in full dataset:")
print(wvs['S002VS'].value_counts().sort_index())

# Check the EVS_1990_wvs_format.csv for any 1981 data
evs_csv = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)
print(f"\nEVS CSV S002VS values: {sorted(evs_csv['S002VS'].unique())}")
print(f"EVS CSV S020 values (years): {sorted(evs_csv['S020'].unique())}")
print(f"EVS CSV COUNTRY_ALPHA: {sorted(evs_csv['COUNTRY_ALPHA'].unique())}")

# Check if GBR is in WVS with wave 1 data under different coding
print("\nChecking for GBR in WVS full dataset:")
gbr_all = wvs[wvs['COUNTRY_ALPHA'] == 'GBR']
print(f"GBR total rows: {len(gbr_all)}")
print(f"GBR waves: {gbr_all['S002VS'].value_counts().sort_index()}")
if len(gbr_all) > 0:
    gbr_f063 = gbr_all[gbr_all['F063'].between(1, 10)]
    for wave, g in gbr_f063.groupby('S002VS'):
        print(f"  GBR wave {wave}: N={len(g)}, %10={(g['F063']==10).mean()*100:.2f}%")
