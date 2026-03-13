"""
Exploration script for Table 6 - attempt to find 1981 European data and fix Italy/Brazil
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
evs_stata_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")
evs_csv_path = os.path.join(base, "data", "EVS_1990_wvs_format.csv")

print("=== Checking EVS_1990_wvs_format.csv ===")
evs_csv = pd.read_csv(evs_csv_path, nrows=5)
print("Columns:", list(evs_csv.columns)[:50])
print()

# Get all rows to find unique countries
evs_csv_full = pd.read_csv(evs_csv_path, low_memory=False)
print("Total rows:", len(evs_csv_full))
if 'S003' in evs_csv_full.columns:
    print("Countries (S003):", sorted(evs_csv_full['S003'].unique()))
if 'S002' in evs_csv_full.columns:
    print("Waves (S002):", sorted(evs_csv_full['S002'].unique()))
if 'S002VS' in evs_csv_full.columns:
    print("Waves VS (S002VS):", sorted(evs_csv_full['S002VS'].unique()))
print()

print("=== Checking WVS Time Series - Wave 1 countries ===")
wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'S020', 'F028'], low_memory=False)
wave1 = wvs[wvs['S002VS'] == 1]
print("Wave 1 country codes (S003):", sorted(wave1['S003'].unique()))

# European country codes we need for 1981
# Belgium=56, Canada=124, France=250, Great Britain=826, Iceland=352, Ireland=372,
# Italy=380, Netherlands=528, Northern Ireland, Norway=578, Spain=724, Sweden=752
european_1981 = {56: 'Belgium', 124: 'Canada', 250: 'France', 826: 'Great Britain',
                  352: 'Iceland', 372: 'Ireland', 380: 'Italy', 528: 'Netherlands',
                  578: 'Norway', 724: 'Spain', 752: 'Sweden'}
for code, name in european_1981.items():
    cnt = len(wave1[wave1['S003'] == code])
    print(f"  {name} (S003={code}): {cnt} rows")

# Check F028 for Brazil wave 3
print("\n=== Brazil Wave 3 (F028 distribution) ===")
wvs_w3 = wvs[wvs['S002VS'] == 3]
brazil = wvs_w3[wvs_w3['S003'] == 76]
print("Brazil wave 3 total rows:", len(brazil))
print("F028 value counts:")
print(brazil['F028'].value_counts().sort_index())

# Check WVS wave 3 F028 for Italy
print("\n=== Italy 1990 WVS/EVS data (S003=380) ===")
wave2 = wvs[wvs['S002VS'] == 2]
italy2 = wave2[wave2['S003'] == 380]
print("Italy wave 2 rows:", len(italy2))
if len(italy2) > 0:
    print("F028 distribution:")
    print(italy2['F028'].value_counts().sort_index())

print("\n=== EVS Stata - Italy (IT) ===")
evs = pd.read_stata(evs_stata_path, convert_categoricals=False,
                     columns=['c_abrv', 'country1', 'q336', 'year'])
italy_evs = evs[evs['c_abrv'] == 'IT']
print("Italy EVS rows:", len(italy_evs))
print("q336 distribution:")
print(italy_evs['q336'].value_counts().sort_index())

# Try different scales
for threshold in [1, 2, 3]:
    monthly = (italy_evs['q336'] <= threshold).sum()
    valid = italy_evs['q336'].isin([1,2,3,4,5,6,7,8]).sum()
    print(f"  Threshold {threshold}: {monthly}/{valid} = {round(monthly/valid*100)}%")

print("\n=== Check EVS for Northern Ireland ===")
ni_evs = evs[evs['c_abrv'] == 'GB-NIR']
print("Northern Ireland EVS rows:", len(ni_evs))
print("c_abrv unique:", evs['c_abrv'].unique()[:30])

print("\n=== Check if EVS has 1981 data ===")
print("EVS year range:", evs['year'].min(), "-", evs['year'].max())
print("EVS year distribution:")
print(evs['year'].value_counts().sort_index())
