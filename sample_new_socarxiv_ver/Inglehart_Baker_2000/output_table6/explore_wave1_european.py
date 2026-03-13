"""
Exhaustively search for European 1981 data in all available datasets
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, "data")

print("=== DATA FILES AVAILABLE ===")
for f in os.listdir(data_dir):
    fpath = os.path.join(data_dir, f)
    size = os.path.getsize(fpath) / 1024 / 1024
    print(f"  {f}: {size:.1f} MB")

print()

# Check WVS Time Series for wave 1 countries
wvs_path = os.path.join(data_dir, "WVS_Time_Series_1981-2022_csv_v5_0.csv")
wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'F028'],
                  low_memory=False)

print("=== WVS WAVE 1 COUNTRIES ===")
w1 = wvs[wvs['S002VS'] == 1]
print(f"Wave 1 total rows: {len(w1)}")
print(f"Countries (S003 + COUNTRY_ALPHA):")
for s003 in sorted(w1['S003'].unique()):
    alpha = w1[w1['S003'] == s003]['COUNTRY_ALPHA'].iloc[0] if len(w1[w1['S003'] == s003]) > 0 else 'N/A'
    yr = w1[w1['S003'] == s003]['S020'].unique().tolist()
    cnt = len(w1[w1['S003'] == s003])
    print(f"  S003={s003} ({alpha}): {cnt} rows, years={yr}")

# Check EVS ZA4460 for any 1981 data
evs_path = os.path.join(data_dir, "ZA4460_v3-0-0.dta")
evs = pd.read_stata(evs_path, convert_categoricals=False,
                    columns=['c_abrv', 'country', 'year', 'q336'])
print(f"\n=== EVS ZA4460 YEARS ===")
print(f"Year distribution:\n{evs['year'].value_counts().sort_index()}")
print(f"\nCountries in 1981 rows:")
evs_1981 = evs[evs['year'] < 1985]
if len(evs_1981) > 0:
    print(evs_1981['c_abrv'].value_counts())
else:
    print("  No rows before 1985")

# Check EVS CSV
evs_csv_path = os.path.join(data_dir, "EVS_1990_wvs_format.csv")
evs_csv = pd.read_csv(evs_csv_path, low_memory=False)
print(f"\n=== EVS CSV years ===")
if 'S020' in evs_csv.columns:
    print(evs_csv['S020'].value_counts().sort_index())
print(f"EVS CSV COUNTRY_ALPHA unique: {sorted(evs_csv['COUNTRY_ALPHA'].unique())}")

# Check for any additional data files
print("\n=== OTHER POTENTIAL DATA FILES ===")
for f in os.listdir(base):
    if any(f.endswith(ext) for ext in ['.csv', '.dta', '.sav', '.xlsx', '.xls', '.rds', '.RData']):
        fpath = os.path.join(base, f)
        size = os.path.getsize(fpath) / 1024 / 1024
        print(f"  {f}: {size:.1f} MB")

# Final check: WVS Time Series wave 1 - European ISO codes
european_s003 = {
    32: 'Argentina', 36: 'Australia', 76: 'Brazil', 124: 'Canada',
    246: 'Finland', 250: 'France', 276: 'Germany', 348: 'Hungary',
    372: 'Ireland', 380: 'Italy', 392: 'Japan', 410: 'South Korea',
    484: 'Mexico', 528: 'Netherlands', 578: 'Norway', 702: 'Singapore',
    710: 'South Africa', 724: 'Spain', 752: 'Sweden', 756: 'Switzerland',
    826: 'UK', 840: 'USA'
}
print(f"\n=== EUROPEAN-LIKE S003 CODES IN WVS WAVE 1 ===")
for s003, name in european_s003.items():
    cnt = len(w1[w1['S003'] == s003])
    if cnt > 0:
        print(f"  {s003} ({name}): {cnt} rows")
