"""
Comprehensive search for 1981 European data
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
evs_stata_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")

# Check WVS with S002 (not S002VS) for 1981 European countries
print("=== WVS all waves (S002 variable) ===")
wvs = pd.read_csv(wvs_path, usecols=['S002', 'S002VS', 'S003', 'S020', 'F028'], low_memory=False, nrows=5)
print("Columns available:", list(wvs.columns))

# Read full data with S002
wvs = pd.read_csv(wvs_path, usecols=['S002', 'S002VS', 'S003', 'S020', 'F028', 'S017'], low_memory=False)
print("S002 unique values:", sorted(wvs['S002'].unique()))
print("S002VS unique values:", sorted(wvs['S002VS'].unique()))
print()

# WVS 1981 European countries in S002 wave 1
# The European countries (Belgium=56, etc.) should be in S002=1 if they exist
european_1981 = {56: 'Belgium', 124: 'Canada', 250: 'France', 826: 'Great Britain',
                  352: 'Iceland', 372: 'Ireland', 380: 'Italy', 528: 'Netherlands',
                  578: 'Norway', 724: 'Spain', 752: 'Sweden'}
extra = {826: 'Great Britain (also in W3)'}

print("Checking S002=1 (WVS wave 1) for European countries:")
w1_s002 = wvs[wvs['S002'] == 1]
print("S002=1 country codes:", sorted(w1_s002['S003'].unique()))
for s003, name in sorted(european_1981.items()):
    cnt = len(w1_s002[w1_s002['S003'] == s003])
    print(f"  {name} (S003={s003}): {cnt} rows")

# Check if EVS ZA4460 has any 1981 data
print("\n=== EVS Stata year range ===")
evs = pd.read_stata(evs_stata_path, convert_categoricals=False,
                     columns=['c_abrv', 'country1', 'q336', 'year'])
print("Year range:", evs['year'].min(), "-", evs['year'].max())
print("Unique years:", sorted(evs['year'].unique()))

# Are any European 1981 rows in WVS with S020 = 1981?
print("\nWVS rows with S020=1981:")
rows_1981 = wvs[wvs['S020'] == 1981]
print("Total rows with S020=1981:", len(rows_1981))
print("Country codes:", sorted(rows_1981['S003'].unique()))

# Check S020 for wave 1 countries
w1s = wvs[wvs['S002VS'] == 1]
print("\nS020 range for WVS wave 1:", w1s['S020'].min(), "-", w1s['S020'].max())
for s003, name in [(32, 'Argentina'), (36, 'Australia'), (246, 'Finland'),
                    (348, 'Hungary'), (392, 'Japan'), (410, 'South Korea'),
                    (484, 'Mexico'), (710, 'South Africa')]:
    c = w1s[w1s['S003'] == s003]
    if len(c) > 0:
        print(f"  {name}: years {c['S020'].min()}-{c['S020'].max()}")

# Check EVS 1981 from the WVS time series notes
# EVS 1981 is coded as S002VS=1 for European countries...
# Actually EVS 1981 = "Wave 1" in EVS but WVS started later
# The WVS Time Series file may have EVS 1981 as S002VS=1 for some countries

# Let's check all S003 codes in wave 1
print("\n=== ALL S003 in S002VS=1 ===")
for s003 in sorted(w1s['S003'].unique()):
    cnt = len(w1s[w1s['S003'] == s003])
    print(f"  S003={s003}: {cnt} rows")

# Also check if there are entries where S003 is a European code in wave 1
# The "missing" countries are 1981 data for EVS Wave 1 countries
# EVS Wave 1 (1981) is sometimes merged into WVS Time Series

# Check what the WVS Time Series PDF says about early waves
print("\n=== Look at S020 1981 1982 rows ===")
early = wvs[(wvs['S020'] >= 1981) & (wvs['S020'] <= 1984)]
print("Rows 1981-1984:", len(early))
print("S003 codes 1981-1984:", sorted(early['S003'].unique()))
for s003 in sorted(early['S003'].unique()):
    name = {32: 'Argentina', 36: 'Australia', 246: 'Finland', 348: 'Hungary',
            392: 'Japan', 410: 'South Korea', 484: 'Mexico', 710: 'South Africa'}.get(s003, f'code={s003}')
    c = early[early['S003'] == s003]
    print(f"  {name} (S003={s003}): S020={sorted(c['S020'].unique())}, wave={sorted(c['S002VS'].unique())}")
