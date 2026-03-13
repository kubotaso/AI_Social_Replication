"""Check if God importance data exists under alternative variable names."""
import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv', low_memory=False,
                   nrows=0)  # Just get column names
print(f"Total columns: {len(wvs.columns)}")

# Look for F063-related columns
f_cols = [c for c in wvs.columns if c.startswith('F06')]
print(f"F06x columns: {f_cols}")

# Look for any columns with 'god' or similar
# Actually, let's load Korea wave 1 and check all available variables
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   low_memory=False)
kor_w1 = wvs[(wvs['COUNTRY_ALPHA'] == 'KOR') & (wvs['S002VS'] == 1)]
kor_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'KOR') & (wvs['S002VS'] == 2)]

print(f"\nKorea wave 1: {len(kor_w1)} rows")
print(f"Korea wave 2: {len(kor_w2)} rows")

# Check all F-series variables for Korea
print("\nKorea wave 1 F-variables with valid data (>= 1):")
for col in sorted([c for c in wvs.columns if c.startswith('F')]):
    valid = kor_w1[kor_w1[col] >= 1][col]
    if len(valid) > 0:
        print(f"  {col}: n={len(valid)}, min={valid.min()}, max={valid.max()}, mean={valid.mean():.2f}")

print("\nKorea wave 2 F-variables with valid data (>= 1):")
for col in sorted([c for c in wvs.columns if c.startswith('F')]):
    valid = kor_w2[kor_w2[col] >= 1][col]
    if len(valid) > 0:
        print(f"  {col}: n={len(valid)}, min={valid.min()}, max={valid.max()}, mean={valid.mean():.2f}")

# Also check WVS wave 2 for EVS 1981 countries
# These countries had EVS 1981 data that's not in our dataset
# But maybe WVS wave 2 overlaps?
print("\n\nChecking WVS wave 1 coverage more broadly:")
for country in ['BEL', 'CAN', 'FRA', 'GBR', 'ISL', 'IRL', 'NIR', 'ITA',
                'NLD', 'NOR', 'ESP', 'SWE', 'USA', 'DEU']:
    w1 = wvs[(wvs['COUNTRY_ALPHA'] == country) & (wvs['S002VS'] == 1)]
    if len(w1) > 0:
        f063_valid = w1[(w1['F063'] >= 1) & (w1['F063'] <= 10)]
        print(f"  {country} wave 1: n={len(w1)}, F063 valid={len(f063_valid)}")
    else:
        # Check if this country is in wave 2 with year 1981-1982
        w_early = wvs[(wvs['COUNTRY_ALPHA'] == country) & (wvs['S020'] <= 1982)]
        if len(w_early) > 0:
            print(f"  {country}: in WVS with year <= 1982: {len(w_early)} rows, S002VS={w_early['S002VS'].unique()}")
        else:
            print(f"  {country}: NOT in WVS pre-1983")
