import pandas as pd
import numpy as np

# Get all column names from WVS
wvs_cols = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv', nrows=0).columns.tolist()
# Find wave-related columns
wave_cols = [c for c in wvs_cols if 's002' in c.lower() or 'wave' in c.lower()]
print("Wave-related columns:", wave_cols)

# Check all S-prefixed columns
s_cols = [c for c in wvs_cols if c.startswith('S0')]
print("S0-prefixed columns:", s_cols[:20])

# Read with the relevant columns
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS', 'S003', 'S020', 'F028'], low_memory=False)

# Check if any 1981-1984 data exists for our missing countries
target_names = {
    56: 'Belgium', 124: 'Canada', 250: 'France', 276: 'Germany',
    352: 'Iceland', 372: 'Ireland', 380: 'Italy', 528: 'Netherlands',
    578: 'Norway', 724: 'Spain', 752: 'Sweden', 826: 'Great Britain', 840: 'United States'
}

for code, name in sorted(target_names.items()):
    sub = wvs[(wvs['S003'] == code) & (wvs['S020'] <= 1985)]
    if len(sub) > 0:
        yr = sorted(sub['S020'].unique())
        waves = sorted(sub['S002VS'].unique())
        print(f"  {name} (S003={code}): N={len(sub)}, years={yr}, waves={waves}")

# Check all data with year <= 1985
print("\n=== All countries with year <= 1985 ===")
early = wvs[wvs['S020'] <= 1985]
for code in sorted(early['S003'].unique()):
    sub = early[early['S003'] == code]
    yr = sorted(sub['S020'].unique())
    waves = sorted(sub['S002VS'].unique())
    print(f"  S003={code}: N={len(sub)}, years={yr}, waves={waves}")
