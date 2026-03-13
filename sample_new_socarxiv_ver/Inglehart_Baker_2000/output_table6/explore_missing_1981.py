import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','S003','F028','X048WVS','S020'], low_memory=False)

# The 14 MISSING cells are ALL 1981 data for EVS-only countries:
# Belgium 1981, Canada 1981, France 1981, Great Britain 1981, Iceland 1981,
# Ireland 1981, Italy 1981, Netherlands 1981, Norway 1981, N.Ireland 1981,
# Spain 1981, Sweden 1981, US 1981, W.Germany 1981

# Wait - the US IS in the WVS wave 1! Let me check more carefully
missing_countries = {
    'Belgium': 56, 'Canada': 124, 'France': 250, 'Great Britain': 826,
    'Iceland': 352, 'Ireland': 372, 'Italy': 380, 'Netherlands': 528,
    'Norway': 578, 'Northern Ireland': 902, 'Spain': 724, 'Sweden': 752,
    'United States': 840, 'West Germany': 276
}

# Check wave 1 for each
w1 = wvs[wvs['S002VS'] == 1]
print("Countries in WVS Wave 1:")
for code in sorted(w1['S003'].unique()):
    sub = w1[w1['S003'] == code]
    yr = sorted(sub['S020'].unique())
    print(f"  S003={code}: N={len(sub)}, years={yr}")

print()
print("=== Checking missing 1981 countries in WVS ===")
for name, code in sorted(missing_countries.items()):
    sub = w1[w1['S003'] == code]
    if len(sub) > 0:
        yr = sorted(sub['S020'].unique())
        print(f"  {name} (S003={code}): FOUND! N={len(sub)}, years={yr}")
    else:
        print(f"  {name} (S003={code}): NOT FOUND in wave 1")

# Also check: is there wave 1 data under a different country code?
# Northern Ireland could be under 826 (Great Britain) or a different code
# Check: what about COW codes?
print()
print("=== Check all S003 codes in WVS wave 1 that aren't in our map ===")
known_codes = set(wvs_country_map.keys() for wvs_country_map in [{}])
for code in sorted(w1['S003'].unique()):
    sub = w1[w1['S003'] == code]
    yr = sorted(sub['S020'].unique())
    print(f"  S003={code}: N={len(sub)}, years={yr}")

# The key insight: the paper says it uses WVS 1981, 1990-1991, 1995-1998
# For 1981, many European countries participated in EVS (1981) not WVS
# The WVS Time Series v5.0 includes EVS data for some waves
# Let me check: does the WVS Time Series have Belgium, etc.?
print()
print("=== All unique S003 codes across ALL waves ===")
for code in sorted(wvs['S003'].unique()):
    waves = sorted(wvs[wvs['S003']==code]['S002VS'].unique())
    if 1 in waves:
        sub = wvs[(wvs['S003']==code)&(wvs['S002VS']==1)]
        yr = sorted(sub['S020'].unique())
        N = len(sub)
        print(f"  S003={code}: wave 1 present, N={N}, years={yr}")
