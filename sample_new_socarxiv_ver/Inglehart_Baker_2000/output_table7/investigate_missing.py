"""
Deep investigation of the 17 missing cells:
1. Korean data in WVS wave 1 and 2
2. European 1981 data - any alternative sources
3. Latvia 1990 in EVS
4. Alternative approaches for Korea
"""
import pandas as pd
import numpy as np

wvs_path = "data/WVS_Time_Series_1981-2022_csv_v5_0.csv"
evs_dta_path = "data/ZA4460_v3-0-0.dta"

# Load WVS with more variables to check Korea
wvs = pd.read_csv(wvs_path, low_memory=False,
                   usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'A006', 'S020', 'G006', 'S017', 'S003',
                             'F025', 'F025_WVS', 'F027', 'F028', 'F028B'])

# Check Korea waves 1 and 2
print("=== KOREA DETAILED ===")
kor_all = wvs[wvs['COUNTRY_ALPHA'] == 'KOR']
print(f"Korea total rows: {len(kor_all)}")
print(f"Korea by wave:")
for wave in sorted(kor_all['S002VS'].unique()):
    sub = kor_all[kor_all['S002VS'] == wave]
    print(f"  Wave {wave}: N={len(sub)}, year={sub['S020'].mode().iloc[0] if len(sub) > 0 else 'N/A'}")

    # Check all available God-related variables
    for var in ['F063', 'A006', 'F025', 'F025_WVS', 'F027', 'F028', 'F028B']:
        if var in sub.columns:
            valid = sub[sub[var].between(1, 10)]
            if len(valid) > 0:
                pct = (valid[var] == 10).mean() * 100
                print(f"    {var}: valid={len(valid)}, %10={pct:.2f}%")

# Check A006 for Korea waves 1-2 (might be on different scale 1-10)
print("\n=== KOREA A006 DETAILS ===")
kor_w1 = wvs[(wvs['COUNTRY_ALPHA'] == 'KOR') & (wvs['S002VS'] == 1)]
kor_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'KOR') & (wvs['S002VS'] == 2)]

print(f"Korea W1 A006 value counts:")
print(kor_w1['A006'].value_counts().sort_index())

print(f"\nKorea W2 A006 value counts:")
print(kor_w2['A006'].value_counts().sort_index())

print(f"\nKorea W2 F063 value counts:")
print(kor_w2['F063'].value_counts().sort_index())

# Check the EVS for Latvia
print("\n=== LATVIA IN EVS ===")
evs = pd.read_stata(evs_dta_path, convert_categoricals=False)
lva = evs[evs['c_abrv'] == 'LV']
print(f"Latvia EVS rows: {len(lva)}")
if len(lva) > 0:
    print(f"Latvia EVS q365 values:")
    print(lva['q365'].value_counts().sort_index())

# Check what other variables might have Latvia data
print("\nChecking all country codes in EVS:")
for code, country in sorted(evs['c_abrv'].value_counts().items()):
    if 'LV' in code or 'LAT' in code:
        print(f"  Code '{code}': {country} rows")

# Check WVS wave 2 for Latvia
print("\n=== LATVIA IN WVS ===")
lva_wvs = wvs[(wvs['COUNTRY_ALPHA'] == 'LVA') & (wvs['S002VS'] == 2)]
print(f"Latvia WVS wave 2 rows: {len(lva_wvs)}")
if len(lva_wvs) > 0:
    f063_valid = lva_wvs[lva_wvs['F063'].between(1, 10)]
    print(f"Latvia WVS wave 2 F063 valid: {len(f063_valid)}")

# Check all EVS countries for Latvia-like codes
print("\nAll EVS c_abrv values:")
print(sorted(evs['c_abrv'].unique()))

# Check c_abrv1 for countries
print("\nc_abrv1 unique values:")
if 'c_abrv1' in evs.columns:
    print(sorted(evs['c_abrv1'].unique()))

# Latvia might be coded differently
for col in evs.columns:
    if 'lv' in col.lower() or 'lat' in col.lower():
        print(f"Column with 'lv' or 'lat': {col}")

# Check what year Latvia was in EVS
lva_country = evs[evs['c_abrv'] == 'LV']
print(f"\nEVS LV country rows: {len(lva_country)}")
if len(lva_country) > 0:
    print(f"EVS LV q365 values: {lva_country['q365'].value_counts().sort_index()}")
    print(f"EVS LV year: {lva_country['year'].unique()}")

# Check if there might be LVA or Latvia coded differently
print("\nSearching for Latvia in all possible columns...")
# Just count c_abrv
for c_abrv in evs['c_abrv'].unique():
    if c_abrv.startswith('LV') or 'LV' in c_abrv:
        print(f"  Found c_abrv: {c_abrv}, N={len(evs[evs['c_abrv']==c_abrv])}")

# Check EVS 1990 WVS CSV for Latvia
evs_csv = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)
lva_csv = evs_csv[evs_csv['COUNTRY_ALPHA'] == 'LVA']
print(f"\nEVS CSV Latvia rows: {len(lva_csv)}")
if len(lva_csv) > 0:
    print(f"EVS CSV Latvia A006 valid: {(lva_csv['A006'].between(1, 10)).sum()}")
    valid = lva_csv[lva_csv['A006'].between(1, 10)]
    if len(valid) > 0:
        pct = (valid['A006'] == 10).mean() * 100
        print(f"EVS CSV Latvia A006 %10: {pct:.2f}%")
    # Also check F063
    f063_valid = lva_csv[lva_csv['F063'].between(1, 10)]
    print(f"EVS CSV Latvia F063 valid: {len(f063_valid)}")

# Check Estonia in EVS (maybe Latvia is coded as Estonia?)
est_evs = evs[evs['c_abrv'] == 'EE']
print(f"\nEVS Estonia rows: {len(est_evs)}")
if len(est_evs) > 0:
    q365_valid = est_evs[est_evs['q365'].between(1, 10)]
    print(f"EVS Estonia q365 valid: {len(q365_valid)}, %10={(q365_valid['q365']==10).mean()*100:.2f}%")
