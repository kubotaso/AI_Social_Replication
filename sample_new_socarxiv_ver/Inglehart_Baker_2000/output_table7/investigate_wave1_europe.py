"""
Investigate WVS wave 1 data for European countries that are currently MISSING for 1981.
Also explore the EVS 1990 data for any 1981 records.
The paper shows 1981 values for: Belgium(9), Canada(36), France(10), Great Britain(20),
Iceland(22), Ireland(29), Northern Ireland(38), Italy(31), Netherlands(11), Norway(19),
Spain(18), Sweden(9), West Germany(16), USA(50), South Korea(29)
"""
import pandas as pd
import numpy as np
import math

wvs_path = "data/WVS_Time_Series_1981-2022_csv_v5_0.csv"
evs_csv_path = "data/EVS_1990_wvs_format.csv"
evs_dta_path = "data/ZA4460_v3-0-0.dta"

wvs = pd.read_csv(wvs_path, low_memory=False,
                  usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'S020', 'G006', 'S017'])

print("=== WVS WAVE 1 EUROPEAN COVERAGE ===")
wave1 = wvs[wvs['S002VS'] == 1]
print(f"All wave 1 countries:\n{wave1['COUNTRY_ALPHA'].value_counts().sort_index()}")

print("\n=== WAVE 1 MISSING EUROPEAN COUNTRIES ===")
# Countries with missing 1981 values that we need:
missing_1981 = ['BEL', 'CAN', 'FRA', 'GBR', 'ISL', 'IRL', 'NIR', 'ITA',
                'NLD', 'NOR', 'ESP', 'SWE', 'DEU', 'USA', 'KOR']
for c in missing_1981:
    sub = wave1[wave1['COUNTRY_ALPHA'] == c]
    if len(sub) > 0:
        valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)]
        pct = (valid['F063'] == 10).mean() * 100 if len(valid) > 0 else 0
        print(f"  {c}: N={len(sub)}, valid F063={len(valid)}, %10={pct:.4f}% -> round={math.floor(pct+0.5)}")
    else:
        print(f"  {c}: NO DATA in WVS wave 1")

print("\n=== EVS 1990 DATA: Check year coverage ===")
evs_csv = pd.read_csv(evs_csv_path, low_memory=False)
print(f"EVS CSV columns: {list(evs_csv.columns[:20])}")
if 'S020' in evs_csv.columns:
    print(f"Year distribution: {evs_csv['S020'].value_counts().sort_index()}")
elif 'year' in evs_csv.columns:
    print(f"Year distribution: {evs_csv['year'].value_counts().sort_index()}")

print("\n=== EVS ZA4460 YEAR DISTRIBUTION ===")
evs = pd.read_stata(evs_dta_path, convert_categoricals=False,
                    columns=['c_abrv', 'country1', 'q365', 'weight_s', 'year'])
print(f"EVS ZA4460 year distribution:\n{evs['year'].value_counts().sort_index()}")
print(f"\nCountries in EVS ZA4460:\n{evs['c_abrv'].value_counts().sort_index()}")

# Check if EVS ZA4460 has any pre-1990 data
early_data = evs[evs['year'] < 1988]
if len(early_data) > 0:
    print(f"\nEVS ZA4460 data before 1988: N={len(early_data)}")
    print(early_data['c_abrv'].value_counts())

print("\n=== FINALIZE: Norway 1981 ===")
# Norway wave 1 data in WVS - what does it show?
nor_w1 = wvs[(wvs['COUNTRY_ALPHA'] == 'NOR') & (wvs['S002VS'] == 1)]
if len(nor_w1) > 0:
    valid = nor_w1[(nor_w1['F063'] >= 1) & (nor_w1['F063'] <= 10)]
    pct = (valid['F063'] == 10).mean() * 100
    print(f"Norway wave 1 N={len(nor_w1)}, valid={len(valid)}, %10={pct:.4f}% -> {math.floor(pct+0.5)}")
    print(f"Year: {nor_w1['S020'].unique()}")
else:
    print("Norway not in WVS wave 1")

print("\n=== Check EVS CSV for 1981 data ===")
evs_csv2 = pd.read_csv(evs_csv_path, low_memory=False, nrows=5)
print(f"EVS CSV first 5 rows country/year:\n{evs_csv2[['COUNTRY_ALPHA', 'S002VS', 'S020'] if all(c in evs_csv2.columns for c in ['COUNTRY_ALPHA','S002VS','S020']) else evs_csv2.columns[:8]]}")
