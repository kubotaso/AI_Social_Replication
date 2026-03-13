"""
Deeper analysis of Italy, Brazil, Finland, and Norway issues
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
evs_stata_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")

# Load all relevant EVS variables
print("=== EVS Stata variable exploration ===")
evs = pd.read_stata(evs_stata_path, convert_categoricals=False)
print("All EVS columns:", list(evs.columns))
print()

# Check Italy in detail
print("=== ITALY (IT) in EVS Stata ===")
it = evs[evs['c_abrv'] == 'IT']
print("Italy rows:", len(it))
# Show all available columns values for Italy
print("year:", dict(it['year'].value_counts()))
# q336 distribution
print("q336 dist:", dict(it['q336'].value_counts().sort_index()))

# The issue: q336 for Italy gives 51% but paper says 47%
# Maybe there is a weight variable for EVS?
evs_weight_cols = [c for c in evs.columns if 'weight' in c.lower() or c.lower().startswith('w_')]
print("Weight columns:", evs_weight_cols)

# Check if there's an Italian weight
if evs_weight_cols:
    for wcol in evs_weight_cols:
        it_w = it[wcol]
        print(f"\nItaly {wcol}:", it_w.describe())
        valid = it[it['q336'].isin([1,2,3,4,5,6,7,8])]
        monthly = it[it['q336'].isin([1,2,3])]
        if valid[wcol].notna().any() and valid[wcol].sum() > 0:
            wpct = round(monthly[wcol].sum() / valid[wcol].sum() * 100)
            print(f"  Italy weighted (q336 <= 3): {wpct}%")

# Check Norway EVS
print("\n=== NORWAY (NO) in EVS Stata ===")
nor = evs[evs['c_abrv'] == 'NO']
print("Norway rows:", len(nor))
print("q336 dist:", dict(nor['q336'].value_counts().sort_index()))
for valid_vals in [[1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7]]:
    valid = nor[nor['q336'].isin(valid_vals)]
    monthly = nor[nor['q336'].isin([1,2,3])]
    if len(valid) > 0:
        pct = round(len(monthly)/len(valid)*100)
        print(f"  Norway q336 {valid_vals}: {pct}%")

# Check Finland EVS
print("\n=== FINLAND (FI) in EVS Stata ===")
fin = evs[evs['c_abrv'] == 'FI']
print("Finland rows:", len(fin))
print("q336 dist:", dict(fin['q336'].value_counts().sort_index()))
for valid_vals in [[1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7]]:
    valid = fin[fin['q336'].isin(valid_vals)]
    monthly = fin[fin['q336'].isin([1,2,3])]
    if len(valid) > 0:
        pct = round(len(monthly)/len(valid)*100)
        print(f"  Finland q336 {valid_vals}: {pct}%")

# Load WVS and check Finland wave 3
wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'S020', 'F028', 'S017'], low_memory=False)
w3 = wvs[wvs['S002VS'] == 3]
fin3 = w3[w3['S003'] == 246]
print(f"Finland WVS wave 3 rows: {len(fin3)}")
if len(fin3) > 0:
    print("F028 dist:", dict(fin3['F028'].value_counts().sort_index()))
    valid = fin3[fin3['F028'].isin([1,2,3,4,5,6,7,8])]
    monthly = fin3[fin3['F028'].isin([1,2,3])]
    print(f"  Finland WVS wave 3: {round(len(monthly)/len(valid)*100)}%")
    if valid['S017'].sum() > 0:
        wpct = round(monthly['S017'].sum() / valid['S017'].sum() * 100)
        print(f"  Finland WVS wave 3 weighted: {wpct}%")

# Check Brazil in WVS wave 2 (which is '1990-1991')
w2 = wvs[wvs['S002VS'] == 2]
bra2 = w2[w2['S003'] == 76]
print(f"\n=== Brazil WVS wave 2 ===")
print(f"Brazil wave 2 rows: {len(bra2)}")
if len(bra2) > 0:
    print("F028 dist:", dict(bra2['F028'].value_counts().sort_index()))
    for valid_vals in [[1,2,3,4,5,6,7,8], [1,2,3,4,5,6]]:
        valid = bra2[bra2['F028'].isin(valid_vals)]
        monthly = bra2[bra2['F028'].isin([1,2,3])]
        if len(valid) > 0:
            pct = round(len(monthly)/len(valid)*100)
            print(f"  Brazil W2 q336 {valid_vals}: {pct}%")
# Paper says Brazil 1990-1991 = 50%

# Check Brazil WVS wave 3 (95-98)
w3 = wvs[wvs['S002VS'] == 3]
bra3 = w3[w3['S003'] == 76]
print(f"\nBrazil WVS wave 3 rows: {len(bra3)}")
print("F028 dist:", dict(bra3['F028'].value_counts().sort_index()))
# Paper says 54%, current gen = 75%
# Try various valid value sets
for valid_vals in [[1,2,3,4,5,6,7,8], [1,2,3,4,5,6], [1,2,3,4,6,8]]:
    valid = bra3[bra3['F028'].isin(valid_vals)]
    monthly = bra3[bra3['F028'].isin([1,2,3])]
    if len(valid) > 0:
        pct = round(len(monthly)/len(valid)*100)
        print(f"  Brazil W3 F028 {valid_vals}: {pct}% ({len(monthly)}/{len(valid)})")

# Check weights
print(f"Brazil W3 S017:", bra3['S017'].describe())
if bra3['S017'].notna().any():
    valid = bra3[bra3['F028'].isin([1,2,3,4,5,6,7,8])]
    monthly = bra3[bra3['F028'].isin([1,2,3])]
    if valid['S017'].sum() > 0:
        wpct = round(monthly['S017'].sum() / valid['S017'].sum() * 100)
        print(f"  Brazil W3 weighted (all 8pt valid): {wpct}%")

    # Try: include only standard 8-point values (not 5, 7 which are missing in Brazil)
    valid2 = bra3[bra3['F028'].isin([1,2,3,4,6,8])]
    monthly2 = bra3[bra3['F028'].isin([1,2,3])]
    if valid2['S017'].sum() > 0:
        wpct2 = round(monthly2['S017'].sum() / valid2['S017'].sum() * 100)
        print(f"  Brazil W3 weighted (valid=[1,2,3,4,6,8]): {wpct2}%")

# WVS wave 3 year info for Brazil
print(f"\nBrazil W3 survey year (S020): {dict(bra3['S020'].value_counts())}")
