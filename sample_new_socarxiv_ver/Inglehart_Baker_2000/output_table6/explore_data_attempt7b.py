"""
Deeper exploration for Table 6 - F028 in EVS CSV and Brazil scale
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
evs_stata_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")
evs_csv_path = os.path.join(base, "data", "EVS_1990_wvs_format.csv")

# Load EVS CSV
evs_csv = pd.read_csv(evs_csv_path, low_memory=False)
print("=== EVS CSV (1990 WVS format) ===")
print("Columns:", list(evs_csv.columns))
print("Total rows:", len(evs_csv))
print("COUNTRY_ALPHA unique:", sorted(evs_csv['COUNTRY_ALPHA'].unique()))
print()

# F028 distribution for each country
print("=== F028 distributions by country in EVS CSV ===")
for country in sorted(evs_csv['COUNTRY_ALPHA'].unique()):
    sub = evs_csv[evs_csv['COUNTRY_ALPHA'] == country]
    valid = sub[sub['F028'].isin([1,2,3,4,5,6,7,8])]
    monthly = sub[sub['F028'].isin([1,2,3])]
    if len(valid) > 0:
        pct = round(len(monthly)/len(valid)*100)
        print(f"  {country}: {pct}% ({len(monthly)}/{len(valid)})")
        # Show distribution
        print(f"    F028 dist:", dict(sub['F028'].value_counts().sort_index()))

print()
print("=== Check Germany in EVS CSV ===")
de = evs_csv[evs_csv['COUNTRY_ALPHA'] == 'DE']
print("Germany rows:", len(de))
if 'X048WVS' in evs_csv.columns:
    print("X048WVS:", de['X048WVS'].value_counts().sort_index())

# Now check Italy specifically
print()
print("=== Italy in EVS CSV ===")
it_csv = evs_csv[evs_csv['COUNTRY_ALPHA'] == 'IT']
print("Italy rows:", len(it_csv))
if 'F028' in evs_csv.columns:
    print("F028 dist:", dict(it_csv['F028'].value_counts().sort_index()))
    for thresh in [3, 4]:
        valid = it_csv[it_csv['F028'].isin([1,2,3,4,5,6,7,8])]
        monthly = it_csv[it_csv['F028'].isin([1,2,3])]
        print(f"  Monthly (<=3): {round(len(monthly)/len(valid)*100)}%")

# Check the S020 (survey year) for Italy
print()
print("Italy S020:", dict(it_csv['S020'].value_counts()))

# Check Brazil in WVS wave 3 more carefully
print()
print("=== Brazil Wave 3 (WVS) ===")
wvs = pd.read_csv(wvs_path, low_memory=False,
                   usecols=['S002VS', 'S003', 'S020', 'F028', 'F063', 'S017'])
wave3 = wvs[wvs['S002VS'] == 3]
brazil3 = wave3[wave3['S003'] == 76]
print("Brazil wave 3 rows:", len(brazil3))
print("F028 dist:", dict(brazil3['F028'].value_counts().sort_index()))
print("F063 dist:", dict(brazil3['F063'].value_counts().sort_index()))

# Understand what codes mean for Brazil's 6-point scale
# Typical 6-point: 1=more than once/week, 2=once/week, 3=once/month, 4=holy days, 5=other, 6=never
# Try treating values 1,2,3 as monthly (matches 8-point scale 1,2,3)
for valid_vals in [[1,2,3,4,5,6,7,8], [1,2,3,4,5,6], [1,2,3,4,6,8]]:
    valid = brazil3[brazil3['F028'].isin(valid_vals)]
    monthly = brazil3[brazil3['F028'].isin([1,2,3])]
    if len(valid) > 0:
        pct = round(len(monthly)/len(valid)*100)
        print(f"  Valid={valid_vals}: {pct}% ({len(monthly)}/{len(valid)})")

# What if we use weights for Brazil?
print("Brazil S017 (weights):", brazil3['S017'].describe())
valid_brazil = brazil3[brazil3['F028'].isin([1,2,3,4,5,6,7,8])]
monthly_brazil = brazil3[brazil3['F028'].isin([1,2,3])]
if valid_brazil['S017'].notna().any() and valid_brazil['S017'].sum() > 0:
    wpct = round(monthly_brazil['S017'].sum() / valid_brazil['S017'].sum() * 100)
    print(f"  Weighted (all 8-point valid): {wpct}%")

# Paper says Brazil W3 = 54%. Current gen = 75%.
# Brazil has F028 scale issue - value 5 and 7 seem to be missing
# Let's check if F063 (alternative church attendance variable) works better
print("\n=== F063 for Brazil ===")
for valid_vals in [[1,2,3,4,5,6,7,8], [1,2,3,4,5,6]]:
    valid = brazil3[brazil3['F063'].isin(valid_vals)]
    monthly = brazil3[brazil3['F063'].isin([1,2,3])]
    if len(valid) > 0:
        pct = round(len(monthly)/len(valid)*100)
        print(f"  F063 valid={valid_vals}: {pct}% ({len(monthly)}/{len(valid)})")

# Actually the issue might be that the Brazil 6-pt scale has different mapping
# Let's try: 1-3 = monthly or more, 4 = special days, 6 = never
# The paper result is 54%, current gives 75%
# If we include values 4-6 as "less than monthly" and 5 is missing...
print("\n=== Brazil WVS scale analysis ===")
print("F028 unique values:", sorted(brazil3['F028'].dropna().unique()))
# Value 5 and 7 are missing, so the 6-pt scale is: 1,2,3,4,6,8
# Possible mapping: 1=more than weekly, 2=weekly, 3=monthly, 4=holy days only, 6=less often, 8=never
# "at least monthly" = 1,2,3 -> 145+269+438 = 852, total = 852+106+94+87=1139, pct = 75%
# But maybe the scale is different: 1=weekly+, 2=monthly, 3=4x/year, 4=holy days, 6=less, 8=never
# At least monthly = 1,2 -> 145+269=414, total=1139, pct=36%
# Or 1=weekly+, 2=weekly, 3=2-3x/month -- this doesn't seem right either

# Let's try WVS wave 3 Brazil with weighted just values 1,2,3
print("\nBrazil wave 3 F028 1,2,3 unweighted:", round((145+269+438)/(145+269+438+106+94+87)*100), "%")
print("Paper says 54%")

# Check if maybe value 3 in Brazil means something different (e.g. quarterly not monthly)
# and we should only use values 1,2
print("If monthly = 1,2 only:", round((145+269)/(145+269+438+106+94+87)*100), "%")  # Would be 36%
