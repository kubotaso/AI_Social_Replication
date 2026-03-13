"""
Try every possible approach to get Italy 1990-1991 = 47%
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
evs_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")
evs_csv_path = os.path.join(base, "data", "EVS_1990_wvs_format.csv")
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")

evs = pd.read_stata(evs_path, convert_categoricals=False)

ita = evs[evs['c_abrv'] == 'IT'].copy()
print(f"Italy EVS rows: {len(ita)}")
print(f"q336:\n{ita['q336'].value_counts().sort_index()}")
print()

# Target: 47%
# Total rows = 2018, NaN count?
print(f"Total NaN: {ita['q336'].isna().sum()}")

# The way to get 47% from 1022 monthly attendees:
# 1022 / X = 0.47 => X = 1022/0.47 = 2174.5 (but only 2018 rows)
# OR: Y / 2018 = 0.47 => Y = 948.5
# OR: Y / 1996 = 0.47 => Y = 938 (948 monthly rounded)
# Check: 939/1996 = 47.0%?
print(f"1022/1996 = {1022/1996*100:.1f}%")  # 51.2%
print(f"939/1996 = {939/1996*100:.1f}%")    # would need 939 monthly

# Count values 1+2 only (very devout, not just monthly)
v12 = ita['q336'].isin([1,2]).sum()
print(f"Values 1+2 (very devout): {v12}/{len(ita)} = {v12/len(ita)*100:.1f}%")

# What if "monthly" means value 3 only?
v3 = ita['q336'].isin([3]).sum()
print(f"Value 3 only: {v3}/{len(ita)} = {v3/len(ita)*100:.1f}%")

# Weighted approaches
for wc in ['weight_g', 'weight_s']:
    try:
        w_all = ita[wc].fillna(1.0)
        valid_ita = ita[ita['q336'].isin([1,2,3,4,5,6,7,8])]
        w_valid = valid_ita[wc].fillna(1.0)
        mw = (valid_ita['q336'].isin([1,2,3]) * w_valid).sum()
        tw = w_valid.sum()
        print(f"Weighted {wc} (8pt): {mw:.1f}/{tw:.1f} = {mw/tw*100:.1f}%")
    except Exception as e:
        print(f"Weight error: {e}")

# What about country1 sub-division?
if 'country1' in evs.columns:
    print(f"\nItaly country1:\n{ita['country1'].value_counts()}")

# What about different year subsets?
print(f"\nItaly year:\n{ita['year'].value_counts()}")

# Check EVS CSV for Italy
print("\n=== EVS CSV for Italy ===")
try:
    evs_csv = pd.read_csv(evs_csv_path, low_memory=False)
    print("EVS CSV columns:", list(evs_csv.columns[:20]))
    ita_csv = evs_csv[evs_csv.get('COUNTRY_ALPHA', evs_csv.get('S003', pd.Series())) == 380]
    print(f"Italy in CSV (S003=380): {len(ita_csv)}")
    if 'COUNTRY_ALPHA' in evs_csv.columns:
        ita_csv = evs_csv[evs_csv['COUNTRY_ALPHA'] == 'ITA']
        print(f"Italy in CSV (COUNTRY_ALPHA='ITA'): {len(ita_csv)}")
except Exception as e:
    print(f"EVS CSV error: {e}")

# WVS wave 2 Italy check
print("\n=== WVS Wave 2 Italy ===")
wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'F028', 'S017', 'COUNTRY_ALPHA'],
                  low_memory=False)
ita_w2 = wvs[(wvs['S002VS'] == 2) & (wvs['S003'] == 380)]
print(f"Italy WVS Wave 2: {len(ita_w2)} rows")
if len(ita_w2) > 0:
    print(f"F028:\n{ita_w2['F028'].value_counts().sort_index()}")

# Bottom line analysis
print("\n=== ANALYSIS: How could Italy be 47%? ===")
# If we exclude some respondents...
# Maybe exclude certain language groups or regions?
print("Possible sources of 47%:")
print("  - Paper may have used different version of EVS (older ZA)")
print("  - Paper may have excluded Northern Italy or subset")
print("  - Paper may have used different definition of 'at least monthly'")
print("  - Different missing data handling")

# What if values 4 and 5 are counted as 'at least monthly' in Italy?
# Value 4 = "once a month" (but that IS monthly)
# Original scale: 1=more than once a week, 2=once a week, 3=once a month,
#                  4=christmas/easter, 5=other holy days, 6=once a year,
#                  7=less often, 8=never

# 47% would require around 938 respondents in numerator out of 1996
# Current: 1022/1996 = 51.2%
# What sub-selection gives us closer to 47%?

# If q336=3 is "once a month" and the question is "AT LEAST once a month"
# Then values 1, 2, 3 should be correct
# But maybe the paper uses "at least once a week" = values 1, 2 only?
v12 = ita['q336'].isin([1,2]).sum()
v123 = ita['q336'].isin([1,2,3]).sum()
print(f"\nAt least weekly (1,2): {v12}/1996 = {v12/1996*100:.1f}%")
print(f"At least monthly (1,2,3): {v123}/1996 = {v123/1996*100:.1f}%")
