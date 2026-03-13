"""
Comprehensive search for 1981 European data - fixed
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
evs_stata_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")

# Read full data without S002
wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'S020', 'F028', 'S017'], low_memory=False)

# Check wave 1 in detail
w1 = wvs[wvs['S002VS'] == 1]
print("WVS wave 1 (S002VS=1) country codes:", sorted(w1['S003'].unique()))
print("WVS wave 1 years (S020):", sorted(w1['S020'].unique()))

# Check EVS 1981 from the ZA4460 Stata file for year
evs = pd.read_stata(evs_stata_path, convert_categoricals=False,
                     columns=['c_abrv', 'country1', 'q336', 'year'])
print("\nEVS years:", sorted(evs['year'].unique()))

# All rows from WVS with S020 between 1981 and 1984
early = wvs[(wvs['S020'] >= 1981) & (wvs['S020'] <= 1984)]
print("\nWVS rows from 1981-1984:")
for s003 in sorted(early['S003'].unique()):
    c = early[early['S003'] == s003]
    print(f"  S003={s003}: {len(c)} rows, years={sorted(c['S020'].unique())}, wave={sorted(c['S002VS'].unique())}")

# Any S003 in the 1981 range that corresponds to European countries?
european_codes = {56: 'Belgium', 124: 'Canada', 250: 'France', 826: 'Great Britain',
                   352: 'Iceland', 372: 'Ireland', 380: 'Italy', 528: 'Netherlands',
                   578: 'Norway', 724: 'Spain', 752: 'Sweden', 276: 'Germany',
                   840: 'United States', 40: 'Austria'}

print("\nEuropean country codes in WVS wave 1?")
for code, name in sorted(european_codes.items()):
    cnt = len(w1[w1['S003'] == code])
    if cnt > 0:
        print(f"  {name} (S003={code}): {cnt} rows FOUND!")
    else:
        print(f"  {name} (S003={code}): 0 rows (not in WVS wave 1)")

# The conclusion is that European countries in 1981 are NOT in WVS
# They are in EVS Wave 1 (1981) which is a separate dataset (ZA4438)
# We only have EVS Wave 2 (1990) in ZA4460

# Score ceiling analysis
# With 14 missing cells (all 1981 European data), max possible:
# If all 14 become FULL, and also all other issues fixed:
print("\n=== Score ceiling analysis ===")
# Current: 62 FULL + 2 PARTIAL + 2 MISS + 14 MISSING (with Finland fix)
# Max possible with current data (fixing all remaining issues):
# Best case: 62 FULL + 0 PARTIAL + 0 MISS + 14 MISSING
# (Turn Italy and Brazil into partial at best)

for full, partial, miss, missing in [
    (62, 2, 2, 14),   # Current (with Finland fix)
    (63, 1, 2, 14),   # Fix one partial
    (64, 0, 2, 14),   # Fix all partials
    (64, 0, 1, 14),   # Fix one miss too
    (64, 0, 0, 14),   # Fix all misses
    (80, 0, 0, 0),    # Perfect with all data
]:
    total = 80
    cats = (total-missing)/total * 20
    vals = (full + partial*0.7 + miss*0.2) / total * 40
    net = (total-missing)/total * 20
    order = 10
    col = 10
    score = round(cats + vals + order + net + col)
    print(f"  F={full} P={partial} M={miss} MISS={missing}: score={score}")

# Key question: if we could get 14 missing cells to FULL, how many points?
# 14 FULL = 14/80 for cats (3.5) + 14/80*40 (7.0) + 14/80*20 for net (3.5) = 14 extra points
# So adding 14 perfect cells would add 14 points, from 85-86 to 99-100
print("\n14 perfect cells from EVS 1981 would add ~14 points (from ~85 to ~99)")
print("Without EVS 1981 data, max achievable score = ~87")
