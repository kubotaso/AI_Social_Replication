"""
Verify Finland fix and investigate Italy more carefully
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
evs_stata_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")

evs = pd.read_stata(evs_stata_path, convert_categoricals=False,
                     columns=['c_abrv', 'country1', 'q336', 'year', 'weight_g', 'weight_s'])

print("=== FINLAND analysis ===")
fin = evs[evs['c_abrv'] == 'FI']
print("Finland rows:", len(fin))
print("q336 dist:", dict(fin['q336'].value_counts().sort_index()))

# The value 8 = "never/practically never" in EVS
# Finland seems to have many people not answering with 8 (89 respondents)
# But are they valid responses or "never attend"?
# The WVS 8-point scale: 1-7 = some attendance, 8 = never
# Excluding 8 from denominator would treat "never" as missing -- wrong!
# But if Finland uses a 7-point scale internally (different coding), excluding 8 makes sense

# What if Finland's q336 scale is actually 7-point (1-7)?
# Then 8 would be a different code (missing/not applicable)?
valid_7pt = fin[fin['q336'].isin([1,2,3,4,5,6,7])]
monthly_7pt = fin[fin['q336'].isin([1,2,3])]
print(f"  7-point (excl 8): {round(len(monthly_7pt)/len(valid_7pt)*100)}% ({len(monthly_7pt)}/{len(valid_7pt)})")
# This gives 13% - matches paper!

valid_8pt = fin[fin['q336'].isin([1,2,3,4,5,6,7,8])]
monthly_8pt = fin[fin['q336'].isin([1,2,3])]
print(f"  8-point (incl 8): {round(len(monthly_8pt)/len(valid_8pt)*100)}% ({len(monthly_8pt)}/{len(valid_8pt)})")

# Check the distribution of each country to determine typical scale
print("\n=== What value does each country use as max? ===")
for alpha in ['FI', 'IT', 'NO', 'GB-GBN', 'BE', 'FR', 'SE', 'NL', 'US']:
    c = evs[evs['c_abrv'] == alpha]
    if len(c) > 0:
        q336_vals = sorted(c['q336'].dropna().unique())
        print(f"  {alpha}: q336 values = {q336_vals}")

# Norway verification
print("\n=== NORWAY analysis ===")
nor = evs[evs['c_abrv'] == 'NO']
valid_8 = nor[nor['q336'].isin([1,2,3,4,5,6,7,8])]
monthly_3 = nor[nor['q336'].isin([1,2,3])]
print(f"  8-point: {round(len(monthly_3)/len(valid_8)*100)}%")  # Should be 13% from before

# Italy: try each possible scale
print("\n=== ITALY analysis ===")
it = evs[evs['c_abrv'] == 'IT']
q336_vals = sorted(it['q336'].dropna().unique())
print(f"Italy q336 values: {q336_vals}")
# Paper says 47%, current gives 51%
# What if Italy uses a different monthly threshold?
print("q336 distribution:")
print(dict(it['q336'].value_counts().sort_index()))
print()

# Check Italy survey year
print(f"Italy year: {dict(it['year'].value_counts())}")

# What if we use F063 (alternative variable name for church attendance) if available?
# Or what if Italy 1990 uses a different question?
# The actual paper says Italy 1981=48, 1990-91=47

# Let's check if there's any other variable for Italy
print("\nOther potential variables for Italy q336:")
# Let's look at what the EVS codebook says about q336
# The variable is "How often do you attend religious services?"
# For Italy, what if value 1 = more than weekly, 2 = weekly, 3 = 2-3x/month, 4 = once/month...
# Then "at least once a month" = 1,2,3,4 not 1,2,3

# Try threshold 4
valid_it = it[it['q336'].isin([1,2,3,4,5,6,7,8])]
monthly_3 = it[it['q336'].isin([1,2,3])]
monthly_4 = it[it['q336'].isin([1,2,3,4])]
print(f"  Italy threshold 3: {round(len(monthly_3)/len(valid_it)*100)}%")
print(f"  Italy threshold 4: {round(len(monthly_4)/len(valid_it)*100)}%")
# paper=47, threshold 3=51, threshold 4=65% - neither works

# What if there's a country-specific weighting that helps Italy?
for wcol in ['weight_g', 'weight_s']:
    w = it[wcol]
    if w.notna().any() and w.sum() > 0:
        valid = it[it['q336'].isin([1,2,3,4,5,6,7,8])]
        monthly = it[it['q336'].isin([1,2,3])]
        wpct = round(monthly[wcol].sum() / valid[wcol].sum() * 100)
        print(f"  Italy {wcol} weighted (<=3): {wpct}%")

# Investigate if WVS wave 2 had Italy
wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'S020', 'F028'], low_memory=False)
w2 = wvs[wvs['S002VS'] == 2]
it2 = w2[w2['S003'] == 380]
print(f"\nItaly WVS wave 2 rows: {len(it2)}")
# Italy was only in EVS in 1990

# What if the paper uses different row count for Italy?
# Italy has 2018 rows in EVS but only 1996 rows have valid q336?
print(f"\nItaly total: {len(it)}")
print(f"Italy valid q336 (1-8): {len(it[it['q336'].isin([1,2,3,4,5,6,7,8])])}")
print(f"Italy missing q336: {it['q336'].isna().sum()}")
print(f"Italy q336 < 0: {(it['q336'] < 0).sum()}")

# Explore if there's any rounding or different calculation that gets 47%
# 47% of 1996 = 939 respondents
# But we get 1022 = 51%
# 47% of total (2018) = 949 = which is 949/2018 = 47%!
print(f"\n47% of 1996 = {round(0.47 * 1996)}")
print(f"47% of 2018 = {round(0.47 * 2018)}")
print(f"len monthly = {len(it[it['q336'].isin([1,2,3])])}")
# If we compute 1022/2018 = 50.6% -> rounds to 51
# But if paper computed 1022/2166 (including some missing)...
# Or maybe the paper subset to attend valid (not missing)?

# Maybe the 22 missing (2018-1996=22) would change the computation if included?
print(f"\nIf denominator = 2018 (all): {round(len(it[it['q336'].isin([1,2,3])])/2018*100)}%")

# Italy has 22 extra rows - what values do they have?
print("Italy q336 not in 1-8:", it[~it['q336'].isin([1,2,3,4,5,6,7,8])]['q336'].value_counts())
