"""Debug7: Check q336 (church attendance) in original EVS for Hungary and other countries."""
import pandas as pd
import numpy as np

evs_orig = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)

# Get variable label info
reader = pd.io.stata.StataReader('data/ZA4460_v3-0-0.dta')
vl = reader.variable_labels()
print(f"q336: {vl.get('q336', 'N/A')}")

# Check the value labels for q336
try:
    val_labels = reader.value_labels()
    for key, labels in val_labels.items():
        if '336' in key or 'q336' in key.lower():
            print(f"Value labels for {key}:")
            for v, l in sorted(labels.items()):
                print(f"  {v}: {l}")
except Exception as e:
    print(f"Error getting value labels: {e}")
reader.close()

print("\n=== q336 overall distribution ===")
print(evs_orig['q336'].value_counts().sort_index())

# Check q336 with categorical labels
evs_cat = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=True,
                          columns=['country', 'c_abrv', 'q336', 'year'])
print("\n=== q336 categorical values (first 20 unique) ===")
print(evs_cat['q336'].unique()[:20])

# Get Hungary data
hun = evs_orig[evs_orig['c_abrv'] == 'HU']
print(f"\n=== HUNGARY q336 (n={len(hun)}) ===")
print(hun['q336'].value_counts().sort_index())

# Get Germany data
deu = evs_orig[evs_orig['c_abrv'] == 'DE']
if len(deu) == 0:
    # Try other codes
    print("\nc_abrv values:", sorted(evs_orig['c_abrv'].dropna().unique()))
    deu = evs_orig[evs_orig['c_abrv'].str.contains('DE', na=False)]

print(f"\n=== GERMANY q336 (n={len(deu)}) ===")
print(deu['q336'].value_counts().sort_index())

# Check if Germany has region variable
region_cols = [c for c in evs_orig.columns if 'region' in c.lower() or 'state' in c.lower() or 'land' in c.lower() or 'gdr' in c.lower() or 'east' in c.lower() or 'west' in c.lower()]
print("\nRegion/state/east/west columns:", region_cols[:20])

# Check c_abrv for possible East/West Germany distinction
print("\n=== c_abrv unique values ===")
print(sorted(evs_orig['c_abrv'].dropna().unique()))

# Check country variable for Germany
print("\n=== country values for Germany region ===")
deu_all = evs_orig[evs_orig['c_abrv'].str.startswith('DE', na=False)]
print(deu_all['country'].value_counts().sort_index())
print(deu_all['country1'].value_counts().sort_index())

# Check q320 which has 1-10 scale (might be "importance of God")
print(f"\n=== Hungary q320 ===")
print(hun['q320'].value_counts().sort_index())

# So q336 is church attendance. What's the coding?
# Based on EVS codebook, q336 should be:
# 1 = more than once a week
# 2 = once a week
# 3 = once a month
# 4 = only on special holy days
# 5 = once a year
# 6 = less often
# 7 = never
# Let's verify by comparing with our EVS_1990_wvs_format.csv F063 values

# Compute percentages with q336 coding where 1-3 = at least monthly
MONTHLY_VALS = [1, 2, 3]
VALID_VALS = [1, 2, 3, 4, 5, 6, 7]

for alpha in ['HU', 'DE', 'IT', 'IE', 'GB', 'PL', 'ES', 'BE', 'NL', 'FR', 'NO', 'SE', 'IS']:
    sub = evs_orig[evs_orig['c_abrv'] == alpha]
    if len(sub) == 0:
        continue
    valid = sub[sub['q336'].isin(VALID_VALS)]
    monthly = sub[sub['q336'].isin(MONTHLY_VALS)]
    if len(valid) > 0:
        pct = round(len(monthly) / len(valid) * 100)
        print(f"  {alpha}: q336 attend% = {pct}% (n_valid={len(valid)}, n_monthly={len(monthly)})")
    else:
        print(f"  {alpha}: no valid q336 data")

# But wait - our EVS_1990_wvs_format.csv maps to 8-point scale (1-8)
# The original EVS q336 uses 7-point scale (1-7)
# Let's check how our CSV was created
print("\n\n=== Compare EVS CSV F063 with Stata q336 for Hungary ===")
evs_csv = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)
hun_csv = evs_csv[evs_csv['COUNTRY_ALPHA'] == 'HUN']
print("CSV F063 distribution:")
print(hun_csv['F063'].value_counts().sort_index())
print("\nStata q336 distribution:")
print(hun['q336'].value_counts().sort_index())
