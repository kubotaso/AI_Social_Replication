import pandas as pd
import numpy as np

evs_long = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
de = evs_long[evs_long['country'] == 276]
print(f'Germany ZA4460: N={len(de)}')

# Check for region variables
for col in evs_long.columns:
    vals = de[col].dropna()
    if len(vals) > 0 and len(vals.unique()) > 1 and len(vals.unique()) <= 20:
        u = sorted(vals.unique())
        if 'q' in col or 'region' in col.lower() or 'v' == col[0]:
            continue  # skip questionnaire items
        print(f'  {col}: {u}')

# Check c_abrv1 - might distinguish East/West
print(f"\nc_abrv values for Germany: {sorted(de['c_abrv'].unique())}")
if 'c_abrv1' in de.columns:
    print(f"c_abrv1 values for Germany: {sorted(de['c_abrv1'].unique())}")
if 'country1' in de.columns:
    print(f"country1 values for Germany: {sorted(de['country1'].unique())}")
if 'cntry_y' in de.columns:
    print(f"cntry_y values for Germany: {sorted(de['cntry_y'].unique())}")
if 'cntry1_y' in de.columns:
    print(f"cntry1_y values for Germany: {sorted(de['cntry1_y'].unique())}")

# Check q336 (church attendance) for all Germany
print(f"\nq336 for Germany:")
print(de['q336'].value_counts().sort_index())
valid = de[de['q336'].isin([1,2,3,4,5,6,7,8])]
monthly = de[de['q336'].isin([1,2,3])]
print(f"Overall pct: {len(monthly)/len(valid)*100:.1f}%")
