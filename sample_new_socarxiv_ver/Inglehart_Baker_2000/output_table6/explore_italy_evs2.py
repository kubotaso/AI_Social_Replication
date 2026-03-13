"""
Deep dive into Italy Wave 2 EVS data - using correct column names
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
evs_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")

evs = pd.read_stata(evs_path, convert_categoricals=False)

# Country column is 'country'
print("country unique sample:", sorted(evs['country'].unique())[:20])
print()

# Italy check
italy_evs = evs[evs['country'] == 380]
print(f"Italy (country=380): {len(italy_evs)} rows")

# Also check c_abrv
if len(italy_evs) == 0:
    print("country1 unique:", sorted(evs['country1'].dropna().unique())[:30])
    ita_c_abrv = evs[evs['c_abrv'] == 'IT']
    print(f"c_abrv='IT': {len(ita_c_abrv)} rows")

# Try country1
ita1 = evs[evs['country1'] == 380]
print(f"Italy (country1=380): {len(ita1)} rows")

# Check c_abrv values
print("\nc_abrv unique:", sorted(evs['c_abrv'].dropna().unique()))

# Get Italy by alpha
ita_alpha = evs[evs['c_abrv'] == 'IT']
print(f"\nItaly by c_abrv='IT': {len(ita_alpha)} rows")
if len(ita_alpha) > 0:
    print(f"q336 value_counts:\n{ita_alpha['q336'].value_counts().sort_index()}")

    # Standard 8pt
    ita8 = ita_alpha[ita_alpha['q336'].isin([1,2,3,4,5,6,7,8])]
    m = ita8['q336'].isin([1,2,3]).sum()
    t = len(ita8)
    print(f"\n8pt: {m}/{t} = {m/t*100:.1f}%")

    # 7pt
    ita7 = ita_alpha[ita_alpha['q336'].isin([1,2,3,4,5,6,7])]
    m7 = ita7['q336'].isin([1,2,3]).sum()
    t7 = len(ita7)
    print(f"7pt: {m7}/{t7} = {m7/t7*100:.1f}%")

    # weighted
    for wc in ['weight_g', 'weight_s']:
        try:
            w = ita8[wc].fillna(1.0)
            mw = (ita8['q336'].isin([1,2,3]) * w).sum()
            tw = w.sum()
            print(f"weighted ({wc}) 8pt: {mw:.1f}/{tw:.1f} = {mw/tw*100:.1f}%")
        except:
            pass

    # Check years
    print(f"\nyear values: {ita_alpha['year'].value_counts().to_dict()}")

    # Check country1 for Italy sub-units
    print(f"country1 values: {ita_alpha['country1'].value_counts().to_dict()}")

# Check other European countries to see which ones Finland maps to
print("\n\nEuropean c_abrv values and q336 stats:")
for alpha in ['FI', 'HU', 'IT', 'FR', 'BE', 'DE', 'ES', 'SE', 'NO']:
    sub = evs[evs['c_abrv'] == alpha]
    if len(sub) > 0:
        valid = sub[sub['q336'].isin([1,2,3,4,5,6,7,8])]
        m = valid['q336'].isin([1,2,3]).sum()
        t = len(valid)
        vals = sorted(sub['q336'].dropna().unique().tolist())
        print(f"  {alpha}: {m}/{t} = {m/t*100:.1f}% | q336 vals: {vals}")
