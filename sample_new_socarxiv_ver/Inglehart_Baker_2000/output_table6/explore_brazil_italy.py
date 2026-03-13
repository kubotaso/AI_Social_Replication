"""
Deep dive into Brazil Wave 3 and Italy Wave 2 to understand discrepancies
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
evs_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")

# === BRAZIL WAVE 3 ===
print("=== BRAZIL WAVE 3 ===")
wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'F028', 'S017'],
                  low_memory=False)
bra3 = wvs[(wvs['S002VS'] == 3) & (wvs['S003'] == 76)]
print(f"Brazil Wave 3: {len(bra3)} rows")
print(f"Years: {sorted(bra3['S020'].unique())}")
print(f"F028 value_counts:\n{bra3['F028'].value_counts().sort_index()}")
print(f"S017 range: {bra3['S017'].min()} to {bra3['S017'].max()}")
print(f"S017 unique count: {bra3['S017'].nunique()}")

# Try with wave 3 weights
bra3_valid = bra3[bra3['F028'].notna() & (bra3['F028'] >= 1)]
print(f"\nBrazil Wave 3 valid rows (F028 >= 1): {len(bra3_valid)}")

# Unweighted
monthly = bra3_valid['F028'].isin([1, 2, 3]).sum()
total = len(bra3_valid)
print(f"Unweighted: {monthly}/{total} = {monthly/total*100:.1f}%")

# Weighted
w = bra3_valid['S017'].fillna(1.0)
monthly_w = (bra3_valid['F028'].isin([1, 2, 3]) * w).sum()
total_w = w.sum()
print(f"Weighted: {monthly_w:.1f}/{total_w:.1f} = {monthly_w/total_w*100:.1f}%")

# What if we include negative codes?
bra3_all = bra3[bra3['F028'].notna()]
print(f"\nAll rows with non-null F028: {len(bra3_all)}")
print(f"F028 all values:\n{bra3_all['F028'].value_counts().sort_index()}")

monthly2 = bra3_all['F028'].isin([1, 2, 3]).sum()
total2 = len(bra3_all)
print(f"With negatives in denom: {monthly2}/{total2} = {monthly2/total2*100:.1f}%")

# What about different year subsets?
for year in sorted(bra3['S020'].unique()):
    yr_data = bra3[(bra3['S020'] == year) & bra3['F028'].notna() & (bra3['F028'] >= 1)]
    if len(yr_data) > 0:
        m = yr_data['F028'].isin([1, 2, 3]).sum()
        t = len(yr_data)
        print(f"  Year {year}: {m}/{t} = {m/t*100:.1f}%")

print()

# === ITALY WAVE 2 (EVS) ===
print("=== ITALY WAVE 2 (EVS) ===")
evs = pd.read_stata(evs_path, convert_categoricals=False)
# Find Italy
# EVS country codes
italy_evs = evs[evs['cntry'] == 380]  # ISO 380 = Italy
print(f"Italy in EVS (cntry=380): {len(italy_evs)} rows")

# What columns are available?
q336_cols = [c for c in evs.columns if 'q336' in c.lower() or 'church' in c.lower() or 'attend' in c.lower()]
print(f"Attendance-related cols: {q336_cols}")

if len(italy_evs) > 0:
    print(f"q336 value_counts:\n{italy_evs['q336'].value_counts().sort_index()}")

    # Standard 8-point
    valid_8pt = [1, 2, 3, 4, 5, 6, 7, 8]
    ita_valid = italy_evs[italy_evs['q336'].isin(valid_8pt)]
    monthly = ita_valid['q336'].isin([1, 2, 3]).sum()
    total = len(ita_valid)
    print(f"\nItaly EVS 8pt: {monthly}/{total} = {monthly/total*100:.1f}%")

    # 7-point
    valid_7pt = [1, 2, 3, 4, 5, 6, 7]
    ita7 = italy_evs[italy_evs['q336'].isin(valid_7pt)]
    monthly7 = ita7['q336'].isin([1, 2, 3]).sum()
    total7 = len(ita7)
    print(f"Italy EVS 7pt: {monthly7}/{total7} = {monthly7/total7*100:.1f}%")

    # What about weighting?
    weight_cols = [c for c in evs.columns if 'weight' in c.lower() or c.startswith('w_') or c == 'gweight' or 'wgt' in c.lower()]
    print(f"\nWeight columns: {weight_cols}")
    for wc in weight_cols[:3]:
        try:
            w = ita_valid[wc].fillna(1.0)
            m_w = (ita_valid['q336'].isin([1, 2, 3]) * w).sum()
            t_w = w.sum()
            print(f"  {wc}: {m_w:.1f}/{t_w:.1f} = {m_w/t_w*100:.1f}%")
        except:
            pass

    # Check country1 for any sub-country issues
    if 'country1' in evs.columns:
        print(f"\ncountry1 for Italy:\n{italy_evs['country1'].value_counts()}")

    # Check year variable
    year_cols = [c for c in evs.columns if 'year' in c.lower() or c == 'S020' or 's020' in c.lower()]
    print(f"Year cols: {year_cols}")
    for yc in year_cols[:3]:
        print(f"  {yc}: {italy_evs[yc].value_counts().to_dict()}")

# Also check WVS for Italy wave 2
print("\n=== Italy WVS wave 2 check ===")
ita_w2 = wvs[(wvs['S002VS'] == 2) & (wvs['S003'] == 380)]
print(f"Italy WVS Wave 2: {len(ita_w2)} rows")
if len(ita_w2) > 0:
    print(f"F028:\n{ita_w2['F028'].value_counts().sort_index()}")
    ita_w2_valid = ita_w2[ita_w2['F028'].isin([1,2,3,4,5,6,7,8])]
    m = ita_w2_valid['F028'].isin([1,2,3]).sum()
    t = len(ita_w2_valid)
    print(f"WVS wave 2 Italy: {m}/{t} = {m/t*100:.1f}%")
