import pandas as pd
import numpy as np

# Check Italy EVS in detail
evs_long = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
it = evs_long[evs_long['country'] == 380]
print(f"Italy ZA4460 N: {len(it)}")
print("q336 value counts:")
print(it['q336'].value_counts().sort_index())
# With 8-point scale:
valid = it[it['q336'].isin([1,2,3,4,5,6,7,8])]
monthly = it[it['q336'].isin([1,2,3])]
print(f"8pt: {len(monthly)/len(valid)*100:.1f}% (N={len(valid)})")
# With 7-point scale (no 5):
valid7 = it[it['q336'].isin([1,2,3,4,6,7,8])]
print(f"7pt: {len(monthly)/len(valid7)*100:.1f}% (N={len(valid7)})")
# Italy paper=47%, we get 51%
# The distribution shows quite a few 5s (58). If 5 is actually "non-standard" for Italy
# it could mean they used 7-point scale. But that gives higher %, not lower.

# Check if there's a weight variable that could help
print(f"\nWeight variables:")
for col in ['weight_g', 'weight_s']:
    if col in evs_long.columns:
        wt = it[col]
        print(f"  {col}: mean={wt.mean():.4f}, std={wt.std():.4f}, min={wt.min():.4f}, max={wt.max():.4f}")

# Try weighted calculation
wt_col = 'weight_s'
if wt_col in evs_long.columns:
    valid_wt = it[it['q336'].isin([1,2,3,4,5,6,7,8])].copy()
    monthly_mask = valid_wt['q336'].isin([1,2,3])
    weighted_monthly = valid_wt.loc[monthly_mask, wt_col].sum()
    weighted_total = valid_wt[wt_col].sum()
    print(f"  Weighted pct (weight_s): {weighted_monthly/weighted_total*100:.1f}%")

wt_col = 'weight_g'
if wt_col in evs_long.columns:
    valid_wt = it[it['q336'].isin([1,2,3,4,5,6,7,8])].copy()
    monthly_mask = valid_wt['q336'].isin([1,2,3])
    weighted_monthly = valid_wt.loc[monthly_mask, wt_col].sum()
    weighted_total = valid_wt[wt_col].sum()
    print(f"  Weighted pct (weight_g): {weighted_monthly/weighted_total*100:.1f}%")

# Check Hungary with weights
print(f"\n=== Hungary ===")
hu = evs_long[evs_long['country'] == 348]
valid_hu = hu[hu['q336'].isin([1,2,3,4,5,6,7,8])].copy()
monthly_hu = valid_hu['q336'].isin([1,2,3])
for wt_col in ['weight_s', 'weight_g']:
    wm = valid_hu.loc[monthly_hu, wt_col].sum()
    wt = valid_hu[wt_col].sum()
    print(f"  Weighted pct ({wt_col}): {wm/wt*100:.1f}%")

# Check all countries with weights to see which match better
print(f"\n=== All countries weighted vs unweighted ===")
paper_vals = {
    'Belgium': 35, 'Canada': 40, 'Finland': 13, 'France': 17,
    'Hungary': 34, 'Iceland': 9, 'Ireland': 88, 'Italy': 47,
    'Latvia': 9, 'Netherlands': 31, 'Norway': 13, 'Poland': 85,
    'Slovenia': 35, 'Spain': 40, 'Sweden': 10,
    'Great Britain': 25, 'United States': 59, 'Northern Ireland': 69,
    'Bulgaria': 9,
}

country_codes = {
    56: 'Belgium', 124: 'Canada', 246: 'Finland', 250: 'France',
    348: 'Hungary', 352: 'Iceland', 372: 'Ireland', 380: 'Italy',
    428: 'Latvia', 528: 'Netherlands', 578: 'Norway', 616: 'Poland',
    705: 'Slovenia', 724: 'Spain', 752: 'Sweden',
    826: 'Great Britain', 840: 'United States', 909: 'Northern Ireland',
    100: 'Bulgaria',
}

for code, name in sorted(country_codes.items(), key=lambda x: x[1]):
    if name not in paper_vals:
        continue
    subset = evs_long[evs_long['country'] == code]
    valid_s = subset[subset['q336'].isin([1,2,3,4,5,6,7,8])].copy()
    monthly_mask = valid_s['q336'].isin([1,2,3])

    unw = round(monthly_mask.sum() / len(valid_s) * 100)

    wm_s = valid_s.loc[monthly_mask, 'weight_s'].sum()
    wt_s = valid_s['weight_s'].sum()
    ws = round(wm_s / wt_s * 100) if wt_s > 0 else None

    wm_g = valid_s.loc[monthly_mask, 'weight_g'].sum()
    wt_g = valid_s['weight_g'].sum()
    wg = round(wm_g / wt_g * 100) if wt_g > 0 else None

    paper = paper_vals[name]
    best = min([abs(unw-paper), abs((ws or 999)-paper), abs((wg or 999)-paper)])

    print(f"{name:<22} paper={paper:>3}  unweighted={unw:>3}  weight_s={ws}  weight_g={wg}  best_diff={best}")
