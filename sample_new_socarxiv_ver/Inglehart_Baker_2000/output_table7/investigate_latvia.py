"""
Investigate Latvia 1990 value - currently MISSING, paper=9.
EVS ZA4460 has Latvia (c_abrv='LV') data.
Also check if EVS has data for Korea.
"""
import pandas as pd
import numpy as np
import math

evs_dta_path = "data/ZA4460_v3-0-0.dta"
evs_csv_path = "data/EVS_1990_wvs_format.csv"

evs = pd.read_stata(evs_dta_path, convert_categoricals=False,
                    columns=['c_abrv', 'country1', 'q365', 'weight_s', 'year'])

print("=== LATVIA IN EVS ZA4460 ===")
lva = evs[evs['c_abrv'] == 'LV']
print(f"Latvia rows: {len(lva)}")
print(f"Year distribution: {lva['year'].value_counts().sort_index()}")
print(f"q365 distribution:\n{lva['q365'].value_counts().sort_index()}")

valid = lva[(lva['q365'] >= 1) & (lva['q365'] <= 10)]
print(f"Valid q365 N={len(valid)}")
if len(valid) > 0:
    pct = (valid['q365'] == 10).mean() * 100
    w = evs.loc[valid.index, 'weight_s']
    pct_w = ((valid['q365'] == 10) * w).sum() / w.sum() * 100 if w.notna().all() and w.gt(0).all() else pct
    print(f"Unweighted %10: {pct:.4f}% -> round={math.floor(pct+0.5)}, floor={math.floor(pct)}, ceil={math.ceil(pct)}")
    print(f"Weighted %10: {pct_w:.4f}% -> round={math.floor(pct_w+0.5)}")
    print(f"Paper=9: does this match?")

print("\n=== LATVIA IN EVS CSV ===")
evs_csv = pd.read_csv(evs_csv_path, low_memory=False)
lva_csv = evs_csv[evs_csv['COUNTRY_ALPHA'] == 'LVA']
print(f"Latvia in EVS CSV rows: {len(lva_csv)}")
if len(lva_csv) > 0:
    print(f"Year dist: {lva_csv['S020'].value_counts().sort_index()}")
    valid_csv = lva_csv[(lva_csv['A006'] >= 1) & (lva_csv['A006'] <= 10)]
    if len(valid_csv) > 0:
        pct = (valid_csv['A006'] == 10).mean() * 100
        print(f"EVS CSV Latvia A006 %10={pct:.4f}% -> round={math.floor(pct+0.5)}")

print("\n=== EVS ZA4460: ALL COUNTRIES AND THEIR GOD IMPORTANCE ===")
za_to_alpha = {
    'US': 'USA', 'GB-GBN': 'GBR', 'GB-NIR': 'NIR', 'IE': 'IRL',
    'BE': 'BEL', 'FR': 'FRA', 'SE': 'SWE', 'NL': 'NLD', 'NO': 'NOR',
    'FI': 'FIN', 'IS': 'ISL', 'ES': 'ESP', 'IT': 'ITA',
    'CA': 'CAN', 'HU': 'HUN', 'BG': 'BGR', 'SI': 'SVN', 'CH': 'CHE',
    'DE': 'DEU', 'LV': 'LVA'
}
evs_valid = evs[(evs['q365'] >= 1) & (evs['q365'] <= 10)]
print(f"\n{'c_abrv':<12} {'alpha':<8} {'N':>6} {'%10':>8} {'round':>6}")
for c_abrv in sorted(evs['c_abrv'].unique()):
    sub = evs_valid[evs_valid['c_abrv'] == c_abrv]
    if len(sub) > 0:
        pct = (sub['q365'] == 10).mean() * 100
        alpha = za_to_alpha.get(c_abrv, '---')
        print(f"  {c_abrv:<10} {alpha:<8} {len(sub):>6} {pct:>8.4f}% {math.floor(pct+0.5):>6}")

print("\n=== DOES LATVIA IN EVS MATCH PAPER? ===")
lva_valid = evs_valid[evs_valid['c_abrv'] == 'LV']
pct = (lva_valid['q365'] == 10).mean() * 100
print(f"Latvia q365 %10: {pct:.4f}% -> round={math.floor(pct+0.5)}, paper=9")
print(f"Exact % comparison: {pct:.2f} vs 9 -> diff={abs(pct-9):.2f}")

print("\n=== SOUTH KOREA IN EVS? ===")
kor_evs = evs[evs['c_abrv'].str.contains('KR', na=False)] if evs['c_abrv'].dtype == object else pd.DataFrame()
print(f"Korea in EVS: {len(kor_evs)}")
# Check all c_abrv values
print(f"All c_abrv values: {sorted(evs['c_abrv'].unique())}")
