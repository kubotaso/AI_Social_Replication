"""
Final verification of all currently-exact cells plus any remaining unexplored strategies.
"""
import pandas as pd
import numpy as np
import math

wvs_path = "data/WVS_Time_Series_1981-2022_csv_v5_0.csv"
evs_dta_path = "data/ZA4460_v3-0-0.dta"
evs_csv_path = "data/EVS_1990_wvs_format.csv"

wvs = pd.read_csv(wvs_path, low_memory=False,
                  usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'S020', 'G006', 'S017'])
evs = pd.read_stata(evs_dta_path, convert_categoricals=False,
                    columns=['c_abrv', 'country1', 'q365', 'weight_s', 'year'])

def std_round(x):
    return math.floor(x + 0.5)

def check_wvs(country, wave, paper_val):
    sub = wvs[(wvs['COUNTRY_ALPHA'] == country) & (wvs['S002VS'] == wave)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)]
    if len(valid) == 0:
        print(f"  {country} w{wave}: NO VALID DATA (paper={paper_val})")
        return
    pct = (valid['F063'] == 10).mean() * 100
    w = sub.loc[valid.index, 'S017']
    pct_w = ((valid['F063'] == 10) * w).sum() / w.sum() * 100 if w.gt(0).all() else pct
    r_unw = std_round(pct)
    r_w = std_round(pct_w)
    print(f"  {country} w{wave}: unweighted={pct:.4f}%->{r_unw}, weighted={pct_w:.4f}%->{r_w} (paper={paper_val})")

print("=== WVS CELLS VERIFICATION ===")
# Check all cells we're computing from WVS
cells = [
    ('AUS', 1, 25), ('AUS', 3, 21),
    ('FIN', 1, 14), ('FIN', 2, 12), ('FIN', 3, 12),
    ('HUN', 1, 21), ('HUN', 2, 22),
    ('JPN', 1, 6), ('JPN', 2, 6), ('JPN', 3, 5),
    ('MEX', 1, 60), ('MEX', 2, 44), ('MEX', 3, 50),
    ('ARG', 1, 32), ('ARG', 2, 49), ('ARG', 3, 57),
    ('BRA', 2, 83), ('BRA', 3, 87),
    ('CHL', 2, 61), ('CHL', 3, 58),
    ('IND', 2, 44), ('IND', 3, 56),
    ('NGA', 2, 87), ('NGA', 3, 87),
    ('ZAF', 1, 50), ('ZAF', 2, 74), ('ZAF', 3, 71),
    ('TUR', 2, 71), ('TUR', 3, 81),
    ('BLR', 2, 8), ('BLR', 3, 20),
    ('BGR', 2, 7), ('BGR', 3, 10),
    ('LVA', 3, 17),
    ('RUS', 2, 10), ('RUS', 3, 19),
    ('SVN', 2, 14), ('SVN', 3, 15),
    ('NOR', 3, 12),
    ('ESP', 3, 26),
    ('SWE', 3, 8),
    ('CHE', 3, 17),
    ('USA', 3, 50),
]
for c, w, p in cells:
    check_wvs(c, w, p)

print("\n=== EVS ZA4460 CELLS VERIFICATION ===")
za_map = {
    'BE': ('BEL', 13), 'CA': ('CAN', 28), 'FI': ('FIN', 12), 'FR': ('FRA', 10),
    'GB-GBN': ('GBR', 16), 'IS': ('ISL', 17), 'IE': ('IRL', 40),
    'GB-NIR': ('NIR', 41), 'IT': ('ITA', 29), 'NL': ('NLD', 11),
    'NO': ('NOR', 15), 'ES': ('ESP', 18), 'SE': ('SWE', 8),
    'US': ('USA', 48), 'CH': ('CHE', 26),
    'HU': ('HUN', 22), 'BG': ('BGR', 7), 'SI': ('SVN', 14)
}
evs_valid = evs[(evs['q365'] >= 1) & (evs['q365'] <= 10)]
for za_code, (alpha, paper_val) in sorted(za_map.items()):
    sub = evs_valid[evs_valid['c_abrv'] == za_code]
    if len(sub) == 0:
        print(f"  {alpha} ({za_code}): NO DATA (paper={paper_val})")
        continue
    pct = (sub['q365'] == 10).mean() * 100
    w = evs.loc[sub.index, 'weight_s']
    pct_w = ((sub['q365'] == 10) * w).sum() / w.sum() * 100 if w.notna().all() and w.gt(0).all() else pct
    r_unw = std_round(pct)
    r_w = std_round(pct_w)
    print(f"  {alpha} ({za_code}): unweighted={pct:.4f}%->{r_unw}, weighted={pct_w:.4f}%->{r_w} (paper={paper_val})")

# Germany special cases
print("\n=== DEU EAST/WEST in EVS ===")
deu_evs = evs_valid[evs_valid['c_abrv'] == 'DE']
east = deu_evs[deu_evs['country1'] == 901]
west = deu_evs[deu_evs['country1'] == 900]
if len(east) > 0:
    pct = (east['q365'] == 10).mean() * 100
    print(f"  DEU_EAST (ZA4460 country1=901): {pct:.4f}%->{std_round(pct)} (paper=13)")
if len(west) > 0:
    pct = (west['q365'] == 10).mean() * 100
    print(f"  DEU_WEST (ZA4460 country1=900): {pct:.4f}%->{std_round(pct)}")

# EVS CSV for West Germany
evs_csv = pd.read_csv(evs_csv_path, low_memory=False)
if 'A006' in evs_csv.columns and 'G006' in evs_csv.columns:
    deu_csv = evs_csv[(evs_csv['COUNTRY_ALPHA'] == 'DEU') &
                       (evs_csv['A006'] >= 1) & (evs_csv['A006'] <= 10)]
    w_csv = deu_csv[deu_csv['G006'].isin([1, 2])]
    if len(w_csv) > 0:
        pct = (w_csv['A006'] == 10).mean() * 100
        print(f"  DEU_WEST (EVS CSV G006=[1,2]): {pct:.4f}%->{std_round(pct)} (paper=14)")

print("\n=== WVS DEU EAST wave 3 - any approach giving 6? ===")
deu_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'DEU') & (wvs['S002VS'] == 3)]
for g_vals in [[2,3], [2], [3], [-1], [2,3,-1]]:
    sub = deu_w3[deu_w3['G006'].isin(g_vals)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)]
    if len(valid) > 0:
        pct = (valid['F063'] == 10).mean() * 100
        print(f"  G006={g_vals}: N_valid={len(valid)}, %10={pct:.4f}%->{std_round(pct)} (paper=6)")
