"""
Check Germany country1 codes in EVS ZA4460 and EVS CSV.
Also check if any other approach gives DEU_EAST 1990 = 13 more precisely.
"""
import pandas as pd
import math

evs = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
evs_csv = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)

def std_round(x):
    return math.floor(x + 0.5)

print("=== GERMANY IN ZA4460 ===")
deu = evs[evs['c_abrv'] == 'DE']
print(f"Total DE: {len(deu)}")
print(f"country1 distribution:\n{deu['country1'].value_counts().sort_index()}")
valid = deu[(deu['q365'] >= 1) & (deu['q365'] <= 10)]
print(f"\nValid q365: {len(valid)}")
for c1 in sorted(deu['country1'].unique()):
    sub = deu[deu['country1'] == c1]
    v = sub[(sub['q365'] >= 1) & (sub['q365'] <= 10)]
    if len(v) > 0:
        pct = (v['q365'] == 10).mean() * 100
        print(f"  country1={c1}: N_valid={len(v)}, %10={pct:.4f}% -> {std_round(pct)}")

print("\n=== GERMANY IN EVS CSV ===")
deu_csv = evs_csv[evs_csv['COUNTRY_ALPHA'] == 'DEU']
print(f"Total DEU: {len(deu_csv)}")
print(f"G006 distribution:\n{deu_csv['G006'].value_counts().sort_index()}")
valid_csv = deu_csv[(deu_csv['A006'] >= 1) & (deu_csv['A006'] <= 10)]
print(f"\nValid A006: {len(valid_csv)}")
for g6 in sorted(deu_csv['G006'].unique()):
    sub = deu_csv[deu_csv['G006'] == g6]
    v = sub[(sub['A006'] >= 1) & (sub['A006'] <= 10)]
    if len(v) > 0:
        pct = (v['A006'] == 10).mean() * 100
        print(f"  G006={g6}: N_valid={len(v)}, %10={pct:.4f}% -> {std_round(pct)}")

print("\nKey: paper DEU_WEST 1990 = 14, DEU_EAST 1990 = 13")
print("Current: EVS CSV G006=[1,2] for WEST=14 (EXACT)")
print("Current: ZA4460 country1=901 for EAST=13 (EXACT)")

# What does EVS CSV G006 say about East?
east_csv = valid_csv[valid_csv['G006'].isin([3, 4, 5])]
if len(east_csv) > 0:
    pct = (east_csv['A006'] == 10).mean() * 100
    print(f"\nEVS CSV G006=[3,4,5] (East?): N={len(east_csv)}, %10={pct:.4f}% -> {std_round(pct)}")

# Try G006=[3] for east in EVS CSV
east_csv3 = valid_csv[valid_csv['G006'] == 3]
if len(east_csv3) > 0:
    pct = (east_csv3['A006'] == 10).mean() * 100
    print(f"EVS CSV G006=3: N={len(east_csv3)}, %10={pct:.4f}% -> {std_round(pct)}")

# Summary of all unique G006 values
print(f"\nAll G006 values in EVS CSV for DEU: {sorted(deu_csv['G006'].unique().tolist())}")
