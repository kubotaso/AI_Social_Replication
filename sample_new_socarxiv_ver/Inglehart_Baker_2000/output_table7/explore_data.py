"""Explore data sources for Table 7 missing values."""
import pandas as pd
import numpy as np

# Check ZA4460 EVS Stata file for God importance variable
print("="*60)
print("EXPLORING ZA4460 EVS STATA FILE")
print("="*60)
evs_dta = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)

# Check q320 and q365 as God importance candidates
for q in ['q320', 'q365']:
    print(f"\n--- Testing {q} ---")
    for c, name in [('US','USA'), ('GB-GBN','GBR'), ('LV','LVA'), ('IE','IRL'),
                     ('BE','BEL'), ('FR','FRA'), ('SE','SWE'), ('NL','NLD'), ('NO','NOR')]:
        s = evs_dta[evs_dta['c_abrv'] == c]
        v = s[q][(s[q] >= 1) & (s[q] <= 10)]
        p10 = (v == 10).mean() * 100 if len(v) > 0 else -1
        print(f'  {q} {name}: pct10={p10:.1f}% n={len(v)}')

# Check Latvia specifically for all 1-10 variables
print("\n\n--- Latvia data in ZA4460 ---")
lva = evs_dta[evs_dta['c_abrv'] == 'LV']
print(f"Latvia rows: {len(lva)}")
for q in ['q320', 'q365']:
    vals = lva[q]
    print(f"  {q}: value_counts = {vals.value_counts().sort_index().to_dict()}")

# Check EVS 1990 CSV for Latvia
print("\n\n" + "="*60)
print("EVS 1990 CSV - Latvia A006")
print("="*60)
evs_csv = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)
lva_csv = evs_csv[evs_csv['COUNTRY_ALPHA'] == 'LVA']
print(f"Latvia rows: {len(lva_csv)}")
print(f"A006 value counts: {lva_csv['A006'].value_counts().sort_index().to_dict()}")
if 'F063' in evs_csv.columns:
    print(f"F063 value counts: {lva_csv['F063'].value_counts().sort_index().to_dict()}")

# Check WVS for Latvia
print("\n\n" + "="*60)
print("WVS DATA - Latvia")
print("="*60)
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F063','A006','S017','S020','G006'],
                   low_memory=False)
lva_wvs = wvs[wvs['COUNTRY_ALPHA'] == 'LVA']
print(f"Latvia total: {len(lva_wvs)}")
for w in sorted(lva_wvs['S002VS'].unique()):
    sub = lva_wvs[lva_wvs['S002VS'] == w]
    f063 = sub['F063'].value_counts().sort_index().to_dict()
    a006 = sub['A006'].value_counts().sort_index().to_dict()
    print(f"  Wave {w}: n={len(sub)}, F063={f063}, A006={a006}")

# Now check all the 1981 missing countries in detail
print("\n\n" + "="*60)
print("1981 MISSING COUNTRIES IN WVS")
print("="*60)
missing_1981 = ['BEL','CAN','FRA','GBR','ISL','IRL','NIR','ITA','NLD','NOR','ESP','SWE','USA','KOR']
for c in missing_1981:
    w1 = wvs[(wvs['COUNTRY_ALPHA'] == c) & (wvs['S002VS'] == 1)]
    if len(w1) > 0:
        f063_valid = w1[(w1['F063'] >= 1) & (w1['F063'] <= 10)]
        a006_valid = w1[(w1['A006'] >= 1)]
        print(f"  {c}: wave1 n={len(w1)}, F063 valid={len(f063_valid)}, A006 valid={len(a006_valid)}")
    else:
        print(f"  {c}: NOT IN WVS WAVE 1")

# Check if the EVS CSV file has an S002VS variable (wave indicator)
print("\n\nEVS CSV columns relevant:", [c for c in evs_csv.columns if c.startswith('S00') or c == 'S020'])
print("EVS CSV S020 unique:", sorted(evs_csv['S020'].unique())[:20] if 'S020' in evs_csv.columns else 'N/A')

# Check raw percentages for close-match countries
print("\n\n" + "="*60)
print("RAW PERCENTAGES FOR CLOSE MATCH COUNTRIES")
print("="*60)

def pct10_raw(data, country, wave_or_period, is_evs=False):
    """Compute % choosing 10, unweighted."""
    if is_evs:
        sub = data[data['COUNTRY_ALPHA'] == country]
        col = 'A006'
    else:
        sub = data[(data['COUNTRY_ALPHA'] == country) & (data['S002VS'] == wave_or_period)]
        col = 'F063'
    valid = sub[(sub[col] >= 1) & (sub[col] <= 10)]
    if len(valid) == 0:
        return None, 0
    return (valid[col] == 10).mean() * 100, len(valid)

# Close matches from attempt 4
close_checks = [
    ('BRA', 2, 'WVS wave 2', False),
    ('GBR', None, 'EVS 1990', True),
    ('ESP', None, 'EVS 1990', True),
    ('NLD', None, 'EVS 1990', True),
    ('USA', None, 'EVS 1990', True),
    ('JPN', 3, 'WVS wave 3', False),
    ('MEX', 3, 'WVS wave 3', False),
    ('NGA', 3, 'WVS wave 3', False),
    ('RUS', 3, 'WVS wave 3', False),
    ('IND', 2, 'WVS wave 2', False),
    ('IND', 3, 'WVS wave 3', False),
    ('ZAF', 1, 'WVS wave 1', False),
    ('ZAF', 2, 'WVS wave 2', False),
    ('ZAF', 3, 'WVS wave 3', False),
]

for country, wave, label, is_evs in close_checks:
    if is_evs:
        pct, n = pct10_raw(evs_csv, country, None, True)
    else:
        pct, n = pct10_raw(wvs, country, wave, False)
    if pct is not None:
        print(f"  {country} {label}: {pct:.2f}% (n={n}), rounds to {round(pct)}")
    else:
        print(f"  {country} {label}: NO DATA")

# Check East Germany coding more carefully
print("\n\n" + "="*60)
print("EAST/WEST GERMANY ANALYSIS")
print("="*60)
deu_wvs = wvs[wvs['COUNTRY_ALPHA'] == 'DEU']
for w in sorted(deu_wvs['S002VS'].unique()):
    sub = deu_wvs[deu_wvs['S002VS'] == w]
    print(f"\nWave {w}: n={len(sub)}")
    print(f"  G006 values: {sub['G006'].value_counts().sort_index().to_dict()}")
    for g, label in [(1,'G006=1'), (2,'G006=2'), (3,'G006=3'), (4,'G006=4')]:
        gsub = sub[sub['G006'] == g]
        valid = gsub[(gsub['F063'] >= 1) & (gsub['F063'] <= 10)]
        if len(valid) > 0:
            pct = (valid['F063'] == 10).mean() * 100
            print(f"  {label}: pct10={pct:.2f}%, n={len(valid)}")

# Also try West=1,2 East=3,4 vs West=1,4 East=2,3 for wave 3
print("\nWave 3 Germany G006 combinations:")
w3_deu = deu_wvs[(deu_wvs['S002VS'] == 3) & (deu_wvs['F063'] >= 1) & (deu_wvs['F063'] <= 10)]
combos = {
    'East(2,3)': [2,3], 'East(3)': [3], 'East(3,4)': [3,4], 'East(2)': [2],
    'West(1,4)': [1,4], 'West(1)': [1], 'West(1,2)': [1,2], 'West(4)': [4],
}
for label, codes in combos.items():
    sub = w3_deu[w3_deu['G006'].isin(codes)]
    if len(sub) > 0:
        pct = (sub['F063'] == 10).mean() * 100
        print(f"  {label}: pct10={pct:.2f}%, n={len(sub)}")

# EVS Germany East/West
print("\nEVS 1990 Germany:")
deu_evs = evs_csv[evs_csv['COUNTRY_ALPHA'] == 'DEU']
for g in sorted(deu_evs['G006'].dropna().unique()):
    gsub = deu_evs[deu_evs['G006'] == g]
    valid = gsub[(gsub['A006'] >= 1) & (gsub['A006'] <= 10)]
    if len(valid) > 0:
        pct = (valid['A006'] == 10).mean() * 100
        print(f"  G006={int(g)}: pct10={pct:.2f}%, n={len(valid)}")

# Also check with S017 weights
print("\n\n" + "="*60)
print("WEIGHTED ANALYSIS FOR KEY DISCREPANCIES")
print("="*60)

def pct10_weighted(df, country, wave):
    sub = df[(df['COUNTRY_ALPHA'] == country) & (df['S002VS'] == wave)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
    if len(valid) == 0:
        return None, None, 0

    w = valid['S017']
    is_10 = (valid['F063'] == 10).astype(float)

    pct_unw = is_10.mean() * 100
    pct_w = (is_10 * w).sum() / w.sum() * 100 if w.sum() > 0 else pct_unw

    return pct_unw, pct_w, len(valid)

for country, wave in [('IND',2), ('IND',3), ('ZAF',1), ('ZAF',2), ('ZAF',3),
                       ('BRA',2), ('NGA',3), ('MEX',3), ('RUS',3), ('USA',2)]:
    uw, wt, n = pct10_weighted(wvs, country, wave)
    if uw is not None:
        print(f"  {country} wave {wave}: unweighted={uw:.2f}%, weighted={wt:.2f}%, n={n}")
