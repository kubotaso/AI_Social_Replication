"""
Check WVS wave 2 coverage for all Table 7 countries.
"""
import pandas as pd
import math

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv', low_memory=False,
                  usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'S020', 'G006', 'S017'])

def std_round(x):
    return math.floor(x + 0.5)

# All Table 7 countries
countries = {
    'AUS': 21, 'BEL': 13, 'CAN': 28, 'FIN': 12, 'FRA': 10,
    'DEU': None, 'GBR': 16, 'ISL': 17, 'IRL': 40, 'NIR': 41,
    'KOR': 39, 'ITA': 29, 'JPN': 6, 'NLD': 11, 'NOR': 15,
    'ESP': 18, 'SWE': 8, 'CHE': 26, 'USA': 48,
    'BLR': 8, 'BGR': 7, 'HUN': 22, 'LVA': 9, 'RUS': 10, 'SVN': 14,
    'ARG': 49, 'BRA': 83, 'CHL': 61, 'IND': 44, 'MEX': 44,
    'NGA': 87, 'ZAF': 74, 'TUR': 71
}

print("=== WVS WAVE 2 COVERAGE FOR TABLE 7 COUNTRIES ===")
wave2 = wvs[wvs['S002VS'] == 2]
print(f"WVS wave 2 countries:\n{wave2['COUNTRY_ALPHA'].value_counts().sort_index()}")

print("\n=== DETAILED CHECK ===")
for country, paper_val in sorted(countries.items()):
    if country == 'DEU':
        continue  # handled separately
    sub = wave2[wave2['COUNTRY_ALPHA'] == country]
    if len(sub) == 0:
        print(f"  {country}: NOT IN WVS WAVE 2")
        continue
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)]
    if len(valid) == 0:
        print(f"  {country}: in WVS wave 2 but F063 all invalid (N={len(sub)})")
        continue
    pct = (valid['F063'] == 10).mean() * 100
    w = sub.loc[valid.index, 'S017']
    pct_w = ((valid['F063'] == 10) * w).sum() / w.sum() * 100 if w.gt(0).all() else pct
    r_unw = std_round(pct)
    r_w = std_round(pct_w)
    match_unw = "EXACT" if r_unw == paper_val else f"diff={r_unw-paper_val:+d}"
    match_w = "EXACT" if r_w == paper_val else f"diff={r_w-paper_val:+d}"
    print(f"  {country}: unw={pct:.2f}%->{r_unw}({match_unw}), w={pct_w:.2f}%->{r_w}({match_w}) (paper={paper_val})")
    print(f"    Year: {sub['S020'].value_counts().sort_index().to_dict()}")

# Germany wave 2 east/west
print("\n=== GERMANY WAVE 2 EAST/WEST ===")
deu_w2 = wave2[wave2['COUNTRY_ALPHA'] == 'DEU']
print(f"DEU wave 2 N={len(deu_w2)}")
if len(deu_w2) > 0:
    print(f"G006 dist: {deu_w2['G006'].value_counts().sort_index().to_dict()}")
    for g_vals, label, paper_val in [([2,3], 'EAST', 13), ([1,4], 'WEST', 14)]:
        sub = deu_w2[deu_w2['G006'].isin(g_vals)]
        valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)]
        if len(valid) > 0:
            pct = (valid['F063'] == 10).mean() * 100
            print(f"  {label}: N={len(valid)}, %10={pct:.4f}%->{std_round(pct)} (paper={paper_val})")
