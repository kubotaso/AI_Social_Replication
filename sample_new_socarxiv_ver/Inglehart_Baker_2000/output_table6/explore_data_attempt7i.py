"""
Check WVS wave 2 Poland and all wave-by-wave coverage
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
evs_stata_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")

wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'S020', 'F028', 'S017'], low_memory=False)
evs = pd.read_stata(evs_stata_path, convert_categoricals=False,
                     columns=['c_abrv', 'country1', 'q336', 'year', 'weight_g'])

# WVS Poland wave 2
print("=== Poland WVS wave 2 ===")
w2 = wvs[wvs['S002VS'] == 2]
pol2 = w2[w2['S003'] == 616]
print("Poland WVS wave 2 rows:", len(pol2))
if len(pol2) > 0:
    print("F028 dist:", dict(pol2['F028'].value_counts().sort_index()))
    valid = pol2[pol2['F028'].isin([1,2,3,4,5,6,7,8])]
    monthly = pol2[pol2['F028'].isin([1,2,3])]
    if len(valid) > 0:
        pct = round(len(monthly)/len(valid)*100)
        print(f"  Poland W2 WVS: {pct}% (paper=85%)")

# Check ALL countries in WVS wave 1 and 2
print("\n=== All WVS Wave 1 countries ===")
w1 = wvs[wvs['S002VS'] == 1]
print("S003 values:", sorted(w1['S003'].unique()))
# Map S003 to names
country_names = {
    32: 'Argentina', 36: 'Australia', 56: 'Belgium', 76: 'Brazil',
    100: 'Bulgaria', 112: 'Belarus', 124: 'Canada', 152: 'Chile',
    246: 'Finland', 250: 'France', 276: 'Germany', 348: 'Hungary',
    356: 'India', 372: 'Ireland', 380: 'Italy', 392: 'Japan',
    410: 'South Korea', 428: 'Latvia', 484: 'Mexico', 528: 'Netherlands',
    566: 'Nigeria', 578: 'Norway', 616: 'Poland', 643: 'Russia',
    705: 'Slovenia', 710: 'South Africa', 724: 'Spain', 752: 'Sweden',
    756: 'Switzerland', 792: 'Turkey', 826: 'Great Britain', 840: 'United States',
    352: 'Iceland', 372: 'Ireland'
}
for s003 in sorted(w1['S003'].unique()):
    name = country_names.get(s003, f'S003={s003}')
    cnt = len(w1[w1['S003'] == s003])
    print(f"  {name}: {cnt} rows")

print("\n=== All WVS Wave 2 countries ===")
for s003 in sorted(w2['S003'].unique()):
    name = country_names.get(s003, f'S003={s003}')
    cnt = len(w2[w2['S003'] == s003])
    print(f"  {name}: {cnt} rows")

print("\n=== All WVS Wave 3 countries ===")
w3 = wvs[wvs['S002VS'] == 3]
for s003 in sorted(w3['S003'].unique()):
    name = country_names.get(s003, f'S003={s003}')
    cnt = len(w3[w3['S003'] == s003])
    print(f"  {name}: {cnt} rows")

# Check if current best has all the correct countries with proper results
print("\n=== Recomputing all results with Finland fix ===")
MONTHLY_VALS = [1, 2, 3]
VALID_8PT = [1, 2, 3, 4, 5, 6, 7, 8]

# EVS countries with special handling
evs_results = {}
evs_map = {
    'BE': 'Belgium', 'CA': 'Canada', 'ES': 'Spain', 'FI': 'Finland',
    'FR': 'France', 'GB-GBN': 'Great Britain', 'GB-NIR': 'Northern Ireland',
    'IS': 'Iceland', 'IE': 'Ireland', 'IT': 'Italy',
    'LV': 'Latvia', 'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland',
    'SE': 'Sweden', 'SI': 'Slovenia', 'US': 'United States', 'HU': 'Hungary',
    'BG': 'Bulgaria',
}

for alpha, name in evs_map.items():
    c = evs[evs['c_abrv'] == alpha]
    if len(c) == 0:
        continue
    # Finland: use 7pt (exclude 8)
    if alpha == 'FI':
        valid = c[c['q336'].isin([1,2,3,4,5,6,7])]
    elif alpha == 'HU':
        valid = c[c['q336'].isin([1,2,3,4,5,6,7])]
    else:
        valid = c[c['q336'].isin(VALID_8PT)]
    monthly = c[c['q336'].isin(MONTHLY_VALS)]
    if len(valid) > 0:
        pct = round(len(monthly)/len(valid)*100)
        evs_results[(name, '1990-1991')] = pct

# Germany from EVS
deu = evs[evs['c_abrv'] == 'DE']
if len(deu) > 0:
    for c1, rname in [(900, 'West Germany'), (901, 'East Germany')]:
        sub = deu[deu['country1'] == c1]
        valid = sub[sub['q336'].isin(VALID_8PT)]
        monthly = sub[sub['q336'].isin(MONTHLY_VALS)]
        if len(valid) > 0:
            evs_results[(rname, '1990-1991')] = round(len(monthly)/len(valid)*100)

# Compare with paper
paper = {
    'Belgium': 35, 'Canada': 40, 'Finland': 13, 'France': 17,
    'Great Britain': 25, 'Iceland': 9, 'Ireland': 88, 'Northern Ireland': 69,
    'Italy': 47, 'Latvia': 9, 'Netherlands': 31, 'Norway': 13, 'Poland': 85,
    'Sweden': 10, 'Slovenia': 35, 'United States': 59, 'Hungary': 34, 'Bulgaria': 9,
    'West Germany': 33, 'East Germany': 20,
}

print(f"{'Country':<25} {'Gen':>6} {'Paper':>6} {'Diff':>6} {'Status':>10}")
print("-"*60)
for name, pval in sorted(paper.items()):
    gen = evs_results.get((name, '1990-1991'), 'N/A')
    diff = abs(gen - pval) if isinstance(gen, int) else 99
    status = 'FULL' if diff <= 1 else ('PARTIAL' if diff <= 3 else 'MISS')
    print(f"  {name:<23} {str(gen):>5} {pval:>6} {diff if isinstance(gen, int) else '':>6} {status:>10}")
