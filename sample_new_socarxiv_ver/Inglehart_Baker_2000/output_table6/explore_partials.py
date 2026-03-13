import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','S003','F028','X048WVS'], low_memory=False)

# Check PARTIAL match countries for improvement potential
cases = [
    (840, 'US', 3, 55),
    (32, 'Argentina', 3, 41),
    (566, 'Nigeria', 3, 87),
    (756, 'Switzerland', 3, 25),
    (410, 'South Korea', 1, 29),
    (246, 'Finland', 2, 13),
]

for code, name, wave, paper_val in cases:
    d = wvs[(wvs['S003'] == code) & (wvs['S002VS'] == wave)]
    if len(d) == 0:
        print(f'{name} W{wave}: NO DATA')
        continue
    print(f'{name} W{wave}: F028 counts:')
    print(d['F028'].value_counts().sort_index().to_string())
    for vset_name, vset in [
        ('7pt', [1,2,3,4,6,7,8]),
        ('8pt', [1,2,3,4,5,6,7,8]),
        ('7pt+neg2', [-2,1,2,3,4,6,7,8]),
        ('8pt+neg2', [-2,1,2,3,4,5,6,7,8]),
    ]:
        v = d[d['F028'].isin(vset)]
        m = d[d['F028'].isin([1,2,3])]
        if len(v) > 0:
            pct = round(len(m)/len(v)*100)
            print(f'  {vset_name}: {len(m)}/{len(v)}={pct}% (paper={paper_val})')
    print()

# Check EVS Finland 1990
evs = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)
fi_evs = evs[evs['COUNTRY_ALPHA'] == 'FIN']
print(f'Finland EVS: F063 counts:')
print(fi_evs['F063'].value_counts().sort_index().to_string())
v = fi_evs[fi_evs['F063'].isin([1,2,3,4,5,6,7,8])]
m = fi_evs[fi_evs['F063'].isin([1,2,3])]
print(f'  pct: {len(m)/len(v)*100:.1f}% (paper=13)')
print()

# Finland WVS wave 2
fi_w2 = wvs[(wvs['S003'] == 246) & (wvs['S002VS'] == 2)]
if len(fi_w2) > 0:
    print(f'Finland WVS W2: F028 counts:')
    print(fi_w2['F028'].value_counts().sort_index().to_string())
    v = fi_w2[fi_w2['F028'].isin([1,2,3,4,5,6,7,8])]
    m = fi_w2[fi_w2['F028'].isin([1,2,3])]
    if len(v) > 0:
        print(f'  pct: {len(m)/len(v)*100:.1f}%')
else:
    print('Finland WVS W2: NO DATA')

# Check East Germany in ZA4460 EVS
evs_long = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
de = evs_long[evs_long['country'] == 276]
print(f'\nGermany ZA4460: N={len(de)}')
# Check if there's a region variable
for col in ['region', 'x048', 'v376', 'eastwest', 'ost', 'q785', 'q786', 'q787']:
    if col in evs_long.columns:
        vals = de[col].dropna().unique()
        print(f'  {col}: {sorted(vals)[:20]}')

# Check all columns that might indicate East/West Germany
for col in evs_long.columns:
    if 'region' in col.lower() or 'ost' in col.lower() or 'east' in col.lower() or 'west' in col.lower() or 'land' in col.lower():
        vals = de[col].dropna().unique()
        if len(vals) > 0:
            print(f'  {col}: {sorted(vals)[:20]}')
