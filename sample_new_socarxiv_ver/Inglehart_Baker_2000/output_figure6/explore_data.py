#!/usr/bin/env python3
"""Explore data availability for Figure 6."""
import pandas as pd
import sys
sys.path.insert(0, '/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_IB_v5')

# Load WVS
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','S020','S003'], low_memory=False)
wvs = wvs[wvs['S002VS'].isin([1,2,3])]

# Load EVS
evs = pd.read_csv('data/EVS_1990_wvs_format.csv')
if 'S002VS' not in evs.columns:
    evs['S002VS'] = 2

# Summary of WVS
wvs_grp = wvs.groupby(['COUNTRY_ALPHA','S002VS'])['S020'].min().reset_index()
evs_grp = evs.groupby('COUNTRY_ALPHA')['S020'].min().reset_index()
evs_grp['S002VS'] = 2

combined = pd.concat([wvs_grp, evs_grp]).drop_duplicates(subset=['COUNTRY_ALPHA','S002VS'])

multi = []
for ca in sorted(combined['COUNTRY_ALPHA'].unique()):
    sub = combined[combined['COUNTRY_ALPHA']==ca].sort_values('S002VS')
    if len(sub) >= 2:
        info = ', '.join([f'w{int(r.S002VS)}({int(r.S020)})' for _,r in sub.iterrows()])
        multi.append(ca)
        print(f'{ca}: {info}')

print(f'\nTotal: {len(multi)} countries with 2+ waves')

# Also check which countries from the figure we can see
fig_countries = [
    'CHN', 'BGR', 'EST', 'RUS', 'BLR', 'LVA', 'LTU', 'SVN',
    'HUN', 'POL', 'JPN', 'KOR', 'DEU',  # East/West Germany
    'SWE', 'NOR', 'FIN', 'NLD', 'CHE',
    'FRA', 'BEL', 'ITA', 'ESP', 'GBR', 'IRL', 'NIR', 'ISL',
    'CAN', 'AUS', 'USA', 'ARG', 'BRA', 'MEX', 'CHL',
    'TUR', 'ZAF', 'IND', 'NGA'
]
print('\nFigure countries coverage:')
for ca in fig_countries:
    sub = combined[combined['COUNTRY_ALPHA']==ca].sort_values('S002VS')
    if len(sub) >= 1:
        info = ', '.join([f'w{int(r.S002VS)}({int(r.S020)})' for _,r in sub.iterrows()])
        print(f'  {ca}: {info} {"<-- multi" if len(sub)>=2 else "SINGLE"}')
    else:
        print(f'  {ca}: NOT FOUND')

# Check for East/West Germany in S003 or S024
print('\nGermany codes in WVS waves 1-3:')
deu = wvs[wvs['COUNTRY_ALPHA']=='DEU']
print(f'  S003 values: {sorted(deu["S003"].unique())}')
if 'S024' in wvs.columns:
    print(f'  S024 values: {sorted(deu[deu.columns.intersection(["S024"])].iloc[:,0].unique()) if "S024" in deu.columns else "N/A"}')

# Also check DEU wave/year combos
deu_grp = deu.groupby(['S002VS','S003'])['S020'].agg(['min','max','count']).reset_index()
print(deu_grp.to_string())
