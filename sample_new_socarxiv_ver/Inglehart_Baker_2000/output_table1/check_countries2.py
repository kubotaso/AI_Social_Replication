#!/usr/bin/env python3
import pandas as pd, csv
with open('data/WVS_Time_Series_1981-2022_csv_v5_0.csv','r') as f:
    header=[h.strip('"') for h in next(csv.reader(f))]
wvs=pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',usecols=['S002VS','COUNTRY_ALPHA'],low_memory=False)
wvs23=wvs[wvs['S002VS'].isin([2,3])]
print('GHA in WVS 2+3:', 'GHA' in wvs23['COUNTRY_ALPHA'].unique())
for w in sorted(wvs['S002VS'].unique()):
    c = sorted(wvs[wvs['S002VS']==w]['COUNTRY_ALPHA'].unique())
    for cname in ['GHA','SLV','ALB']:
        if cname in c:
            print(f'{cname} in wave {w}')

evs = pd.read_csv('data/EVS_1990_wvs_format.csv')
evsc = sorted(evs['COUNTRY_ALPHA'].unique())
all_c = sorted(set(wvs23['COUNTRY_ALPHA'].unique()) | set(evsc))
for exclude in [['MNE'], ['MNE','ALB','SLV','MLT'], ['MNE','ALB','SLV']]:
    remain = [c for c in all_c if c not in exclude]
    print(f'Excluding {exclude}: {len(remain)} countries')

# The paper's 65 societies from Figure 1:
paper_countries = [
    'ARG', 'ARM', 'AUS', 'AUT', 'AZE', 'BGD', 'BEL', 'BLR', 'BIH', 'BRA',
    'BGR', 'CAN', 'CHL', 'CHN', 'COL', 'HRV', 'CZE', 'DNK', 'DOM', 'EST',
    'FIN', 'FRA', 'GEO', 'DEU',  # East+West Germany = 2 in paper but 1 here
    'GHA', 'GBR', 'HUN', 'ISL', 'IND', 'IRL', 'ITA', 'JPN', 'KOR',
    'LVA', 'LTU', 'MKD', 'MEX', 'MDA', 'NLD', 'NZL', 'NGA', 'NIR', 'NOR',
    'PAK', 'PER', 'PHL', 'POL', 'PRT', 'PRI', 'ROU', 'RUS', 'SRB',  # Yugoslavia in paper
    'SVK', 'SVN', 'ZAF', 'ESP', 'SWE', 'CHE', 'TWN', 'TUR', 'UKR', 'URY', 'USA', 'VEN'
]
# That's 64 + East/West Germany = 65 (counting Germany as 2)

in_paper_not_us = [c for c in paper_countries if c not in all_c]
in_us_not_paper = [c for c in all_c if c not in paper_countries and c not in ['MNE']]
print(f"\nIn paper but not in our data: {in_paper_not_us}")
print(f"In our data but not in paper: {in_us_not_paper}")
