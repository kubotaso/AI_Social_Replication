"""Check WVS waves 1 and 2 in detail"""
import pandas as pd
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs = pd.read_csv(os.path.join(base, 'data', 'WVS_Time_Series_1981-2022_csv_v5_0.csv'),
                   usecols=['S002VS', 'S003', 'COUNTRY_ALPHA', 'F001', 'S020', 'S017'])

for wave in [1, 2]:
    w = wvs[wvs['S002VS'] == wave]
    print(f'Wave {wave}: years = {sorted(w["S020"].unique())}')
    for alpha in sorted(w['COUNTRY_ALPHA'].unique()):
        sub = w[w['COUNTRY_ALPHA'] == alpha]
        years = sorted(sub['S020'].unique())
        f001_valid = (sub['F001'] > 0).sum()
        print(f'  {alpha}: n={len(sub)}, F001_valid={f001_valid}, years={years}')
    print()

# Also check: are there any European countries with S020 <= 1985 in any wave?
early = wvs[wvs['S020'] <= 1985]
print(f'All entries with S020 <= 1985:')
for alpha in sorted(early['COUNTRY_ALPHA'].unique()):
    sub = early[early['COUNTRY_ALPHA'] == alpha]
    print(f'  {alpha}: n={len(sub)}, wave={sorted(sub["S002VS"].unique())}, years={sorted(sub["S020"].unique())}')

# Check European country codes in wave 2
print()
print('Wave 2 European countries:')
european_codes = ['BEL', 'CAN', 'FRA', 'DEU', 'GBR', 'ISL', 'IRL', 'NIR', 'ITA', 'NLD', 'NOR', 'ESP', 'SWE', 'USA']
w2 = wvs[wvs['S002VS'] == 2]
for code in european_codes:
    sub = w2[w2['COUNTRY_ALPHA'] == code]
    if len(sub) > 0:
        f001_valid = (sub['F001'] > 0).sum()
        print(f'  {code}: n={len(sub)}, F001_valid={f001_valid}, years={sorted(sub["S020"].unique())}')
    else:
        print(f'  {code}: NOT FOUND in Wave 2')

# Also check what S003 codes exist for countries like Belgium, France etc
print()
print('All unique S003 codes and COUNTRY_ALPHA pairs for key countries:')
for code in ['BEL', 'CAN', 'FRA', 'DEU', 'GBR', 'ISL', 'IRL', 'NIR', 'ITA', 'NLD', 'NOR', 'ESP', 'SWE']:
    sub = wvs[wvs['COUNTRY_ALPHA'] == code]
    if len(sub) > 0:
        waves = sorted(sub['S002VS'].unique())
        s003_vals = sorted(sub['S003'].unique())
        print(f'  {code}: waves={waves}, S003={s003_vals}')
