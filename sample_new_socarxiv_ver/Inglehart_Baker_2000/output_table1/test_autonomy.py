#!/usr/bin/env python3
"""Test different autonomy index constructions."""
import pandas as pd, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import clean_missing, get_latest_per_country, varimax

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
    usecols=['S002VS','COUNTRY_ALPHA','S020','A029','A032','A034','A042',
             'F063','A006','F120','G006','E018','Y002','A008','E025','F118','A165'],
    low_memory=False)
wvs = wvs[wvs['S002VS'].isin([2,3])]
wvs['_s'] = 'w'

evs = pd.read_csv('data/EVS_1990_wvs_format.csv')
evs['_s'] = 'e'

df = pd.concat([wvs, evs], ignore_index=True, sort=False)
df = get_latest_per_country(df)

for c in ['A029','A032','A034','A042','F063','A006','F120','G006','E018','Y002','A008','E025','F118','A165']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df.loc[df[c] < 0, c] = np.nan

for c in ['A029','A032','A034','A042']:
    if c in df.columns:
        df.loc[df[c] == 2, c] = 0

df['god'] = np.where(df['_s'] == 'e', df['A006'], df['F063'])
df['F120r'] = 11 - df['F120']
df['G006r'] = 5 - df['G006']
df['E018r'] = 4 - df['E018']
df['Y002r'] = 4 - df['Y002']
df['F118r'] = 11 - df['F118']

cm = df.groupby('COUNTRY_ALPHA')[['god','F120r','G006r','E018r','Y002r','A008',
    'E025','F118r','A165','A042','A034','A029','A032']].mean()

# Test different autonomy constructions
tests = {
    'obedience_only': cm['A042'],
    '3item_diff': cm['A042'] + cm['A034'] - cm['A029'],
    '4item_diff': cm['A042'] + cm['A034'] - cm['A029'] - cm['A032'],
    'trad_sum': cm['A042'] + cm['A034'],
    'ratio': (cm['A042'] + cm['A034']) / (cm['A042'] + cm['A034'] + cm['A029'] + cm['A032']),
}

for name, auto_series in tests.items():
    cm_test = cm.copy()
    cm_test['auto'] = auto_series
    cols = ['god', 'auto', 'F120r', 'G006r', 'E018r', 'Y002r', 'A008', 'E025', 'F118r', 'A165']
    data = cm_test[cols].dropna(thresh=7)
    for c2 in cols:
        data[c2] = data[c2].fillna(data[c2].mean())

    scaled = (data - data.mean()) / data.std()
    corr = np.corrcoef(scaled.values.T)
    ev, evec = np.linalg.eigh(corr)
    idx = np.argsort(ev)[::-1]
    ev = ev[idx]; evec = evec[:, idx]
    L = evec[:, :2] * np.sqrt(ev[:2])
    L, _ = varimax(L)

    ti = [0, 1, 2, 3, 4]
    f1 = sum(abs(L[j, 0]) for j in ti)
    f2 = sum(abs(L[j, 1]) for j in ti)
    tc = 0 if f1 > f2 else 1
    if np.mean([L[j, tc] for j in ti]) < 0:
        L[:, tc] = -L[:, tc]

    print(f'{name:20s}: N={len(data):2d}  auto_load={L[1,tc]:.3f}  god_load={L[0,tc]:.3f}  '
          f'all_trad: {[f"{L[j,tc]:.2f}" for j in ti]}')
