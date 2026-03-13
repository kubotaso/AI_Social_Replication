#!/usr/bin/env python3
"""Test 5-item autonomy index."""
import pandas as pd, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import clean_missing, get_latest_per_country, varimax

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
    usecols=['S002VS','COUNTRY_ALPHA','S020','A029','A030','A032','A034','A042',
             'F063','A006','F120','G006','E018','Y002','A008','E025','F118','A165'],
    low_memory=False)
wvs = wvs[wvs['S002VS'].isin([2, 3])]
wvs['_s'] = 'w'
evs = pd.read_csv('data/EVS_1990_wvs_format.csv')
evs['_s'] = 'e'
df = pd.concat([wvs, evs], ignore_index=True, sort=False)
df = get_latest_per_country(df)

for c in ['A029','A030','A032','A034','A042','F063','A006','F120','G006','E018','Y002','A008','E025','F118','A165']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df.loc[df[c] < 0, c] = np.nan
for c in ['A029','A030','A032','A034','A042']:
    if c in df.columns: df.loc[df[c]==2, c] = 0

df['god'] = np.where(df['_s']=='e', df['A006'], df['F063'])
df['F120r'] = 11-df['F120']; df['G006r'] = 5-df['G006']; df['E018r'] = 4-df['E018']
df['Y002r'] = 4-df['Y002']; df['F118r'] = 11-df['F118']

cm = df.groupby('COUNTRY_ALPHA')[['god','F120r','G006r','E018r','Y002r','A008',
    'E025','F118r','A165','A042','A034','A029','A032','A030']].mean()

# Test: 5-item conformity index (obedience + faith - independence - determination - imagination)
# For EVS countries without A032 and A030, use partial and rescale
tests = {
    '4item': cm['A042']+cm['A034']-cm['A029']-cm['A032'],
    '5item': cm['A042']+cm['A034']-cm['A029']-cm['A032']-cm['A030'],
    '2item_pos': cm['A042']+cm['A034'],
    '2item_diff_obe_ind': cm['A042']-cm['A029'],
    '5item_filled': (cm['A042']+cm['A034']-cm['A029']
                     -cm['A032'].fillna(cm['A032'].mean())
                     -cm['A030'].fillna(cm['A030'].mean())),
}

for name, auto in tests.items():
    cm_t = cm.copy()
    cm_t['auto'] = auto
    cols = ['god','auto','F120r','G006r','E018r','Y002r','A008','E025','F118r','A165']
    data = cm_t[cols].dropna(thresh=7)
    for c2 in cols: data[c2] = data[c2].fillna(data[c2].mean())
    corr = np.corrcoef(((data-data.mean())/data.std()).values.T)
    ev, evec = np.linalg.eigh(corr)
    idx = np.argsort(ev)[::-1]; ev=ev[idx]; evec=evec[:,idx]
    L = evec[:,:2]*np.sqrt(ev[:2])
    L, _ = varimax(L)
    ti = [0,1,2,3,4]
    f1=sum(abs(L[j,0]) for j in ti); f2=sum(abs(L[j,1]) for j in ti)
    tc=0 if f1>f2 else 1
    if np.mean([L[j,tc] for j in ti])<0: L[:,tc]=-L[:,tc]
    sc = 1-tc
    if np.mean([L[j,sc] for j in [5,6,7,8,9]])<0: L[:,sc]=-L[:,sc]

    v1 = sum(L[j,tc]**2 for j in range(10))/10*100
    v2 = sum(L[j,sc]**2 for j in range(10))/10*100
    print(f'{name:20s}: N={len(data):2d}  auto={L[1,tc]:.3f}  god={L[0,tc]:.3f}  '
          f'var={v1:.1f}/{v2:.1f}%  '
          f'trad=[{", ".join(f"{L[j,tc]:.2f}" for j in range(5))}]  '
          f'surv=[{", ".join(f"{L[j,sc]:.2f}" for j in range(5,10))}]')
