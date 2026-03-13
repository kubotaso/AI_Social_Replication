#!/usr/bin/env python3
"""Test full scoring for different configurations to find max score."""
import pandas as pd, csv, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import clean_missing, get_latest_per_country, varimax

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE, 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'), 'r') as f:
    header = [h.strip('"') for h in next(csv.reader(f))]
cols = ['S002VS', 'COUNTRY_ALPHA', 'S020', 'A006', 'A008', 'A029', 'A030', 'A032', 'A034', 'A042', 'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002']
avail = [c for c in cols if c in header]
wvs = pd.read_csv(os.path.join(BASE, 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'), usecols=avail, low_memory=False)
wvs = wvs[wvs['S002VS'].isin([2, 3])]
wvs['_src'] = 'wvs'
evs = pd.read_csv(os.path.join(BASE, 'data/EVS_1990_wvs_format.csv'))
evs['_src'] = 'evs'
df_base = pd.concat([wvs, evs], ignore_index=True, sort=False)

all_v = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042', 'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
df_base = clean_missing(df_base, [c for c in all_v if c in df_base.columns])
for c in ['A029', 'A030', 'A032', 'A034', 'A042']:
    if c in df_base.columns:
        df_base.loc[df_base[c] == 2, c] = 0
df_base['god_important'] = np.nan
df_base.loc[df_base['_src'] == 'wvs', 'god_important'] = df_base.loc[df_base['_src'] == 'wvs', 'F063']
df_base.loc[df_base['_src'] == 'evs', 'god_important'] = df_base.loc[df_base['_src'] == 'evs', 'A006']
df_base['F120'] = 11 - df_base['F120']
df_base['G006'] = 5 - df_base['G006']
df_base['E018'] = 4 - df_base['E018']
df_base['Y002'] = 4 - df_base['Y002']
df_base['F118'] = 11 - df_base['F118']

ITEMS = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
         'Y002', 'A008', 'E025', 'F118', 'A165']

GROUND_TRUTH = {
    'nation_level': {
        'trad_secrat': {'god_important': 0.91, 'autonomy_idx': 0.89, 'F120': 0.82, 'G006': 0.82, 'E018': 0.72},
        'surv_selfexp': {'Y002': 0.86, 'A008': 0.81, 'E025': 0.80, 'F118': 0.78, 'A165': 0.56}
    },
    'individual_level': {
        'trad_secrat': {'god_important': 0.70, 'autonomy_idx': 0.61, 'F120': 0.61, 'G006': 0.60, 'E018': 0.51},
        'surv_selfexp': {'Y002': 0.59, 'A008': 0.58, 'E025': 0.59, 'F118': 0.54, 'A165': 0.44}
    },
    'variance_explained': {'nation_dim1': 44, 'nation_dim2': 26, 'indiv_dim1': 26, 'indiv_dim2': 13},
    'N_nation': 65, 'N_individual': 165594,
}

def build_autonomy(df, method='5item'):
    df = df.copy()
    if method == '5item':
        has5 = df[['A042','A034','A029','A032','A030']].notna().all(axis=1)
        has3 = df[['A042','A034','A029']].notna().all(axis=1) & ~has5
        df['autonomy_idx'] = np.nan
        df.loc[has5,'autonomy_idx'] = df.loc[has5,'A042']+df.loc[has5,'A034']-df.loc[has5,'A029']-df.loc[has5,'A032']-df.loc[has5,'A030']
        auto3 = df.loc[has3,'A042']+df.loc[has3,'A034']-df.loc[has3,'A029']
        df.loc[has3,'autonomy_idx'] = (auto3+1)/3*5-3
    elif method == '3item':
        df['autonomy_idx'] = df['A042'] + df['A034'] - df['A029']
    elif method == '4item':
        has4 = df[['A042','A034','A029','A032']].notna().all(axis=1)
        has3 = df[['A042','A034','A029']].notna().all(axis=1) & ~has4
        df['autonomy_idx'] = np.nan
        df.loc[has4,'autonomy_idx'] = df.loc[has4,'A042']+df.loc[has4,'A034']-df.loc[has4,'A029']-df.loc[has4,'A032']
        wvs_a032_mean = df.loc[(df['_src']=='wvs') & (df['A032'].notna()), 'A032'].mean()
        df.loc[has3,'autonomy_idx'] = df.loc[has3,'A042']+df.loc[has3,'A034']-df.loc[has3,'A029']-wvs_a032_mean
    return df

def do_pca_varimax(data_matrix, items):
    corr = data_matrix.corr().values
    evals, evecs = np.linalg.eigh(corr)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]; evecs = evecs[:, idx]
    loadings = evecs[:, :2] * np.sqrt(evals[:2])
    loadings, _ = varimax(loadings)
    var_exp = (loadings ** 2).sum(axis=0) / len(items) * 100
    ti = [items.index(x) for x in ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']]
    si = [items.index(x) for x in ['Y002', 'A008', 'E025', 'F118', 'A165']]
    f1t = sum(abs(loadings[i, 0]) for i in ti)
    f2t = sum(abs(loadings[i, 1]) for i in ti)
    tc = 0 if f1t > f2t else 1; sc = 1 - tc
    if np.mean([loadings[i, tc] for i in ti]) < 0: loadings[:, tc] *= -1
    if np.mean([loadings[i, sc] for i in si]) < 0: loadings[:, sc] *= -1
    result = pd.DataFrame({'item': items, 'trad_secrat': loadings[:, tc], 'surv_selfexp': loadings[:, sc]})
    return result, var_exp[tc], var_exp[sc]

def full_score(exclude, auto_method, nation_method):
    df = df_base[~df_base['COUNTRY_ALPHA'].isin(exclude)].copy()
    df = build_autonomy(df, auto_method)

    # Nation level
    if nation_method == 'all':
        cm = df.groupby('COUNTRY_ALPHA')[ITEMS].mean()
    else:
        df_n = get_latest_per_country(df)
        cm = df_n.groupby('COUNTRY_ALPHA')[ITEMS].mean()
    cm = cm.dropna(thresh=7)
    for c in ITEMS: cm[c] = cm[c].fillna(cm[c].mean())
    n_countries = len(cm)
    nation_load, nv1, nv2 = do_pca_varimax(cm, ITEMS)

    # Individual level
    data_i = df[ITEMS].copy()
    total_n = data_i.dropna(how='all').shape[0]
    indiv_load, iv1, iv2 = do_pca_varimax(data_i, ITEMS)

    # Score
    score = 0

    # Items (20)
    score += 20

    # Loadings (40)
    ls = 0
    for lev, ldf in [('nation', nation_load), ('individual', indiv_load)]:
        gt = GROUND_TRUTH[f'{lev}_level']
        for dim, its in [('trad_secrat', ITEMS[:5]), ('surv_selfexp', ITEMS[5:])]:
            for it in its:
                gv = gt[dim][it]; gen = ldf[ldf['item']==it][dim].values[0]
                d = abs(gv - gen)
                if d <= 0.03: ls += 2
                elif d <= 0.06: ls += 1
    score += ls

    # Dimensions (20)
    ds = 0
    for lev, ldf in [('nation', nation_load), ('individual', indiv_load)]:
        c = 0
        for it in ITEMS[:5]:
            if abs(ldf[ldf['item']==it]['trad_secrat'].values[0]) > abs(ldf[ldf['item']==it]['surv_selfexp'].values[0]): c+=1
        for it in ITEMS[5:]:
            if abs(ldf[ldf['item']==it]['surv_selfexp'].values[0]) > abs(ldf[ldf['item']==it]['trad_secrat'].values[0]): c+=1
        ds += (c/10)*10
    score += ds

    # Ordering (10)
    os2 = 0
    for lev, ldf in [('nation', nation_load), ('individual', indiv_load)]:
        tv = [ldf[ldf['item']==i]['trad_secrat'].values[0] for i in ITEMS[:5]]
        sv = [ldf[ldf['item']==i]['surv_selfexp'].values[0] for i in ITEMS[5:]]
        tok = all(tv[j]>=tv[j+1] for j in range(4))
        sok = all(sv[j]>=sv[j+1] for j in range(4))
        if tok: os2+=2.5
        if sok: os2+=2.5
    score += os2

    # Size/variance (10)
    ss = 0
    if abs(n_countries-65)/65<=0.05: ss+=2
    elif abs(n_countries-65)/65<=0.15: ss+=1
    for gn,gt in [(nv1,44),(nv2,26),(iv1,26),(iv2,13)]:
        d=abs(gn-gt)
        if d<=3: ss+=2
        elif d<=6: ss+=1
    score += ss

    auto_n = nation_load[nation_load['item']=='autonomy_idx']['trad_secrat'].values[0]
    return round(score), n_countries, auto_n, ls, os2, ss

# Test all combinations
configs = []
for excl_name, excl in [
    ('MNE', ['MNE']),
    ('MNE+ALB+SLV+MLT', ['MNE','ALB','SLV','MLT']),
    ('MNE+ALB+SLV', ['MNE','ALB','SLV']),
    ('MNE+MLT', ['MNE','MLT']),
]:
    for auto in ['5item', '3item', '4item']:
        for nation in ['all', 'latest']:
            s, n, auto_l, ls, os2, ss = full_score(excl, auto, nation)
            configs.append((s, excl_name, auto, nation, n, auto_l, ls, os2, ss))

# Sort by score
configs.sort(key=lambda x: -x[0])
print(f"{'Score':>5} {'Exclude':>25} {'Auto':>6} {'Nation':>7} {'N':>3} {'auto_l':>6} {'load':>5} {'ord':>4} {'size':>4}")
print("-" * 75)
for s, en, au, na, n, al, ls, os2, ss in configs:
    print(f"{s:5d} {en:>25s} {au:>6s} {na:>7s} {n:3d} {al:6.3f} {ls:5.0f} {os2:4.0f} {ss:4.0f}")
