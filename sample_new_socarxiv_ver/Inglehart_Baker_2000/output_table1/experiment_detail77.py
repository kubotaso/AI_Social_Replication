#!/usr/bin/env python3
"""Detailed analysis of the best-scoring configuration."""
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
df = pd.concat([wvs, evs], ignore_index=True, sort=False)
df = df[~df['COUNTRY_ALPHA'].isin(['MNE', 'MLT'])]

all_v = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042', 'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
df = clean_missing(df, [c for c in all_v if c in df.columns])
for c in ['A029', 'A030', 'A032', 'A034', 'A042']:
    if c in df.columns:
        df.loc[df[c] == 2, c] = 0
df['god_important'] = np.nan
df.loc[df['_src'] == 'wvs', 'god_important'] = df.loc[df['_src'] == 'wvs', 'F063']
df.loc[df['_src'] == 'evs', 'god_important'] = df.loc[df['_src'] == 'evs', 'A006']
df['F120'] = 11 - df['F120']
df['G006'] = 5 - df['G006']
df['E018'] = 4 - df['E018']
df['Y002'] = 4 - df['Y002']
df['F118'] = 11 - df['F118']

# Config 1: MNE+MLT, 5item, latest
has5 = df[['A042','A034','A029','A032','A030']].notna().all(axis=1)
has3 = df[['A042','A034','A029']].notna().all(axis=1) & ~has5
df['autonomy_idx'] = np.nan
df.loc[has5,'autonomy_idx'] = df.loc[has5,'A042']+df.loc[has5,'A034']-df.loc[has5,'A029']-df.loc[has5,'A032']-df.loc[has5,'A030']
auto3 = df.loc[has3,'A042']+df.loc[has3,'A034']-df.loc[has3,'A029']
df.loc[has3,'autonomy_idx'] = (auto3+1)/3*5-3

ITEMS = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
         'Y002', 'A008', 'E025', 'F118', 'A165']

GT = {
    'nation_level': {
        'trad_secrat': {'god_important': 0.91, 'autonomy_idx': 0.89, 'F120': 0.82, 'G006': 0.82, 'E018': 0.72},
        'surv_selfexp': {'Y002': 0.86, 'A008': 0.81, 'E025': 0.80, 'F118': 0.78, 'A165': 0.56}
    },
    'individual_level': {
        'trad_secrat': {'god_important': 0.70, 'autonomy_idx': 0.61, 'F120': 0.61, 'G006': 0.60, 'E018': 0.51},
        'surv_selfexp': {'Y002': 0.59, 'A008': 0.58, 'E025': 0.59, 'F118': 0.54, 'A165': 0.44}
    },
}

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

# Nation: latest per country
df_n = get_latest_per_country(df)
cm = df_n.groupby('COUNTRY_ALPHA')[ITEMS].mean()
cm = cm.dropna(thresh=7)
for c in ITEMS: cm[c] = cm[c].fillna(cm[c].mean())
print(f"N countries: {len(cm)}")

nl, nv1, nv2 = do_pca_varimax(cm, ITEMS)

# Individual
data_i = df[ITEMS].copy()
il, iv1, iv2 = do_pca_varimax(data_i, ITEMS)
total_n = data_i.dropna(how='all').shape[0]

print(f"\n{'Item':20s} {'Paper':>6s} {'Nation':>7s} {'d':>5s} {'Tag':>7s} {'Paper':>6s} {'Indiv':>7s} {'d':>5s} {'Tag':>7s}")
print("-" * 75)
for dim, its in [('trad_secrat', ITEMS[:5]), ('surv_selfexp', ITEMS[5:])]:
    for it in its:
        gn = GT['nation_level'][dim][it]
        gi = GT['individual_level'][dim][it]
        gen_n = nl[nl['item']==it][dim].values[0]
        gen_i = il[il['item']==it][dim].values[0]
        dn = abs(gn-gen_n)
        di = abs(gi-gen_i)
        tn = "MATCH" if dn<=0.03 else "PARTIAL" if dn<=0.06 else "MISS"
        ti2 = "MATCH" if di<=0.03 else "PARTIAL" if di<=0.06 else "MISS"
        print(f"{it:20s} {gn:6.2f} {gen_n:7.3f} {dn:5.3f} {tn:>7s} {gi:6.2f} {gen_i:7.3f} {di:5.3f} {ti2:>7s}")
    print()

print(f"Var nation: {nv1:.1f}/{nv2:.1f} (paper: 44/26)")
print(f"Var indiv:  {iv1:.1f}/{iv2:.1f} (paper: 26/13)")
print(f"N ind: {total_n}")

# Now also test Config 2: MNE+MLT, 3item, all
print("\n\n=== Config 2: MNE+MLT, 3-item, all respondents ===")
df2 = df.copy()
df2['autonomy_idx'] = df2['A042'] + df2['A034'] - df2['A029']
cm2 = df2.groupby('COUNTRY_ALPHA')[ITEMS].mean()
cm2 = cm2.dropna(thresh=7)
for c in ITEMS: cm2[c] = cm2[c].fillna(cm2[c].mean())
print(f"N countries: {len(cm2)}")
nl2, nv1_2, nv2_2 = do_pca_varimax(cm2, ITEMS)
il2, iv1_2, iv2_2 = do_pca_varimax(df2[ITEMS].copy(), ITEMS)

print(f"\n{'Item':20s} {'Paper':>6s} {'Nation':>7s} {'d':>5s} {'Tag':>7s} {'Paper':>6s} {'Indiv':>7s} {'d':>5s} {'Tag':>7s}")
print("-" * 75)
for dim, its in [('trad_secrat', ITEMS[:5]), ('surv_selfexp', ITEMS[5:])]:
    for it in its:
        gn = GT['nation_level'][dim][it]
        gi = GT['individual_level'][dim][it]
        gen_n = nl2[nl2['item']==it][dim].values[0]
        gen_i = il2[il2['item']==it][dim].values[0]
        dn = abs(gn-gen_n)
        di = abs(gi-gen_i)
        tn = "MATCH" if dn<=0.03 else "PARTIAL" if dn<=0.06 else "MISS"
        ti2 = "MATCH" if di<=0.03 else "PARTIAL" if di<=0.06 else "MISS"
        print(f"{it:20s} {gn:6.2f} {gen_n:7.3f} {dn:5.3f} {tn:>7s} {gi:6.2f} {gen_i:7.3f} {di:5.3f} {ti2:>7s}")
    print()

print(f"Var nation: {nv1_2:.1f}/{nv2_2:.1f} (paper: 44/26)")
print(f"Var indiv:  {iv1_2:.1f}/{iv2_2:.1f} (paper: 26/13)")
