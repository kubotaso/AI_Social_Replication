#!/usr/bin/env python3
"""Test the effect of excluding different countries."""
import sys, os, csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import clean_missing, varimax

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WVS_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE, "data/EVS_1990_wvs_format.csv")

ITEMS = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
         'Y002', 'A008', 'E025', 'F118', 'A165']

GT_N = {'god_important': 0.91, 'autonomy_idx': 0.89, 'F120': 0.82, 'G006': 0.82, 'E018': 0.72,
        'Y002': 0.86, 'A008': 0.81, 'E025': 0.80, 'F118': 0.78, 'A165': 0.56}
GT_I = {'god_important': 0.70, 'autonomy_idx': 0.61, 'F120': 0.61, 'G006': 0.60, 'E018': 0.51,
        'Y002': 0.59, 'A008': 0.58, 'E025': 0.59, 'F118': 0.54, 'A165': 0.44}

cols = ['S002VS', 'COUNTRY_ALPHA', 'S020',
        'A006', 'A008', 'A029', 'A030', 'A032', 'A034', 'A042',
        'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002']

with open(WVS_PATH, 'r') as f:
    header = [h.strip('"') for h in next(csv.reader(f))]
avail = [c for c in cols if c in header]

wvs = pd.read_csv(WVS_PATH, usecols=avail, low_memory=False)
wvs = wvs[wvs['S002VS'].isin([2, 3])]
wvs['_src'] = 'wvs'

evs = pd.read_csv(EVS_PATH)
evs['_src'] = 'evs'

df_all = pd.concat([wvs, evs], ignore_index=True, sort=False)
df_all = df_all[~df_all['COUNTRY_ALPHA'].isin(['MNE'])]

all_v = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042',
         'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
df_all = clean_missing(df_all, [c for c in all_v if c in df_all.columns])
for c in ['A029', 'A030', 'A032', 'A034', 'A042']:
    if c in df_all.columns:
        df_all.loc[df_all[c] == 2, c] = 0

df_all['god_important'] = np.nan
df_all.loc[df_all['_src'] == 'wvs', 'god_important'] = df_all.loc[df_all['_src'] == 'wvs', 'F063']
df_all.loc[df_all['_src'] == 'evs', 'god_important'] = df_all.loc[df_all['_src'] == 'evs', 'A006']

# 5-item autonomy
child_vars = ['A042', 'A034', 'A029', 'A032', 'A030']
has5 = df_all[child_vars].notna().all(axis=1)
has3 = df_all[['A042', 'A034', 'A029']].notna().all(axis=1) & ~has5
df_all['autonomy_idx'] = np.nan
df_all.loc[has5, 'autonomy_idx'] = (
    df_all.loc[has5, 'A042'] + df_all.loc[has5, 'A034']
    - df_all.loc[has5, 'A029'] - df_all.loc[has5, 'A032'] - df_all.loc[has5, 'A030']
)
auto3 = df_all.loc[has3, 'A042'] + df_all.loc[has3, 'A034'] - df_all.loc[has3, 'A029']
df_all.loc[has3, 'autonomy_idx'] = (auto3 + 1) / 3 * 5 - 3

df_all['F120'] = 11 - df_all['F120']
df_all['G006'] = 5 - df_all['G006']
df_all['E018'] = 4 - df_all['E018']
df_all['Y002'] = 4 - df_all['Y002']
df_all['F118'] = 11 - df_all['F118']


def do_pca(data_matrix, items):
    corr = data_matrix.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    loadings = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    loadings, _ = varimax(loadings)
    var_exp = (loadings ** 2).sum(axis=0) / len(items) * 100
    trad_idx = [items.index(x) for x in ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']]
    surv_idx = [items.index(x) for x in ['Y002', 'A008', 'E025', 'F118', 'A165']]
    f1t = sum(abs(loadings[i, 0]) for i in trad_idx)
    f2t = sum(abs(loadings[i, 1]) for i in trad_idx)
    tc = 0 if f1t > f2t else 1; sc = 1 - tc
    if np.mean([loadings[i, tc] for i in trad_idx]) < 0: loadings[:, tc] *= -1
    if np.mean([loadings[i, sc] for i in surv_idx]) < 0: loadings[:, sc] *= -1
    return loadings, var_exp[tc], var_exp[sc]


def score_variant(df_in, exclude_list, label):
    df = df_in[~df_in['COUNTRY_ALPHA'].isin(exclude_list)]
    cm = df.groupby('COUNTRY_ALPHA')[ITEMS].mean()
    cm = cm.dropna(thresh=7)
    for c in ITEMS: cm[c] = cm[c].fillna(cm[c].mean())
    n = len(cm)

    nl, nv1, nv2 = do_pca(cm, ITEMS)
    il, iv1, iv2 = do_pca(df[ITEMS], ITEMS)

    score = 20
    ls = 0
    for gt, loadings in [(GT_N, nl), (GT_I, il)]:
        for i, it in enumerate(ITEMS):
            dim = 0 if i < 5 else 1
            gv = gt[it]
            gen = loadings[i, dim]
            d = abs(gv - gen)
            if d <= 0.03: ls += 2
            elif d <= 0.06: ls += 1
    score += ls

    ds = 0
    for loadings in [nl, il]:
        c = 0
        for i in range(5):
            if abs(loadings[i, 0]) > abs(loadings[i, 1]): c += 1
        for i in range(5, 10):
            if abs(loadings[i, 1]) > abs(loadings[i, 0]): c += 1
        ds += (c / 10) * 10
    score += ds

    os2 = 0
    for loadings in [nl, il]:
        tv = [loadings[i, 0] for i in range(5)]
        sv = [loadings[i, 1] for i in range(5, 10)]
        if all(tv[j] >= tv[j+1] for j in range(4)): os2 += 2.5
        if all(sv[j] >= sv[j+1] for j in range(4)): os2 += 2.5
    score += os2

    ss = 0
    if abs(n-65)/65 <= 0.05: ss += 2
    elif abs(n-65)/65 <= 0.15: ss += 1
    for gn, gt in [(nv1, 44), (nv2, 26), (iv1, 26), (iv2, 13)]:
        dv = abs(gn - gt)
        if dv <= 3: ss += 2
        elif dv <= 6: ss += 1
    score += ss

    auto_n = nl[1, 0]
    print(f"{label:<45}: score={score:>3.0f}, N={n:>2}, load={ls:>2}/40, dim={ds:.0f}/20, ord={os2:.1f}/10, size={ss}/10, auto_n={auto_n:.3f}")
    return score

# Baseline
score_variant(df_all, [], "Baseline (5-item + all, excl MNE)")

# Try excluding individual countries
countries = sorted(df_all['COUNTRY_ALPHA'].unique())
results = []
for c in countries:
    s = score_variant(df_all, [c], f"Excl {c}")
    results.append((s, c))

print("\nBest exclusions:")
for s, c in sorted(results, reverse=True)[:10]:
    print(f"  {s:.0f}: excl {c}")

# Try excluding pairs
print("\nTrying top single exclusions as pairs:")
top5 = [c for s, c in sorted(results, reverse=True)[:5]]
for i, c1 in enumerate(top5):
    for c2 in top5[i+1:]:
        score_variant(df_all, [c1, c2], f"Excl {c1}+{c2}")
