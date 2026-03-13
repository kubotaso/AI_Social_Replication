#!/usr/bin/env python3
"""Test all combinations of autonomy construction x aggregation method."""
import sys, os, csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import clean_missing, get_latest_per_country, varimax

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WVS_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE, "data/EVS_1990_wvs_format.csv")

ITEMS = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
         'Y002', 'A008', 'E025', 'F118', 'A165']

GROUND_TRUTH_N = {'god_important': 0.91, 'autonomy_idx': 0.89, 'F120': 0.82, 'G006': 0.82, 'E018': 0.72,
                  'Y002': 0.86, 'A008': 0.81, 'E025': 0.80, 'F118': 0.78, 'A165': 0.56}
GROUND_TRUTH_I = {'god_important': 0.70, 'autonomy_idx': 0.61, 'F120': 0.61, 'G006': 0.60, 'E018': 0.51,
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

df = pd.concat([wvs, evs], ignore_index=True, sort=False)
df = df[~df['COUNTRY_ALPHA'].isin(['MNE'])]

all_v = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042',
         'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
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


def make_autonomy_3item(df_in):
    """3-item: obey + faith - independence"""
    d = df_in.copy()
    d['autonomy_idx'] = np.nan
    has3 = d[['A042', 'A034', 'A029']].notna().all(axis=1)
    d.loc[has3, 'autonomy_idx'] = d.loc[has3, 'A042'] + d.loc[has3, 'A034'] - d.loc[has3, 'A029']
    return d

def make_autonomy_5item(df_in):
    """5-item for WVS, 3-item rescaled for EVS"""
    d = df_in.copy()
    child_vars = ['A042', 'A034', 'A029', 'A032', 'A030']
    has5 = d[child_vars].notna().all(axis=1)
    has3 = d[['A042', 'A034', 'A029']].notna().all(axis=1) & ~has5
    d['autonomy_idx'] = np.nan
    d.loc[has5, 'autonomy_idx'] = (
        d.loc[has5, 'A042'] + d.loc[has5, 'A034']
        - d.loc[has5, 'A029'] - d.loc[has5, 'A032'] - d.loc[has5, 'A030']
    )
    auto3 = d.loc[has3, 'A042'] + d.loc[has3, 'A034'] - d.loc[has3, 'A029']
    d.loc[has3, 'autonomy_idx'] = (auto3 + 1) / 3 * 5 - 3
    return d

def make_autonomy_4item(df_in):
    """4-item for WVS (obey+faith-indep-determ), 3-item rescaled for EVS"""
    d = df_in.copy()
    has4 = d[['A042', 'A034', 'A029', 'A032']].notna().all(axis=1)
    has3 = d[['A042', 'A034', 'A029']].notna().all(axis=1) & ~has4
    d['autonomy_idx'] = np.nan
    d.loc[has4, 'autonomy_idx'] = (
        d.loc[has4, 'A042'] + d.loc[has4, 'A034']
        - d.loc[has4, 'A029'] - d.loc[has4, 'A032']
    )
    auto3 = d.loc[has3, 'A042'] + d.loc[has3, 'A034'] - d.loc[has3, 'A029']
    d.loc[has3, 'autonomy_idx'] = (auto3 + 1) / 3 * 4 - 2
    return d

def make_autonomy_4item_mean_imp(df_in):
    """4-item for WVS, 3-item + mean-imputed A032 for EVS"""
    d = df_in.copy()
    # Get mean A032 from WVS
    a032_mean = d.loc[d['A032'].notna(), 'A032'].mean()
    has4 = d[['A042', 'A034', 'A029', 'A032']].notna().all(axis=1)
    has3 = d[['A042', 'A034', 'A029']].notna().all(axis=1) & ~has4
    d['autonomy_idx'] = np.nan
    d.loc[has4, 'autonomy_idx'] = (
        d.loc[has4, 'A042'] + d.loc[has4, 'A034']
        - d.loc[has4, 'A029'] - d.loc[has4, 'A032']
    )
    d.loc[has3, 'autonomy_idx'] = (
        d.loc[has3, 'A042'] + d.loc[has3, 'A034']
        - d.loc[has3, 'A029'] - a032_mean
    )
    return d


def do_pca_varimax(data_matrix, items):
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


def score_combo(auto_fn, agg_method, label):
    """Score a combination and return total."""
    d = auto_fn(df)

    if agg_method == 'latest':
        dn = get_latest_per_country(d)
        cm = dn.groupby('COUNTRY_ALPHA')[ITEMS].mean()
    elif agg_method == 'all':
        cm = d.groupby('COUNTRY_ALPHA')[ITEMS].mean()

    cm = cm.dropna(thresh=7)
    for c in ITEMS: cm[c] = cm[c].fillna(cm[c].mean())

    nl, nv1, nv2 = do_pca_varimax(cm, ITEMS)
    il, iv1, iv2 = do_pca_varimax(d[ITEMS], ITEMS)

    # Score
    score = 20  # items always present
    ls = 0
    for gt, loadings in [(GROUND_TRUTH_N, nl), (GROUND_TRUTH_I, il)]:
        for i, it in enumerate(ITEMS):
            dim = 0 if i < 5 else 1
            gv = gt[it]
            gen = loadings[i, dim]
            d_val = abs(gv - gen)
            if d_val <= 0.03: ls += 2
            elif d_val <= 0.06: ls += 1
    score += ls

    # Dimension
    ds = 0
    for loadings in [nl, il]:
        c = 0
        for i in range(5):
            if abs(loadings[i, 0]) > abs(loadings[i, 1]): c += 1
        for i in range(5, 10):
            if abs(loadings[i, 1]) > abs(loadings[i, 0]): c += 1
        ds += (c / 10) * 10
    score += ds

    # Ordering
    os2 = 0
    for loadings in [nl, il]:
        tv = [loadings[i, 0] for i in range(5)]
        sv = [loadings[i, 1] for i in range(5, 10)]
        if all(tv[j] >= tv[j+1] for j in range(4)): os2 += 2.5
        if all(sv[j] >= sv[j+1] for j in range(4)): os2 += 2.5
    score += os2

    # Size
    ss = 0
    n = len(cm)
    if abs(n-65)/65 <= 0.05: ss += 2
    elif abs(n-65)/65 <= 0.15: ss += 1
    for gn, gt in [(nv1, 44), (nv2, 26), (iv1, 26), (iv2, 13)]:
        dv = abs(gn - gt)
        if dv <= 3: ss += 2
        elif dv <= 6: ss += 1
    score += ss

    auto_n = nl[1, 0]
    auto_i = il[1, 0]
    print(f"{label:<50}: score={score:>3}, load={ls:>2}/40, dim={ds:.0f}/20, ord={os2:.1f}/10, size={ss}/10  auto_n={auto_n:.3f} auto_i={auto_i:.3f} var_n={nv1:.1f}/{nv2:.1f} var_i={iv1:.1f}/{iv2:.1f}")
    return score


combos = [
    (make_autonomy_3item, 'latest', '3-item + latest'),
    (make_autonomy_3item, 'all', '3-item + all'),
    (make_autonomy_4item, 'latest', '4-item + latest'),
    (make_autonomy_4item, 'all', '4-item + all'),
    (make_autonomy_4item_mean_imp, 'latest', '4-item-mean-imp + latest'),
    (make_autonomy_4item_mean_imp, 'all', '4-item-mean-imp + all'),
    (make_autonomy_5item, 'latest', '5-item + latest'),
    (make_autonomy_5item, 'all', '5-item + all'),
]

print(f"{'Combination':<50}: {'score':>5}, {'load':>5}, {'dim':>5}, {'ord':>5}, {'size':>5}")
print("=" * 120)
results = []
for auto_fn, agg, label in combos:
    s = score_combo(auto_fn, agg, label)
    results.append((s, label))

print()
print("Ranked:")
for s, l in sorted(results, reverse=True):
    print(f"  {s:>3}: {l}")
