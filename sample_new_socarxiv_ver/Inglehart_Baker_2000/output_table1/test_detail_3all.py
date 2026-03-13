#!/usr/bin/env python3
"""Detail test for 3-item + all combination."""
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
        'A006', 'A008', 'A029', 'A034', 'A042',
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

all_v = ['A006', 'F063', 'A029', 'A034', 'A042', 'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
df = clean_missing(df, [c for c in all_v if c in df.columns])
for c in ['A029', 'A034', 'A042']:
    df.loc[df[c] == 2, c] = 0

df['god_important'] = np.nan
df.loc[df['_src'] == 'wvs', 'god_important'] = df.loc[df['_src'] == 'wvs', 'F063']
df.loc[df['_src'] == 'evs', 'god_important'] = df.loc[df['_src'] == 'evs', 'A006']

df['autonomy_idx'] = np.nan
has3 = df[['A042', 'A034', 'A029']].notna().all(axis=1)
df.loc[has3, 'autonomy_idx'] = df.loc[has3, 'A042'] + df.loc[has3, 'A034'] - df.loc[has3, 'A029']

df['F120'] = 11 - df['F120']
df['G006'] = 5 - df['G006']
df['E018'] = 4 - df['E018']
df['Y002'] = 4 - df['Y002']
df['F118'] = 11 - df['F118']

# Nation level: all observations
cm = df.groupby('COUNTRY_ALPHA')[ITEMS].mean()
cm = cm.dropna(thresh=7)
for c in ITEMS: cm[c] = cm[c].fillna(cm[c].mean())

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
    result = pd.DataFrame({'item': items, 'trad': loadings[:, tc], 'surv': loadings[:, sc]})
    return result, var_exp[tc], var_exp[sc]

nl, nv1, nv2 = do_pca(cm, ITEMS)
il, iv1, iv2 = do_pca(df[ITEMS], ITEMS)

print("3-item + all combination - detailed scoring:")
print(f"N={len(cm)} countries")
print(f"\nNation-level (var: {nv1:.1f}/{nv2:.1f}%):")
for it in ITEMS:
    dim = 'trad' if ITEMS.index(it) < 5 else 'surv'
    gt = GT_N[it]
    gen = nl[nl['item']==it][dim].values[0]
    d = abs(gt - gen)
    tag = "MATCH" if d <= 0.03 else "PARTIAL" if d <= 0.06 else "MISS"
    pts = 2 if d <= 0.03 else 1 if d <= 0.06 else 0
    print(f"  {it:<20}: paper={gt:.2f} gen={gen:.3f} d={d:.3f} {tag} ({pts}pts)")

print(f"\nIndividual-level (var: {iv1:.1f}/{iv2:.1f}%):")
for it in ITEMS:
    dim = 'trad' if ITEMS.index(it) < 5 else 'surv'
    gt = GT_I[it]
    gen = il[il['item']==it][dim].values[0]
    d = abs(gt - gen)
    tag = "MATCH" if d <= 0.03 else "PARTIAL" if d <= 0.06 else "MISS"
    pts = 2 if d <= 0.03 else 1 if d <= 0.06 else 0
    print(f"  {it:<20}: paper={gt:.2f} gen={gen:.3f} d={d:.3f} {tag} ({pts}pts)")

print("\nCountry list:")
for c in sorted(cm.index):
    print(f"  {c}: god={cm.loc[c,'god_important']:.2f} auto={cm.loc[c,'autonomy_idx']:.2f}")
