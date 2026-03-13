#!/usr/bin/env python3
"""Test: use pairwise correlation for individual-level PCA."""
import sys, os, csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import clean_missing, get_latest_per_country, varimax

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WVS_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE, "data/EVS_1990_wvs_format.csv")

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

ITEMS = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
         'Y002', 'A008', 'E025', 'F118', 'A165']

all_v = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042',
         'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
df = clean_missing(df, [c for c in all_v if c in df.columns])
for c in ['A029', 'A030', 'A032', 'A034', 'A042']:
    if c in df.columns:
        df.loc[df[c] == 2, c] = 0

df['god_important'] = np.nan
df.loc[df['_src'] == 'wvs', 'god_important'] = df.loc[df['_src'] == 'wvs', 'F063']
df.loc[df['_src'] == 'evs', 'god_important'] = df.loc[df['_src'] == 'evs', 'A006']

# 3-item autonomy for all
df['autonomy_idx'] = df['A042'] + df['A034'] - df['A029']

# Recode
df['F120'] = 11 - df['F120']
df['G006'] = 5 - df['G006']
df['E018'] = 4 - df['E018']
df['Y002'] = 4 - df['Y002']
df['F118'] = 11 - df['F118']

# Individual level: pairwise correlation
data_i = df[ITEMS].copy()
print(f"Individual N (any item): {data_i.dropna(how='all').shape[0]:,}")
print(f"Individual N per item:")
for c in ITEMS:
    print(f"  {c}: {data_i[c].dropna().shape[0]:,}")

# Default .corr() uses pairwise complete
corr_pairwise = data_i.corr().values
print(f"\nPairwise correlation matrix:")
print(pd.DataFrame(corr_pairwise, index=ITEMS, columns=ITEMS).round(3))

# Compare with listwise deletion
data_complete = data_i.dropna()
print(f"\nListwise N: {len(data_complete):,}")
corr_listwise = data_complete.corr().values

# PCA on pairwise
def do_pca(corr, items, label):
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

    print(f"\n{label}: Var {var_exp[tc]:.1f}/{var_exp[sc]:.1f}%")
    gt_n = {'god_important': 0.70, 'autonomy_idx': 0.61, 'F120': 0.61, 'G006': 0.60, 'E018': 0.51}
    gt_s = {'Y002': 0.59, 'A008': 0.58, 'E025': 0.59, 'F118': 0.54, 'A165': 0.44}
    for i, it in enumerate(items):
        dim = tc if i < 5 else sc
        gt = gt_n.get(it, gt_s.get(it, 0))
        d = loadings[i, dim] - gt
        match = "MATCH" if abs(d) <= 0.03 else "PARTIAL" if abs(d) <= 0.06 else "MISS"
        print(f"  {it:<20}: {loadings[i, dim]:.3f} (paper: {gt:.2f}, d={d:+.3f}) {match}")

do_pca(corr_pairwise, ITEMS, "Pairwise correlation")
do_pca(corr_listwise, ITEMS, "Listwise deletion")

# Also test: what N do we get? Paper says 165,594 total, 146,789 smallest
print(f"\nN comparison:")
print(f"  Paper total: 165,594  Our total: {data_i.dropna(how='all').shape[0]:,}")
print(f"  Paper smallest: 146,789  Our smallest: {min(data_i[c].count() for c in ITEMS):,}")
