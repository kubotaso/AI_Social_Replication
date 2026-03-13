#!/usr/bin/env python3
"""Test: what happens with different wave selections."""
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

ITEMS = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
         'Y002', 'A008', 'E025', 'F118', 'A165']

def prepare(df):
    all_v = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042',
             'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
    df = clean_missing(df, [c for c in all_v if c in df.columns])
    for c in ['A029', 'A030', 'A032', 'A034', 'A042']:
        if c in df.columns:
            df.loc[df[c] == 2, c] = 0
    df['god_important'] = np.nan
    df.loc[df['_src'] == 'wvs', 'god_important'] = df.loc[df['_src'] == 'wvs', 'F063']
    df.loc[df['_src'] == 'evs', 'god_important'] = df.loc[df['_src'] == 'evs', 'A006']
    # 3-item autonomy
    df['autonomy_idx'] = df['A042'] + df['A034'] - df['A029']
    # Recode
    df['F120'] = 11 - df['F120']
    df['G006'] = 5 - df['G006']
    df['E018'] = 4 - df['E018']
    df['Y002'] = 4 - df['Y002']
    df['F118'] = 11 - df['F118']
    return df

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

# Test 1: Wave 3 only + EVS
print("=== Test 1: Wave 3 only + EVS ===")
wvs3 = wvs[wvs['S002VS'] == 3].copy()
wvs3['_src'] = 'wvs'
evs = pd.read_csv(EVS_PATH)
evs['_src'] = 'evs'
df1 = pd.concat([wvs3, evs], ignore_index=True, sort=False)
df1 = df1[~df1['COUNTRY_ALPHA'].isin(['MNE'])]
df1 = prepare(df1)
df1n = get_latest_per_country(df1)
cm1 = df1n.groupby('COUNTRY_ALPHA')[ITEMS].mean()
cm1 = cm1.dropna(thresh=7)
for c in ITEMS: cm1[c] = cm1[c].fillna(cm1[c].mean())
loadings1, v1, v2 = do_pca_varimax(cm1, ITEMS)
print(f"N={len(cm1)}, Var: {v1:.1f}/{v2:.1f}%")
for i, it in enumerate(ITEMS):
    tc = 0 if i < 5 else 1
    print(f"  {it}: {loadings1[i, tc]:.3f}")
print(f"  Individual N: {df1[ITEMS].dropna(how='all').shape[0]:,}")

# Test 2: Wave 2+3, prefer wave 3 (current approach)
print("\n=== Test 2: Wave 2+3 + EVS (prefer latest) ===")
wvs23 = wvs[wvs['S002VS'].isin([2, 3])].copy()
wvs23['_src'] = 'wvs'
evs2 = pd.read_csv(EVS_PATH)
evs2['_src'] = 'evs'
df2 = pd.concat([wvs23, evs2], ignore_index=True, sort=False)
df2 = df2[~df2['COUNTRY_ALPHA'].isin(['MNE'])]
df2 = prepare(df2)
df2n = get_latest_per_country(df2)
cm2 = df2n.groupby('COUNTRY_ALPHA')[ITEMS].mean()
cm2 = cm2.dropna(thresh=7)
for c in ITEMS: cm2[c] = cm2[c].fillna(cm2[c].mean())
loadings2, v1b, v2b = do_pca_varimax(cm2, ITEMS)
print(f"N={len(cm2)}, Var: {v1b:.1f}/{v2b:.1f}%")
for i, it in enumerate(ITEMS):
    tc = 0 if i < 5 else 1
    print(f"  {it}: {loadings2[i, tc]:.3f}")
print(f"  Individual N: {df2[ITEMS].dropna(how='all').shape[0]:,}")

# Test 3: Use ALL data (wave 2+3 + EVS), not just latest wave per country
print("\n=== Test 3: Wave 2+3 + EVS (ALL observations, not just latest) ===")
# For nation level, use all observations from a country across waves
cm3 = df2.groupby('COUNTRY_ALPHA')[ITEMS].mean()
cm3 = cm3.dropna(thresh=7)
for c in ITEMS: cm3[c] = cm3[c].fillna(cm3[c].mean())
loadings3, v1c, v2c = do_pca_varimax(cm3, ITEMS)
print(f"N={len(cm3)}, Var: {v1c:.1f}/{v2c:.1f}%")
for i, it in enumerate(ITEMS):
    tc = 0 if i < 5 else 1
    print(f"  {it}: {loadings3[i, tc]:.3f}")

# Test 4: Use standardized items for PCA
print("\n=== Test 4: Standardized items (z-score) for nation-level PCA ===")
cm4 = cm2.copy()
for c in ITEMS:
    cm4[c] = (cm4[c] - cm4[c].mean()) / cm4[c].std()
loadings4, v1d, v2d = do_pca_varimax(cm4, ITEMS)
print(f"N={len(cm4)}, Var: {v1d:.1f}/{v2d:.1f}%")
for i, it in enumerate(ITEMS):
    tc = 0 if i < 5 else 1
    print(f"  {it}: {loadings4[i, tc]:.3f}")
# Note: z-scoring shouldn't matter for correlation-based PCA
