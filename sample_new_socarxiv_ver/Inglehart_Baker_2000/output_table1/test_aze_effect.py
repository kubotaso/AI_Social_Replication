#!/usr/bin/env python3
"""Check what happens when AZE is excluded -- why does ordering improve?"""
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

child_vars = ['A042', 'A034', 'A029', 'A032', 'A030']
has5 = df[child_vars].notna().all(axis=1)
has3 = df[['A042', 'A034', 'A029']].notna().all(axis=1) & ~has5
df['autonomy_idx'] = np.nan
df.loc[has5, 'autonomy_idx'] = (
    df.loc[has5, 'A042'] + df.loc[has5, 'A034']
    - df.loc[has5, 'A029'] - df.loc[has5, 'A032'] - df.loc[has5, 'A030']
)
auto3 = df.loc[has3, 'A042'] + df.loc[has3, 'A034'] - df.loc[has3, 'A029']
df.loc[has3, 'autonomy_idx'] = (auto3 + 1) / 3 * 5 - 3

df['F120'] = 11 - df['F120']
df['G006'] = 5 - df['G006']
df['E018'] = 4 - df['E018']
df['Y002'] = 4 - df['Y002']
df['F118'] = 11 - df['F118']

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

# With AZE
print("=== WITH AZE ===")
cm = df.groupby('COUNTRY_ALPHA')[ITEMS].mean()
cm = cm.dropna(thresh=7)
for c in ITEMS: cm[c] = cm[c].fillna(cm[c].mean())
nl, nv1, nv2 = do_pca(cm, ITEMS)
print(f"Nation ordering:")
tv = [nl[i, 0] for i in range(5)]
sv = [nl[i, 1] for i in range(5, 10)]
for i, it in enumerate(ITEMS[:5]):
    print(f"  trad {it}: {tv[i]:.3f}")
for i, it in enumerate(ITEMS[5:]):
    print(f"  surv {it}: {sv[i]:.3f}")
print(f"  trad ok: {all(tv[j]>=tv[j+1] for j in range(4))}")
print(f"  surv ok: {all(sv[j]>=sv[j+1] for j in range(4))}")

# Without AZE
print("\n=== WITHOUT AZE ===")
df2 = df[df['COUNTRY_ALPHA'] != 'AZE']
cm2 = df2.groupby('COUNTRY_ALPHA')[ITEMS].mean()
cm2 = cm2.dropna(thresh=7)
for c in ITEMS: cm2[c] = cm2[c].fillna(cm2[c].mean())
nl2, nv1b, nv2b = do_pca(cm2, ITEMS)
tv2 = [nl2[i, 0] for i in range(5)]
sv2 = [nl2[i, 1] for i in range(5, 10)]
for i, it in enumerate(ITEMS[:5]):
    print(f"  trad {it}: {tv2[i]:.3f}")
for i, it in enumerate(ITEMS[5:]):
    print(f"  surv {it}: {sv2[i]:.3f}")
print(f"  trad ok: {all(tv2[j]>=tv2[j+1] for j in range(4))}")
print(f"  surv ok: {all(sv2[j]>=sv2[j+1] for j in range(4))}")

# Check AZE's values
print("\n=== AZE values ===")
aze = cm.loc['AZE']
print(aze)

# Check individual-level ordering with and without AZE
il, iv1, iv2 = do_pca(df[ITEMS], ITEMS)
il2, iv1b, iv2b = do_pca(df2[ITEMS], ITEMS)
print("\n=== Individual ordering WITH AZE ===")
sv_i = [il[i, 1] for i in range(5, 10)]
for i, it in enumerate(ITEMS[5:]):
    print(f"  {it}: {sv_i[i]:.3f}")
print(f"  surv ok: {all(sv_i[j]>=sv_i[j+1] for j in range(4))}")

print("\n=== Individual ordering WITHOUT AZE ===")
sv_i2 = [il2[i, 1] for i in range(5, 10)]
for i, it in enumerate(ITEMS[5:]):
    print(f"  {it}: {sv_i2[i]:.3f}")
print(f"  surv ok: {all(sv_i2[j]>=sv_i2[j+1] for j in range(4))}")
