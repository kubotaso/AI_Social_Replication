#!/usr/bin/env python3
"""Test different factor extraction methods for Table 1."""
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
df = df[~df['COUNTRY_ALPHA'].isin(['MNE'])]
all_v = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042', 'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
df = clean_missing(df, [c for c in all_v if c in df.columns])
for c in ['A029', 'A030', 'A032', 'A034', 'A042']:
    if c in df.columns:
        df.loc[df[c] == 2, c] = 0
df['god_important'] = np.nan
df.loc[df['_src'] == 'wvs', 'god_important'] = df.loc[df['_src'] == 'wvs', 'F063']
df.loc[df['_src'] == 'evs', 'god_important'] = df.loc[df['_src'] == 'evs', 'A006']

# Recode
df['F120'] = 11 - df['F120']
df['G006'] = 5 - df['G006']
df['E018'] = 4 - df['E018']
df['Y002'] = 4 - df['Y002']
df['F118'] = 11 - df['F118']

# Use 5-item autonomy (best so far)
has5 = df[['A042','A034','A029','A032','A030']].notna().all(axis=1)
has3 = df[['A042','A034','A029']].notna().all(axis=1) & ~has5
df['autonomy_idx'] = np.nan
df.loc[has5,'autonomy_idx'] = df.loc[has5,'A042']+df.loc[has5,'A034']-df.loc[has5,'A029']-df.loc[has5,'A032']-df.loc[has5,'A030']
auto3 = df.loc[has3,'A042']+df.loc[has3,'A034']-df.loc[has3,'A029']
df.loc[has3,'autonomy_idx'] = (auto3+1)/3*5-3

ITEMS = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
         'Y002', 'A008', 'E025', 'F118', 'A165']

# Prepare nation-level data
df_nation = get_latest_per_country(df)
cm = df_nation.groupby('COUNTRY_ALPHA')[ITEMS].mean()
cm = cm.dropna(thresh=7)
for c in ITEMS: cm[c] = cm[c].fillna(cm[c].mean())
print(f"N countries: {len(cm)}")

def show_loadings(loadings, items, var1, var2, method_name):
    paper_trad = {'god_important': 0.91, 'autonomy_idx': 0.89, 'F120': 0.82, 'G006': 0.82, 'E018': 0.72}
    paper_surv = {'Y002': 0.86, 'A008': 0.81, 'E025': 0.80, 'F118': 0.78, 'A165': 0.56}

    trad_idx = [items.index(x) for x in ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']]
    surv_idx = [items.index(x) for x in ['Y002', 'A008', 'E025', 'F118', 'A165']]

    f1t = sum(abs(loadings[i, 0]) for i in trad_idx)
    f2t = sum(abs(loadings[i, 1]) for i in trad_idx)
    tc = 0 if f1t > f2t else 1; sc = 1 - tc

    if np.mean([loadings[i, tc] for i in trad_idx]) < 0: loadings[:, tc] *= -1
    if np.mean([loadings[i, sc] for i in surv_idx]) < 0: loadings[:, sc] *= -1

    print(f"\n--- {method_name} ---")
    print(f"Variance: {var1:.1f}% / {var2:.1f}% (paper: 44/26)")
    for i, item in enumerate(items):
        if item in paper_trad:
            paper_v = paper_trad[item]
            gen_v = loadings[i, tc]
        else:
            paper_v = paper_surv[item]
            gen_v = loadings[i, sc]
        d = abs(paper_v - gen_v)
        tag = "MATCH" if d <= 0.03 else "PARTIAL" if d <= 0.06 else "MISS"
        print(f"  {item:20s} paper={paper_v:.2f} gen={gen_v:.3f} d={d:.3f} {tag}")

# Method 1: PCA on correlation matrix (current approach)
print("\n" + "="*60)
corr = cm.corr().values
evals, evecs = np.linalg.eigh(corr)
idx = np.argsort(evals)[::-1]
evals = evals[idx]; evecs = evecs[:, idx]
loadings = evecs[:, :2] * np.sqrt(evals[:2])
loadings, _ = varimax(loadings)
var_exp = (loadings ** 2).sum(axis=0) / len(ITEMS) * 100
show_loadings(loadings, ITEMS, var_exp[0], var_exp[1], "PCA on correlation (current)")

# Method 2: PCA on covariance matrix
from sklearn.decomposition import PCA as skPCA
print("\n" + "="*60)
# Standardize data
cm_std = (cm - cm.mean()) / cm.std()
pca = skPCA(n_components=2)
scores = pca.fit_transform(cm_std.values)
loadings_raw = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_rot, _ = varimax(loadings_raw)
var_exp = (loadings_rot ** 2).sum(axis=0) / len(ITEMS) * 100
show_loadings(loadings_rot, ITEMS, var_exp[0], var_exp[1], "sklearn PCA + varimax")

# Method 3: Principal Axis Factoring (PAF)
# PAF uses communalities instead of 1s on the diagonal
print("\n" + "="*60)
corr_matrix = cm.corr().values.copy()
# PAF: iterate to estimate communalities
for iteration in range(100):
    old_diag = np.diag(corr_matrix).copy()
    evals_paf, evecs_paf = np.linalg.eigh(corr_matrix)
    idx = np.argsort(evals_paf)[::-1]
    evals_paf = evals_paf[idx]; evecs_paf = evecs_paf[:, idx]
    # Keep only positive eigenvalues
    pos = evals_paf > 0
    loadings_paf = evecs_paf[:, :2] * np.sqrt(np.maximum(evals_paf[:2], 0))
    communalities = (loadings_paf ** 2).sum(axis=1)
    np.fill_diagonal(corr_matrix, communalities)
    if np.max(np.abs(np.diag(corr_matrix) - old_diag)) < 1e-6:
        break
loadings_paf_rot, _ = varimax(loadings_paf)
var_exp_paf = (loadings_paf_rot ** 2).sum(axis=0) / len(ITEMS) * 100
show_loadings(loadings_paf_rot, ITEMS, var_exp_paf[0], var_exp_paf[1], "Principal Axis Factoring (PAF)")

# Method 4: Use polychoric-style approach - treat items as ordinal
# First, let's try PCA on Spearman correlations instead of Pearson
print("\n" + "="*60)
spearman_corr = cm.rank().corr().values
evals_sp, evecs_sp = np.linalg.eigh(spearman_corr)
idx = np.argsort(evals_sp)[::-1]
evals_sp = evals_sp[idx]; evecs_sp = evecs_sp[:, idx]
loadings_sp = evecs_sp[:, :2] * np.sqrt(evals_sp[:2])
loadings_sp_rot, _ = varimax(loadings_sp)
var_exp_sp = (loadings_sp_rot ** 2).sum(axis=0) / len(ITEMS) * 100
show_loadings(loadings_sp_rot, ITEMS, var_exp_sp[0], var_exp_sp[1], "PCA on Spearman correlations")

# Method 5: Try using ALL respondents in waves 2+3 (not just latest per country)
# for computing country means
print("\n" + "="*60)
cm_all = df.groupby('COUNTRY_ALPHA')[ITEMS].mean()
cm_all = cm_all.dropna(thresh=7)
for c in ITEMS: cm_all[c] = cm_all[c].fillna(cm_all[c].mean())
cm_all = cm_all[~cm_all.index.isin(['MNE'])]
corr_all = cm_all.corr().values
evals_a, evecs_a = np.linalg.eigh(corr_all)
idx = np.argsort(evals_a)[::-1]
evals_a = evals_a[idx]; evecs_a = evecs_a[:, idx]
loadings_a = evecs_a[:, :2] * np.sqrt(evals_a[:2])
loadings_a_rot, _ = varimax(loadings_a)
var_exp_a = (loadings_a_rot ** 2).sum(axis=0) / len(ITEMS) * 100
show_loadings(loadings_a_rot, ITEMS, var_exp_a[0], var_exp_a[1], f"PCA all respondents (N={len(cm_all)})")

# Let's also check: what does the correlation matrix look like?
print("\n\n=== Correlation matrix for nation-level data ===")
print(cm.corr().round(2).to_string())

# Check autonomy's correlation with other traditional items
print(f"\n\nAutonomy correlations with traditional items:")
for item in ['god_important', 'F120', 'G006', 'E018']:
    r = cm['autonomy_idx'].corr(cm[item])
    print(f"  autonomy vs {item}: {r:.3f}")
