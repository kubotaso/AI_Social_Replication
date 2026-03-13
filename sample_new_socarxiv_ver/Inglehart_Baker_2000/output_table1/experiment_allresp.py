#!/usr/bin/env python3
"""Test using all respondents (not just latest per country) for nation means."""
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
df['F120'] = 11 - df['F120']
df['G006'] = 5 - df['G006']
df['E018'] = 4 - df['E018']
df['Y002'] = 4 - df['Y002']
df['F118'] = 11 - df['F118']

# 5-item autonomy (current best)
has5 = df[['A042','A034','A029','A032','A030']].notna().all(axis=1)
has3 = df[['A042','A034','A029']].notna().all(axis=1) & ~has5
df['autonomy_idx'] = np.nan
df.loc[has5,'autonomy_idx'] = df.loc[has5,'A042']+df.loc[has5,'A034']-df.loc[has5,'A029']-df.loc[has5,'A032']-df.loc[has5,'A030']
auto3 = df.loc[has3,'A042']+df.loc[has3,'A034']-df.loc[has3,'A029']
df.loc[has3,'autonomy_idx'] = (auto3+1)/3*5-3

ITEMS = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
         'Y002', 'A008', 'E025', 'F118', 'A165']

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

# Method A: Latest per country (current approach)
df_latest = get_latest_per_country(df)
cm_latest = df_latest.groupby('COUNTRY_ALPHA')[ITEMS].mean()
cm_latest = cm_latest.dropna(thresh=7)
for c in ITEMS: cm_latest[c] = cm_latest[c].fillna(cm_latest[c].mean())

# Method B: All respondents
cm_all = df.groupby('COUNTRY_ALPHA')[ITEMS].mean()
cm_all = cm_all.dropna(thresh=7)
for c in ITEMS: cm_all[c] = cm_all[c].fillna(cm_all[c].mean())

# What countries differ?
countries_latest = set(cm_latest.index)
countries_all = set(cm_all.index)
print(f"Latest: {len(countries_latest)} countries")
print(f"All resp: {len(countries_all)} countries")
print(f"Diff: only in latest={countries_latest-countries_all}, only in all={countries_all-countries_latest}")

# Compare autonomy means
print("\nComparing autonomy_idx means:")
for c in sorted(countries_latest):
    if c in cm_all.index:
        v1 = cm_latest.loc[c, 'autonomy_idx']
        v2 = cm_all.loc[c, 'autonomy_idx']
        diff = v2 - v1
        if abs(diff) > 0.01:
            print(f"  {c}: latest={v1:.4f} all={v2:.4f} diff={diff:+.4f}")

# Run PCA for both
r_latest, v1l, v2l = do_pca_varimax(cm_latest, ITEMS)
r_all, v1a, v2a = do_pca_varimax(cm_all, ITEMS)

paper_vals = {'god_important': (0.91, 0.70), 'autonomy_idx': (0.89, 0.61),
              'F120': (0.82, 0.61), 'G006': (0.82, 0.60), 'E018': (0.72, 0.51),
              'Y002': (0.86, 0.59), 'A008': (0.81, 0.58), 'E025': (0.80, 0.59),
              'F118': (0.78, 0.54), 'A165': (0.56, 0.44)}

print(f"\n{'Item':20s} {'Paper':>6s} {'Latest':>7s} {'All':>7s}")
print("-" * 45)
for item in ITEMS:
    dim = 'trad_secrat' if item in ['god_important','autonomy_idx','F120','G006','E018'] else 'surv_selfexp'
    paper_v = paper_vals[item][0]
    latest_v = r_latest[r_latest['item']==item][dim].values[0]
    all_v = r_all[r_all['item']==item][dim].values[0]
    d_latest = abs(paper_v - latest_v)
    d_all = abs(paper_v - all_v)
    better = "<--" if d_all < d_latest else ""
    print(f"  {item:20s} {paper_v:6.2f} {latest_v:7.3f} {all_v:7.3f} {better}")
print(f"\nVariance Latest: {v1l:.1f}/{v2l:.1f}  All: {v1a:.1f}/{v2a:.1f}  Paper: 44/26")

# Now try: all respondents but different autonomy constructions
print("\n\n=== Testing autonomy with all respondents ===")

# 3-item for all
df2 = df.copy()
df2['autonomy_idx'] = df2['A042'] + df2['A034'] - df2['A029']
cm3 = df2.groupby('COUNTRY_ALPHA')[ITEMS].mean().dropna(thresh=7)
for c in ITEMS: cm3[c] = cm3[c].fillna(cm3[c].mean())
r3, v13, v23 = do_pca_varimax(cm3, ITEMS)
auto3_load = r3[r3['item']=='autonomy_idx']['trad_secrat'].values[0]
print(f"3-item all resp: auto={auto3_load:.3f}")

# 4-item mean impute
df2 = df.copy()
wvs_a032_mean = df2.loc[(df2['_src']=='wvs') & (df2['A032'].notna()), 'A032'].mean()
has4 = df2[['A042','A034','A029','A032']].notna().all(axis=1)
has3 = df2[['A042','A034','A029']].notna().all(axis=1) & ~has4
df2['autonomy_idx'] = np.nan
df2.loc[has4,'autonomy_idx'] = df2.loc[has4,'A042']+df2.loc[has4,'A034']-df2.loc[has4,'A029']-df2.loc[has4,'A032']
df2.loc[has3,'autonomy_idx'] = df2.loc[has3,'A042']+df2.loc[has3,'A034']-df2.loc[has3,'A029']-wvs_a032_mean
cm4 = df2.groupby('COUNTRY_ALPHA')[ITEMS].mean().dropna(thresh=7)
for c in ITEMS: cm4[c] = cm4[c].fillna(cm4[c].mean())
r4, v14, v24 = do_pca_varimax(cm4, ITEMS)
auto4_load = r4[r4['item']=='autonomy_idx']['trad_secrat'].values[0]
print(f"4-item mean impute all resp: auto={auto4_load:.3f}")

# Now check if using all respondents improves individual level too
print("\n=== Individual level comparison ===")

# Individual level with pairwise correlation
data_i = df[ITEMS].copy()
# Pairwise correlation (the paper says "pairwise deletion")
corr_i = data_i.corr().values  # pandas .corr() uses pairwise by default
evals_i, evecs_i = np.linalg.eigh(corr_i)
idx = np.argsort(evals_i)[::-1]
evals_i = evals_i[idx]; evecs_i = evecs_i[:, idx]
loadings_i = evecs_i[:, :2] * np.sqrt(evals_i[:2])
loadings_i, _ = varimax(loadings_i)
var_exp_i = (loadings_i ** 2).sum(axis=0) / len(ITEMS) * 100
ti = [ITEMS.index(x) for x in ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']]
si = [ITEMS.index(x) for x in ['Y002', 'A008', 'E025', 'F118', 'A165']]
f1t = sum(abs(loadings_i[i, 0]) for i in ti)
f2t = sum(abs(loadings_i[i, 1]) for i in ti)
tc = 0 if f1t > f2t else 1; sc = 1 - tc
if np.mean([loadings_i[i, tc] for i in ti]) < 0: loadings_i[:, tc] *= -1
if np.mean([loadings_i[i, sc] for i in si]) < 0: loadings_i[:, sc] *= -1

print(f"{'Item':20s} {'Paper':>6s} {'Indiv':>7s}")
for item in ITEMS:
    dim_col = tc if item in ['god_important','autonomy_idx','F120','G006','E018'] else sc
    paper_v = paper_vals[item][1]
    gen_v = loadings_i[ITEMS.index(item), dim_col]
    d = abs(paper_v - gen_v)
    tag = "MATCH" if d <= 0.03 else "PARTIAL" if d <= 0.06 else "MISS"
    print(f"  {item:20s} {paper_v:6.2f} {gen_v:7.3f} d={d:.3f} {tag}")
print(f"Variance: {var_exp_i[tc]:.1f}/{var_exp_i[sc]:.1f} (paper: 26/13)")
print(f"Total N: {len(data_i.dropna(how='all'))}, Min N: {data_i.count().min()}")
