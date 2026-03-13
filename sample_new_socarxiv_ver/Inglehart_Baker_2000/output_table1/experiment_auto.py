#!/usr/bin/env python3
"""Test different autonomy constructions for Table 1."""
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
df['F120_r'] = 11 - df['F120']
df['G006_r'] = 5 - df['G006']
df['E018_r'] = 4 - df['E018']
df['Y002_r'] = 4 - df['Y002']
df['F118_r'] = 11 - df['F118']

ITEMS = ['god_important', 'autonomy_idx', 'F120_r', 'G006_r', 'E018_r', 'Y002_r', 'A008', 'E025', 'F118_r', 'A165']

def do_nation_pca(df2, items):
    df_n = get_latest_per_country(df2)
    cm = df_n.groupby('COUNTRY_ALPHA')[items].mean()
    cm = cm.dropna(thresh=7)
    for c in items: cm[c] = cm[c].fillna(cm[c].mean())
    corr = cm.corr().values
    evals, evecs = np.linalg.eigh(corr)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]; evecs = evecs[:, idx]
    loadings = evecs[:, :2] * np.sqrt(evals[:2])
    loadings, _ = varimax(loadings)
    var_exp = (loadings ** 2).sum(axis=0) / len(items) * 100
    ti = [items.index(x) for x in ['god_important', 'autonomy_idx', 'F120_r', 'G006_r', 'E018_r']]
    si = [items.index(x) for x in ['Y002_r', 'A008', 'E025', 'F118_r', 'A165']]
    f1t = sum(abs(loadings[i, 0]) for i in ti)
    f2t = sum(abs(loadings[i, 1]) for i in ti)
    tc = 0 if f1t > f2t else 1; sc = 1 - tc
    if np.mean([loadings[i, tc] for i in ti]) < 0: loadings[:, tc] *= -1
    if np.mean([loadings[i, sc] for i in si]) < 0: loadings[:, sc] *= -1
    result = {}
    for i, item in enumerate(items):
        result[item] = (loadings[i, tc], loadings[i, sc])
    return len(cm), result, var_exp[tc], var_exp[sc]

print(f"{'Method':45s} {'N':>3s} {'auto':>6s} {'god':>6s} {'F120':>6s} {'G006':>6s} {'E018':>6s} {'Y002':>6s} {'A008':>6s} {'E025':>6s} {'F118':>6s} {'A165':>6s} {'v1':>5s} {'v2':>5s}")
print("-" * 120)

def show(name, df2):
    n, r, v1, v2 = do_nation_pca(df2, ITEMS)
    vals = [r[item][0] if item in ['god_important','autonomy_idx','F120_r','G006_r','E018_r'] else r[item][1] for item in ITEMS]
    print(f"{name:45s} {n:3d} " + " ".join(f"{v:6.3f}" for v in vals) + f" {v1:5.1f} {v2:5.1f}")

# Paper values for reference
print(f"{'PAPER':45s} {'65':>3s}  0.910  0.890  0.820  0.820  0.720  0.860  0.810  0.800  0.780  0.560  44.0  26.0")
print()

# 1. 5-item rescaled (current best)
df2 = df.copy()
has5 = df2[['A042','A034','A029','A032','A030']].notna().all(axis=1)
has3 = df2[['A042','A034','A029']].notna().all(axis=1) & ~has5
df2['autonomy_idx'] = np.nan
df2.loc[has5,'autonomy_idx'] = df2.loc[has5,'A042']+df2.loc[has5,'A034']-df2.loc[has5,'A029']-df2.loc[has5,'A032']-df2.loc[has5,'A030']
auto3 = df2.loc[has3,'A042']+df2.loc[has3,'A034']-df2.loc[has3,'A029']
df2.loc[has3,'autonomy_idx'] = (auto3+1)/3*5-3
show("5-item rescale", df2)

# 2. 3-item for all
df2 = df.copy()
df2['autonomy_idx'] = df2['A042'] + df2['A034'] - df2['A029']
show("3-item all", df2)

# 3. 4-item mean impute
df2 = df.copy()
wvs_a032_mean = df2.loc[(df2['_src']=='wvs') & (df2['A032'].notna()), 'A032'].mean()
has4 = df2[['A042','A034','A029','A032']].notna().all(axis=1)
has3 = df2[['A042','A034','A029']].notna().all(axis=1) & ~has4
df2['autonomy_idx'] = np.nan
df2.loc[has4,'autonomy_idx'] = df2.loc[has4,'A042']+df2.loc[has4,'A034']-df2.loc[has4,'A029']-df2.loc[has4,'A032']
df2.loc[has3,'autonomy_idx'] = df2.loc[has3,'A042']+df2.loc[has3,'A034']-df2.loc[has3,'A029']-wvs_a032_mean
show("4-item mean impute A032", df2)

# 4. Obedience only
df2 = df.copy()
df2['autonomy_idx'] = df2['A042']
show("Obedience only", df2)

# 5. 3-to-4 regression
df2 = df.copy()
wvs_mask = df2['_src']=='wvs'
has4w = wvs_mask & df2[['A042','A034','A029','A032']].notna().all(axis=1)
auto4w = df2.loc[has4w,'A042']+df2.loc[has4w,'A034']-df2.loc[has4w,'A029']-df2.loc[has4w,'A032']
auto3w = df2.loc[has4w,'A042']+df2.loc[has4w,'A034']-df2.loc[has4w,'A029']
slope = auto3w.cov(auto4w)/auto3w.var()
intercept = auto4w.mean() - slope*auto3w.mean()
auto3_all = df2['A042']+df2['A034']-df2['A029']
df2['autonomy_idx'] = slope*auto3_all + intercept
has4 = df2[['A042','A034','A029','A032']].notna().all(axis=1)
df2.loc[has4,'autonomy_idx'] = df2.loc[has4,'A042']+df2.loc[has4,'A034']-df2.loc[has4,'A029']-df2.loc[has4,'A032']
show("3-to-4 regression", df2)

# 6. Independence doubled as proxy for determination
df2 = df.copy()
has4 = df2[['A042','A034','A029','A032']].notna().all(axis=1)
has3 = df2[['A042','A034','A029']].notna().all(axis=1) & ~has4
df2['autonomy_idx'] = np.nan
df2.loc[has4,'autonomy_idx'] = df2.loc[has4,'A042']+df2.loc[has4,'A034']-df2.loc[has4,'A029']-df2.loc[has4,'A032']
df2.loc[has3,'autonomy_idx'] = df2.loc[has3,'A042']+df2.loc[has3,'A034']-2*df2.loc[has3,'A029']
show("4-item A029 doubled", df2)

# 7. Faith + obedience as % (binary sum)
df2 = df.copy()
df2['autonomy_idx'] = (df2['A042'] + df2['A034']) / 2
show("(obed+faith)/2", df2)

# 8. (obed+faith)/(obed+faith+indep+det) ratio - percentage based
df2 = df.copy()
has4 = df2[['A042','A034','A029','A032']].notna().all(axis=1)
has3 = df2[['A042','A034','A029']].notna().all(axis=1) & ~has4
df2['autonomy_idx'] = np.nan
denom4 = df2.loc[has4,'A042']+df2.loc[has4,'A034']+df2.loc[has4,'A029']+df2.loc[has4,'A032']
denom4 = denom4.replace(0, np.nan)
df2.loc[has4,'autonomy_idx'] = (df2.loc[has4,'A042']+df2.loc[has4,'A034']) / denom4
denom3 = df2.loc[has3,'A042']+df2.loc[has3,'A034']+df2.loc[has3,'A029']
denom3 = denom3.replace(0, np.nan)
df2.loc[has3,'autonomy_idx'] = (df2.loc[has3,'A042']+df2.loc[has3,'A034']) / denom3
show("Ratio (obed+faith)/total", df2)

# 9. Only WVS data - no EVS
df2 = df[df['_src']=='wvs'].copy()
has4 = df2[['A042','A034','A029','A032']].notna().all(axis=1)
df2['autonomy_idx'] = np.nan
df2.loc[has4,'autonomy_idx'] = df2.loc[has4,'A042']+df2.loc[has4,'A034']-df2.loc[has4,'A029']-df2.loc[has4,'A032']
show("WVS only, 4-item", df2)

# 10. Use wave 3 only (not waves 2+3)
wvs3 = pd.read_csv(os.path.join(BASE, 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'), usecols=avail, low_memory=False)
wvs3 = wvs3[wvs3['S002VS'] == 3]
wvs3['_src'] = 'wvs'
df3 = pd.concat([wvs3, evs], ignore_index=True, sort=False)
df3 = df3[~df3['COUNTRY_ALPHA'].isin(['MNE'])]
df3 = clean_missing(df3, [c for c in all_v if c in df3.columns])
for c in ['A029', 'A030', 'A032', 'A034', 'A042']:
    if c in df3.columns:
        df3.loc[df3[c] == 2, c] = 0
df3['god_important'] = np.nan
df3.loc[df3['_src'] == 'wvs', 'god_important'] = df3.loc[df3['_src'] == 'wvs', 'F063']
df3.loc[df3['_src'] == 'evs', 'god_important'] = df3.loc[df3['_src'] == 'evs', 'A006']
df3['F120_r'] = 11 - df3['F120']
df3['G006_r'] = 5 - df3['G006']
df3['E018_r'] = 4 - df3['E018']
df3['Y002_r'] = 4 - df3['Y002']
df3['F118_r'] = 11 - df3['F118']
has5 = df3[['A042','A034','A029','A032','A030']].notna().all(axis=1)
has3 = df3[['A042','A034','A029']].notna().all(axis=1) & ~has5
df3['autonomy_idx'] = np.nan
df3.loc[has5,'autonomy_idx'] = df3.loc[has5,'A042']+df3.loc[has5,'A034']-df3.loc[has5,'A029']-df3.loc[has5,'A032']-df3.loc[has5,'A030']
auto3 = df3.loc[has3,'A042']+df3.loc[has3,'A034']-df3.loc[has3,'A029']
df3.loc[has3,'autonomy_idx'] = (auto3+1)/3*5-3
show("Wave 3 only + EVS, 5-item", df3)
