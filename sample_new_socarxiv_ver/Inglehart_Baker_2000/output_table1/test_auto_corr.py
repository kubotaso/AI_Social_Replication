#!/usr/bin/env python3
"""Test different autonomy constructions to find the best one."""
import sys, os, csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import clean_missing, get_latest_per_country, varimax

# Load data
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

all_v = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042',
         'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
df = clean_missing(df, [c for c in all_v if c in df.columns])

for c in ['A029', 'A030', 'A032', 'A034', 'A042']:
    if c in df.columns:
        df.loc[df[c] == 2, c] = 0

# God importance
df['god_important'] = np.nan
df.loc[df['_src'] == 'wvs', 'god_important'] = df.loc[df['_src'] == 'wvs', 'F063']
df.loc[df['_src'] == 'evs', 'god_important'] = df.loc[df['_src'] == 'evs', 'A006']

# Recode
df['F120'] = 11 - df['F120']
df['G006'] = 5 - df['G006']
df['E018'] = 4 - df['E018']
df['Y002'] = 4 - df['Y002']
df['F118'] = 11 - df['F118']

df_nation = get_latest_per_country(df)

# Test various autonomy constructions
other_items = ['god_important', 'F120', 'G006', 'E018']

def test_autonomy(name, auto_series):
    """Test how well an autonomy variant correlates with traditional items."""
    items = ['god_important', auto_series.name, 'F120', 'G006', 'E018',
             'Y002', 'A008', 'E025', 'F118', 'A165']

    cm = df_nation.groupby('COUNTRY_ALPHA')[other_items + ['Y002', 'A008', 'E025', 'F118', 'A165']].mean()
    cm[auto_series.name] = auto_series
    cm = cm.dropna(thresh=7)
    for c in items:
        if c in cm.columns:
            cm[c] = cm[c].fillna(cm[c].mean())

    # Do PCA+varimax
    data_matrix = cm[items]
    corr = data_matrix.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    loadings = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    loadings, _ = varimax(loadings)

    var_exp = (loadings ** 2).sum(axis=0) / len(items) * 100

    trad_idx = [items.index(x) for x in ['god_important', auto_series.name, 'F120', 'G006', 'E018']]
    surv_idx = [items.index(x) for x in ['Y002', 'A008', 'E025', 'F118', 'A165']]

    f1t = sum(abs(loadings[i, 0]) for i in trad_idx)
    f2t = sum(abs(loadings[i, 1]) for i in trad_idx)
    tc = 0 if f1t > f2t else 1
    sc = 1 - tc

    if np.mean([loadings[i, tc] for i in trad_idx]) < 0:
        loadings[:, tc] *= -1
    if np.mean([loadings[i, sc] for i in surv_idx]) < 0:
        loadings[:, sc] *= -1

    auto_loading = loadings[1, tc]  # autonomy is item 1
    god_loading = loadings[0, tc]

    print(f"\n{name}: N={len(cm)}")
    print(f"  Autonomy loading (trad): {auto_loading:.3f} (paper: 0.89)")
    print(f"  God loading (trad): {god_loading:.3f} (paper: 0.91)")
    print(f"  Var explained: {var_exp[tc]:.1f}% / {var_exp[sc]:.1f}%")
    print(f"  All trad loadings: ", end="")
    for i, it in enumerate(items[:5]):
        print(f"{it}={loadings[i,tc]:.2f}", end=" ")
    print()
    return auto_loading

# Construction 1: 4-item (obedience + faith - independence - determination)
# Only works for WVS countries
auto4 = df_nation.groupby('COUNTRY_ALPHA').apply(
    lambda g: g['A042'].mean() + g['A034'].mean() - g['A029'].mean() - g['A032'].mean()
    if g['A032'].notna().sum() > 10 else np.nan
)
auto4.name = 'autonomy_idx'
test_autonomy("4-item (WVS only, drop EVS missing)", auto4)

# Construction 2: 4-item with mean imputation of A032 for EVS
# Use overall mean of A032 from WVS
a032_mean = df_nation[df_nation['A032'].notna()].groupby('COUNTRY_ALPHA')['A032'].mean().mean()
print(f"\nA032 overall mean: {a032_mean:.3f}")

auto4_imp = df_nation.groupby('COUNTRY_ALPHA').apply(
    lambda g: g['A042'].mean() + g['A034'].mean() - g['A029'].mean() - (
        g['A032'].mean() if g['A032'].notna().sum() > 10 else a032_mean
    )
)
auto4_imp.name = 'autonomy_idx'
test_autonomy("4-item (A032 mean imputed for EVS)", auto4_imp)

# Construction 3: 3-item (obedience + faith - independence) for ALL
auto3 = df_nation.groupby('COUNTRY_ALPHA').apply(
    lambda g: g['A042'].mean() + g['A034'].mean() - g['A029'].mean()
)
auto3.name = 'autonomy_idx'
test_autonomy("3-item (obey + faith - indep) all countries", auto3)

# Construction 4: 2-item (obedience - independence)
auto2 = df_nation.groupby('COUNTRY_ALPHA').apply(
    lambda g: g['A042'].mean() - g['A029'].mean()
)
auto2.name = 'autonomy_idx'
test_autonomy("2-item (obey - indep) all countries", auto2)

# Construction 5: obedience only
auto1 = df_nation.groupby('COUNTRY_ALPHA')['A042'].mean()
auto1.name = 'autonomy_idx'
test_autonomy("1-item (obedience only)", auto1)

# Construction 6: 5-item (obey + faith - indep - determ - imagination) for WVS, 3-item rescaled for EVS
def make_auto5(g):
    if g['A032'].notna().sum() > 10 and g['A030'].notna().sum() > 10:
        return g['A042'].mean() + g['A034'].mean() - g['A029'].mean() - g['A032'].mean() - g['A030'].mean()
    else:
        # 3-item: rescale from [-1,2] to [-3,2]
        v3 = g['A042'].mean() + g['A034'].mean() - g['A029'].mean()
        return (v3 + 1) / 3 * 5 - 3

auto5 = df_nation.groupby('COUNTRY_ALPHA').apply(make_auto5)
auto5.name = 'autonomy_idx'
test_autonomy("5-item (with rescaled 3-item for EVS)", auto5)

# Construction 7: 4-item with A032 imputed using regression from WVS
# Fit regression: A032_mean ~ A029_mean + A034_mean + A042_mean using WVS countries
cm_wvs = df_nation[df_nation['A032'].notna()].groupby('COUNTRY_ALPHA')[['A029', 'A030', 'A032', 'A034', 'A042']].mean()
cm_wvs = cm_wvs.dropna()
X = cm_wvs[['A029', 'A034', 'A042']].values
y = cm_wvs['A032'].values
X_aug = np.column_stack([X, np.ones(len(X))])
coefs = np.linalg.lstsq(X_aug, y, rcond=None)[0]

def make_auto4_reg(g):
    if g['A032'].notna().sum() > 10:
        return g['A042'].mean() + g['A034'].mean() - g['A029'].mean() - g['A032'].mean()
    else:
        a032_pred = coefs[0]*g['A029'].mean() + coefs[1]*g['A034'].mean() + coefs[2]*g['A042'].mean() + coefs[3]
        return g['A042'].mean() + g['A034'].mean() - g['A029'].mean() - a032_pred

auto4_reg = df_nation.groupby('COUNTRY_ALPHA').apply(make_auto4_reg)
auto4_reg.name = 'autonomy_idx'
test_autonomy("4-item (A032 regression-imputed for EVS)", auto4_reg)

# Construction 8: Use nation-level mean of 4-item autonomy for EVS countries
# that have 3-item, rescale using the relationship between 3-item and 4-item means in WVS
wvs_cm3 = df_nation[df_nation['A032'].notna()].groupby('COUNTRY_ALPHA').apply(
    lambda g: g['A042'].mean() + g['A034'].mean() - g['A029'].mean()
)
wvs_cm4 = df_nation[df_nation['A032'].notna()].groupby('COUNTRY_ALPHA').apply(
    lambda g: g['A042'].mean() + g['A034'].mean() - g['A029'].mean() - g['A032'].mean()
)
# Fit linear: 4-item = a * 3-item + b
valid_both = pd.DataFrame({'v3': wvs_cm3, 'v4': wvs_cm4}).dropna()
X_lin = np.column_stack([valid_both['v3'].values, np.ones(len(valid_both))])
y_lin = valid_both['v4'].values
ab = np.linalg.lstsq(X_lin, y_lin, rcond=None)[0]
print(f"\n3-item to 4-item conversion: 4-item = {ab[0]:.3f} * 3-item + {ab[1]:.3f}")
print(f"  R2 = {1 - np.var(y_lin - X_lin @ ab)/np.var(y_lin):.3f}")

def make_auto_linear(g):
    if g['A032'].notna().sum() > 10:
        return g['A042'].mean() + g['A034'].mean() - g['A029'].mean() - g['A032'].mean()
    else:
        v3 = g['A042'].mean() + g['A034'].mean() - g['A029'].mean()
        return ab[0] * v3 + ab[1]

auto_lin = df_nation.groupby('COUNTRY_ALPHA').apply(make_auto_linear)
auto_lin.name = 'autonomy_idx'
test_autonomy("4-item (linear conversion from 3-item for EVS)", auto_lin)
