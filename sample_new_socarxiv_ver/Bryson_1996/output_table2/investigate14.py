import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('gss1993_clean.csv')

minority_genres = ['rap', 'reggae', 'blues', 'jazz', 'gospel', 'latin']
remaining_genres = ['musicals', 'oldies', 'classicl', 'bigband', 'newage', 'opera',
                    'blugrass', 'folk', 'moodeasy', 'conrock', 'hvymetal', 'country']
for g in minority_genres + remaining_genres:
    df[g] = pd.to_numeric(df[g], errors='coerce')

minority_valid = df[minority_genres].isin([1,2,3,4,5]).all(axis=1)
remaining_valid = df[remaining_genres].isin([1,2,3,4,5]).all(axis=1)
df['dv1'] = np.nan
df.loc[minority_valid, 'dv1'] = (df.loc[minority_valid, minority_genres] >= 4).sum(axis=1)
df['dv2'] = np.nan
df.loc[remaining_valid, 'dv2'] = (df.loc[remaining_valid, remaining_genres] >= 4).sum(axis=1)

for v in ['racmost','busing','racdif1','racdif2','racdif3','educ','realinc','hompop','prestg80','age','sex','race','ethnic','fund','relig','region']:
    df[v] = pd.to_numeric(df[v], errors='coerce')

df['education'] = df['educ']
df['income_pc'] = df['realinc'] / df['hompop']
df['occ_prestige'] = df['prestg80']
df['female'] = (df['sex'] == 2).astype(int)
df['age_var'] = df['age']
df['black'] = (df['race'] == 2).astype(int)
df['hispanic'] = df['ethnic'].isin([17, 22, 25]).astype(int)
df['other_race'] = (df['race'] == 3).astype(int)
df['conservative_protestant'] = (df['fund'] == 1).astype(int)
df['no_religion'] = (df['relig'] == 4).astype(int)
df['southern'] = (df['region'] == 3).astype(int)

# Strategy: 4-item scale (drop racdif3 entirely)
# This gives a cleaner scale and might work better for both models
df['r1'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
df['r2'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
df['r3'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
df['r4'] = np.where(df['racdif2'].notna(), (df['racdif2'] == 2).astype(float), np.nan)

coded_4 = ['r1','r2','r3','r4']
n_valid_4 = pd.concat([df[c] for c in coded_4], axis=1).notna().sum(axis=1)

# Approach: 4 items, require 3+ valid, NAs as 0, scale to 0-5 range
racism_4_na0 = sum(df[c].fillna(0) for c in coded_4)
# Scale to 0-5: multiply by 5/4
racism_4_scaled = racism_4_na0 * (5.0/4.0)
racism_4_scaled[n_valid_4 < 3] = np.nan
df['racism_4_scaled'] = racism_4_scaled

# Also try: 4 items, strict (all 4 required), unscaled (range 0-4)
racism_4_strict = pd.concat([df[c] for c in coded_4], axis=1).sum(axis=1)
all4_valid = pd.concat([df[c] for c in coded_4], axis=1).notna().all(axis=1)
racism_4_strict[~all4_valid] = np.nan
df['racism_4_strict'] = racism_4_strict

# And: 4 items, strict, scaled to 0-5
df['racism_4_strict_scaled'] = racism_4_strict * (5.0/4.0)

preds_base = ['education', 'income_pc', 'occ_prestige',
         'female', 'age_var', 'black', 'hispanic', 'other_race',
         'conservative_protestant', 'no_religion', 'southern']

configs = {
    '4-item NA0 scaled': ['racism_4_scaled'] + preds_base,
    '4-item strict': ['racism_4_strict'] + preds_base,
    '4-item strict scaled': ['racism_4_strict_scaled'] + preds_base,
}

for label, preds in configs.items():
    print(f'\n=== {label} ===')
    for dv, dv_label in [('dv1', 'Model 1'), ('dv2', 'Model 2')]:
        mdf = df[[dv] + preds].dropna()
        N = len(mdf)
        y = mdf[dv].values
        X = mdf[preds].values
        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        X_zc = sm.add_constant(X_z)
        res = sm.OLS(y_z, X_zc).fit()
        X_c = sm.add_constant(X)
        res_raw = sm.OLS(y, X_c).fit()
        racism_beta = res.params[1]
        racism_p = res.pvalues[1]
        r2 = res_raw.rsquared
        const = res_raw.params[0]
        racism_mean = mdf[preds[0]].mean()
        racism_sd = mdf[preds[0]].std(ddof=1)
        sig = '***' if racism_p < 0.001 else '**' if racism_p < 0.01 else '*' if racism_p < 0.05 else ''
        print(f'  {dv_label}: N={N}, racism={racism_beta:.3f}{sig}, R2={r2:.3f}, const={const:.3f}')
        print(f'    racism stats: mean={racism_mean:.2f}, SD={racism_sd:.2f}')

# Also try: 5 items with racdif3 = original direction, but drop racdif2
# (racdif2 has low variance and negative corrs with others)
df['r5_orig'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 1).astype(float), np.nan)
coded_no_racdif2 = ['r1','r2','r3','r5_orig']
n_valid_no2 = pd.concat([df[c] for c in coded_no_racdif2], axis=1).notna().sum(axis=1)
all4_no2 = pd.concat([df[c] for c in coded_no_racdif2], axis=1).notna().all(axis=1)

racism_no2 = pd.concat([df[c] for c in coded_no_racdif2], axis=1).sum(axis=1)
racism_no2[~all4_no2] = np.nan
# Scale to 0-5
df['racism_no_racdif2'] = racism_no2 * (5.0/4.0)

preds_no2 = ['racism_no_racdif2'] + preds_base
print('\n=== 4-item: racmost+busing+racdif1+racdif3(orig), strict, scaled 0-5 ===')
for dv, dv_label in [('dv1', 'Model 1'), ('dv2', 'Model 2')]:
    mdf = df[[dv] + preds_no2].dropna()
    N = len(mdf)
    y = mdf[dv].values
    X = mdf[preds_no2].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    X_zc = sm.add_constant(X_z)
    res = sm.OLS(y_z, X_zc).fit()
    X_c = sm.add_constant(X)
    res_raw = sm.OLS(y, X_c).fit()
    racism_beta = res.params[1]
    racism_p = res.pvalues[1]
    r2 = res_raw.rsquared
    sig = '***' if racism_p < 0.001 else '**' if racism_p < 0.01 else '*' if racism_p < 0.05 else ''
    print(f'  {dv_label}: N={N}, racism={racism_beta:.3f}{sig}, R2={r2:.3f}')
    print(f'    racism stats: mean={mdf[preds_no2[0]].mean():.2f}, SD={mdf[preds_no2[0]].std(ddof=1):.2f}')

# Try: drop racdif2, keep racdif3 flipped (racmost+busing+racdif1+racdif3_flip)
df['r5_flip'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 2).astype(float), np.nan)
coded_no2_flip = ['r1','r2','r3','r5_flip']
all4_no2_flip = pd.concat([df[c] for c in coded_no2_flip], axis=1).notna().all(axis=1)
racism_no2_flip = pd.concat([df[c] for c in coded_no2_flip], axis=1).sum(axis=1)
racism_no2_flip[~all4_no2_flip] = np.nan
df['racism_no2_flip'] = racism_no2_flip * (5.0/4.0)

preds_no2_flip = ['racism_no2_flip'] + preds_base
print('\n=== 4-item: racmost+busing+racdif1+racdif3(flip), strict, scaled 0-5 ===')
for dv, dv_label in [('dv1', 'Model 1'), ('dv2', 'Model 2')]:
    mdf = df[[dv] + preds_no2_flip].dropna()
    N = len(mdf)
    y = mdf[dv].values
    X = mdf[preds_no2_flip].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    X_zc = sm.add_constant(X_z)
    res = sm.OLS(y_z, X_zc).fit()
    X_c = sm.add_constant(X)
    res_raw = sm.OLS(y, X_c).fit()
    racism_beta = res.params[1]
    racism_p = res.pvalues[1]
    r2 = res_raw.rsquared
    sig = '***' if racism_p < 0.001 else '**' if racism_p < 0.01 else '*' if racism_p < 0.05 else ''
    print(f'  {dv_label}: N={N}, racism={racism_beta:.3f}{sig}, R2={r2:.3f}')
    print(f'    racism stats: mean={mdf[preds_no2_flip[0]].mean():.2f}, SD={mdf[preds_no2_flip[0]].std(ddof=1):.2f}')
