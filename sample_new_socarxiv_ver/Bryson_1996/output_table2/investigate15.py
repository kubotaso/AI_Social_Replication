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

# The paper reports: range 0-5, mean=2.65, SD=1.56, alpha=0.54
# 4-item (racmost+busing+racdif1+racdif3_flip) scaled to 5: mean=2.45, SD=1.52, alpha=0.48
# This is the closest match to the paper's descriptive stats!

# Full 5-item coding options to try:
# Key idea: maybe the correct coding is:
# racmost=1 (object to school, racist)
# busing=2 (oppose busing, racist)
# racdif1=2 (NOT due to discrimination, racist)
# racdif2 - EXCLUDED (factor analysis removed it)
# racdif3=2 (in our dataset, this means YES lack motivation, racist)
# Plus a 5th item from outside the standard set?

# Actually - what if the paper has 5 items but one is racdif4?
# Paper says: "factor analysis suggested the removal of one item"
# Started with 6 items, removed 1, left with 5
# Original 6: racmost, busing, racdif1, racdif2, racdif3, racdif4
# Removed racdif2 (low variance, p=0.870, negative correlations)
# Remaining 5: racmost, busing, racdif1, racdif3, racdif4

df['r1'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
df['r2'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
df['r3'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
df['r5'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 1).astype(float), np.nan)
df['r5f'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 2).astype(float), np.nan)
df['racdif4'] = pd.to_numeric(df['racdif4'], errors='coerce')
df['r6'] = np.where(df['racdif4'].notna(), (df['racdif4'] == 1).astype(float), np.nan)

preds_base = ['education', 'income_pc', 'occ_prestige', 'female', 'age_var',
              'black', 'hispanic', 'other_race', 'conservative_protestant',
              'no_religion', 'southern']

configs = {
    # 5 items: racmost+busing+racdif1+racdif3(orig)+racdif4
    'racmost+busing+racdif1+racdif3+racdif4': ['r1','r2','r3','r5','r6'],
    # 5 items: racmost+busing+racdif1+racdif3(flip)+racdif4
    'racmost+busing+racdif1+racdif3f+racdif4': ['r1','r2','r3','r5f','r6'],
}

for label, coded in configs.items():
    n_valid = pd.concat([df[c] for c in coded], axis=1).notna().sum(axis=1)
    all_valid = pd.concat([df[c] for c in coded], axis=1).notna().all(axis=1)

    # Strict (all 5 valid)
    racism_strict = pd.concat([df[c] for c in coded], axis=1).sum(axis=1)
    racism_strict[~all_valid] = np.nan

    # Alpha
    valid_data = pd.concat([df[c] for c in coded], axis=1).loc[all_valid]
    k = 5
    iv = valid_data.var(ddof=1).sum()
    tv = valid_data.sum(axis=1).var(ddof=1)
    alpha = (k/(k-1)) * (1 - iv/tv)

    print(f'\n=== {label} (strict) ===')
    print(f'  All valid: {all_valid.sum()}, mean={racism_strict.dropna().mean():.2f}, SD={racism_strict.dropna().std(ddof=1):.2f}, alpha={alpha:.2f}')

    # With 4+ valid, NAs as 0
    racism_na0 = sum(df[c].fillna(0) for c in coded)
    racism_na0[n_valid < 4] = np.nan
    df['racism_test'] = racism_na0

    preds = ['racism_test'] + preds_base
    for dv, dv_label in [('dv1', 'Model 1'), ('dv2', 'Model 2')]:
        mdf = df[[dv] + preds].dropna()
        N = len(mdf)
        if N < 50:
            continue
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
        sig = '***' if racism_p < 0.001 else '**' if racism_p < 0.01 else '*' if racism_p < 0.05 else ''
        r2 = res_raw.rsquared
        adj_r2 = res_raw.rsquared_adj
        const = res_raw.params[0]
        print(f'  {dv_label} (4+valid, NA0): N={N}, racism={racism_beta:.3f}{sig}, R2={r2:.3f}, AdjR2={adj_r2:.3f}, const={const:.3f}')

    # Person-mean imputation, 4+ valid
    coded_df = pd.concat([df[c] for c in coded], axis=1)
    pmean = coded_df.mean(axis=1)
    for c in coded:
        coded_df[c] = coded_df[c].fillna(pmean)
    racism_pmi = coded_df.sum(axis=1)
    racism_pmi[n_valid < 4] = np.nan
    df['racism_test'] = racism_pmi

    for dv, dv_label in [('dv1', 'Model 1'), ('dv2', 'Model 2')]:
        mdf = df[[dv] + preds].dropna()
        N = len(mdf)
        if N < 50:
            continue
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
        sig = '***' if racism_p < 0.001 else '**' if racism_p < 0.01 else '*' if racism_p < 0.05 else ''
        r2 = res_raw.rsquared
        const = res_raw.params[0]
        print(f'  {dv_label} (PMI): N={N}, racism={racism_beta:.3f}{sig}, R2={r2:.3f}, const={const:.3f}')
