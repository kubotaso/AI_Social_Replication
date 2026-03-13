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

for v in ['racmost','busing','racdif1','racdif3','racdif4','educ','realinc','hompop','prestg80','age','sex','race','ethnic','fund','relig','region']:
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

# Correct scale: racmost, busing, racdif1, racdif3(flip=2), racdif4
df['r1'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
df['r2'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
df['r3'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
df['r4'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 2).astype(float), np.nan)
df['r5'] = np.where(df['racdif4'].notna(), (df['racdif4'] == 1).astype(float), np.nan)

coded = ['r1','r2','r3','r4','r5']
n_valid = pd.concat([df[c] for c in coded], axis=1).notna().sum(axis=1)

preds_base = ['education', 'income_pc', 'occ_prestige', 'female', 'age_var',
              'black', 'hispanic', 'other_race', 'conservative_protestant',
              'no_religion', 'southern']

# Test: NAs as 0, various thresholds
# NAs as 0 means: if a racism item is NA, code it as 0 (non-racist)
for min_req in [1, 2, 3, 4]:
    racism_na0 = sum(df[c].fillna(0) for c in coded)
    racism_na0[n_valid < min_req] = np.nan
    df['racism'] = racism_na0
    preds = ['racism'] + preds_base

    for dv, dv_label in [('dv1', 'M1'), ('dv2', 'M2')]:
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
        r_beta = res.params[1]
        r_p = res.pvalues[1]
        sig = '***' if r_p<0.001 else '**' if r_p<0.01 else '*' if r_p<0.05 else ''
        print(f'NA0 {min_req}+ {dv_label}: N={N}, racism={r_beta:.3f}{sig}, R2={res_raw.rsquared:.3f}, AdjR2={res_raw.rsquared_adj:.3f}')

print()

# Best bet: the paper likely requires all 5 items valid for those who were asked all 5
# But for those NOT asked racmost (different ballot), they're excluded
# This gives N=453/431 (too low)

# Alternative: maybe the data was originally extracted from a DIFFERENT GSS source
# that has more complete racmost data

# Let's try the approach that gives N closest to 644 with best Model 1 match
# require 3+ gives 660 (16 too high)
# require 4+ gives 634 (10 too low)
# We need 644 exactly

# What if we require 4+ items but use a slightly different other IV setup?
# E.g., no prestige requirement, or different income

# Try: replace income_pc with just realinc (no hompop division)
for inc_label, inc_col in [('income_pc', 'income_pc'), ('realinc', 'realinc')]:
    for min_req in [3, 4]:
        racism_na0 = sum(df[c].fillna(0) for c in coded)
        racism_na0[n_valid < min_req] = np.nan
        df['racism'] = racism_na0
        preds = ['racism', 'education', inc_col, 'occ_prestige', 'female', 'age_var',
                 'black', 'hispanic', 'other_race', 'conservative_protestant',
                 'no_religion', 'southern']
        for dv, dv_label in [('dv1', 'M1'), ('dv2', 'M2')]:
            mdf = df[[dv] + preds].dropna()
            N = len(mdf)
            print(f'{inc_label} {min_req}+ {dv_label}: N={N}')

# Try: what if the income variable is income91 (categorical)?
df['income91'] = pd.to_numeric(df['income91'], errors='coerce')
for inc_label, inc_col in [('income91', 'income91')]:
    for min_req in [3, 4]:
        racism_na0 = sum(df[c].fillna(0) for c in coded)
        racism_na0[n_valid < min_req] = np.nan
        df['racism'] = racism_na0
        preds = ['racism', 'education', inc_col, 'occ_prestige', 'female', 'age_var',
                 'black', 'hispanic', 'other_race', 'conservative_protestant',
                 'no_religion', 'southern']
        for dv, dv_label in [('dv1', 'M1'), ('dv2', 'M2')]:
            mdf = df[[dv] + preds].dropna()
            N = len(mdf)
            print(f'{inc_label} {min_req}+ {dv_label}: N={N}')
