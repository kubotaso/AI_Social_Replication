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

# Correct 5-item scale: racmost, busing, racdif1, racdif3(flip), racdif4
df['r1'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
df['r2'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
df['r3'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
df['r4'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 2).astype(float), np.nan)  # FLIPPED
df['r5'] = np.where(df['racdif4'].notna(), (df['racdif4'] == 1).astype(float), np.nan)

coded = ['r1','r2','r3','r4','r5']
n_valid = pd.concat([df[c] for c in coded], axis=1).notna().sum(axis=1)

preds_base = ['education', 'income_pc', 'occ_prestige', 'female', 'age_var',
              'black', 'hispanic', 'other_race', 'conservative_protestant',
              'no_religion', 'southern']

# Test different minimum valid requirements
for min_req in [3, 4, 5]:
    # NAs as 0
    racism_na0 = sum(df[c].fillna(0) for c in coded)
    racism_na0[n_valid < min_req] = np.nan
    df['racism'] = racism_na0

    preds = ['racism'] + preds_base
    print(f'\n=== NAs as 0, require {min_req}+ items ===')
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
        sig = '***' if racism_p < 0.001 else '**' if racism_p < 0.01 else '*' if racism_p < 0.05 else ''
        r2 = res_raw.rsquared
        adj_r2 = res_raw.rsquared_adj
        const = res_raw.params[0]
        education_beta = res.params[2]
        black_beta = res.params[7]
        age_beta = res.params[6]
        print(f'  {dv_label}: N={N}, racism={racism_beta:.3f}{sig} (p={racism_p:.4f}), R2={r2:.3f}, AdjR2={adj_r2:.3f}')
        print(f'    const={const:.3f}, educ={education_beta:.3f}, black={black_beta:.3f}, age={age_beta:.3f}')
        print(f'    racism mean={mdf["racism"].mean():.2f}, SD={mdf["racism"].std(ddof=1):.2f}')

    # Person-mean imputation
    coded_df = pd.concat([df[c] for c in coded], axis=1)
    pmean = coded_df.mean(axis=1)
    for c in coded:
        coded_df[c] = coded_df[c].fillna(pmean)
    racism_pmi = coded_df.sum(axis=1)
    racism_pmi[n_valid < min_req] = np.nan
    df['racism'] = racism_pmi

    print(f'\n=== PMI, require {min_req}+ items ===')
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
        sig = '***' if racism_p < 0.001 else '**' if racism_p < 0.01 else '*' if racism_p < 0.05 else ''
        r2 = res_raw.rsquared
        adj_r2 = res_raw.rsquared_adj
        const = res_raw.params[0]
        education_beta = res.params[2]
        black_beta = res.params[7]
        print(f'  {dv_label}: N={N}, racism={racism_beta:.3f}{sig} (p={racism_p:.4f}), R2={r2:.3f}, AdjR2={adj_r2:.3f}')
        print(f'    const={const:.3f}, educ={education_beta:.3f}, black={black_beta:.3f}')
