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

for v in ['racmost','busing','racdif1','racdif2','racdif3','racdif4','educ','realinc','hompop','prestg80','age','sex','race','ethnic','fund','relig','region']:
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

# Try 6-item scale: all of racmost, busing, racdif1, racdif2, racdif3(flip), racdif4
df['r1'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
df['r2'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
df['r3'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
df['r4'] = np.where(df['racdif2'].notna(), (df['racdif2'] == 2).astype(float), np.nan)
df['r5'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 2).astype(float), np.nan)
df['r6'] = np.where(df['racdif4'].notna(), (df['racdif4'] == 1).astype(float), np.nan)

# Various scale configurations with PMI
configs = {
    '5-item orig+flip PMI 4+': (['r1','r2','r3','r4','r5'], 4),
    '5-item racdif4 NA0 4+': (['r1','r2','r3','r5','r6'], 4),
    '6-item all PMI 4+': (['r1','r2','r3','r4','r5','r6'], 4),
    '6-item all PMI 5+': (['r1','r2','r3','r4','r5','r6'], 5),
    '6-item all NA0 4+': (['r1','r2','r3','r4','r5','r6'], 4),
    '6-item all NA0 5+': (['r1','r2','r3','r4','r5','r6'], 5),
}

preds_base = ['education', 'income_pc', 'occ_prestige', 'female', 'age_var',
              'black', 'hispanic', 'other_race', 'conservative_protestant',
              'no_religion', 'southern']

for label, (coded, min_req) in configs.items():
    n_valid = pd.concat([df[c] for c in coded], axis=1).notna().sum(axis=1)

    if 'PMI' in label:
        coded_df = pd.concat([df[c] for c in coded], axis=1)
        pmean = coded_df.mean(axis=1)
        for c in coded:
            coded_df[c] = coded_df[c].fillna(pmean)
        racism = coded_df.sum(axis=1)
    else:
        racism = sum(df[c].fillna(0) for c in coded)

    racism[n_valid < min_req] = np.nan

    # Scale to 0-5 range if not already
    n_items = len(coded)
    if n_items != 5:
        racism = racism * (5.0 / n_items)

    df['racism'] = racism
    preds = ['racism'] + preds_base

    for dv, dv_label in [('dv1', 'M1'), ('dv2', 'M2')]:
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
        r_beta = res.params[1]
        r_p = res.pvalues[1]
        sig = '***' if r_p<0.001 else '**' if r_p<0.01 else '*' if r_p<0.05 else ''
        bl_beta = res.params[7]  # black
        bl_p = res.pvalues[7]
        bl_sig = '***' if bl_p<0.001 else '**' if bl_p<0.01 else '*' if bl_p<0.05 else ''
        print(f'{label:<30} {dv_label}: N={N:>4}, racism={r_beta:>7.3f}{sig:<3}, R2={res_raw.rsquared:.3f}, AdjR2={res_raw.rsquared_adj:.3f}, black={bl_beta:.3f}{bl_sig}')
