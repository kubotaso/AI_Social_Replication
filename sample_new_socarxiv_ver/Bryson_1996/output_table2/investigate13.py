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

for v in ['racmost','busing','racdif1','racdif2','racdif3','educ','realinc','hompop','prestg80','age','sex','race','ethnic','fund','relig','region','income91']:
    df[v] = pd.to_numeric(df[v], errors='coerce')

# Racism: try ORIGINAL coding (racdif3=1=racist) with person-mean imputation
df['r1'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
df['r2'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
df['r3'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
df['r4'] = np.where(df['racdif2'].notna(), (df['racdif2'] == 2).astype(float), np.nan)
df['r5_orig'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 1).astype(float), np.nan)
df['r5_flip'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 2).astype(float), np.nan)

coded_orig = ['r1','r2','r3','r4','r5_orig']
coded_flip = ['r1','r2','r3','r4','r5_flip']

# IVs
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

# Try: income_pc rescaled (divide by 10000)
df['income_pc_10k'] = df['income_pc'] / 10000

# Try: log income
df['log_income_pc'] = np.log(df['income_pc'].clip(lower=1))

# For each coding, compute racism with 4+ valid, NAs as 0
for label, coded in [('ORIGINAL', coded_orig), ('FLIPPED', coded_flip)]:
    n_valid = pd.concat([df[c] for c in coded], axis=1).notna().sum(axis=1)

    # Person-mean imputation
    coded_df = pd.concat([df[c] for c in coded], axis=1)
    pmean = coded_df.mean(axis=1)
    for c in coded:
        coded_df[c] = coded_df[c].fillna(pmean)
    racism = coded_df.sum(axis=1)
    racism[n_valid < 4] = np.nan
    df[f'racism_{label}'] = racism

    # NAs-as-0 approach
    racism_na0 = sum(df[c].fillna(0) for c in coded)
    racism_na0[n_valid < 4] = np.nan
    df[f'racism_na0_{label}'] = racism_na0

# Test different combinations
print('Testing various configurations:')
print(f'{"Config":<60} {"M1_N":>5} {"M1_racism":>10} {"M1_R2":>7} | {"M2_N":>5} {"M2_racism":>10} {"M2_R2":>7}')

configs = [
    ('Flipped, PMI, income_pc', 'racism_FLIPPED', 'income_pc'),
    ('Flipped, NA0, income_pc', 'racism_na0_FLIPPED', 'income_pc'),
    ('Original, PMI, income_pc', 'racism_ORIGINAL', 'income_pc'),
    ('Original, NA0, income_pc', 'racism_na0_ORIGINAL', 'income_pc'),
    ('Flipped, PMI, realinc', 'racism_FLIPPED', 'realinc'),
    ('Flipped, NA0, realinc', 'racism_na0_FLIPPED', 'realinc'),
    ('Original, PMI, realinc', 'racism_ORIGINAL', 'realinc'),
    ('Original, NA0, realinc', 'racism_na0_ORIGINAL', 'realinc'),
    ('Flipped, PMI, income91', 'racism_FLIPPED', 'income91'),
    ('Flipped, NA0, income91', 'racism_na0_FLIPPED', 'income91'),
]

for config_label, racism_col, income_col in configs:
    preds = [racism_col, 'education', income_col, 'occ_prestige',
             'female', 'age_var', 'black', 'hispanic', 'other_race',
             'conservative_protestant', 'no_religion', 'southern']

    results = []
    for dv in ['dv1', 'dv2']:
        mdf = df[[dv] + preds].dropna()
        N = len(mdf)
        if N < 50:
            results.append((N, 0, 0))
            continue
        y = mdf[dv].values
        X = mdf[preds].values
        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        X_zc = sm.add_constant(X_z)
        res = sm.OLS(y_z, X_zc).fit()
        X_c = sm.add_constant(X)
        res_raw = sm.OLS(y, X_c).fit()
        results.append((N, res.params[1], res_raw.rsquared))

    m1 = results[0]
    m2 = results[1]
    print(f'{config_label:<60} {m1[0]:>5} {m1[1]:>10.3f} {m1[2]:>7.3f} | {m2[0]:>5} {m2[1]:>10.3f} {m2[2]:>7.3f}')
