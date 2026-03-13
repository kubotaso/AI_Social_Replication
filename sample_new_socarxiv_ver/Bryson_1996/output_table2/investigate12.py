import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('gss1993_clean.csv')

# Setup all variables
minority_genres = ['rap', 'reggae', 'blues', 'jazz', 'gospel', 'latin']
remaining_genres = ['musicals', 'oldies', 'classicl', 'bigband', 'newage', 'opera',
                    'blugrass', 'folk', 'moodeasy', 'conrock', 'hvymetal', 'country']
for g in minority_genres + remaining_genres:
    df[g] = pd.to_numeric(df[g], errors='coerce')

minority_valid = df[minority_genres].isin([1,2,3,4,5]).all(axis=1)
remaining_valid = df[remaining_genres].isin([1,2,3,4,5]).all(axis=1)
df['dv_minority'] = np.nan
df.loc[minority_valid, 'dv_minority'] = (df.loc[minority_valid, minority_genres] >= 4).sum(axis=1)
df['dv_remaining'] = np.nan
df.loc[remaining_valid, 'dv_remaining'] = (df.loc[remaining_valid, remaining_genres] >= 4).sum(axis=1)

for v in ['racmost','busing','racdif1','racdif2','racdif3','educ','realinc','hompop','prestg80','age','sex','race','ethnic','fund','relig','region']:
    df[v] = pd.to_numeric(df[v], errors='coerce')

df['education'] = df['educ']
df['income_pc'] = df['realinc'] / df['hompop']
df['occ_prestige'] = df['prestg80']
df['female'] = (df['sex'] == 2).astype(int)
df['age_var'] = df['age']
df['black'] = (df['race'] == 2).astype(int)
df['hispanic'] = pd.to_numeric(df['ethnic'], errors='coerce').isin([17, 22, 25]).astype(int)
df['other_race'] = (df['race'] == 3).astype(int)
df['conservative_protestant'] = (df['fund'] == 1).astype(int)
df['no_religion'] = (df['relig'] == 4).astype(int)
df['southern'] = (df['region'] == 3).astype(int)

# Test approach: racdif3 coded as 1=racist (ORIGINAL), all 5 required
# BUT also try flipped version
for flip_label, racdif3_racist_val in [('Original (1=racist)', 1), ('Flipped (2=racist)', 2)]:
    df['r_racmost'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
    df['r_busing'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
    df['r_racdif1'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
    df['r_racdif2'] = np.where(df['racdif2'].notna(), (df['racdif2'] == 2).astype(float), np.nan)
    df['r_racdif3'] = np.where(df['racdif3'].notna(), (df['racdif3'] == racdif3_racist_val).astype(float), np.nan)

    coded = ['r_racmost','r_busing','r_racdif1','r_racdif2','r_racdif3']
    all5 = df[coded].notna().all(axis=1)
    racism = df.loc[all5, coded].sum(axis=1)
    df['racism_score'] = np.nan
    df.loc[all5, 'racism_score'] = racism.values

    predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
                  'female', 'age_var', 'black', 'hispanic', 'other_race',
                  'conservative_protestant', 'no_religion', 'southern']

    print(f'\n=== {flip_label}, All 5 required ===')
    for dv_name, label in [('dv_minority','Model 1'), ('dv_remaining','Model 2')]:
        model_df = df[[dv_name] + predictors].dropna()
        N = len(model_df)
        y = model_df[dv_name].values
        X = model_df[predictors].values

        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        X_z_c = sm.add_constant(X_z)
        res = sm.OLS(y_z, X_z_c).fit()

        X_c = sm.add_constant(X)
        res_raw = sm.OLS(y, X_c).fit()

        print(f'  {label}: N={N}, R2={res_raw.rsquared:.3f}, AdjR2={res_raw.rsquared_adj:.3f}')
        print(f'    Racism: beta={res.params[1]:.3f}, p={res.pvalues[1]:.4f}')
        print(f'    Education: beta={res.params[2]:.3f}')
        print(f'    Black: beta={res.params[7]:.3f}')
        print(f'    Constant: {res_raw.params[0]:.3f}')
        print(f'    Racism mean={model_df["racism_score"].mean():.2f}, SD={model_df["racism_score"].std(ddof=1):.2f}')
