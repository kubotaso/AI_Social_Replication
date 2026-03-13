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
df['dv_minority'] = np.nan
df.loc[minority_valid, 'dv_minority'] = (df.loc[minority_valid, minority_genres] >= 4).sum(axis=1)
df['dv_remaining'] = np.nan
df.loc[remaining_valid, 'dv_remaining'] = (df.loc[remaining_valid, remaining_genres] >= 4).sum(axis=1)

for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())
df['r_racdif3'] = (df['racdif3'] == 2).astype(float).where(df['racdif3'].notna())

coded = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3']

racism_vals = []
for idx in df.index:
    vals = [df.loc[idx, c] for c in coded]
    n_v = sum(1 for v in vals if not np.isnan(v))
    if n_v >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        racism_vals.append(sum(v if not np.isnan(v) else pm for v in vals))
    else:
        racism_vals.append(np.nan)
df['racism_score'] = racism_vals

df['education'] = pd.to_numeric(df['educ'], errors='coerce')
df['realinc'] = pd.to_numeric(df['realinc'], errors='coerce')
df['hompop'] = pd.to_numeric(df['hompop'], errors='coerce')
df['income91'] = pd.to_numeric(df['income91'], errors='coerce')
df['occ_prestige'] = pd.to_numeric(df['prestg80'], errors='coerce')
df['female'] = (pd.to_numeric(df['sex'], errors='coerce') == 2).astype(int)
df['age_var'] = pd.to_numeric(df['age'], errors='coerce')
df['black'] = (pd.to_numeric(df['race'], errors='coerce') == 2).astype(int)
df['ethnic'] = pd.to_numeric(df['ethnic'], errors='coerce')
df['hispanic'] = df['ethnic'].isin([17, 22, 25]).astype(int)
df['other_race'] = (pd.to_numeric(df['race'], errors='coerce') == 3).astype(int)
df['conservative_protestant'] = (pd.to_numeric(df['fund'], errors='coerce') == 1).astype(int)
df['no_religion'] = (pd.to_numeric(df['relig'], errors='coerce') == 4).astype(int)
df['southern'] = (pd.to_numeric(df['region'], errors='coerce') == 3).astype(int)

# Different income formulations
income_variants = {
    'realinc/hompop': df['realinc'] / df['hompop'],
    'income91/hompop': df['income91'] / df['hompop'],
    'income91 (no div)': df['income91'],
    'log(realinc/hompop)': np.log(df['realinc'] / df['hompop']).replace([np.inf, -np.inf], np.nan),
    'realinc (no div)': df['realinc'],
    'sqrt(realinc/hompop)': np.sqrt(df['realinc'] / df['hompop']),
}

base_preds = ['racism_score', 'education', 'INCOME', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']

pred_labels = ['Racism score', 'Education', 'Household income per cap',
               'Occupational prestige', 'Female', 'Age', 'Black', 'Hispanic',
               'Other race', 'Conservative Protestant', 'No religion', 'Southern']

for inc_name, inc_series in income_variants.items():
    df['income_test'] = inc_series
    preds = base_preds.copy()
    preds[preds.index('INCOME')] = 'income_test'

    for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
        model_df = df[[dv] + preds].dropna()
        N = len(model_df)

        y = model_df[dv].values
        X = model_df[preds].values
        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        model = sm.OLS(y_z, sm.add_constant(X_z)).fit()

        X_const = sm.add_constant(X)
        model_raw = sm.OLS(y, X_const).fit()

        betas = dict(zip(pred_labels, model.params[1:]))
        pvals = dict(zip(pred_labels, model.pvalues[1:]))

        racism_b = betas['Racism score']
        income_b = betas['Household income per cap']
        black_b = betas['Black']
        age_b = betas['Age']
        r2 = model_raw.rsquared
        adj_r2 = model_raw.rsquared_adj

        print(f"{inc_name:25s} {dv_name}: N={N:3d}, rac={racism_b:6.3f}, inc={income_b:6.3f}, blk={black_b:6.3f}, age={age_b:6.3f}, R2={r2:.3f}, adjR2={adj_r2:.3f}")
