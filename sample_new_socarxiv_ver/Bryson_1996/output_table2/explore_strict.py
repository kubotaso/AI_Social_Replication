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

for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())
df['r_racdif3'] = (df['racdif3'] == 2).astype(float).where(df['racdif3'].notna())
df['r_racdif4'] = (df['racdif4'] == 1).astype(float).where(df['racdif4'].notna())

df['education'] = pd.to_numeric(df['educ'], errors='coerce')
df['realinc'] = pd.to_numeric(df['realinc'], errors='coerce')
df['hompop'] = pd.to_numeric(df['hompop'], errors='coerce')
df['income_pc'] = df['realinc'] / df['hompop']
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

predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']

pred_labels = ['Racism score', 'Education', 'Household income per cap',
               'Occupational prestige', 'Female', 'Age', 'Black', 'Hispanic',
               'Other race', 'Conservative Protestant', 'No religion', 'Southern']

# Correct 5-item, strict (require all 5)
coded_correct5 = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif3', 'r_racdif4']
df['racism_score'] = df[coded_correct5].sum(axis=1, min_count=5)
print("=== Correct 5-item, strict (all 5 required) ===")
for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    model_df = df[[dv] + predictors].dropna()
    N = len(model_df)
    y = model_df[dv].values
    X = model_df[predictors].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
    X_c = sm.add_constant(X)
    model_raw = sm.OLS(y, X_c).fit()
    betas = dict(zip(pred_labels, model.params[1:]))
    pvals = dict(zip(pred_labels, model.pvalues[1:]))
    print(f"  {dv_name}: N={N}, R2={model_raw.rsquared:.3f}")
    for label in pred_labels:
        b = betas[label]
        p = pvals[label]
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"    {label:<30} {b:7.3f} {sig}")

# What if we use old 5-item but with only 3 items required (more permissive)?
print("\n=== Old 5-item (racdif2 included), min 3, person-mean imputation ===")
coded_old5 = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3']
racism_vals = []
for idx in df.index:
    vals = [df.loc[idx, c] for c in coded_old5]
    n_v = sum(1 for v in vals if not np.isnan(v))
    if n_v >= 3:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        racism_vals.append(sum(v if not np.isnan(v) else pm for v in vals))
    else:
        racism_vals.append(np.nan)
df['racism_score'] = racism_vals
for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    model_df = df[[dv] + predictors].dropna()
    N = len(model_df)
    y = model_df[dv].values
    X = model_df[predictors].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
    X_c = sm.add_constant(X)
    model_raw = sm.OLS(y, X_c).fit()
    betas = dict(zip(pred_labels, model.params[1:]))
    pvals = dict(zip(pred_labels, model.pvalues[1:]))
    print(f"  {dv_name}: N={N}, R2={model_raw.rsquared:.3f}")
    for label in pred_labels:
        b = betas[label]
        p = pvals[label]
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"    {label:<30} {b:7.3f} {sig}")

# Try: what if we treat race==3 as "other" but also make Hispanic = race==3 AND ethnic in Hispanic codes?
# This would separate "Other race non-Hispanic" from "Hispanic (who could be any race)"
print("\n=== Hispanic-Other race overlap test ===")
df['hisp_strict'] = (df['ethnic'].isin([17, 22, 25]) & (df['race'] != 3)).astype(int)
df['other_strict'] = ((df['race'] == 3) & ~df['ethnic'].isin([17, 22, 25])).astype(int)
print(f"Hispanic (standard): {df['hispanic'].sum()}")
print(f"Hispanic (non-other race): {df['hisp_strict'].sum()}")
print(f"Other race (non-hispanic): {df['other_strict'].sum()}")
print(f"Hispanic AND other race: {((df['ethnic'].isin([17, 22, 25])) & (df['race'] == 3)).sum()}")

# Test with strict coding
racism_vals = []
for idx in df.index:
    vals = [df.loc[idx, c] for c in coded_old5]
    n_v = sum(1 for v in vals if not np.isnan(v))
    if n_v >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        racism_vals.append(sum(v if not np.isnan(v) else pm for v in vals))
    else:
        racism_vals.append(np.nan)
df['racism_score'] = racism_vals

predictors_strict = ['racism_score', 'education', 'income_pc', 'occ_prestige',
                     'female', 'age_var', 'black', 'hisp_strict', 'other_strict',
                     'conservative_protestant', 'no_religion', 'southern']

print("\n=== With strict Hispanic/Other race coding ===")
for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    model_df = df[[dv] + predictors_strict].dropna()
    N = len(model_df)
    y = model_df[dv].values
    X = model_df[predictors_strict].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
    X_c = sm.add_constant(X)
    model_raw = sm.OLS(y, X_c).fit()
    betas = dict(zip(pred_labels, model.params[1:]))
    pvals = dict(zip(pred_labels, model.pvalues[1:]))
    print(f"  {dv_name}: N={N}, R2={model_raw.rsquared:.3f}")
    for label in pred_labels:
        b = betas[label]
        p = pvals[label]
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"    {label:<30} {b:7.3f} {sig}")
