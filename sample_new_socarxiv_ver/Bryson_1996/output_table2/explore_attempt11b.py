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

# Common IVs
df['education'] = pd.to_numeric(df['educ'], errors='coerce')
df['realinc'] = pd.to_numeric(df['realinc'], errors='coerce')
df['hompop'] = pd.to_numeric(df['hompop'], errors='coerce')
df['income_pc'] = df['realinc'] / df['hompop']
df['income91'] = pd.to_numeric(df['income91'], errors='coerce')
df['occ_prestige'] = pd.to_numeric(df['prestg80'], errors='coerce')
df['female'] = (pd.to_numeric(df['sex'], errors='coerce') == 2).astype(int)
df['age_var'] = pd.to_numeric(df['age'], errors='coerce')
df['race'] = pd.to_numeric(df['race'], errors='coerce')
df['black'] = (df['race'] == 2).astype(int)
df['ethnic'] = pd.to_numeric(df['ethnic'], errors='coerce')
df['hispanic'] = (df['ethnic'].isin([17, 22, 25]) & (df['race'] != 3)).astype(int)
df['other_race'] = ((df['race'] == 3) & ~df['ethnic'].isin([17, 22, 25])).astype(int)
df['conservative_protestant'] = (pd.to_numeric(df['fund'], errors='coerce') == 1).astype(int)
df['no_religion'] = (pd.to_numeric(df['relig'], errors='coerce') == 4).astype(int)
df['southern'] = (pd.to_numeric(df['region'], errors='coerce') == 3).astype(int)

# Racism items
for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())
df['r_racdif3'] = (df['racdif3'] == 2).astype(float).where(df['racdif3'].notna())
df['r_racdif4'] = (df['racdif4'] == 1).astype(float).where(df['racdif4'].notna())

pred_labels = ['Racism score', 'Education', 'Household income per cap',
               'Occupational prestige', 'Female', 'Age', 'Black', 'Hispanic',
               'Other race', 'Conservative Protestant', 'No religion', 'Southern']

# Test income91 / hompop vs realinc / hompop with old 5-item scale
coded = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3']
racism_vals = []
for idx in df.index:
    vals = [df.loc[idx, c] for c in coded]
    n_valid = sum(1 for v in vals if not np.isnan(v))
    if n_valid >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        racism_vals.append(sum(v if not np.isnan(v) else pm for v in vals))
    else:
        racism_vals.append(np.nan)
df['racism_score'] = racism_vals

# What does income91 look like?
print("income91 distribution:")
print(df['income91'].describe())
print(f"\nincome91 / hompop correlation with realinc/hompop: {df['income_pc'].corr(df['income91']/df['hompop']):.3f}")

# Test with income91/hompop
df['income_pc_91'] = df['income91'] / df['hompop']
print(f"\nrealinc/hompop valid: {df['income_pc'].notna().sum()}")
print(f"income91/hompop valid: {df['income_pc_91'].notna().sum()}")

for inc_var, inc_label in [('income_pc', 'realinc/hompop'), ('income_pc_91', 'income91/hompop')]:
    predictors = ['racism_score', 'education', inc_var, 'occ_prestige',
                  'female', 'age_var', 'black', 'hispanic', 'other_race',
                  'conservative_protestant', 'no_religion', 'southern']
    print(f"\n=== Income: {inc_label} ===")
    for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
        model_df = df[[dv] + predictors].dropna()
        N = len(model_df)
        y = model_df[dv].values
        X = model_df[predictors].values
        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
        model_raw = sm.OLS(y, sm.add_constant(X)).fit()
        betas = dict(zip(pred_labels, model.params[1:]))
        pvals = dict(zip(pred_labels, model.pvalues[1:]))
        print(f"  {dv_name}: N={N}, R2={model_raw.rsquared:.3f}, AdjR2={model_raw.rsquared_adj:.3f}")
        for lab in ['Racism score', 'Black', 'Age', 'Conservative Protestant']:
            b = betas[lab]
            p = pvals[lab]
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"    {lab:<30} {b:7.3f} {sig:<4} (p={p:.4f})")

# Also test: what if we use log(income_pc)?
df['log_income_pc'] = np.log(df['income_pc'].clip(lower=1))
predictors_log = ['racism_score', 'education', 'log_income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']
print(f"\n=== Income: log(realinc/hompop) ===")
for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    model_df = df[[dv] + predictors_log].dropna()
    N = len(model_df)
    y = model_df[dv].values
    X = model_df[predictors_log].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
    model_raw = sm.OLS(y, sm.add_constant(X)).fit()
    betas = dict(zip(pred_labels, model.params[1:]))
    pvals = dict(zip(pred_labels, model.pvalues[1:]))
    print(f"  {dv_name}: N={N}, R2={model_raw.rsquared:.3f}, AdjR2={model_raw.rsquared_adj:.3f}")
    for lab in ['Racism score', 'Black', 'Age', 'Conservative Protestant', 'Household income per cap']:
        b = betas[lab]
        p = pvals[lab]
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"    {lab:<30} {b:7.3f} {sig:<4} (p={p:.4f})")

# Test: what if we exclude age >= 89 (GSS top-code)?
print("\n\n=== AGE EXCLUSION TEST ===")
print(f"Respondents with age >= 89: {(df['age_var'] >= 89).sum()}")
df_noold = df[df['age_var'] < 89].copy()
predictors_std = ['racism_score', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']
print(f"\n=== Excluding age >= 89 ===")
for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    model_df = df_noold[[dv] + predictors_std].dropna()
    N = len(model_df)
    y = model_df[dv].values
    X = model_df[predictors_std].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
    model_raw = sm.OLS(y, sm.add_constant(X)).fit()
    betas = dict(zip(pred_labels, model.params[1:]))
    pvals = dict(zip(pred_labels, model.pvalues[1:]))
    print(f"  {dv_name}: N={N}, R2={model_raw.rsquared:.3f}, AdjR2={model_raw.rsquared_adj:.3f}")
    for lab in pred_labels:
        b = betas[lab]
        p = pvals[lab]
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"    {lab:<30} {b:7.3f} {sig:<4} (p={p:.4f})")
