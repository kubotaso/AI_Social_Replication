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

# IVs
df['education'] = pd.to_numeric(df['educ'], errors='coerce')
df['realinc'] = pd.to_numeric(df['realinc'], errors='coerce')
df['hompop'] = pd.to_numeric(df['hompop'], errors='coerce')
df['income_pc'] = df['realinc'] / df['hompop']
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

# Current best: old 5-item, min 4
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

pred_labels = ['Racism score', 'Education', 'Household income per cap',
               'Occupational prestige', 'Female', 'Age', 'Black', 'Hispanic',
               'Other race', 'Conservative Protestant', 'No religion', 'Southern']

predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']

# What if we use South Atlantic + East South Central + West South Central as "Southern"?
# GSS region codes: 1=NE, 2=Mid Atl, 3=East N Cent, 4=West N Cent, 5=South Atl, 6=East S Cent, 7=West S Cent, 8=Mountain, 9=Pacific
df['region'] = pd.to_numeric(df['region'], errors='coerce')
print("Region distribution:")
print(df['region'].value_counts().sort_index())

# Try different Southern definitions
df['southern_broad'] = df['region'].isin([5, 6, 7]).astype(int)
print(f"\nSouthern (region==3 only): {df['southern'].sum()}")
print(f"Southern (region 5,6,7): {df['southern_broad'].sum()}")

# What about prestg80 vs prestige?
for col in df.columns:
    if 'prest' in col.lower() or 'sei' in col.lower() or 'occ' in col.lower():
        print(f"\nPrestige column: {col}")
        vals = pd.to_numeric(df[col], errors='coerce')
        print(f"  Valid: {vals.notna().sum()}, Mean: {vals.mean():.1f}")

# What about using region==3 (E N Central) vs actual Southern states?
# Actually in 9-region GSS coding:
# region=5 is South Atlantic, region=6 is East South Central, region=7 is West South Central
# These are the Census South regions. region=3 is East North Central (Midwest)!
# The paper says "Southern" which should be regions 5,6,7 NOT region 3!
print("\n\nCRITICAL: GSS 9-region codes:")
print("1=New England, 2=Middle Atlantic, 3=East North Central,")
print("4=West North Central, 5=South Atlantic, 6=East South Central,")
print("7=West South Central, 8=Mountain, 9=Pacific")
print(f"\nregion==3 is EAST NORTH CENTRAL (Midwest), NOT Southern!")
print(f"True Southern (5,6,7) count: {df['southern_broad'].sum()}")

# Test with correct Southern coding
predictors_fixed_south = ['racism_score', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern_broad']

pred_labels_s = pred_labels.copy()

print("\n=== With CORRECT Southern coding (region 5,6,7) ===")
for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    model_df = df[[dv] + predictors_fixed_south].dropna()
    N = len(model_df)
    y = model_df[dv].values
    X = model_df[predictors_fixed_south].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
    model_raw = sm.OLS(y, sm.add_constant(X)).fit()
    betas = dict(zip(pred_labels_s, model.params[1:]))
    pvals = dict(zip(pred_labels_s, model.pvalues[1:]))
    print(f"  {dv_name}: N={N}, R2={model_raw.rsquared:.3f}, AdjR2={model_raw.rsquared_adj:.3f}")
    for lab in pred_labels_s:
        b = betas[lab]
        p = pvals[lab]
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"    {lab:<30} {b:7.3f} {sig:<4} (p={p:.4f})")

print("\n=== With OLD Southern coding (region==3, E N Central) ===")
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
    for lab in pred_labels:
        b = betas[lab]
        p = pvals[lab]
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"    {lab:<30} {b:7.3f} {sig:<4} (p={p:.4f})")
