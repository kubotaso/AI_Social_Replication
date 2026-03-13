import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('gss1993_clean.csv')

minority_genres = ['rap', 'reggae', 'blues', 'jazz', 'gospel', 'latin']
remaining_genres = ['musicals', 'oldies', 'classicl', 'bigband', 'newage', 'opera',
                    'blugrass', 'folk', 'moodeasy', 'conrock', 'hvymetal', 'country']

for g in minority_genres + remaining_genres:
    df[g] = pd.to_numeric(df[g], errors='coerce')

# Standard IVs
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

# Racism
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
    n_valid = sum(1 for v in vals if not np.isnan(v))
    if n_valid >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        racism_vals.append(sum(v if not np.isnan(v) else pm for v in vals))
    else:
        racism_vals.append(np.nan)
df['racism_score'] = racism_vals

predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']
pred_labels = ['Racism score', 'Education', 'Household income per cap',
               'Occupational prestige', 'Female', 'Age', 'Black', 'Hispanic',
               'Other race', 'Conservative Protestant', 'No religion', 'Southern']

# Check ethnic distribution to see if we're missing Hispanic codes
print("Ethnic code distribution (top 20):")
print(df['ethnic'].value_counts().head(20))
print(f"\nHispanic ethnic codes in GSS:")
print("  17 = Mexican")
print("  22 = Puerto Rican")
print("  25 = Spanish")
print("  Also: 2 = Not applicable? 97 = American? etc.")

# Check if there are other Hispanic codes
print(f"\nHispanic (ethnic in 17,22,25 & race!=3): {df['hispanic'].sum()}")
# What about adding code 38 (Other Spanish)?
print(f"ethnic==38: {(df['ethnic']==38).sum()}")
# What about hispanic variable if it exists?
if 'hispanic' in df.columns:
    print(f"GSS hispanic var: exists")
for col in df.columns:
    if 'hisp' in col.lower():
        print(f"  Found: {col}")

# Test different DV thresholds
print("\n\n=== DV THRESHOLD TEST ===")
for threshold, label in [(4, '>=4 (dislike+strongly)'), (5, '==5 (strongly dislike only)'), (3, '>=3 (neutral+)')]:
    if threshold == 3:
        # Count of NOT liked (< 4)
        pass

    minority_valid = df[minority_genres].isin([1,2,3,4,5]).all(axis=1)
    remaining_valid = df[remaining_genres].isin([1,2,3,4,5]).all(axis=1)

    if threshold <= 4:
        dv1 = np.nan * np.ones(len(df))
        dv1[minority_valid] = (df.loc[minority_valid, minority_genres] >= threshold).sum(axis=1)
        dv2 = np.nan * np.ones(len(df))
        dv2[remaining_valid] = (df.loc[remaining_valid, remaining_genres] >= threshold).sum(axis=1)
    else:
        dv1 = np.nan * np.ones(len(df))
        dv1[minority_valid] = (df.loc[minority_valid, minority_genres] == threshold).sum(axis=1)
        dv2 = np.nan * np.ones(len(df))
        dv2[remaining_valid] = (df.loc[remaining_valid, remaining_genres] == threshold).sum(axis=1)

    df['dv1_test'] = dv1
    df['dv2_test'] = dv2

    print(f"\n  Threshold: {label}")
    print(f"  DV1 mean: {df['dv1_test'].mean():.2f}, DV2 mean: {df['dv2_test'].mean():.2f}")

    for dv, dv_name in [('dv1_test', 'M1'), ('dv2_test', 'M2')]:
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
        print(f"    {dv_name}: N={N}, R2={model_raw.rsquared:.3f}")
        for lab in ['Racism score', 'Education', 'Age', 'Black']:
            print(f"      {lab}: {betas[lab]:.3f} (p={pvals[lab]:.4f})")

# Also test: what if "Other race" includes Hispanic people coded as race==3?
# i.e., overlapping coding but in a different direction
print("\n\n=== OTHER RACE CODING VARIANTS ===")
# Variant 1: Standard (overlapping)
df['hisp_std'] = df['ethnic'].isin([17, 22, 25]).astype(int)
df['other_std'] = (df['race'] == 3).astype(int)
print(f"Standard (overlapping): Hispanic={df['hisp_std'].sum()}, Other={df['other_std'].sum()}")

# Variant 2: Non-overlapping (current best)
print(f"Non-overlapping: Hispanic={df['hispanic'].sum()}, Other={df['other_race'].sum()}")

# Variant 3: Hispanic includes all race==3 who are also ethnic Hispanic
df['hisp_v3'] = df['ethnic'].isin([17, 22, 25]).astype(int)
df['other_v3'] = ((df['race'] == 3) & ~df['ethnic'].isin([17, 22, 25])).astype(int)
print(f"Variant 3 (hisp=all ethnic, other=race3-non-hisp): Hispanic={df['hisp_v3'].sum()}, Other={df['other_v3'].sum()}")
