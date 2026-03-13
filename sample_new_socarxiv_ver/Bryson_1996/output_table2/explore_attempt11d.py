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
for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

# Test with racdif3 coded as 1=racist (instruction_summary's original interpretation)
df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())

pred_labels = ['Racism score', 'Education', 'Household income per cap',
               'Occupational prestige', 'Female', 'Age', 'Black', 'Hispanic',
               'Other race', 'Conservative Protestant', 'No religion', 'Southern']
predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']

# racdif3: 1="yes, lack motivation" (racist) vs 2="no" (not racist)
# In the dataset, let's check the actual values
print("racdif3 value counts:")
print(df['racdif3'].value_counts())
print("\nracdif3 cross-tab with racmost (should be positively correlated if racist=1):")

# With racdif3=1 as racist (standard GSS interpretation):
df['r_racdif3_v1'] = (df['racdif3'] == 1).astype(float).where(df['racdif3'].notna())
# With racdif3=2 as racist (our current flipped interpretation):
df['r_racdif3_v2'] = (df['racdif3'] == 2).astype(float).where(df['racdif3'].notna())

# Inter-item correlations
items_v1 = pd.DataFrame({
    'racmost': df['r_racmost'],
    'busing': df['r_busing'],
    'racdif1': df['r_racdif1'],
    'racdif2': df['r_racdif2'],
    'racdif3_v1': df['r_racdif3_v1'],
    'racdif3_v2': df['r_racdif3_v2'],
})
print("\nCorrelation matrix:")
print(items_v1.corr().round(3))

# Test both with old 5-item scale
for version, racdif3_col in [('racdif3=1 racist', 'r_racdif3_v1'), ('racdif3=2 racist', 'r_racdif3_v2')]:
    coded = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', racdif3_col]
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

    print(f"\n=== Old 5-item, {version}, min 4, PM imputation ===")
    print(f"  Racism score: mean={df['racism_score'].mean():.2f}, SD={df['racism_score'].std():.2f}")
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

# Also check: what if we use the MEAN of all 5 items instead of the SUM?
# The standardized beta would be the same, but person-mean imputation behavior differs
print("\n\n=== What if we use MEAN instead of SUM? ===")
coded = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3_v2']
coded_df = pd.DataFrame({c: df[c] for c in coded})
n_valid = coded_df.notna().sum(axis=1)
df['racism_mean'] = coded_df.mean(axis=1)
df.loc[n_valid < 4, 'racism_mean'] = np.nan
print(f"  Racism mean: mean={df['racism_mean'].mean():.3f}, SD={df['racism_mean'].std():.3f}")

predictors_mean = ['racism_mean', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']
for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    model_df = df[[dv] + predictors_mean].dropna()
    N = len(model_df)
    y = model_df[dv].values
    X = model_df[predictors_mean].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
    print(f"  {dv_name}: N={N}, Racism beta={model.params[1]:.3f} (p={model.pvalues[1]:.4f})")
