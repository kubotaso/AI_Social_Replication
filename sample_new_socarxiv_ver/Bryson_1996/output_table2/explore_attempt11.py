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

predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']

def test_config(label, coded_items, min_valid, scale_factor=1.0):
    coded_df = pd.DataFrame({c: df[c] for c in coded_items})
    n_valid = coded_df.notna().sum(axis=1)
    pmean = coded_df.mean(axis=1)
    for c in coded_items:
        coded_df[c] = coded_df[c].fillna(pmean)
    racism_raw = coded_df.sum(axis=1) * scale_factor
    df['racism_score'] = racism_raw
    df.loc[n_valid < min_valid, 'racism_score'] = np.nan

    print(f"\n=== {label} ===")
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
        for lab in ['Racism score', 'Education', 'Age', 'Black', 'Hispanic', 'Other race', 'Conservative Protestant']:
            b = betas[lab]
            p = pvals[lab]
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"    {lab:<30} {b:7.3f} {sig:<4} (p={p:.4f})")

# Config 1: correct 5-item, min 3, person-mean imputation, scale to 0-5
test_config("Correct 5-item, min 3, PM imputation",
            ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif3', 'r_racdif4'],
            min_valid=3, scale_factor=1.0)

# Config 2: correct 5-item, min 4, person-mean imputation
test_config("Correct 5-item, min 4, PM imputation",
            ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif3', 'r_racdif4'],
            min_valid=4, scale_factor=1.0)

# Config 3: old 5-item (best so far), min 4
test_config("Old 5-item, min 4, PM imputation (CURRENT BEST)",
            ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3'],
            min_valid=4, scale_factor=1.0)

# Config 4: old 5-item, min 3
test_config("Old 5-item, min 3, PM imputation",
            ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3'],
            min_valid=3, scale_factor=1.0)

# Config 5: 6-item scale, min 4, scaled to 0-5
test_config("6-item, min 4, PM imputation, scaled to 0-5",
            ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3', 'r_racdif4'],
            min_valid=4, scale_factor=5.0/6.0)

# Config 6: 6-item scale, min 5, scaled to 0-5
test_config("6-item, min 5, PM imputation, scaled to 0-5",
            ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3', 'r_racdif4'],
            min_valid=5, scale_factor=5.0/6.0)

# Also check: what if we use income91 instead of realinc/hompop?
print("\n\n=== INCOME VARIABLE TEST ===")
df['income91'] = pd.to_numeric(df.get('income91', df.get('rincome', pd.Series())), errors='coerce')
print(f"income91 valid: {df['income91'].notna().sum()}")
print(f"realinc valid: {df['realinc'].notna().sum()}")

# Check if there's a coninc variable
for col in df.columns:
    if 'inc' in col.lower() or 'earn' in col.lower():
        print(f"  Found income-related column: {col}, valid={pd.to_numeric(df[col], errors='coerce').notna().sum()}")
