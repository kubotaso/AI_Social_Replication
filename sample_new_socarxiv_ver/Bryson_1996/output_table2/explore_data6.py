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

coded = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3', 'r_racdif4']

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

# Strategy: the 6-item scale with min 4 gives N=655/614
# The old 5-item scale (without racdif4, racdif3 flipped) with min 4 gives N=645/600
# The N targets are 644/605
#
# What if we use 5-item scale (racmost, busing, racdif1, racdif2, racdif3(flipped))
# with person-mean imputation, then ALSO add racdif4 as a separate racism item?
# No - that's the same as 6-item scale but differently weighted

# What determines the N? The number of missing values in the COMBINED predictors
# The racism scale min_valid determines how many people have valid racism scores
# Then listwise deletion on all predictors determines final N

# Let me check: how many people are lost due to EACH predictor being missing?
print("=== Missing data by predictor ===")
for pred in predictors:
    if pred == 'racism_score':
        continue
    n_miss = df[pred].isna().sum()
    print(f"{pred}: {n_miss} missing")

# The racism score is the main filter. Let's see counts
for min_v in [3, 4, 5, 6]:
    n_valid_racism = pd.concat([df[c] for c in coded], axis=1).notna().sum(axis=1)
    mask = n_valid_racism >= min_v
    racism_vals = []
    for idx in df.index:
        vals = [df.loc[idx, c] for c in coded]
        n_v = sum(1 for v in vals if not np.isnan(v))
        if n_v >= min_v:
            valid_v = [v for v in vals if not np.isnan(v)]
            pm = np.mean(valid_v)
            racism_vals.append(sum(v if not np.isnan(v) else pm for v in vals) * 5/6)
        else:
            racism_vals.append(np.nan)
    df['racism_score'] = racism_vals

    for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
        model_df = df[[dv] + predictors].dropna()
        print(f"6-item min_valid={min_v}, {dv_name}: N={len(model_df)}")

# Now try: maybe the issue is that ethnic NaN people should be excluded
# (as missing, not coded as 0 for Hispanic)
print("\n=== What if ethnic NaN is treated as missing? ===")
df['hispanic_strict'] = np.where(df['ethnic'].isna(), np.nan, df['ethnic'].isin([17, 22, 25]).astype(float))
predictors_strict = predictors.copy()
predictors_strict[predictors_strict.index('hispanic')] = 'hispanic_strict'

df['racism_score'] = [np.nan] * len(df)
for idx in df.index:
    vals = [df.loc[idx, c] for c in coded]
    n_v = sum(1 for v in vals if not np.isnan(v))
    if n_v >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        df.loc[idx, 'racism_score'] = sum(v if not np.isnan(v) else pm for v in vals) * 5/6

for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    model_df = df[[dv] + predictors_strict].dropna()
    print(f"ethnic NaN=missing, {dv_name}: N={len(model_df)}")

    y = model_df[dv].values
    X = model_df[predictors_strict].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
    print(f"  racism={model.params[1]:.3f}, education={model.params[2]:.3f}, black={model.params[7]:.3f}, age={model.params[6]:.3f}")

    X_const = sm.add_constant(X)
    model_raw = sm.OLS(y, X_const).fit()
    print(f"  R2={model_raw.rsquared:.3f}, adjR2={model_raw.rsquared_adj:.3f}")

# What if we also require fund to be non-missing?
print("\n=== Testing fund NaN = missing ===")
df['fund_val'] = pd.to_numeric(df['fund'], errors='coerce')
df['conservative_protestant_strict'] = np.where(df['fund_val'].isna(), np.nan, (df['fund_val'] == 1).astype(float))
predictors_strict2 = predictors.copy()
predictors_strict2[predictors_strict2.index('conservative_protestant')] = 'conservative_protestant_strict'

for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    model_df = df[[dv] + predictors_strict2].dropna()
    print(f"fund NaN=missing, {dv_name}: N={len(model_df)}")

# What about treating ethnic NaN as missing AND fund NaN as missing?
predictors_strict3 = predictors.copy()
predictors_strict3[predictors_strict3.index('hispanic')] = 'hispanic_strict'
predictors_strict3[predictors_strict3.index('conservative_protestant')] = 'conservative_protestant_strict'

for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    model_df = df[[dv] + predictors_strict3].dropna()
    print(f"both strict, {dv_name}: N={len(model_df)}")

# What if age 89 is coded as missing? (GSS codes 89+ as 89)
print(f"\nAge distribution top:")
print(df['age_var'].describe())
print(f"Age >= 89: {(df['age_var'] >= 89).sum()}")

# Check the old 5-item scale approach that got best score
# It used racmost, busing, racdif1, racdif2, racdif3(flipped)
coded_old = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3']
for idx in df.index:
    vals = [df.loc[idx, c] for c in coded_old]
    n_v = sum(1 for v in vals if not np.isnan(v))
    if n_v >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        df.loc[idx, 'racism_score'] = sum(v if not np.isnan(v) else pm for v in vals)
    else:
        df.loc[idx, 'racism_score'] = np.nan

for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    model_df = df[[dv] + predictors].dropna()
    print(f"\nOld 5-item (racdif2, no racdif4), {dv_name}: N={len(model_df)}")

    y = model_df[dv].values
    X = model_df[predictors].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()

    X_const = sm.add_constant(X)
    model_raw = sm.OLS(y, X_const).fit()

    print(f"  racism={model.params[1]:.3f}(p={model.pvalues[1]:.4f}), black={model.params[7]:.3f}, age={model.params[6]:.3f}")
    print(f"  R2={model_raw.rsquared:.3f}, adjR2={model_raw.rsquared_adj:.3f}")
    print(f"  Racism stats: mean={model_df['racism_score'].mean():.3f}, SD={model_df['racism_score'].std(ddof=1):.3f}")
