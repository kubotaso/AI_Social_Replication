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

# Other IVs
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

predictor_labels = ['Racism score', 'Education', 'Household income per cap',
                    'Occupational prestige', 'Female', 'Age', 'Black', 'Hispanic',
                    'Other race', 'Conservative Protestant', 'No religion', 'Southern']

def build_scale(df, coded_items, min_valid, rescale_to=None):
    vals_list = []
    for idx in df.index:
        vals = [df.loc[idx, c] for c in coded_items]
        n_v = sum(1 for v in vals if not np.isnan(v))
        if n_v >= min_valid:
            valid_v = [v for v in vals if not np.isnan(v)]
            pm = np.mean(valid_v)
            raw = sum(v if not np.isnan(v) else pm for v in vals)
            if rescale_to is not None:
                raw = raw * rescale_to / len(coded_items)
            vals_list.append(raw)
        else:
            vals_list.append(np.nan)
    return pd.Series(vals_list, index=df.index)

def run_model(df, dv_col, predictors, predictor_labels):
    vars_needed = [dv_col] + predictors
    model_df = df[vars_needed].dropna()
    N = len(model_df)

    y = model_df[dv_col].values
    X = model_df[predictors].values

    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()

    X_const = sm.add_constant(X)
    model_raw = sm.OLS(y, X_const).fit()

    return {
        'N': N,
        'R2': model.rsquared,
        'adjR2': model_raw.rsquared_adj,
        'betas': dict(zip(predictor_labels, model.params[1:])),
        'pvals': dict(zip(predictor_labels, model.pvalues[1:])),
        'constant': model_raw.params[0],
        'constant_p': model_raw.pvalues[0]
    }

# Test many min_valid and rescale combinations for 6-item scale
print("=== Testing 6-item scale (racdif3 flipped) with different min_valid ===")
for min_v in [4, 5, 6]:
    df['racism_score'] = build_scale(df, coded, min_v, rescale_to=5)

    m1 = run_model(df, 'dv_minority', predictors, predictor_labels)
    m2 = run_model(df, 'dv_remaining', predictors, predictor_labels)

    print(f"min_valid={min_v}: M1 N={m1['N']}, racism={m1['betas']['Racism score']:.3f}, R2={m1['R2']:.3f}")
    print(f"            M2 N={m2['N']}, racism={m2['betas']['Racism score']:.3f}, black={m2['betas']['Black']:.3f}, R2={m2['R2']:.3f}, age={m2['betas']['Age']:.3f}")

# Try NOT rescaling (keep 0-6 range)
print("\n=== 6-item without rescaling ===")
for min_v in [4, 5]:
    df['racism_score'] = build_scale(df, coded, min_v, rescale_to=None)

    m1 = run_model(df, 'dv_minority', predictors, predictor_labels)
    m2 = run_model(df, 'dv_remaining', predictors, predictor_labels)

    print(f"min_valid={min_v}: M1 N={m1['N']}, racism={m1['betas']['Racism score']:.3f}, R2={m1['R2']:.3f}")
    print(f"            M2 N={m2['N']}, racism={m2['betas']['Racism score']:.3f}, black={m2['betas']['Black']:.3f}, R2={m2['R2']:.3f}")

# Standardized coefficients don't depend on rescaling so these should be the same
# Let me verify
print("\n=== Verification: rescaling shouldn't matter for std betas ===")
df['racism_a'] = build_scale(df, coded, 4, rescale_to=5)
df['racism_b'] = build_scale(df, coded, 4, rescale_to=None)
mask = df[['racism_a', 'racism_b']].notna().all(axis=1)
print(f"Correlation: {df.loc[mask, 'racism_a'].corr(df.loc[mask, 'racism_b']):.6f}")

# Key insight: the standardized betas are the same regardless of linear rescaling
# The ONLY thing that matters is which items are in the scale and the imputation method

# Let me try a completely different approach: what if NAs in racism items are treated
# as the non-racist response (0) rather than imputed with person mean?
print("\n=== NA as 0 (non-racist) with different min counts ===")
for min_v in [3, 4, 5]:
    raw_sum = sum(df[c].fillna(0) for c in coded)
    n_valid = pd.concat([df[c] for c in coded], axis=1).notna().sum(axis=1)
    df['racism_score'] = raw_sum
    df.loc[n_valid < min_v, 'racism_score'] = np.nan

    m1 = run_model(df, 'dv_minority', predictors, predictor_labels)
    m2 = run_model(df, 'dv_remaining', predictors, predictor_labels)

    print(f"min_valid={min_v}: M1 N={m1['N']}, racism={m1['betas']['Racism score']:.3f}, R2={m1['R2']:.3f}")
    print(f"            M2 N={m2['N']}, racism={m2['betas']['Racism score']:.3f}, black={m2['betas']['Black']:.3f}, R2={m2['R2']:.3f}")

# Try: what if we exclude Black respondents from the racism scale computation?
# Some studies do this because racism items may function differently for Black respondents
# No - that doesn't make sense, they'd still be in the regression

# What if the issue is that we need to use DIFFERENT Hispanic coding?
# Try broader Hispanic definition
print("\n=== Testing broader Hispanic definition ===")
df['racism_score'] = build_scale(df, coded, 4, rescale_to=5)

# ethnic 2=Spain, 15=Central/South American
for hisp_codes, label in [([17, 22, 25], 'standard'),
                           ([2, 17, 22, 25], '+Spain'),
                           ([2, 15, 17, 22, 25], '+Spain+CSA')]:
    df['hispanic'] = df['ethnic'].isin(hisp_codes).astype(int)

    m1 = run_model(df, 'dv_minority', predictors, predictor_labels)
    m2 = run_model(df, 'dv_remaining', predictors, predictor_labels)

    print(f"Hispanic {label}: M1 hisp={m1['betas']['Hispanic']:.3f}, M2 hisp={m2['betas']['Hispanic']:.3f}")
    print(f"  M2 black={m2['betas']['Black']:.3f}, racism={m2['betas']['Racism score']:.3f}")

# Reset
df['hispanic'] = df['ethnic'].isin([17, 22, 25]).astype(int)

# What if we try the exact SPSS standardization approach?
# Some SPSS implementations standardize differently (ddof=0 vs ddof=1)
print("\n=== Testing ddof=0 standardization ===")
df['racism_score'] = build_scale(df, coded, 4, rescale_to=5)
vars_needed = ['dv_remaining'] + predictors
model_df = df[vars_needed].dropna()
y = model_df['dv_remaining'].values
X = model_df[predictors].values
y_z = (y - y.mean()) / y.std(ddof=0)  # population std
X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)  # population std
model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
print(f"ddof=0: racism={model.params[1]:.3f}, black={model.params[7]:.3f}, R2={model.rsquared:.3f}")

# ddof=1 (sample std)
y_z = (y - y.mean()) / y.std(ddof=1)
X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
print(f"ddof=1: racism={model.params[1]:.3f}, black={model.params[7]:.3f}, R2={model.rsquared:.3f}")
# These should give same betas

# What about using statsmodels' built-in method to get standardized coefficients?
# beta_i = b_i * (sx_i / sy)
X_const = sm.add_constant(X)
model_raw = sm.OLS(y, X_const).fit()
b_unstd = model_raw.params[1:]
sx = X.std(axis=0, ddof=1)
sy = y.std(ddof=1)
betas_manual = b_unstd * sx / sy
print(f"Manual std (ddof=1): racism={betas_manual[0]:.3f}, black={betas_manual[6]:.3f}")

# Maybe the issue is something structural about the DV
# Let's check: what if DV2 should be 0-12 but we're counting wrong?
print(f"\n=== DV2 distribution check ===")
dv2_valid = model_df['dv_remaining']
print(f"DV2 range: {dv2_valid.min():.0f}-{dv2_valid.max():.0f}")
print(f"DV2 mean: {dv2_valid.mean():.3f}, SD: {dv2_valid.std(ddof=1):.3f}")
print(f"DV2 value counts:")
print(dv2_valid.value_counts().sort_index())
