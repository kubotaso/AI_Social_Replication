#!/usr/bin/env python3
"""Debug the experience coefficient sign issue."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

df = pd.read_csv('data/psid_panel.csv')
df['education_years'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(
    {**EDUC_MAP, 9: np.nan}
)
df = df[df['education_years'].notna()].copy()
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df['tenure'] = df['tenure_topel'] - 1

df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
grp = df.groupby(['person_id', 'job_id'])
df['prev_year'] = grp['year'].shift(1)
df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
df['prev_tenure'] = grp['tenure'].shift(1)
df['prev_experience'] = grp['experience'].shift(1)

within = df[
    (df['prev_year'].notna()) &
    (df['year'] - df['prev_year'] == 1) &
    df['experience'].notna() &
    df['prev_experience'].notna() &
    (df['experience'] >= 1)
].copy()
within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']
within = within[within['d_log_wage'].between(-2, 2)].copy()

# Experience stats
print("=== EXPERIENCE DISTRIBUTION ===")
print(f"Mean experience: {within['experience'].mean():.1f}")
print(f"Mean prev_experience: {within['prev_experience'].mean():.1f}")
print(f"d_experience = experience - prev_experience:")
within['d_exp'] = within['experience'] - within['prev_experience']
print(f"  Mean: {within['d_exp'].mean():.4f}")
print(f"  Unique values: {sorted(within['d_exp'].unique())[:10]}")
# Should be 1 for all within-job obs

# d_exp_sq = exp^2 - prev_exp^2
within['d_exp_sq'] = within['experience']**2 - within['prev_experience']**2
# This should equal 2*exp - 1 when d_exp = 1
within['d_exp_sq_check'] = 2 * within['experience'] - 1
print(f"\nd_exp_sq stats:")
print(f"  Mean: {within['d_exp_sq'].mean():.1f}")
print(f"  Min: {within['d_exp_sq'].min():.1f}")
print(f"  Max: {within['d_exp_sq'].max():.1f}")

# Check if d_exp != 1 for some observations
bad_dexp = within[within['d_exp'] != 1]
print(f"\nObs where d_experience != 1: {len(bad_dexp)}")
if len(bad_dexp) > 0:
    print(f"  d_exp values: {sorted(bad_dexp['d_exp'].unique())[:20]}")
    print(f"  These obs break the identification assumption!")
    # Look at some examples
    for _, row in bad_dexp.head(5).iterrows():
        print(f"  person={row['person_id']}, year={row['year']}, "
              f"age={row['age']}, educ={row['education_years']}, "
              f"exp={row['experience']}, prev_exp={row['prev_experience']}, "
              f"d_exp={row['d_exp']}")

# The paper says "experience and tenure progress at the same rate"
# i.e., d_tenure = d_experience = 1 for ALL within-job observations
# If d_experience != 1, that's a problem

# How many have d_exp exactly 1?
exact_1 = (within['d_exp'] == 1).sum()
print(f"\nObs with d_exp == 1: {exact_1} out of {len(within)} ({100*exact_1/len(within):.1f}%)")

# Now try the model WITHOUT observations where d_exp != 1
within_clean = within[within['d_exp'] == 1].copy()
print(f"\nAfter keeping only d_exp==1: N={len(within_clean)}")

# Recompute
t = within_clean['tenure'].values.astype(float)
pt = within_clean['prev_tenure'].values.astype(float)
e = within_clean['experience'].values.astype(float)
pe = within_clean['prev_experience'].values.astype(float)

within_clean['d_tenure'] = t - pt
within_clean['d_exp_sq'] = e**2 - pe**2
within_clean['d_exp_cu'] = e**3 - pe**3
within_clean['d_exp_qu'] = e**4 - pe**4

yr_dum = pd.get_dummies(within_clean['year'], prefix='yr', dtype=float)
yr_cols = sorted(yr_dum.columns.tolist())[1:]

X = pd.concat([within_clean[['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']].reset_index(drop=True),
                yr_dum[yr_cols].reset_index(drop=True)], axis=1)
y = within_clean['d_log_wage'].values
valid = np.isfinite(X.values).all(axis=1) & np.isfinite(y)

m = sm.OLS(y[valid], X.loc[valid].values, hasconst=True).fit()
print(f"\nModel 1 (d_exp==1 only):")
print(f"  Delta Tenure: {m.params[0]:.4f}")
print(f"  d_exp_sq (x100): {m.params[1]*100:.4f}")
print(f"  d_exp_cu (x1000): {m.params[2]*1000:.4f}")
print(f"  d_exp_qu (x10000): {m.params[3]*10000:.4f}")
print(f"  R^2: {m.rsquared:.4f}")
print(f"  SE: {np.sqrt(m.mse_resid):.4f}")
print(f"  N: {int(m.nobs)}")

# Check correlation matrix of regressors
print("\n=== CORRELATION MATRIX ===")
corr_vars = ['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
corr_df = within_clean[corr_vars + ['d_log_wage']]
print(corr_df.corr().round(3))

# What is the VIF for d_exp_sq?
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_check = within_clean[corr_vars].values
for i, name in enumerate(corr_vars):
    vif = variance_inflation_factor(X_check, i)
    print(f"VIF({name}): {vif:.1f}")
