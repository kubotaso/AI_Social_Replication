#!/usr/bin/env python3
"""
Deep investigation: why are experience polynomial coefficients wrong?
The issue is likely in how experience is constructed vs the paper's approach.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

df = pd.read_csv('data/psid_panel.csv')

# Fix education
df['educ_raw'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'educ_raw'] = df.loc[cat_mask, 'education_clean'].map({**EDUC_MAP, 9: np.nan})

def get_fixed_educ(group):
    good = group[group['year'].isin([1975, 1976])]['educ_raw'].dropna()
    if len(good) > 0:
        return good.iloc[0]
    mapped = group['educ_raw'].dropna()
    if len(mapped) > 0:
        modes = mapped.mode()
        return modes.iloc[0] if len(modes) > 0 else mapped.median()
    return np.nan

person_educ = df.groupby('person_id').apply(get_fixed_educ)
df['education_fixed'] = df['person_id'].map(person_educ)
df = df[df['education_fixed'].notna()].copy()

df['experience'] = df['age'] - df['education_fixed'] - 6
df['tenure'] = df['tenure_topel'] - 1

# Within-job
df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
grp = df.groupby(['person_id', 'job_id'])
df['prev_year'] = grp['year'].shift(1)
df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
df['prev_tenure'] = grp['tenure'].shift(1)
df['prev_experience'] = grp['experience'].shift(1)

within = df[
    (df['prev_year'].notna()) &
    (df['year'] - df['prev_year'] == 1)
].copy()
within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']
within['d_exp'] = within['experience'] - within['prev_experience']

# Use d_exp == 1 filter
base = within[within['d_exp'] == 1].copy()

# Apply 2-SD trim (to get N ~ 8683)
m0, s0 = base['d_log_wage'].mean(), base['d_log_wage'].std()
w = base[(base['d_log_wage'] >= m0 - 2*s0) & (base['d_log_wage'] <= m0 + 2*s0)].copy()

print(f"N = {len(w)}")
print(f"Mean experience: {w['experience'].mean():.2f}")
print(f"Mean prev_experience: {w['prev_experience'].mean():.2f}")

# Paper says Table A1 has mean experience X = 20.021
# Our mean is lower because we're missing years 1968-1970

# KEY INSIGHT: The polynomial differences are:
# d_exp_sq = exp^2 - (exp-1)^2 = 2*exp - 1
# d_exp_cu = exp^3 - (exp-1)^3 = 3*exp^2 - 3*exp + 1
# d_exp_qu = exp^4 - (exp-1)^4 = 4*exp^3 - 6*exp^2 + 4*exp - 1

# These are EXACT functions of exp level.
# The OLS coefficient on d_exp_sq estimates the SECOND DERIVATIVE of g(X)
# Since d_exp_sq = 2*exp - 1, it's essentially a linear function of experience
# The coefficient on d_exp_sq is g''(mean_exp) approximately

# With DIFFERENT mean experience, we get DIFFERENT polynomial coefficients
# because we're evaluating the polynomial derivatives at different points!

# To verify: let's check what coefficients we'd get if we SHIFTED experience
# to match the paper's mean of ~20

# Current mean experience in within-job sample
mean_exp = w['experience'].mean()
print(f"\nMean experience in sample: {mean_exp:.2f}")
print(f"Paper's mean experience: ~20.021")

# What if we SHIFT experience to match?
# exp_shifted = experience + (20.021 - mean_exp)
shift = 20.021 - mean_exp
print(f"Shift needed: {shift:.2f}")

# But shifting experience doesn't change d_exp_sq!
# d_exp_sq = (exp+shift)^2 - (exp+shift-1)^2 = 2*(exp+shift) - 1 = 2*exp + 2*shift - 1
# The change is just a constant added to d_exp_sq
# Since d_tenure = 1 (constant), this constant gets absorbed into d_tenure coefficient!

# Wait, that means shifting experience should NOT change the exp polynomial coefficients
# if there's an intercept (d_tenure = 1 acts as intercept)

# Actually, there's no explicit intercept in the model - d_tenure IS the constant
# And d_exp_sq = 2*exp - 1 is NOT a constant - it varies with exp
# But shifting exp changes d_exp_sq by a constant (2*shift), which IS absorbed by d_tenure

# So the exp polynomial coefficients should be invariant to experience shifts?
# Let me verify...

# WAIT: The issue might be that d_exp_sq, d_exp_cu, d_exp_qu are highly correlated
# when experience is small (because polynomial differences converge to similar functions
# at low experience levels). With higher experience, the polynomial terms spread out more,
# giving better identification.

# Let's check the variation in our polynomial terms vs what the paper likely had
t = w['tenure'].values.astype(float)
pt = w['prev_tenure'].values.astype(float)
e = w['experience'].values.astype(float)
pe = w['prev_experience'].values.astype(float)

w['d_tenure'] = t - pt
w['d_tenure_sq'] = t**2 - pt**2
w['d_exp_sq'] = e**2 - pe**2
w['d_exp_cu'] = e**3 - pe**3
w['d_exp_qu'] = e**4 - pe**4

print(f"\nPolynomial term statistics:")
for var in ['d_exp_sq', 'd_exp_cu', 'd_exp_qu']:
    print(f"  {var}: mean={w[var].mean():.2f}, std={w[var].std():.2f}, "
          f"min={w[var].min():.0f}, max={w[var].max():.0f}")

# Correlation matrix for experience polynomial terms
corr = w[['d_exp_sq', 'd_exp_cu', 'd_exp_qu']].corr()
print(f"\nCorrelation matrix (experience terms):")
print(corr.to_string(float_format='%.4f'))

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_exp = w[['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']].values
print(f"\nVIF (Model 1):")
for i, name in enumerate(['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']):
    vif = variance_inflation_factor(X_exp, i)
    print(f"  {name}: {vif:.1f}")

# HYPOTHESIS: The experience polynomial coefficients change because
# the experience DISTRIBUTION is different (not just the mean).
# Our data has experience from -1 to 54; paper likely has 1 to ~40.
# The presence of extreme experience values (high outliers) can strongly
# influence the polynomial coefficients.

# Let's check experience distribution
print(f"\nExperience distribution:")
print(w['experience'].describe())
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p}: {w['experience'].quantile(p/100):.0f}")

# Try restricting experience range to 1-40 (reasonable working life)
# and see if coefficients improve
w_restricted = w[(w['experience'] >= 1) & (w['experience'] <= 40)].copy()
print(f"\nAfter exp 1-40: N={len(w_restricted)}")

# Run regression on restricted sample
t = w_restricted['tenure'].values.astype(float)
pt = w_restricted['prev_tenure'].values.astype(float)
e = w_restricted['experience'].values.astype(float)
pe = w_restricted['prev_experience'].values.astype(float)

w_restricted['d_tenure'] = t - pt
w_restricted['d_exp_sq'] = e**2 - pe**2
w_restricted['d_exp_cu'] = e**3 - pe**3
w_restricted['d_exp_qu'] = e**4 - pe**4

yd = pd.get_dummies(w_restricted['year'], prefix='yr', dtype=float)
yc = sorted(yd.columns.tolist())[1:]
yv = w_restricted['d_log_wage'].values

vl = ['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
X = pd.concat([w_restricted[vl].reset_index(drop=True), yd[yc].reset_index(drop=True)], axis=1)
valid = np.isfinite(X.values).all(axis=1) & np.isfinite(yv)
md = sm.OLS(yv[valid], X.loc[valid].values, hasconst=True).fit()

print(f"Model 1 (exp 1-40):")
print(f"  N={int(md.nobs)}, SE_reg={np.sqrt(md.mse_resid):.4f}, R^2={md.rsquared:.4f}")
for i, var in enumerate(vl):
    c, s = md.params[i], md.bse[i]
    scale = {0: 1, 1: 100, 2: 1000, 3: 10000}[i]
    gt = {0: 0.1242, 1: -0.6051, 2: 0.1460, 3: 0.0131}[i]
    print(f"  {var}: {c*scale:.4f} (paper: {gt:.4f})")

# NOW: let's try SCALING experience differently
# What if experience is measured in decades (experience/10)?
# This shouldn't change coefficients if we adjust scaling properly...
# But let me try it numerically to be sure

# Actually, let me reconsider the EDUCATION coding.
# Maybe the education mapping is wrong, which would shift all experience levels
# The paper says mean education = 12.645
# Let's check what we get
print(f"\n\nMean education_fixed: {df['education_fixed'].mean():.3f} (paper: 12.645)")
print(f"Education distribution:")
print(df['education_fixed'].value_counts().sort_index())

# Our mean is close. Let's try the RAW education (not fixed) for 1975/1976
raw_75_76 = df[df['year'].isin([1975, 1976])]['education_clean']
print(f"\n1975/1976 education (raw, should be years):")
print(raw_75_76.describe())
print(raw_75_76.value_counts().sort_index())

# What if education_clean in 1975/1976 is ALSO categorical, not years?
# Let's check the range
print(f"\nMax education_clean in 1975: {df[df['year']==1975]['education_clean'].max()}")
print(f"Max education_clean in 1976: {df[df['year']==1976]['education_clean'].max()}")
print(f"Max education_clean in 1977: {df[df['year']==1977]['education_clean'].max()}")
