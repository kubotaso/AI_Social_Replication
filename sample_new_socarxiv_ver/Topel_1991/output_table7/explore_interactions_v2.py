#!/usr/bin/env python3
"""
Deep dive into interaction term specification.
The key problem: exp_sq_interaction is -0.000001 but should be -0.00061.
This is a factor of 600x off.

Hypothesis: The paper might define the interaction differently.
Maybe it's ct * exp (not ct * exp^2), or there's a scaling issue.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/data/psid_panel.csv')

EDUC = {0:0, 1:3, 2:7, 3:10, 4:12, 5:12, 6:14, 7:16, 8:17, 9:17}
df['ed_yrs'] = df['education_clean'].copy()
for yr in df['year'].unique():
    m = df['year'] == yr
    if df.loc[m, 'education_clean'].max() <= 9:
        df.loc[m, 'ed_yrs'] = df.loc[m, 'education_clean'].map(EDUC)

df['exp'] = (df['age'] - df['ed_yrs'] - 6).clip(lower=1)
df['exp_sq'] = df['exp'] ** 2

# Education dummies
df['ed_cat'] = pd.cut(df['ed_yrs'], bins=[-1, 11, 12, 15, 20], labels=['lt12', '12', '13_15', '16plus'])
ed_dummies = pd.get_dummies(df['ed_cat'], prefix='ed', drop_first=True, dtype=float)
for col in ed_dummies.columns:
    df[col] = ed_dummies[col]
ed_dum_cols = list(ed_dummies.columns)

CPS = {1968:1.0, 1969:1.032, 1970:1.091, 1971:1.115, 1972:1.113,
       1973:1.151, 1974:1.167, 1975:1.188, 1976:1.117, 1977:1.121,
       1978:1.133, 1979:1.128, 1980:1.128, 1981:1.109, 1982:1.103, 1983:1.089}

df['lw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['union'] = df['union_member'].fillna(0)
df['disability'] = df['disabled'].fillna(0)
df['smsa'] = df['lives_in_smsa'].fillna(0)
df['married_d'] = df['married'].fillna(0)

df['tenure_var'] = df['tenure_topel'].astype(float)
df['ct_obs'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)
df['ct_x_censor'] = df['ct_obs'] * df['censor']

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

all_vars = ['lw_cps', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor'] + control_vars
sample = df.dropna(subset=all_vars).copy()

y = sample['lw_cps']
base_vars = ['exp', 'exp_sq', 'tenure_var']

print(f"Sample: {len(sample)}")
print(f"\nMean values: exp={sample['exp'].mean():.1f}, exp_sq={sample['exp_sq'].mean():.1f}, ct={sample['ct_obs'].mean():.1f}, tenure={sample['tenure_var'].mean():.1f}")

# Test different interaction definitions
print("\n=== INTERACTION DEFINITIONS ===")

# 1. Standard: ct * exp^2
sample['int1'] = sample['ct_obs'] * sample['exp_sq']
sample['int1_t'] = sample['ct_obs'] * sample['tenure_var']
X1 = sm.add_constant(sample[base_vars + ['ct_obs', 'ct_x_censor', 'int1', 'int1_t'] + control_vars])
m1 = sm.OLS(y, X1).fit()
print(f"1. ct * exp^2: coef={m1.params['int1']:.8f}, se={m1.bse['int1']:.8f}, t={m1.tvalues['int1']:.3f}")
print(f"   ct * tenure: coef={m1.params['int1_t']:.6f}, se={m1.bse['int1_t']:.6f}")

# 2. ct * exp (not exp^2)
sample['int2'] = sample['ct_obs'] * sample['exp']
sample['int2_t'] = sample['ct_obs'] * sample['tenure_var']
X2 = sm.add_constant(sample[base_vars + ['ct_obs', 'ct_x_censor', 'int2', 'int2_t'] + control_vars])
m2 = sm.OLS(y, X2).fit()
print(f"\n2. ct * exp: coef={m2.params['int2']:.8f}, se={m2.bse['int2']:.8f}, t={m2.tvalues['int2']:.3f}")
print(f"   ct * tenure: coef={m2.params['int2_t']:.6f}, se={m2.bse['int2_t']:.6f}")

# 3. ct * exp^2 / 100 (scaled)
sample['int3'] = sample['ct_obs'] * sample['exp_sq'] / 100
sample['int3_t'] = sample['ct_obs'] * sample['tenure_var']
X3 = sm.add_constant(sample[base_vars + ['ct_obs', 'ct_x_censor', 'int3', 'int3_t'] + control_vars])
m3 = sm.OLS(y, X3).fit()
print(f"\n3. ct * exp^2/100: coef={m3.params['int3']:.8f}, se={m3.bse['int3']:.8f}")

# 4. exp^2 / 100 interaction (experience enters as x100)
# The paper reports experience^2 coefficient as -0.00079. If we multiply exp by 1/10,
# then exp^2 coef becomes -0.00079 * 100 = -0.079. But that doesn't match either.
# WAIT: the paper says "Experience^2" coefficient is -.00079.
# If experience is in raw years (~18), exp^2 is ~324.
# Coef * exp^2 contribution at mean = -0.00079 * 324 = -0.256 (over career)
# That's reasonable for a quadratic term.

# 5. Maybe the paper defines the interaction in terms of the MARGINAL effect
# Let's try: d(log_w)/d(experience) depends on ct
# The unrestricted model has: ... + gamma_x * ct * exp^2 + gamma_t * ct * tenure
# So d(log_w)/d(exp) = beta_x + 2*beta_x2*exp + 2*gamma_x*ct*exp
# The "Experience^2 interaction" row reports gamma_x (coefficient on ct * exp^2)

# 6. What if the paper uses exp/10 in the interaction?
sample['int6'] = sample['ct_obs'] * (sample['exp'] / 10) ** 2
X6 = sm.add_constant(sample[base_vars + ['ct_obs', 'ct_x_censor', 'int6', 'int1_t'] + control_vars])
m6 = sm.OLS(y, X6).fit()
print(f"\n6. ct * (exp/10)^2: coef={m6.params['int6']:.8f}, se={m6.bse['int6']:.8f}")

# 7. The key ratio: our coef / target coef
ratio = m1.params['int1'] / (-0.00061)
print(f"\n7. Ratio our/target: {ratio:.6f}")
print(f"   Mean ct_obs * exp_sq: {sample['int1'].mean():.1f}")
print(f"   If we scale interaction by 1/{1/ratio:.0f}: {m1.params['int1'] * (1/ratio):.8f}")

# 8. IMPORTANT: What if the model from the paper actually uses:
# w = ... + delta * T_bar + gamma_x * T_bar * (X^2 - X_bar^2) + gamma_t * T_bar * (T - T_bar)
# i.e., demeaned interactions? Let's try demeaning.
exp_sq_mean = sample['exp_sq'].mean()
tenure_mean = sample['tenure_var'].mean()
sample['int8_x'] = sample['ct_obs'] * (sample['exp_sq'] - exp_sq_mean)
sample['int8_t'] = sample['ct_obs'] * (sample['tenure_var'] - tenure_mean)
X8 = sm.add_constant(sample[base_vars + ['ct_obs', 'ct_x_censor', 'int8_x', 'int8_t'] + control_vars])
m8 = sm.OLS(y, X8).fit()
print(f"\n8. Demeaned interactions:")
print(f"   ct * (exp^2 - mean): coef={m8.params['int8_x']:.8f}, se={m8.bse['int8_x']:.8f}")
print(f"   ct * (tenure - mean): coef={m8.params['int8_t']:.6f}, se={m8.bse['int8_t']:.6f}")

# 9. Maybe the paper reports coefficients divided by 1000?
# So -0.00061 really means the coefficient on ct*exp^2 IS -0.00061,
# and our -0.000001 is just wrong because of the data/specification.
# Let me check: what is the implied effect at 10 years tenure, 20 years experience?
# Paper: -0.00061 * 10 * 400 = -2.44 (too large??)
# Us: -0.000001 * 10 * 400 = -0.004
# Paper at mean ct=8, exp_sq=500: -0.00061 * 8 * 500 = -2.44 (way too large)
# That can't be right for a log wage equation!
#
# UNLESS: the coefficient is on ct*exp^2/1000 or similar
# Or the coefficient IS small but our exp_sq variable is wrong.
#
# If the paper uses experience/10 (so "experience" ranges 0-5 instead of 0-50):
# Then exp^2 ranges 0-25, ct*exp^2 ranges 0-25*13=325
# Coefficient -0.00061 * 325 = -0.2 (reasonable)
#
# Our exp ranges 1-54, exp_sq ranges 1-2916, ct*exp_sq ranges ~1-30000
# Coefficient needs to be much smaller to keep the contribution manageable.
# Our -0.000001 * 30000 = -0.03

# 10. What if the paper divides experience by 10 throughout?
sample['exp10'] = sample['exp'] / 10
sample['exp10_sq'] = sample['exp10'] ** 2
sample['int10_x'] = sample['ct_obs'] * sample['exp10_sq']
sample['int10_t'] = sample['ct_obs'] * sample['tenure_var']
X10 = sm.add_constant(sample[['exp10', 'exp10_sq', 'tenure_var', 'ct_obs', 'ct_x_censor', 'int10_x', 'int10_t'] + control_vars])
m10 = sm.OLS(y, X10).fit()
print(f"\n10. exp/10: exp10_sq coef={m10.params['exp10_sq']:.6f}, int_x={m10.params['int10_x']:.8f}, int_t={m10.params['int10_t']:.6f}")
print(f"    Target: exp_sq=-0.00072, int_x=-0.00061, int_t=0.0142")

# 11. Try dividing by 100
sample['exp100'] = sample['exp'] / 100
sample['exp100_sq'] = sample['exp100'] ** 2
sample['int100_x'] = sample['ct_obs'] * sample['exp100_sq']
X100 = sm.add_constant(sample[['exp100', 'exp100_sq', 'tenure_var', 'ct_obs', 'ct_x_censor', 'int100_x', 'int1_t'] + control_vars])
m100 = sm.OLS(y, X100).fit()
print(f"\n11. exp/100: exp100_sq coef={m100.params['exp100_sq']:.6f}, int_x={m100.params['int100_x']:.8f}")

# 12. Let me think about this differently.
# The paper's unrestricted model (eq 18) allows completed tenure to interact
# with the wage-tenure and wage-experience profiles.
# The way to write this is:
# log(w) = beta_x * X + beta_x2 * X^2 + beta_t * T + delta * T_bar
#         + gamma_x * T_bar * X^2 + gamma_t * T_bar * T + controls
#
# But maybe the paper parameterizes it differently:
# log(w) = (beta_x + delta_x * T_bar) * X + (beta_x2 + gamma_x * T_bar) * X^2
#         + (beta_t + gamma_t * T_bar) * T + delta * T_bar + controls
#
# This is equivalent. gamma_x IS the coefficient on T_bar * X^2.
# And our coefficient is -0.000001. Theirs is -0.00061.
#
# The ratio is about 600. What if we're computing exp_sq wrong?
# Our exp ranges 1-54, exp_sq 1-2916
# If they use pre-sample experience starting from 1968 (we start 1971),
# experience for someone age 30 with 12 yrs education in 1975:
# Their exp = 30 - 12 - 6 = 12 (same)
# Nope, same formula.

# Let me try: maybe the issue is in HOW completed tenure is computed
# What if ct_obs should be max_tenure minus CURRENT tenure?
# i.e., remaining tenure rather than total tenure
sample['remaining_t'] = sample['ct_obs'] - sample['tenure_var']
sample['int_rem_x'] = sample['remaining_t'] * sample['exp_sq']
sample['int_rem_t'] = sample['remaining_t'] * sample['tenure_var']
X_rem = sm.add_constant(sample[base_vars + ['remaining_t', 'ct_x_censor', 'int_rem_x', 'int_rem_t'] + control_vars])
m_rem = sm.OLS(y, X_rem).fit()
print(f"\n12. Remaining tenure interactions:")
print(f"    rem_t: coef={m_rem.params['remaining_t']:.6f}")
print(f"    rem_t * exp^2: coef={m_rem.params['int_rem_x']:.8f}, se={m_rem.bse['int_rem_x']:.8f}")
print(f"    rem_t * tenure: coef={m_rem.params['int_rem_t']:.6f}, se={m_rem.bse['int_rem_t']:.6f}")

# 13. What if the interaction term in the paper is ct * experience (not exp^2)?
# and the "Experience^2" label is misleading?
sample['int_exp'] = sample['ct_obs'] * sample['exp']
X_exp = sm.add_constant(sample[base_vars + ['ct_obs', 'ct_x_censor', 'int_exp', 'int1_t'] + control_vars])
m_exp = sm.OLS(y, X_exp).fit()
print(f"\n13. ct * experience (linear):")
print(f"    coef={m_exp.params['int_exp']:.8f}, se={m_exp.bse['int_exp']:.8f}")
print(f"    Target: -0.00061")

# 14. VIF analysis for the interaction terms
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = sample[base_vars + ['ct_obs', 'ct_x_censor', 'int1', 'int1_t']]
print("\n14. VIF for interaction model:")
for i, col in enumerate(X_vif.columns):
    try:
        vif = variance_inflation_factor(sm.add_constant(X_vif).values, i+1)
        print(f"    {col}: VIF={vif:.1f}")
    except:
        print(f"    {col}: VIF=NA")

# 15. Correlation matrix
print("\n15. Correlation matrix (key vars):")
corr = sample[['exp_sq', 'tenure_var', 'ct_obs', 'int1', 'int1_t']].corr()
print(corr.round(3))
