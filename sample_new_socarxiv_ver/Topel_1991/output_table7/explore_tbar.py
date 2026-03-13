#!/usr/bin/env python3
"""
Explore: What if T_bar is the MEAN tenure within each job spell?

From eq (18): y = X_0*beta1 + (T - T_bar)(beta1+beta2) + T_bar(beta1+beta2) + T*theta + epsilon

T_bar = average tenure on job j
T* = completed tenure (= last observed tenure + predicted residual life)

The table reports coefficients on:
- Experience: from X_0*beta1
- Experience^2: from X_0*beta1
- Tenure: either (T-T_bar) or just T depending on column
- Completed tenure: T* or T^L
- x censor: T^L * censor_dummy
- Experience^2 (interaction): T* * X^2 or T^L * X^2
- Tenure (interaction): T* * T or T^L * T

But WAIT - eq (18) says the model has:
X_0*beta1 + (T - T_bar)(beta1+beta2) + T_bar(beta1+beta2) + T*theta

This is NOT the same as adding T* interactions with X^2 and T.
The unrestricted version allows different coefficients on (T-T_bar) and T_bar.

Actually let me reread. The text says:
"Least squares applied to (17) is equivalent to estimating (18) and imposing
the restriction that the coefficients on T - T_bar and T_bar are identical."

So the RESTRICTED model (cols 2, 4) estimates:
  y = X*beta + T*gamma + T_bar*delta + controls
where T_bar is average tenure on job j

The UNRESTRICTED model (cols 3, 5) allows T_bar to interact with X^2 and T.

BUT the actual Table 7 clearly has separate rows for:
- Observed completed tenure (cols 2-3): This is T^L (last observed tenure)
- x censor (cols 2-3): T^L * censor
- Experience^2 interaction (cols 3, 5)
- Tenure interaction (cols 3, 5)

Hmm, maybe the unrestricted model isn't exactly eq (18). The text on p. 171 says:
"Columns 3 and 5... These estimates are derived by applying (18) and solving
for beta_2 from estimates of beta_1 + beta_2 and beta_1."

So the unrestricted model does use eq (18) but with completed tenure T* allowed
to interact with the profile.

Let me try BOTH interpretations:
1. T_bar = max(tenure) on each job (our current approach)
2. T_bar = mean(tenure) on each job
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

# Two definitions of completed tenure
df['ct_max'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)  # max tenure
df['ct_mean'] = df.groupby('job_id')['tenure_topel'].transform('mean').astype(float)  # mean tenure

df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

all_vars = ['lw_cps', 'exp', 'exp_sq', 'tenure_var'] + control_vars
sample = df.dropna(subset=all_vars).copy()
y = sample['lw_cps']

print(f"Sample: {len(sample)}")
print(f"ct_max mean: {sample['ct_max'].mean():.2f}, ct_mean mean: {sample['ct_mean'].mean():.2f}")
print(f"tenure mean: {sample['tenure_var'].mean():.2f}")

# ==== Test with ct_max (our current approach) ====
print("\n" + "="*80)
print("=== USING ct_max (max tenure on job) ===")
ct = 'ct_max'
sample[f'{ct}_x_censor'] = sample[ct] * sample['censor']
sample[f'{ct}_x_exp_sq'] = sample[ct] * sample['exp_sq']
sample[f'{ct}_x_tenure'] = sample[ct] * sample['tenure_var']

# Col 3: unrestricted with observed CT
X3 = sm.add_constant(sample[['exp', 'exp_sq', 'tenure_var', ct, f'{ct}_x_censor',
                              f'{ct}_x_exp_sq', f'{ct}_x_tenure'] + control_vars])
m3 = sm.OLS(y, X3).fit()
print(f"Col 3: exp_sq_int={m3.params[f'{ct}_x_exp_sq']:.8f} (target -0.00061)")
print(f"       tenure_int={m3.params[f'{ct}_x_tenure']:.6f} (target 0.0142)")
print(f"       ct={m3.params[ct]:.6f}, tenure={m3.params['tenure_var']:.6f}")

# ==== Test with ct_mean ====
print("\n" + "="*80)
print("=== USING ct_mean (mean tenure on job) ===")
ct = 'ct_mean'
sample[f'{ct}_x_censor'] = sample[ct] * sample['censor']
sample[f'{ct}_x_exp_sq'] = sample[ct] * sample['exp_sq']
sample[f'{ct}_x_tenure'] = sample[ct] * sample['tenure_var']

X3m = sm.add_constant(sample[['exp', 'exp_sq', 'tenure_var', ct, f'{ct}_x_censor',
                               f'{ct}_x_exp_sq', f'{ct}_x_tenure'] + control_vars])
m3m = sm.OLS(y, X3m).fit()
print(f"Col 3: exp_sq_int={m3m.params[f'{ct}_x_exp_sq']:.8f} (target -0.00061)")
print(f"       tenure_int={m3m.params[f'{ct}_x_tenure']:.6f} (target 0.0142)")
print(f"       ct={m3m.params[ct]:.6f}, tenure={m3m.params['tenure_var']:.6f}")

# ==== What about (T - T_bar) model from eq 18? ====
print("\n" + "="*80)
print("=== EQUATION (18) PARAMETERIZATION ===")
# From eq (18): y = X_0*beta1 + (T-T_bar)(beta1+beta2) + T_bar(beta1+beta2) + T*theta + epsilon
# T_bar_j = average tenure on job j, T = current tenure, T* = completed tenure
# The model has: X_0, (T - T_bar), T_bar, T*
# Unrestricted: allow different coeffs on (T-T_bar) and T_bar, plus T* interactions

sample['t_minus_tbar_max'] = sample['tenure_var'] - sample['ct_max']
sample['t_minus_tbar_mean'] = sample['tenure_var'] - sample['ct_mean']

# With T_bar = mean tenure
print("\nT_bar = mean tenure:")
t_m_tbar = 't_minus_tbar_mean'
tbar = 'ct_mean'
ct_obs = 'ct_max'  # T* = completed tenure (different from T_bar!)

# Restricted: same coeff on (T-T_bar) and T_bar
# Unrestricted: different coefficients, plus T* interactions
sample['ct_max_x_censor_mean'] = sample['ct_max'] * sample['censor']
X_unr = sm.add_constant(sample[['exp', 'exp_sq', t_m_tbar, tbar, ct_obs,
                                 'ct_max_x_censor_mean',
                                 f'ct_max_x_exp_sq', f'ct_max_x_tenure'] + control_vars])
m_unr = sm.OLS(y, X_unr).fit()
print(f"  (T-T_bar)={m_unr.params[t_m_tbar]:.6f}")
print(f"  T_bar={m_unr.params[tbar]:.6f}")
print(f"  T*={m_unr.params[ct_obs]:.6f}")
print(f"  T*_x_exp_sq={m_unr.params['ct_max_x_exp_sq']:.8f}")
print(f"  T*_x_tenure={m_unr.params['ct_max_x_tenure']:.6f}")

# ===== Try: maybe the interaction is T_bar * X^2 not T* * X^2 =====
print("\n" + "="*80)
print("=== INTERACTION WITH T_BAR (mean) instead of T* ===")
sample['tbar_x_exp_sq'] = sample['ct_mean'] * sample['exp_sq']
sample['tbar_x_tenure'] = sample['ct_mean'] * sample['tenure_var']
X_tbar_int = sm.add_constant(sample[['exp', 'exp_sq', 'tenure_var', 'ct_mean',
                                      'ct_mean_x_censor',
                                      'tbar_x_exp_sq', 'tbar_x_tenure'] + control_vars])
m_tbar_int = sm.OLS(y, X_tbar_int).fit()
print(f"  T_bar_x_exp_sq={m_tbar_int.params['tbar_x_exp_sq']:.8f}")
print(f"  T_bar_x_tenure={m_tbar_int.params['tbar_x_tenure']:.6f}")

# ===== What if completed tenure is defined as total years the job spans? =====
# Not max(tenure) but max(year) - min(year) + 1
print("\n" + "="*80)
print("=== CT = job duration in years (max_year - min_year + 1) ===")
df['job_min_yr'] = df.groupby('job_id')['year'].transform('min')
df['job_max_yr'] = df.groupby('job_id')['year'].transform('max')
df['ct_duration'] = (df['job_max_yr'] - df['job_min_yr'] + 1).astype(float)
sample['ct_duration'] = df.loc[sample.index, 'ct_duration']

print(f"ct_duration mean: {sample['ct_duration'].mean():.2f}, ct_max mean: {sample['ct_max'].mean():.2f}")
print(f"Correlation: {sample['ct_duration'].corr(sample['ct_max']):.4f}")

sample['ct_dur_x_censor'] = sample['ct_duration'] * sample['censor']
sample['ct_dur_x_exp_sq'] = sample['ct_duration'] * sample['exp_sq']
sample['ct_dur_x_tenure'] = sample['ct_duration'] * sample['tenure_var']

X_dur = sm.add_constant(sample[['exp', 'exp_sq', 'tenure_var', 'ct_duration', 'ct_dur_x_censor',
                                 'ct_dur_x_exp_sq', 'ct_dur_x_tenure'] + control_vars])
m_dur = sm.OLS(y, X_dur).fit()
print(f"  exp_sq_int={m_dur.params['ct_dur_x_exp_sq']:.8f}")
print(f"  tenure_int={m_dur.params['ct_dur_x_tenure']:.6f}")
print(f"  ct_dur={m_dur.params['ct_duration']:.6f}")
print(f"  tenure={m_dur.params['tenure_var']:.6f}")

# ===== What if "Experience^2" in table means the interaction is with EXP^2 / 100? =====
# The paper uses exp^2/100 sometimes for readability
print("\n" + "="*80)
print("=== SCALING: ct * exp_sq_scaled ===")
for scale in [1, 10, 100, 1000]:
    sample['int_scaled'] = sample['ct_max'] * sample['exp_sq'] / scale
    X_s = sm.add_constant(sample[['exp', 'exp_sq', 'tenure_var', 'ct_max', 'ct_max_x_censor',
                                   'int_scaled', 'ct_max_x_tenure'] + control_vars])
    m_s = sm.OLS(y, X_s).fit()
    print(f"  scale={scale}: int_coef={m_s.params['int_scaled']:.8f}, se={m_s.bse['int_scaled']:.8f}")

# ===== What if the interaction is NOT with ct but with REMAINING tenure? =====
print("\n" + "="*80)
print("=== REMAINING TENURE = ct - current_tenure ===")
sample['rem_tenure'] = (sample['ct_max'] - sample['tenure_var']).clip(lower=0)
sample['rem_x_exp_sq'] = sample['rem_tenure'] * sample['exp_sq']
sample['rem_x_tenure'] = sample['rem_tenure'] * sample['tenure_var']
sample['rem_x_censor'] = sample['rem_tenure'] * sample['censor']

X_rem = sm.add_constant(sample[['exp', 'exp_sq', 'tenure_var', 'rem_tenure', 'rem_x_censor',
                                 'rem_x_exp_sq', 'rem_x_tenure'] + control_vars])
m_rem = sm.OLS(y, X_rem).fit()
print(f"  rem_x_exp_sq={m_rem.params['rem_x_exp_sq']:.8f}")
print(f"  rem_x_tenure={m_rem.params['rem_x_tenure']:.6f}")
print(f"  rem_tenure={m_rem.params['rem_tenure']:.6f}")
print(f"  tenure={m_rem.params['tenure_var']:.6f}")

# ===== What if we use first tenure (initial tenure on job)? =====
print("\n" + "="*80)
print("=== T^0 (initial tenure on job, first observed) ===")
df['t0'] = df.groupby('job_id')['tenure_topel'].transform('min').astype(float)
sample['t0'] = df.loc[sample.index, 't0']
print(f"T^0 stats: mean={sample['t0'].mean():.2f}, min={sample['t0'].min()}, max={sample['t0'].max()}")

# From footnote 22: T_bar = (T^0 + T^L)/2
sample['tbar_fn22'] = (sample['t0'] + sample['ct_max']) / 2
print(f"T_bar (fn22) mean: {sample['tbar_fn22'].mean():.2f}")

sample['tbar22_x_exp_sq'] = sample['tbar_fn22'] * sample['exp_sq']
sample['tbar22_x_tenure'] = sample['tbar_fn22'] * sample['tenure_var']
sample['ct_x_cen_22'] = sample['ct_max'] * sample['censor']

X_22 = sm.add_constant(sample[['exp', 'exp_sq', 'tenure_var', 'ct_max', 'ct_x_cen_22',
                                'tbar22_x_exp_sq', 'tbar22_x_tenure'] + control_vars])
m_22 = sm.OLS(y, X_22).fit()
print(f"  tbar22_x_exp_sq={m_22.params['tbar22_x_exp_sq']:.8f}")
print(f"  tbar22_x_tenure={m_22.params['tbar22_x_tenure']:.6f}")
