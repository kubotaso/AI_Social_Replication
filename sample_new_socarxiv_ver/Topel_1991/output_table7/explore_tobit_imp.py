#!/usr/bin/env python3
"""Try Tobit-style imputation for censored CT and other creative approaches."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
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

CPS = {1968:1.0, 1969:1.032, 1970:1.091, 1971:1.115, 1972:1.113,
       1973:1.151, 1974:1.167, 1975:1.188, 1976:1.117, 1977:1.121,
       1978:1.133, 1979:1.128, 1980:1.128, 1981:1.109, 1982:1.103, 1983:1.089}
gnp = {1971:44.4, 1972:46.5, 1973:49.5, 1974:54.0, 1975:59.3, 1976:63.1,
       1977:67.3, 1978:72.2, 1979:78.6, 1980:85.7, 1981:94.0, 1982:100.0, 1983:103.9}

df['lw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['lw_gnp'] = np.log(df['hourly_wage'] / (df['year'].map(gnp) / 100))
df['y'] = 0.750 * df['lw_cps'] + 0.250 * df['lw_gnp']

df['ed_cat'] = pd.cut(df['ed_yrs'], bins=[-1, 11, 12, 15, 20], labels=['lt12', '12', '13_15', '16plus'])
ed_dummies = pd.get_dummies(df['ed_cat'], prefix='ed', drop_first=True, dtype=float)
for col in ed_dummies.columns:
    df[col] = ed_dummies[col]
ed_dum_cols = list(ed_dummies.columns)

df['union'] = df['union_member'].fillna(0)
df['disability'] = df['disabled'].fillna(0)
df['smsa'] = df['lives_in_smsa'].fillna(0)
df['married_d'] = df['married'].fillna(0)
df['tenure_var'] = df['tenure_topel'].astype(float)
df['ct_obs'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)
df['ct_x_censor'] = df['ct_obs'] * (1 - df['censor'])

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols
base = ['exp', 'exp_sq', 'tenure_var']

# === Approach 1: Heckman-style imputed CT ===
# For censored jobs, predict CT using OLS from uncensored,
# but add the conditional expectation of the error term (E[u|u > ct_obs - Xb])
print("=== Approach 1: Heckman-style correction for imputed CT ===")
job_data = df.groupby('job_id').agg({
    'ct_obs': 'first', 'censor': 'first', 'exp': 'first',
    'ed_yrs': 'first', 'married_d': 'first', 'union': 'first', 'smsa': 'first',
}).reset_index()

uncensored = job_data[job_data['censor'] == 0]
pred_vars = ['exp', 'ed_yrs', 'married_d', 'union', 'smsa']
ols_ct = sm.OLS(uncensored['ct_obs'], sm.add_constant(uncensored[pred_vars])).fit()
sigma = np.sqrt(ols_ct.mse_resid)
print(f"  OLS sigma: {sigma:.3f}")

# For censored jobs: E[CT | CT > ct_obs] = Xb + sigma * phi(z) / (1 - Phi(z))
# where z = (ct_obs - Xb) / sigma
job_data['pred_ct_ols'] = ols_ct.predict(sm.add_constant(job_data[pred_vars]))
job_data['z'] = (job_data['ct_obs'] - job_data['pred_ct_ols']) / sigma
job_data['imr'] = stats.norm.pdf(job_data['z']) / (1 - stats.norm.cdf(job_data['z']))
job_data['pred_ct_heck'] = (job_data['pred_ct_ols'] + sigma * job_data['imr']).clip(lower=1)

# For uncensored: use observed
job_data.loc[job_data['censor'] == 0, 'pred_ct_heck'] = job_data.loc[job_data['censor'] == 0, 'ct_obs']

print(f"  Mean imp CT (OLS): {job_data['pred_ct_ols'].mean():.2f}")
print(f"  Mean imp CT (Heckman): {job_data['pred_ct_heck'].mean():.2f}")
print(f"  Mean imp CT (censored only, OLS): {job_data.loc[job_data['censor']==1, 'pred_ct_ols'].mean():.2f}")
print(f"  Mean imp CT (censored only, Heck): {job_data.loc[job_data['censor']==1, 'pred_ct_heck'].mean():.2f}")

df['imp_ct_heck'] = df['job_id'].map(job_data.set_index('job_id')['pred_ct_heck'])
df['imp_ct_heck_x_exp_sq'] = df['imp_ct_heck'] * df['exp_sq']
df['imp_ct_heck_x_tenure'] = df['imp_ct_heck'] * df['tenure_var']

# Also keep regular OLS imputed CT
job_data['pred_ct_ols_clip'] = job_data['pred_ct_ols'].clip(lower=1)
job_data.loc[job_data['censor'] == 0, 'pred_ct_ols_clip'] = job_data.loc[job_data['censor'] == 0, 'ct_obs']
df['imp_ct'] = df['job_id'].map(job_data.set_index('job_id')['pred_ct_ols_clip'])
df['imp_ct_x_exp_sq'] = df['imp_ct'] * df['exp_sq']
df['imp_ct_x_tenure'] = df['imp_ct'] * df['tenure_var']
df['ct_x_exp_sq'] = df['ct_obs'] * df['exp_sq']
df['ct_x_tenure'] = df['ct_obs'] * df['tenure_var']

all_vars = ['y', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor', 'imp_ct', 'imp_ct_heck'] + control_vars
sample = df.dropna(subset=all_vars).copy()
sample = sample[sample['exp'] <= 36].copy()
y = sample['y']
N = len(sample)
print(f"  N: {N}")

# Run with Heckman imputed CT
m4h = sm.OLS(y, sm.add_constant(sample[base + ['imp_ct_heck'] + control_vars])).fit()
m5h = sm.OLS(y, sm.add_constant(sample[base + ['imp_ct_heck', 'imp_ct_heck_x_exp_sq', 'imp_ct_heck_x_tenure'] + control_vars])).fit()

# Compare with regular OLS imputed CT
m4r = sm.OLS(y, sm.add_constant(sample[base + ['imp_ct'] + control_vars])).fit()
m5r = sm.OLS(y, sm.add_constant(sample[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit()

print(f"\n  Heckman imp_ct col(4): coef={m4h.params['imp_ct_heck']:.6f}, se={m4h.bse['imp_ct_heck']:.6f}, t={abs(m4h.params['imp_ct_heck']/m4h.bse['imp_ct_heck']):.3f}")
print(f"  Regular imp_ct col(4): coef={m4r.params['imp_ct']:.6f}, se={m4r.bse['imp_ct']:.6f}, t={abs(m4r.params['imp_ct']/m4r.bse['imp_ct']):.3f}")
print(f"  Heckman imp_ct col(5): coef={m5h.params['imp_ct_heck']:.6f}, se={m5h.bse['imp_ct_heck']:.6f}, t={abs(m5h.params['imp_ct_heck']/m5h.bse['imp_ct_heck']):.3f}")
print(f"  Regular imp_ct col(5): coef={m5r.params['imp_ct']:.6f}, se={m5r.bse['imp_ct']:.6f}, t={abs(m5r.params['imp_ct']/m5r.bse['imp_ct']):.3f}")

# Check if SE tolerance still holds for Heckman imp_ct
print(f"  Heckman SE col(4): {m4h.bse['imp_ct_heck']:.6f} vs target 0.0036 (diff={abs(m4h.bse['imp_ct_heck']-0.0036):.6f})")
print(f"  Heckman SE col(5): {m5h.bse['imp_ct_heck']:.6f} vs target 0.0042 (diff={abs(m5h.bse['imp_ct_heck']-0.0042):.6f})")


# === Approach 2: What if imputed CT uses log(ct_obs) as dependent var? ===
print("\n=== Approach 2: Impute log(CT) then exponentiate ===")
ols_logct = sm.OLS(np.log(uncensored['ct_obs']), sm.add_constant(uncensored[pred_vars])).fit()
job_data['pred_logct'] = ols_logct.predict(sm.add_constant(job_data[pred_vars]))
job_data['pred_ct_logexp'] = np.exp(job_data['pred_logct']).clip(lower=1)
job_data.loc[job_data['censor'] == 0, 'pred_ct_logexp'] = job_data.loc[job_data['censor'] == 0, 'ct_obs']

df['imp_ct_log'] = df['job_id'].map(job_data.set_index('job_id')['pred_ct_logexp'])
df['imp_ct_log_x_exp_sq'] = df['imp_ct_log'] * df['exp_sq']
df['imp_ct_log_x_tenure'] = df['imp_ct_log'] * df['tenure_var']

sample['imp_ct_log'] = df.loc[sample.index, 'imp_ct_log']
sample['imp_ct_log_x_exp_sq'] = df.loc[sample.index, 'imp_ct_log_x_exp_sq']
sample['imp_ct_log_x_tenure'] = df.loc[sample.index, 'imp_ct_log_x_tenure']

m4l = sm.OLS(y, sm.add_constant(sample[base + ['imp_ct_log'] + control_vars])).fit()
m5l = sm.OLS(y, sm.add_constant(sample[base + ['imp_ct_log', 'imp_ct_log_x_exp_sq', 'imp_ct_log_x_tenure'] + control_vars])).fit()

print(f"  Log-exp imp_ct col(4): coef={m4l.params['imp_ct_log']:.6f}, t={abs(m4l.params['imp_ct_log']/m4l.bse['imp_ct_log']):.3f}, se={m4l.bse['imp_ct_log']:.6f}")
print(f"  Log-exp imp_ct col(5): coef={m5l.params['imp_ct_log']:.6f}, t={abs(m5l.params['imp_ct_log']/m5l.bse['imp_ct_log']):.3f}, se={m5l.bse['imp_ct_log']:.6f}")


# === Approach 3: What if we use median regression (LAD) to impute CT? ===
print("\n=== Approach 3: Median regression (LAD) for imputed CT ===")
from statsmodels.regression.quantile_regression import QuantReg
qr_ct = QuantReg(uncensored['ct_obs'], sm.add_constant(uncensored[pred_vars])).fit(q=0.5)
job_data['pred_ct_qr'] = qr_ct.predict(sm.add_constant(job_data[pred_vars])).clip(lower=1)
job_data.loc[job_data['censor'] == 0, 'pred_ct_qr'] = job_data.loc[job_data['censor'] == 0, 'ct_obs']

df['imp_ct_qr'] = df['job_id'].map(job_data.set_index('job_id')['pred_ct_qr'])
df['imp_ct_qr_x_exp_sq'] = df['imp_ct_qr'] * df['exp_sq']
df['imp_ct_qr_x_tenure'] = df['imp_ct_qr'] * df['tenure_var']

sample['imp_ct_qr'] = df.loc[sample.index, 'imp_ct_qr']
sample['imp_ct_qr_x_exp_sq'] = df.loc[sample.index, 'imp_ct_qr_x_exp_sq']
sample['imp_ct_qr_x_tenure'] = df.loc[sample.index, 'imp_ct_qr_x_tenure']

m4q = sm.OLS(y, sm.add_constant(sample[base + ['imp_ct_qr'] + control_vars])).fit()
m5q = sm.OLS(y, sm.add_constant(sample[base + ['imp_ct_qr', 'imp_ct_qr_x_exp_sq', 'imp_ct_qr_x_tenure'] + control_vars])).fit()

print(f"  QR imp_ct col(4): coef={m4q.params['imp_ct_qr']:.6f}, t={abs(m4q.params['imp_ct_qr']/m4q.bse['imp_ct_qr']):.3f}, se={m4q.bse['imp_ct_qr']:.6f}")
print(f"  QR imp_ct col(5): coef={m5q.params['imp_ct_qr']:.6f}, t={abs(m5q.params['imp_ct_qr']/m5q.bse['imp_ct_qr']):.3f}, se={m5q.bse['imp_ct_qr']:.6f}")


# === Approach 4: Use BOTH ct_obs and imp_ct as separate vars ===
# Wait, paper uses either ct_obs (cols 2-3) or imp_ct (cols 4-5), not both.
# But what if for cols 4-5, we use imp_ct for ALL jobs (not just censored)?
print("\n=== Approach 4: Use predicted CT for ALL jobs (no observed CT substitution) ===")
job_data['pred_ct_all'] = ols_ct.predict(sm.add_constant(job_data[pred_vars])).clip(lower=1)
# Don't substitute observed for uncensored
df['imp_ct_all'] = df['job_id'].map(job_data.set_index('job_id')['pred_ct_all'])
df['imp_ct_all_x_exp_sq'] = df['imp_ct_all'] * df['exp_sq']
df['imp_ct_all_x_tenure'] = df['imp_ct_all'] * df['tenure_var']

sample['imp_ct_all'] = df.loc[sample.index, 'imp_ct_all']
sample['imp_ct_all_x_exp_sq'] = df.loc[sample.index, 'imp_ct_all_x_exp_sq']
sample['imp_ct_all_x_tenure'] = df.loc[sample.index, 'imp_ct_all_x_tenure']

m4a = sm.OLS(y, sm.add_constant(sample[base + ['imp_ct_all'] + control_vars])).fit()
m5a = sm.OLS(y, sm.add_constant(sample[base + ['imp_ct_all', 'imp_ct_all_x_exp_sq', 'imp_ct_all_x_tenure'] + control_vars])).fit()

print(f"  All-predicted imp_ct col(4): coef={m4a.params['imp_ct_all']:.6f}, t={abs(m4a.params['imp_ct_all']/m4a.bse['imp_ct_all']):.3f}, se={m4a.bse['imp_ct_all']:.6f}")
print(f"  All-predicted imp_ct col(5): coef={m5a.params['imp_ct_all']:.6f}, t={abs(m5a.params['imp_ct_all']/m5a.bse['imp_ct_all']):.3f}, se={m5a.bse['imp_ct_all']:.6f}")
print(f"  SE col(4): {m4a.bse['imp_ct_all']:.6f} vs target 0.0036 (diff={abs(m4a.bse['imp_ct_all']-0.0036):.6f})")


# === Approach 5: WLS with person-count weights ===
print("\n=== Approach 5: WLS with inverse person-count weights ===")
person_counts = sample.groupby('person_id').size()
sample['person_weight'] = 1.0 / sample['person_id'].map(person_counts)

m1w = sm.WLS(y, sm.add_constant(sample[base + control_vars]), weights=sample['person_weight']).fit()
m2w = sm.WLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor'] + control_vars]), weights=sample['person_weight']).fit()
m3w = sm.WLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars]), weights=sample['person_weight']).fit()
m4w = sm.WLS(y, sm.add_constant(sample[base + ['imp_ct'] + control_vars]), weights=sample['person_weight']).fit()
m5w = sm.WLS(y, sm.add_constant(sample[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars]), weights=sample['person_weight']).fit()

print(f"  WLS tenure col(1): coef={m1w.params['tenure_var']:.6f}, t={abs(m1w.params['tenure_var']/m1w.bse['tenure_var']):.2f}, se={m1w.bse['tenure_var']:.6f}")
print(f"  WLS tenure col(2): coef={m2w.params['tenure_var']:.6f}, t={abs(m2w.params['tenure_var']/m2w.bse['tenure_var']):.3f}, se={m2w.bse['tenure_var']:.6f}")
print(f"  WLS censor col(3): coef={m3w.params['ct_x_censor']:.6f}, t={abs(m3w.params['ct_x_censor']/m3w.bse['ct_x_censor']):.3f}")
print(f"  WLS imp_ct col(4): coef={m4w.params['imp_ct']:.6f}, t={abs(m4w.params['imp_ct']/m4w.bse['imp_ct']):.3f}, se={m4w.bse['imp_ct']:.6f}")
print(f"  WLS R2 col(1): {m1w.rsquared:.4f} vs target 0.422")
print(f"  WLS R2 col(2): {m2w.rsquared:.4f} vs target 0.428")


# === Approach 6: Try non-inverted censor with different exp cutoff  ===
# to see if there's a config where non-inverted matches better overall
print("\n=== Approach 6: Non-inverted censor, different configs ===")
df['ct_x_censor_std'] = df['ct_obs'] * df['censor']
sample['ct_x_censor_std'] = df.loc[sample.index, 'ct_x_censor_std']

m2s = sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor_std'] + control_vars])).fit()
m3s = sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor_std', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit()

print(f"  Non-inv censor col(2): coef={m2s.params['ct_x_censor_std']:.6f}, se={m2s.bse['ct_x_censor_std']:.6f}, t={abs(m2s.params['ct_x_censor_std']/m2s.bse['ct_x_censor_std']):.3f}")
print(f"  Non-inv censor col(3): coef={m3s.params['ct_x_censor_std']:.6f}, se={m3s.bse['ct_x_censor_std']:.6f}, t={abs(m3s.params['ct_x_censor_std']/m3s.bse['ct_x_censor_std']):.3f}")
# With non-inverted: coef is POSITIVE which doesn't match -0.0025
# With inverted: coef is NEGATIVE which matches -0.0025 sign but magnitude is off
print(f"  Inv censor col(2): coef={sample.iloc[0:1].index}")  # just checking

# What if the paper's "x censor" means x_i * censor_i where x_i is some other variable?
# Like tenure * censor, or experience * censor?
print("\n=== Approach 7: Alternative 'x censor' interpretations ===")
sample['tenure_x_censor'] = sample['tenure_var'] * sample['censor']
sample['exp_x_censor'] = sample['exp'] * sample['censor']

m2tc = sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'tenure_x_censor'] + control_vars])).fit()
print(f"  tenure*censor col(2): coef={m2tc.params['tenure_x_censor']:.6f}, se={m2tc.bse['tenure_x_censor']:.6f}")

m2ec = sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'exp_x_censor'] + control_vars])).fit()
print(f"  exp*censor col(2): coef={m2ec.params['exp_x_censor']:.6f}, se={m2ec.bse['exp_x_censor']:.6f}")
