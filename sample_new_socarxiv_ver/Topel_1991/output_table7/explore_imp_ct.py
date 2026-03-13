#!/usr/bin/env python3
"""Explore different imputed CT methods to get closer to paper values."""
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
df['ed_cat'] = pd.cut(df['ed_yrs'], bins=[-1, 11, 12, 15, 20], labels=['lt12', '12', '13_15', '16plus'])
ed_dummies = pd.get_dummies(df['ed_cat'], prefix='ed', drop_first=True, dtype=float)
for col in ed_dummies.columns:
    df[col] = ed_dummies[col]
ed_dum_cols = list(ed_dummies.columns)

CPS = {1968:1.0, 1969:1.032, 1970:1.091, 1971:1.115, 1972:1.113,
       1973:1.151, 1974:1.167, 1975:1.188, 1976:1.117, 1977:1.121,
       1978:1.133, 1979:1.128, 1980:1.128, 1981:1.109, 1982:1.103, 1983:1.089}
gnp = {1971:44.4, 1972:46.5, 1973:49.5, 1974:54.0, 1975:59.3, 1976:63.1,
       1977:67.3, 1978:72.2, 1979:78.6, 1980:85.7, 1981:94.0, 1982:100.0, 1983:103.9}

df['lw_blend'] = 0.745 * (df['log_hourly_wage'] - np.log(df['year'].map(CPS))) + \
                 0.255 * np.log(df['hourly_wage'] / (df['year'].map(gnp) / 100))

df['union'] = df['union_member'].fillna(0)
df['disability'] = df['disabled'].fillna(0)
df['smsa'] = df['lives_in_smsa'].fillna(0)
df['married_d'] = df['married'].fillna(0)
df['tenure_var'] = df['tenure_topel'].astype(float)
df['ct_obs'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

# Prepare for analysis
all_vars = ['lw_blend', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor'] + control_vars

# Method 1: Current hazard model approach (from attempt 6)
print("=== METHOD 1: Hazard model ===")
df_sorted = df.sort_values(['person_id', 'job_id', 'year']).copy()
df_sorted['next_year_in_job'] = df_sorted.groupby(['person_id', 'job_id'])['year'].shift(-1)
df_sorted['job_max_year'] = df_sorted.groupby('job_id')['year'].transform('max')
df_sorted['job_ends'] = 0
df_sorted.loc[
    (df_sorted['next_year_in_job'].isna()) & (df_sorted['job_max_year'] < 1983),
    'job_ends'
] = 1

haz_vars = ['exp', 'ed_yrs', 'married_d', 'tenure_var']
for v in haz_vars:
    df_sorted[v] = df_sorted[v].fillna(0)

haz_sample = df_sorted.dropna(subset=['job_ends']).copy()
X_haz = sm.add_constant(haz_sample[haz_vars])
y_haz = haz_sample['job_ends']
haz_model = sm.Logit(y_haz, X_haz).fit(disp=0, method='bfgs')

X_all = sm.add_constant(df_sorted[haz_vars])
df_sorted['haz_prob'] = haz_model.predict(X_all).clip(0.01, 0.99)
df_sorted['exp_rem'] = ((1 - df_sorted['haz_prob']) / df_sorted['haz_prob']).clip(upper=30)
df_sorted['imp_ct_1'] = df_sorted['tenure_var'] + df_sorted['exp_rem']
df_sorted.loc[df_sorted['censor'] == 0, 'imp_ct_1'] = df_sorted.loc[df_sorted['censor'] == 0, 'ct_obs']
print(f"  Mean imp_ct_1: {df_sorted['imp_ct_1'].mean():.2f}")
print(f"  Censored mean: {df_sorted.loc[df_sorted['censor']==1, 'imp_ct_1'].mean():.2f}")
print(f"  Uncensored mean: {df_sorted.loc[df_sorted['censor']==0, 'imp_ct_1'].mean():.2f}")

# Method 2: OLS prediction of CT from uncensored jobs
print("\n=== METHOD 2: OLS prediction from uncensored jobs ===")
# Get one row per job
job_data = df.groupby('job_id').agg({
    'ct_obs': 'first',
    'censor': 'first',
    'exp': 'first',  # experience at start of job
    'ed_yrs': 'first',
    'married_d': 'first',
    'union': 'first',
    'smsa': 'first',
    'disability': 'first',
}).reset_index()

uncensored = job_data[job_data['censor'] == 0].copy()
censored = job_data[job_data['censor'] == 1].copy()
print(f"  Uncensored jobs: {len(uncensored)}, censored: {len(censored)}")

pred_vars = ['exp', 'ed_yrs', 'married_d', 'union', 'smsa']
X_unc = sm.add_constant(uncensored[pred_vars])
y_unc = uncensored['ct_obs']
ols_ct = sm.OLS(y_unc, X_unc).fit()
print(f"  OLS R2: {ols_ct.rsquared:.3f}")

# Predict for all jobs
X_all_jobs = sm.add_constant(job_data[pred_vars])
job_data['pred_ct'] = ols_ct.predict(X_all_jobs).clip(lower=1)
# For uncensored: use observed ct
job_data.loc[job_data['censor'] == 0, 'pred_ct'] = job_data.loc[job_data['censor'] == 0, 'ct_obs']

# Map back to person-year
df['imp_ct_2'] = df['job_id'].map(job_data.set_index('job_id')['pred_ct'])
print(f"  Mean imp_ct_2: {df['imp_ct_2'].mean():.2f}")

# Method 3: For censored jobs, use ct_obs + small increment (based on hazard)
print("\n=== METHOD 3: ct_obs + small prediction for censored ===")
# For censored jobs, the completed tenure T* = T^L + R
# where T^L is last observed tenure and R is predicted remaining
# Use a simpler model: R ~ exp + ed + tenure_at_censor
last_obs = df.groupby('job_id').tail(1).copy()
cens_last = last_obs[last_obs['censor'] == 1].copy()
uncens_last = last_obs[last_obs['censor'] == 0].copy()

# For uncensored jobs, "remaining" is 0 by definition
# For prediction, use the fact that longer-tenured jobs tend to last longer
# Simple approach: predict R = 0 for censored (i.e., use ct_obs as-is)
df['imp_ct_3'] = df['ct_obs'].copy()  # Just use observed CT for everyone
print(f"  Mean imp_ct_3 (just ct_obs): {df['imp_ct_3'].mean():.2f}")

# Method 4: Clipped hazard with smaller expected remaining
print("\n=== METHOD 4: Hazard with smaller R cap ===")
for cap in [5, 10, 15, 20, 30]:
    df_sorted['exp_rem_c'] = ((1 - df_sorted['haz_prob']) / df_sorted['haz_prob']).clip(upper=cap)
    df_sorted[f'imp_ct_4_{cap}'] = df_sorted['tenure_var'] + df_sorted['exp_rem_c']
    df_sorted.loc[df_sorted['censor'] == 0, f'imp_ct_4_{cap}'] = df_sorted.loc[df_sorted['censor'] == 0, 'ct_obs']
    print(f"  cap={cap}: mean={df_sorted[f'imp_ct_4_{cap}'].mean():.2f}")

# Now test each method in the actual regression (col 4: restricted with imp_ct)
sample = df.dropna(subset=all_vars).copy()
sample['lw_blend'] = 0.745 * (sample['log_hourly_wage'] - np.log(sample['year'].map(CPS))) + \
                      0.255 * np.log(sample['hourly_wage'] / (sample['year'].map(gnp) / 100))

y = sample['lw_blend']

# Map imp_ct methods to sample
sample['imp_ct_1'] = df_sorted.loc[sample.index, 'imp_ct_1'] if 'imp_ct_1' in df_sorted.columns else np.nan
sample['imp_ct_2'] = df.loc[sample.index, 'imp_ct_2']
sample['imp_ct_3'] = df.loc[sample.index, 'imp_ct_3']

for cap in [5, 10, 15, 20, 30]:
    col = f'imp_ct_4_{cap}'
    if col in df_sorted.columns:
        sample[col] = df_sorted.loc[sample.index, col]

base = ['exp', 'exp_sq', 'tenure_var']

print("\n=== REGRESSION RESULTS (Col 4: restricted) ===")
print(f"Target: imp_ct coef = 0.0053, tenure coef = 0.0060")

for method, col in [('M1: hazard', 'imp_ct_1'),
                     ('M2: OLS', 'imp_ct_2'),
                     ('M3: ct_obs', 'imp_ct_3')]:
    s2 = sample.dropna(subset=[col])
    X4 = sm.add_constant(s2[base + [col] + control_vars])
    m4 = sm.OLS(s2['lw_blend'], X4).fit()
    print(f"\n{method}:")
    print(f"  imp_ct coef: {m4.params[col]:.6f} (target 0.0053)")
    print(f"  tenure coef: {m4.params['tenure_var']:.6f} (target 0.0060)")
    print(f"  exp_sq coef: {m4.params['exp_sq']:.6f}")

for cap in [5, 10, 15, 20, 30]:
    col = f'imp_ct_4_{cap}'
    s2 = sample.dropna(subset=[col])
    X4 = sm.add_constant(s2[base + [col] + control_vars])
    m4 = sm.OLS(s2['lw_blend'], X4).fit()
    print(f"\nM4 cap={cap}: imp_ct={m4.params[col]:.6f}, tenure={m4.params['tenure_var']:.6f}")

# Also check: what if we use ct_obs without censor correction in col 2?
# Maybe the "x censor" row is actually T^L * (1-censor) = ct for complete jobs only
print("\n=== ALTERNATIVE CENSOR DEFINITIONS ===")
# Option A: ct * (1 - censor) instead of ct * censor
sample['ct_x_nocensor'] = sample['ct_obs'] * (1 - sample['censor'])
X2a = sm.add_constant(sample[base + ['ct_obs', 'ct_x_nocensor'] + control_vars])
m2a = sm.OLS(y, X2a).fit()
print(f"A: ct_obs * (1-censor): ct={m2a.params['ct_obs']:.6f}, x_nocensor={m2a.params['ct_x_nocensor']:.6f}")

# Option B: separate censor dummy (not interaction)
sample['censor_d'] = sample['censor']
X2b = sm.add_constant(sample[base + ['ct_obs', 'censor_d'] + control_vars])
m2b = sm.OLS(y, X2b).fit()
print(f"B: separate censor dummy: ct={m2b.params['ct_obs']:.6f}, censor={m2b.params['censor_d']:.6f}")

# Option C: T^L alone plus censor dummy (no interaction)
# Paper says: "the interaction of T^L with an indicator that is one for jobs that censor"
# This IS ct * censor. But maybe we should also include a main censor effect?
X2c = sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor', 'censor_d'] + control_vars])
m2c = sm.OLS(y, X2c).fit()
print(f"C: ct + ct*censor + censor: ct={m2c.params['ct_obs']:.6f}, ct_x_censor={m2c.params['ct_x_censor']:.6f}, censor={m2c.params['censor_d']:.6f}")
