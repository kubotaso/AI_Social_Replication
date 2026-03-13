#!/usr/bin/env python3
"""Explore different completed tenure definitions and their impact on interaction terms."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/data/psid_panel.csv')

# Basic tenure stats
print("=== TENURE STATS ===")
print(f"tenure_topel: min={df['tenure_topel'].min()}, max={df['tenure_topel'].max()}, mean={df['tenure_topel'].mean():.2f}")

# Job-level stats
job_stats = df.groupby('job_id').agg(
    min_yr=('year','min'), max_yr=('year','max'),
    n_obs=('year','count'),
    min_ten=('tenure_topel','min'), max_ten=('tenure_topel','max'),
).reset_index()
job_stats['yr_span'] = job_stats['max_yr'] - job_stats['min_yr'] + 1
job_stats['is_censored'] = job_stats['max_yr'] >= 1983

print(f"\n=== JOB STATS ===")
print(f"N jobs: {len(job_stats)}")
print(f"Mean max_tenure: {job_stats['max_ten'].mean():.3f}")
print(f"Mean year_span: {job_stats['yr_span'].mean():.3f}")
print(f"Mean n_obs: {job_stats['n_obs'].mean():.3f}")
print(f"Censored: {job_stats['is_censored'].sum()} / {len(job_stats)}")

# Tenure at start of jobs
print(f"\nMin tenure per job (T^0):")
print(job_stats['min_ten'].describe())

# Different CT definitions
# Definition A: max(tenure_topel) per job [current approach]
# Definition B: year_span (max_yr - min_yr + 1)
# Definition C: max_tenure + 1
# Definition D: n_obs (number of panel observations)
# Definition E: T^L = last observed tenure (same as max_tenure for ascending tenure)

# Now test how these different definitions affect interaction terms
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

df['union'] = df['union_member'].fillna(0)
df['disability'] = df['disabled'].fillna(0)
df['smsa'] = df['lives_in_smsa'].fillna(0)
df['married_d'] = df['married'].fillna(0)
df['tenure_var'] = df['tenure_topel'].astype(float)

# Different CT definitions
df['ct_max_ten'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
df['ct_yr_span'] = df.groupby('job_id')['year'].transform(lambda x: x.max() - x.min() + 1).astype(float)
df['ct_n_obs'] = df.groupby('job_id')['year'].transform('count').astype(float)
df['ct_max_ten_p1'] = df['ct_max_ten'] + 1

df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)

# Try INVERTED censor: 1 = uncensored (job ended), 0 = censored (still active)
df['uncensor'] = 1 - df['censor']

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

ALPHA = 0.750
df['lw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['lw_gnp'] = np.log(df['hourly_wage'] / (df['year'].map(gnp) / 100))
df['lw'] = ALPHA * df['lw_cps'] + (1 - ALPHA) * df['lw_gnp']

all_vars = ['lw', 'exp', 'exp_sq', 'tenure_var', 'ct_max_ten', 'censor'] + control_vars
base = df.dropna(subset=all_vars).copy()
base = base[base['exp'] <= 39].copy()

print(f"\nSample: N={len(base)}")

# Test each CT definition with col(3) unrestricted model
ct_defs = {
    'max_tenure': 'ct_max_ten',
    'yr_span': 'ct_yr_span',
    'n_obs': 'ct_n_obs',
    'max_ten+1': 'ct_max_ten_p1',
}

print("\n=== COLUMN 3 (UNRESTRICTED) WITH DIFFERENT CT DEFINITIONS ===")
print(f"Target: exp_sq_int=-0.00061, tenure_int=0.0142, ct=0.0316, censor=-0.0024")

for name, ct_col in ct_defs.items():
    s = base.copy()
    s['ct'] = s[ct_col]
    s['ct_x_cen'] = s['ct'] * s['censor']
    s['ct_x_esq'] = s['ct'] * s['exp_sq']
    s['ct_x_t'] = s['ct'] * s['tenure_var']

    X = sm.add_constant(s[['exp', 'exp_sq', 'tenure_var', 'ct', 'ct_x_cen', 'ct_x_esq', 'ct_x_t'] + control_vars])
    m = sm.OLS(s['lw'], X).fit()

    print(f"\n  {name:15s}: R2={m.rsquared:.4f}")
    print(f"    ct_coef={m.params['ct']:.6f} (SE={m.bse['ct']:.6f})")
    print(f"    ct_x_cen={m.params['ct_x_cen']:.6f} (SE={m.bse['ct_x_cen']:.6f})")
    print(f"    ct_x_esq={m.params['ct_x_esq']:.8f} (SE={m.bse['ct_x_esq']:.8f})")
    print(f"    ct_x_t={m.params['ct_x_t']:.6f} (SE={m.bse['ct_x_t']:.6f})")
    print(f"    tenure={m.params['tenure_var']:.6f}")
    print(f"    exp_sq={m.params['exp_sq']:.6f}")

# Now test inverted censor
print("\n=== INVERTED CENSOR (1=uncensored, 0=censored) ===")
for name, ct_col in [('max_tenure', 'ct_max_ten')]:
    s = base.copy()
    s['ct'] = s[ct_col]
    s['ct_x_cen'] = s['ct'] * s['uncensor']  # inverted!
    s['ct_x_esq'] = s['ct'] * s['exp_sq']
    s['ct_x_t'] = s['ct'] * s['tenure_var']

    X = sm.add_constant(s[['exp', 'exp_sq', 'tenure_var', 'ct', 'ct_x_cen', 'ct_x_esq', 'ct_x_t'] + control_vars])
    m = sm.OLS(s['lw'], X).fit()

    print(f"  {name:15s} (inv censor): R2={m.rsquared:.4f}")
    print(f"    ct_coef={m.params['ct']:.6f} (SE={m.bse['ct']:.6f})")
    print(f"    ct_x_cen={m.params['ct_x_cen']:.6f} (SE={m.bse['ct_x_cen']:.6f})")
    print(f"    ct_x_esq={m.params['ct_x_esq']:.8f} (SE={m.bse['ct_x_esq']:.8f})")
    print(f"    ct_x_t={m.params['ct_x_t']:.6f} (SE={m.bse['ct_x_t']:.6f})")

# Test exp_sq scaling: what if experience is in decades?
print("\n=== DIFFERENT EXPERIENCE SCALING ===")
for scale_name, exp_scale in [('raw', 1), ('/10', 10), ('/100', 100)]:
    s = base.copy()
    s['exp_s'] = s['exp'] / exp_scale
    s['exp_sq_s'] = s['exp_s'] ** 2
    s['ct'] = s['ct_max_ten']
    s['ct_x_cen'] = s['ct'] * s['censor']
    s['ct_x_esq'] = s['ct'] * s['exp_sq_s']
    s['ct_x_t'] = s['ct'] * s['tenure_var']

    X = sm.add_constant(s[['exp_s', 'exp_sq_s', 'tenure_var', 'ct', 'ct_x_cen', 'ct_x_esq', 'ct_x_t'] + control_vars])
    m = sm.OLS(s['lw'], X).fit()

    print(f"\n  exp{exp_scale}: R2={m.rsquared:.4f}")
    print(f"    exp={m.params['exp_s']:.6f}, exp_sq={m.params['exp_sq_s']:.6f}")
    print(f"    ct_x_esq={m.params['ct_x_esq']:.8f}")
    print(f"    ct_x_t={m.params['ct_x_t']:.6f}")
    print(f"    tenure={m.params['tenure_var']:.6f}")

# Test: what if interaction is with T_bar (average tenure) instead of completed tenure?
print("\n=== INTERACTION WITH T_BAR (mean tenure on job) ===")
df['t_bar'] = df.groupby('job_id')['tenure_topel'].transform('mean').astype(float)
base['t_bar'] = df.loc[base.index, 't_bar']

s = base.copy()
s['ct'] = s['ct_max_ten']
s['ct_x_cen'] = s['ct'] * s['censor']
s['tbar_x_esq'] = s['t_bar'] * s['exp_sq']
s['tbar_x_t'] = s['t_bar'] * s['tenure_var']

X = sm.add_constant(s[['exp', 'exp_sq', 'tenure_var', 'ct', 'ct_x_cen', 'tbar_x_esq', 'tbar_x_t'] + control_vars])
m = sm.OLS(s['lw'], X).fit()
print(f"  T_bar interactions: R2={m.rsquared:.4f}")
print(f"    ct={m.params['ct']:.6f}")
print(f"    ct_x_cen={m.params['ct_x_cen']:.6f}")
print(f"    tbar_x_esq={m.params['tbar_x_esq']:.8f}")
print(f"    tbar_x_t={m.params['tbar_x_t']:.6f}")
print(f"    tenure={m.params['tenure_var']:.6f}")

# Test clustered SEs (person-level)
print("\n=== CLUSTERED SEs (PERSON) FOR COL 1 ===")
s = base.copy()
X1 = sm.add_constant(s[['exp', 'exp_sq', 'tenure_var'] + control_vars])
m1_ols = sm.OLS(s['lw'], X1).fit()
m1_clust = sm.OLS(s['lw'], X1).fit(cov_type='cluster', cov_kwds={'groups': s['person_id']})
m1_hc1 = sm.OLS(s['lw'], X1).fit(cov_type='HC1')

print(f"  OLS SE(tenure):     {m1_ols.bse['tenure_var']:.6f}")
print(f"  Cluster(person) SE: {m1_clust.bse['tenure_var']:.6f}")
print(f"  HC1 SE(tenure):     {m1_hc1.bse['tenure_var']:.6f}")
print(f"  Target SE(tenure):  0.0052")
print(f"\n  OLS SE(exp):     {m1_ols.bse['exp']:.6f}")
print(f"  Cluster(person): {m1_clust.bse['exp']:.6f}")
print(f"  Target SE(exp):  0.0013")

# Also test job-level clustering
m1_job_clust = sm.OLS(s['lw'], X1).fit(cov_type='cluster', cov_kwds={'groups': s['job_id']})
print(f"\n  Cluster(job) SE(tenure): {m1_job_clust.bse['tenure_var']:.6f}")
print(f"  Cluster(job) SE(exp):    {m1_job_clust.bse['exp']:.6f}")
