#!/usr/bin/env python3
"""Check tenure variable properties and test alternative specifications."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/data/psid_panel.csv')

# Check if tenure_topel increases by exactly 1 each year
df_s = df.sort_values(['person_id', 'job_id', 'year'])
df_s['ten_diff'] = df_s.groupby(['person_id', 'job_id'])['tenure_topel'].diff()

print("Tenure differences within job (should be 1.0):")
print(df_s['ten_diff'].dropna().value_counts().head(10))
print(f"\nMean diff: {df_s['ten_diff'].dropna().mean():.4f}")
print(f"Std diff: {df_s['ten_diff'].dropna().std():.4f}")

# Check if there's also a raw tenure column
print("\nColumns with 'tenure' in name:")
for c in df.columns:
    if 'tenure' in c.lower():
        print(f"  {c}: min={df[c].min()}, max={df[c].max()}, mean={df[c].mean():.2f}")

# Check what happens with PSID's original tenure variable
print("\nChecking 'tenure' column (if different from tenure_topel):")
if 'tenure' in df.columns:
    print(f"  tenure: min={df['tenure'].min()}, max={df['tenure'].max()}, mean={df['tenure'].mean():.2f}")
    df_s2 = df.sort_values(['person_id', 'job_id', 'year'])
    df_s2['raw_ten_diff'] = df_s2.groupby(['person_id', 'job_id'])['tenure'].diff()
    print(f"  Raw tenure diffs:")
    print(df_s2['raw_ten_diff'].dropna().value_counts().head(10))

# Also check if there's a tenure_with_employer or similar
for c in df.columns:
    if 'ten' in c.lower() or 'senior' in c.lower() or 'dur' in c.lower():
        print(f"  {c}")

# Now the key question: is our completed tenure (max per job) the same as the paper's T^L?
# The paper says T^L is the last observed job tenure for a particular job
# Our max(tenure_topel) should equal the last observed value since tenure increases monotonically

# Verify: for each job, does max(tenure) = last year's tenure?
job_last = df_s.groupby('job_id').last().reset_index()
job_max = df.groupby('job_id')['tenure_topel'].max().reset_index()
job_max.columns = ['job_id', 'max_ten']
merged = job_last[['job_id', 'tenure_topel']].merge(job_max, on='job_id')
print(f"\nLast year tenure == max tenure: {(merged['tenure_topel'] == merged['max_ten']).all()}")

# The paper also mentions T_bar = (T^0 + T^L)/2, average tenure on the job
# For our data, T^0 = 1 for all jobs, T^L = max(tenure)
# So T_bar = (1 + max_tenure) / 2

# The critical equation (18) says:
# y = X_0*beta_1 + (T - T_bar)(beta_1 + beta_2) + T_bar(beta_1 + beta_2) + T*theta
# where T* is completed tenure
#
# If we rearrange: coefficients on T and T_bar should be separated
# T has coefficient (beta_1 + beta_2) [from the (T-T_bar) part]
# T_bar has ANOTHER coefficient
# T* has coefficient theta
#
# The RESTRICTED model (17) constrains coefficients on (T-T_bar) and T_bar to be equal
# The UNRESTRICTED model allows them to differ

# Let me try the exact eq(18) parameterization
EDUC = {0:0, 1:3, 2:7, 3:10, 4:12, 5:12, 6:14, 7:16, 8:17, 9:17}
df['ed_yrs'] = df['education_clean'].copy()
for yr in df['year'].unique():
    m_ = df['year'] == yr
    if df.loc[m_, 'education_clean'].max() <= 9:
        df.loc[m_, 'ed_yrs'] = df.loc[m_, 'education_clean'].map(EDUC)

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

ALPHA = 0.750
df['lw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['lw_gnp'] = np.log(df['hourly_wage'] / (df['year'].map(gnp) / 100))
df['lw'] = ALPHA * df['lw_cps'] + (1 - ALPHA) * df['lw_gnp']

df['ct_obs'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)
df['t_bar'] = df.groupby('job_id')['tenure_topel'].transform('mean').astype(float)

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

all_vars = ['lw', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor', 't_bar'] + control_vars
base = df.dropna(subset=all_vars).copy()
base = base[base['exp'] <= 39].copy()

# Equation (18) parameterization: separate T-T_bar and T_bar
base['t_minus_tbar'] = base['tenure_var'] - base['t_bar']

print(f"\n=== EQUATION (18) EXACT PARAMETERIZATION ===")
print(f"N = {len(base)}")

# Unrestricted model with T-T_bar and T_bar as separate regressors plus T*
# Plus interactions: T_bar * exp_sq and T_bar * tenure (or T* * exp_sq and T* * tenure)
base['tbar_x_esq'] = base['t_bar'] * base['exp_sq']
base['tbar_x_t'] = base['t_bar'] * base['tenure_var']
base['ct_x_cen'] = base['ct_obs'] * base['censor']
base['ct_x_esq'] = base['ct_obs'] * base['exp_sq']
base['ct_x_t'] = base['ct_obs'] * base['tenure_var']

# Try eq(18) with T-T_bar, T_bar, T* as separate variables
X_eq18 = sm.add_constant(base[['exp', 'exp_sq', 't_minus_tbar', 't_bar', 'ct_obs', 'ct_x_cen'] + control_vars])
m_eq18 = sm.OLS(base['lw'], X_eq18).fit()
print(f"\nEq(18) with T-T_bar and T_bar:")
for v in ['exp', 'exp_sq', 't_minus_tbar', 't_bar', 'ct_obs', 'ct_x_cen']:
    print(f"  {v:20s}: {m_eq18.params[v]:10.6f} (SE={m_eq18.bse[v]:.6f})")
print(f"  R2 = {m_eq18.rsquared:.4f}")

# And the full unrestricted with interactions on T_bar
X_eq18_full = sm.add_constant(base[['exp', 'exp_sq', 't_minus_tbar', 't_bar', 'ct_obs', 'ct_x_cen',
                                     'tbar_x_esq', 'tbar_x_t'] + control_vars])
m_eq18_full = sm.OLS(base['lw'], X_eq18_full).fit()
print(f"\nEq(18) full unrestricted with T_bar interactions:")
for v in ['exp', 'exp_sq', 't_minus_tbar', 't_bar', 'ct_obs', 'ct_x_cen', 'tbar_x_esq', 'tbar_x_t']:
    print(f"  {v:20s}: {m_eq18_full.params[v]:10.8f} (SE={m_eq18_full.bse[v]:.8f})")
print(f"  R2 = {m_eq18_full.rsquared:.4f}")

# Also try: interactions between T* (ct_obs) and the full T and X^2
# but with T_bar as a SEPARATE additional regressor
X_hybrid = sm.add_constant(base[['exp', 'exp_sq', 'tenure_var', 't_bar', 'ct_obs', 'ct_x_cen',
                                  'ct_x_esq', 'ct_x_t'] + control_vars])
m_hybrid = sm.OLS(base['lw'], X_hybrid).fit()
print(f"\nHybrid: T, T_bar, T*, T*xX^2, T*xT:")
for v in ['exp', 'exp_sq', 'tenure_var', 't_bar', 'ct_obs', 'ct_x_cen', 'ct_x_esq', 'ct_x_t']:
    print(f"  {v:20s}: {m_hybrid.params[v]:10.8f} (SE={m_hybrid.bse[v]:.8f})")
print(f"  R2 = {m_hybrid.rsquared:.4f}")
