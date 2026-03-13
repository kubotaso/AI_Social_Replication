#!/usr/bin/env python3
"""Test: What if the model includes tenure^2 (even though paper says linear)?"""
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
df['tenure_sq'] = df['tenure_var'] ** 2

df['ct_obs'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)
df['ct_x_censor'] = df['ct_obs'] * df['censor']
df['ct_x_exp_sq'] = df['ct_obs'] * df['exp_sq']
df['ct_x_tenure'] = df['ct_obs'] * df['tenure_var']

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

all_vars = ['lw_blend', 'exp', 'exp_sq', 'tenure_var', 'tenure_sq', 'ct_obs', 'censor'] + control_vars
sample = df.dropna(subset=all_vars).copy()
y = sample['lw_blend']

base_vars_tsq = ['exp', 'exp_sq', 'tenure_var', 'tenure_sq']

print(f"Sample: {len(sample)}")
print("\n=== Col 1 with tenure^2 ===")
X1 = sm.add_constant(sample[base_vars_tsq + control_vars])
m1 = sm.OLS(y, X1).fit()
print(f"exp: {m1.params['exp']:.5f} (SE {m1.bse['exp']:.5f}) [target 0.0418, SE 0.0013]")
print(f"exp_sq: {m1.params['exp_sq']:.6f} (SE {m1.bse['exp_sq']:.6f}) [target -0.00079, SE 0.00003]")
print(f"tenure: {m1.params['tenure_var']:.5f} (SE {m1.bse['tenure_var']:.5f}) [target 0.0138, SE 0.0052]")
print(f"tenure_sq: {m1.params['tenure_sq']:.6f} (SE {m1.bse['tenure_sq']:.6f}) [not in table]")
print(f"R2: {m1.rsquared:.4f} [target 0.422]")

# Check: does t_sq coefficient match the paper's implicit polynomial?
# Paper Tables 3-6 use quartic tenure: T + T^2/10 + T^3/100 + T^4/1000
# Table 7 says "linear only". But maybe the 0.0138 implicitly includes
# the average effect of the quadratic terms.

print("\n=== Col 2 with tenure^2 (restricted with observed CT) ===")
X2 = sm.add_constant(sample[base_vars_tsq + ['ct_obs', 'ct_x_censor'] + control_vars])
m2 = sm.OLS(y, X2).fit()
print(f"exp: {m2.params['exp']:.5f} (SE {m2.bse['exp']:.5f})")
print(f"exp_sq: {m2.params['exp_sq']:.6f} (SE {m2.bse['exp_sq']:.6f})")
print(f"tenure: {m2.params['tenure_var']:.5f} (SE {m2.bse['tenure_var']:.5f}) [target -0.0015, SE 0.0015]")
print(f"tenure_sq: {m2.params['tenure_sq']:.6f}")
print(f"ct_obs: {m2.params['ct_obs']:.5f} (SE {m2.bse['ct_obs']:.5f}) [target 0.0165, SE 0.0016]")
print(f"ct_x_censor: {m2.params['ct_x_censor']:.5f} (SE {m2.bse['ct_x_censor']:.5f}) [target -0.0025, SE 0.0073]")

print("\n=== Col 3 with tenure^2 (unrestricted with observed CT) ===")
X3 = sm.add_constant(sample[base_vars_tsq + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])
m3 = sm.OLS(y, X3).fit()
print(f"exp: {m3.params['exp']:.5f}")
print(f"exp_sq: {m3.params['exp_sq']:.6f}")
print(f"tenure: {m3.params['tenure_var']:.5f} (SE {m3.bse['tenure_var']:.5f}) [target 0.0137, SE 0.0038]")
print(f"tenure_sq: {m3.params['tenure_sq']:.6f}")
print(f"ct_obs: {m3.params['ct_obs']:.5f}")
print(f"ct_x_censor: {m3.params['ct_x_censor']:.5f}")
print(f"ct_x_exp_sq: {m3.params['ct_x_exp_sq']:.8f} [target -0.00061]")
print(f"ct_x_tenure: {m3.params['ct_x_tenure']:.5f} [target 0.0142]")
print(f"R2: {m3.rsquared:.4f}")

# Now check significance matching
print("\n=== SIGNIFICANCE CHECK ===")
print("Col 1:")
for v in ['exp', 'exp_sq', 'tenure_var']:
    pv = m1.pvalues[v]
    sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
    print(f"  {v}: coef={m1.params[v]:.6f}, SE={m1.bse[v]:.6f}, p={pv:.4f} {sig}")

# Compare all 5 SEs for Col 1 (without vs with tenure^2)
print("\n=== SE COMPARISON: without vs with tenure^2 ===")
m1_no_tsq = sm.OLS(y, sm.add_constant(sample[['exp', 'exp_sq', 'tenure_var'] + control_vars])).fit()
print(f"{'Variable':15s} {'SE(no tsq)':>12s} {'SE(with tsq)':>12s} {'Target':>10s}")
for v, t in [('exp', 0.0013), ('exp_sq', 0.00003), ('tenure_var', 0.0052)]:
    print(f"{v:15s} {m1_no_tsq.bse[v]:12.6f} {m1.bse[v]:12.6f} {t:10.6f}")

# Also check: what about including a tenure cubic?
sample['tenure_cu'] = sample['tenure_var'] ** 3
X1c = sm.add_constant(sample[['exp', 'exp_sq', 'tenure_var', 'tenure_sq', 'tenure_cu'] + control_vars])
m1c = sm.OLS(y, X1c).fit()
print(f"\n=== With tenure + tenure^2 + tenure^3 ===")
print(f"tenure SE: {m1c.bse['tenure_var']:.5f} (target 0.0052)")
print(f"tenure coef: {m1c.params['tenure_var']:.5f}")
