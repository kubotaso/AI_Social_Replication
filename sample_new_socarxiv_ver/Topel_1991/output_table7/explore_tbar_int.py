#!/usr/bin/env python3
"""Test if the interaction terms use T_bar (average tenure) instead of CT."""
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

# T_bar = average tenure within job = (1 + ct_obs) / 2  (since all start at 1)
df['t_bar'] = (1 + df['ct_obs']) / 2.0

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

all_vars = ['lw', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor'] + control_vars
base = df.dropna(subset=all_vars).copy()
base = base[base['exp'] <= 39].copy()

print(f"N = {len(base)}")

# Test different interaction variable definitions
# Interpretation 1: ct_obs * exp_sq and ct_obs * tenure (current)
# Interpretation 2: t_bar * exp_sq and t_bar * tenure
# Interpretation 3: The unrestricted model includes (T-T_bar) and T_bar separately
#                    with T_bar interactions with X^2 and T

# Interpretation 1: current approach
base['ct_x_cen'] = base['ct_obs'] * (1 - base['censor'])
base['ct_x_esq'] = base['ct_obs'] * base['exp_sq']
base['ct_x_t'] = base['ct_obs'] * base['tenure_var']

X3_1 = sm.add_constant(base[['exp', 'exp_sq', 'tenure_var', 'ct_obs', 'ct_x_cen', 'ct_x_esq', 'ct_x_t'] + control_vars])
m3_1 = sm.OLS(base['lw'], X3_1).fit()
print(f"\nInt 1 (ct*esq, ct*t): ct_x_esq={m3_1.params['ct_x_esq']:.8f}, ct_x_t={m3_1.params['ct_x_t']:.6f}")

# Interpretation 2: t_bar interactions
base['tb_x_esq'] = base['t_bar'] * base['exp_sq']
base['tb_x_t'] = base['t_bar'] * base['tenure_var']

X3_2 = sm.add_constant(base[['exp', 'exp_sq', 'tenure_var', 'ct_obs', 'ct_x_cen', 'tb_x_esq', 'tb_x_t'] + control_vars])
m3_2 = sm.OLS(base['lw'], X3_2).fit()
print(f"Int 2 (tb*esq, tb*t): tb_x_esq={m3_2.params['tb_x_esq']:.8f}, tb_x_t={m3_2.params['tb_x_t']:.6f}")

# Interpretation 3: eq(18) with T-T_bar and T_bar as separate regressors, plus T_bar interactions
base['t_minus_tb'] = base['tenure_var'] - base['t_bar']
base['tb_x_esq2'] = base['t_bar'] * base['exp_sq']
base['tb_x_t2'] = base['t_bar'] * base['tenure_var']

X3_3 = sm.add_constant(base[['exp', 'exp_sq', 't_minus_tb', 't_bar', 'ct_obs', 'ct_x_cen', 'tb_x_esq2', 'tb_x_t2'] + control_vars])
m3_3 = sm.OLS(base['lw'], X3_3).fit()
print(f"Int 3 (eq18 T-Tb,Tb + Tb*esq, Tb*t):")
print(f"  T-Tb={m3_3.params['t_minus_tb']:.6f}, Tb={m3_3.params['t_bar']:.6f}")
print(f"  tb_x_esq={m3_3.params['tb_x_esq2']:.8f}, tb_x_t={m3_3.params['tb_x_t2']:.6f}")

# Interpretation 4: Include T, CT, CT*censor, and have SEPARATE interaction of CT with X^2 and T
# But use experience/10 for the interaction
base['exp_dec'] = base['exp'] / 10.0
base['exp_dec_sq'] = base['exp_dec'] ** 2
base['ct_x_edec_sq'] = base['ct_obs'] * base['exp_dec_sq']

X3_4 = sm.add_constant(base[['exp', 'exp_sq', 'tenure_var', 'ct_obs', 'ct_x_cen', 'ct_x_edec_sq', 'ct_x_t'] + control_vars])
m3_4 = sm.OLS(base['lw'], X3_4).fit()
print(f"\nInt 4 (ct*exp_dec^2, ct*t): ct_x_edec_sq={m3_4.params['ct_x_edec_sq']:.6f}, ct_x_t={m3_4.params['ct_x_t']:.6f}")
# Note: the coefficient on ct_x_edec_sq should be 100x the coefficient on ct_x_esq
# since (exp/10)^2 = exp^2/100

# Interpretation 5: What if the paper includes BOTH main exp_sq AND ct*exp_sq?
# In the unrestricted model, the coefficient on exp_sq ABSORBS the ct effect
# when ct*exp_sq is small. What if the paper includes exp_sq twice?
# That doesn't make sense...

# Interpretation 6: What if the interaction terms are reported PER UNIT of completed tenure?
# I.e., the "Experience^2 interaction" row shows the TOTAL effect of CT on the
# experience^2 profile, divided by mean(CT)?
# If ct_x_esq coef = -0.0000085 and mean(ct) = 4.4, then per-unit: -0.0000085/4.4 = -0.0000019
# That makes it worse, not better.

# Interpretation 7: What if the interaction uses (exp^2 - mean_exp^2) demeaned?
base['exp_sq_dm'] = base['exp_sq'] - base['exp_sq'].mean()
base['ten_dm'] = base['tenure_var'] - base['tenure_var'].mean()
base['ct_x_esq_dm'] = base['ct_obs'] * base['exp_sq_dm']
base['ct_x_t_dm'] = base['ct_obs'] * base['ten_dm']

X3_7 = sm.add_constant(base[['exp', 'exp_sq', 'tenure_var', 'ct_obs', 'ct_x_cen', 'ct_x_esq_dm', 'ct_x_t_dm'] + control_vars])
m3_7 = sm.OLS(base['lw'], X3_7).fit()
print(f"\nInt 7 (demeaned): ct_x_esq_dm={m3_7.params['ct_x_esq_dm']:.8f}, ct_x_t_dm={m3_7.params['ct_x_t_dm']:.6f}")

# Print comparison
print(f"\n=== COMPARISON (targets: esq_int=-0.00061, ten_int=0.0142) ===")
print(f"Current (ct*esq):     {m3_1.params['ct_x_esq']:>12.8f}  (600x too small)")
print(f"T_bar*esq:            {m3_2.params['tb_x_esq']:>12.8f}")
print(f"Eq18 T_bar*esq:       {m3_3.params['tb_x_esq2']:>12.8f}")
print(f"ct*(exp/10)^2:        {m3_4.params['ct_x_edec_sq']:>12.8f}")
print(f"ct*esq_demeaned:      {m3_7.params['ct_x_esq_dm']:>12.8f}")
print()
print(f"Current (ct*t):       {m3_1.params['ct_x_t']:>12.8f}  (wrong sign)")
print(f"T_bar*t:              {m3_2.params['tb_x_t']:>12.8f}")
print(f"Eq18 T_bar*t:         {m3_3.params['tb_x_t2']:>12.8f}")
print(f"ct*t_demeaned:        {m3_7.params['ct_x_t_dm']:>12.8f}")
