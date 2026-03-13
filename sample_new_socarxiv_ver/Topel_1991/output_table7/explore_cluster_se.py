#!/usr/bin/env python3
"""Explore clustered standard errors to match paper's SEs."""
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

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

all_vars = ['lw_blend', 'exp', 'exp_sq', 'tenure_var'] + control_vars
sample = df.dropna(subset=all_vars).copy()
y = sample['lw_blend']

base_vars = ['exp', 'exp_sq', 'tenure_var']
X = sm.add_constant(sample[base_vars + control_vars])

# Targets for Col 1
# exp SE: 0.0013, exp_sq SE: 0.00003, tenure SE: 0.0052

# Regular OLS
m_ols = sm.OLS(y, X).fit()
print("=== REGULAR OLS ===")
print(f"exp SE: {m_ols.bse['exp']:.5f} (target 0.0013)")
print(f"exp_sq SE: {m_ols.bse['exp_sq']:.6f} (target 0.00003)")
print(f"tenure SE: {m_ols.bse['tenure_var']:.5f} (target 0.0052)")

# HC0 (White)
m_hc0 = sm.OLS(y, X).fit(cov_type='HC0')
print("\n=== HC0 (White) ===")
print(f"exp SE: {m_hc0.bse['exp']:.5f}")
print(f"exp_sq SE: {m_hc0.bse['exp_sq']:.6f}")
print(f"tenure SE: {m_hc0.bse['tenure_var']:.5f}")

# HC1
m_hc1 = sm.OLS(y, X).fit(cov_type='HC1')
print("\n=== HC1 ===")
print(f"exp SE: {m_hc1.bse['exp']:.5f}")
print(f"exp_sq SE: {m_hc1.bse['exp_sq']:.6f}")
print(f"tenure SE: {m_hc1.bse['tenure_var']:.5f}")

# HC3
m_hc3 = sm.OLS(y, X).fit(cov_type='HC3')
print("\n=== HC3 ===")
print(f"exp SE: {m_hc3.bse['exp']:.5f}")
print(f"exp_sq SE: {m_hc3.bse['exp_sq']:.6f}")
print(f"tenure SE: {m_hc3.bse['tenure_var']:.5f}")

# Clustered by person
m_cl_person = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': sample['person_id']})
print("\n=== Clustered by person ===")
print(f"exp SE: {m_cl_person.bse['exp']:.5f}")
print(f"exp_sq SE: {m_cl_person.bse['exp_sq']:.6f}")
print(f"tenure SE: {m_cl_person.bse['tenure_var']:.5f}")

# Clustered by job
m_cl_job = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': sample['job_id']})
print("\n=== Clustered by job ===")
print(f"exp SE: {m_cl_job.bse['exp']:.5f}")
print(f"exp_sq SE: {m_cl_job.bse['exp_sq']:.6f}")
print(f"tenure SE: {m_cl_job.bse['tenure_var']:.5f}")

# Which SE method best matches the paper's SEs for all 3 key variables?
print("\n=== SUMMARY OF SE COMPARISON ===")
print(f"{'Method':<20s} {'exp_SE':>10s} {'exp_sq_SE':>12s} {'tenure_SE':>12s}")
print(f"{'Target':.<20s} {'0.0013':>10s} {'0.000030':>12s} {'0.0052':>12s}")
for name, m in [('OLS', m_ols), ('HC0', m_hc0), ('HC1', m_hc1), ('HC3', m_hc3),
                ('Cluster person', m_cl_person), ('Cluster job', m_cl_job)]:
    print(f"{name:<20s} {m.bse['exp']:>10.5f} {m.bse['exp_sq']:>12.6f} {m.bse['tenure_var']:>12.5f}")

# Check: the paper's tenure SE=0.0052 is 3.4x our OLS SE of 0.0015
# But exp SE=0.0013 matches our OLS SE=0.0013 almost exactly
# This is puzzling: exp SE matches but tenure SE doesn't
# Possible explanation: the paper uses a specification with more tenure terms
# (e.g., tenure^2) that increases the SE on the linear tenure term
# In Table 7, only LINEAR tenure is used. So this SE mismatch is unusual.

# Let me check what happens with tenure^2 included
sample['tenure_sq'] = sample['tenure_var'] ** 2
X_t2 = sm.add_constant(sample[base_vars + ['tenure_sq'] + control_vars])
m_t2 = sm.OLS(y, X_t2).fit()
print(f"\n=== With tenure^2 ===")
print(f"tenure SE: {m_t2.bse['tenure_var']:.5f} (target 0.0052)")
print(f"tenure coef: {m_t2.params['tenure_var']:.5f}")
print(f"tenure_sq coef: {m_t2.params['tenure_sq']:.6f}")

# Maybe the paper uses tenure starting at 0?
sample['t0'] = sample['tenure_var'] - 1
sample['t0_sq'] = sample['t0'] ** 2
X_t0 = sm.add_constant(sample[['exp', 'exp_sq', 't0'] + control_vars])
m_t0 = sm.OLS(y, X_t0).fit()
print(f"\n=== Tenure from 0 ===")
print(f"tenure SE: {m_t0.bse['t0']:.5f}")

# Compute significance for paper's values with different SE methods
print("\n=== SIGNIFICANCE IMPLICATIONS ===")
# Paper col 1: tenure = 0.0138, SE = 0.0052
# t = 0.0138/0.0052 = 2.65 -> ** (p<0.01)
# Our col 1: tenure = 0.0246, SE_ols = 0.0015 -> t = 16.4 -> ***
# Our col 1 with cluster-person: SE = 0.0046 -> t = 0.0246/0.0046 = 5.35 -> ***
# Hmm, still ***

# The paper's tenure SE of 0.0052 in col 1 might reflect that the paper's
# tenure coefficient is MUCH smaller (0.0138 vs our 0.0246).
# With the paper's coefficient and SE: t = 0.0138/0.0052 = 2.65 -> ** or ***
# So it's ** (p<0.01 but p>0.001 since t<3.29)

# For col 2: tenure = -0.0015, SE = 0.0015
# t = -0.0015/0.0015 = 1.0 -> not significant
# Our: 0.0053, SE_ols = 0.0022 -> t = 2.45 -> *

# The key issue: our tenure coefficient is too large (0.025 vs 0.014)
# because we're missing the 1968-1970 years that would add observations
# with very short tenure spells.
