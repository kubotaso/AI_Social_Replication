#!/usr/bin/env python3
"""Test different SE correction methods to match paper's SE pattern."""
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
df['ct_x_censor'] = df['ct_obs'] * df['censor']

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

all_vars = ['lw', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor'] + control_vars
base = df.dropna(subset=all_vars).copy()
base = base[base['exp'] <= 39].copy()

# Column 1 with different SE methods
X1 = sm.add_constant(base[['exp', 'exp_sq', 'tenure_var'] + control_vars])
y = base['lw']

print("=== COLUMN 1 SEs UNDER DIFFERENT METHODS ===")
print(f"Target SEs: exp=0.0013, exp_sq=0.00003, tenure=0.0052")
print()

# OLS
m_ols = sm.OLS(y, X1).fit()
print(f"OLS:           exp_SE={m_ols.bse['exp']:.5f}  esq_SE={m_ols.bse['exp_sq']:.6f}  ten_SE={m_ols.bse['tenure_var']:.5f}")

# HC0-HC3
for hc in ['HC0', 'HC1', 'HC2', 'HC3']:
    m = sm.OLS(y, X1).fit(cov_type=hc)
    print(f"{hc}:            exp_SE={m.bse['exp']:.5f}  esq_SE={m.bse['exp_sq']:.6f}  ten_SE={m.bse['tenure_var']:.5f}")

# Cluster by person
m_cp = sm.OLS(y, X1).fit(cov_type='cluster', cov_kwds={'groups': base['person_id']})
print(f"Cluster(pers): exp_SE={m_cp.bse['exp']:.5f}  esq_SE={m_cp.bse['exp_sq']:.6f}  ten_SE={m_cp.bse['tenure_var']:.5f}")

# Cluster by job
m_cj = sm.OLS(y, X1).fit(cov_type='cluster', cov_kwds={'groups': base['job_id']})
print(f"Cluster(job):  exp_SE={m_cj.bse['exp']:.5f}  esq_SE={m_cj.bse['exp_sq']:.6f}  ten_SE={m_cj.bse['tenure_var']:.5f}")

# HAC (Newey-West)
for maxlags in [1, 3, 5, 10]:
    m_hac = sm.OLS(y, X1).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    print(f"HAC(lag={maxlags:2d}):    exp_SE={m_hac.bse['exp']:.5f}  esq_SE={m_hac.bse['exp_sq']:.6f}  ten_SE={m_hac.bse['tenure_var']:.5f}")

# Now test Column 2 with inverted censor
print("\n=== COLUMN 2 SEs ===")
print(f"Target SEs: tenure=0.0015, ct=0.0016, censor_int=0.0073")

base['ct_x_uncensor'] = base['ct_obs'] * (1 - base['censor'])
X2 = sm.add_constant(base[['exp', 'exp_sq', 'tenure_var', 'ct_obs', 'ct_x_censor'] + control_vars])
X2_inv = sm.add_constant(base[['exp', 'exp_sq', 'tenure_var', 'ct_obs', 'ct_x_uncensor'] + control_vars])

m2_ols = sm.OLS(y, X2).fit()
m2_inv = sm.OLS(y, X2_inv).fit()

print(f"OLS:           ten_SE={m2_ols.bse['tenure_var']:.5f}  ct_SE={m2_ols.bse['ct_obs']:.5f}  cen_SE={m2_ols.bse['ct_x_censor']:.5f}")
print(f"OLS(inv_cen):  ten_SE={m2_inv.bse['tenure_var']:.5f}  ct_SE={m2_inv.bse['ct_obs']:.5f}  ucen_SE={m2_inv.bse['ct_x_uncensor']:.5f}")

m2_cp = sm.OLS(y, X2).fit(cov_type='cluster', cov_kwds={'groups': base['person_id']})
print(f"Cluster(pers): ten_SE={m2_cp.bse['tenure_var']:.5f}  ct_SE={m2_cp.bse['ct_obs']:.5f}  cen_SE={m2_cp.bse['ct_x_censor']:.5f}")

m2_cj = sm.OLS(y, X2).fit(cov_type='cluster', cov_kwds={'groups': base['job_id']})
print(f"Cluster(job):  ten_SE={m2_cj.bse['tenure_var']:.5f}  ct_SE={m2_cj.bse['ct_obs']:.5f}  cen_SE={m2_cj.bse['ct_x_censor']:.5f}")

# Test: what significance does clustering give?
print("\n=== SIGNIFICANCE WITH CLUSTERING (Person) FOR COL 1 ===")
for var in ['exp', 'exp_sq', 'tenure_var']:
    c_ols = m_ols.params[var]
    se_ols = m_ols.bse[var]
    se_clp = m_cp.bse[var]
    t_ols = abs(c_ols / se_ols)
    t_clp = abs(c_ols / se_clp)
    stars_ols = '***' if t_ols > 3.291 else '**' if t_ols > 2.576 else '*' if t_ols > 1.96 else ''
    stars_clp = '***' if t_clp > 3.291 else '**' if t_clp > 2.576 else '*' if t_clp > 1.96 else ''
    print(f"  {var}: OLS t={t_ols:.2f} ({stars_ols}) | Cluster t={t_clp:.2f} ({stars_clp})")

print("\n=== TARGET SIGNIFICANCE FOR COL 1 ===")
targets = [('exp', 0.0418, 0.0013), ('exp_sq', -0.00079, 0.00003), ('tenure', 0.0138, 0.0052)]
for name, c, se in targets:
    t = abs(c/se)
    stars = '***' if t > 3.291 else '**' if t > 2.576 else '*' if t > 1.96 else ''
    print(f"  {name}: t={t:.2f} ({stars})")
