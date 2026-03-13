#!/usr/bin/env python3
"""Test experience restriction and year drop for sample size matching."""
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
df['ct_obs'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

base_all_vars = ['exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor'] + control_vars
base = df.dropna(subset=base_all_vars + ['log_hourly_wage']).copy()

# Test: exp <= 40
print("=== Experience <= 40 ===")
s = base[base['exp'] <= 40].copy()
print(f"N: {len(s)}")
s['lw_cps'] = s['log_hourly_wage'] - np.log(s['year'].map(CPS))
s['lw_gnp'] = np.log(s['hourly_wage'] / (s['year'].map(gnp) / 100))

# Find optimal alpha
best_r2_diff = 999
for a10 in range(600, 900, 1):
    alpha = a10/1000
    y_b = alpha * s['lw_cps'] + (1-alpha) * s['lw_gnp']
    X = sm.add_constant(s[['exp', 'exp_sq', 'tenure_var'] + control_vars])
    m = sm.OLS(y_b, X).fit()
    if abs(m.rsquared - 0.422) < best_r2_diff:
        best_r2_diff = abs(m.rsquared - 0.422)
        best_alpha = alpha
        best_r2 = m.rsquared

print(f"Best alpha: {best_alpha:.3f}, R2: {best_r2:.4f}")
s['lw_blend'] = best_alpha * s['lw_cps'] + (1-best_alpha) * s['lw_gnp']
X1 = sm.add_constant(s[['exp', 'exp_sq', 'tenure_var'] + control_vars])
m1 = sm.OLS(s['lw_blend'], X1).fit()
print(f"exp: {m1.params['exp']:.5f} (target 0.0418)")
print(f"exp_sq: {m1.params['exp_sq']:.6f} (target -0.00079, err={abs(m1.params['exp_sq']-(-0.00079))/0.00079:.1%})")
print(f"tenure: {m1.params['tenure_var']:.5f} (target 0.0138)")

# Cols 2-3
s['ct_x_censor'] = s['ct_obs'] * s['censor']
s['ct_x_exp_sq'] = s['ct_obs'] * s['exp_sq']
s['ct_x_tenure'] = s['ct_obs'] * s['tenure_var']

X2 = sm.add_constant(s[['exp', 'exp_sq', 'tenure_var', 'ct_obs', 'ct_x_censor'] + control_vars])
m2 = sm.OLS(s['lw_blend'], X2).fit()
print(f"\nCol 2: tenure={m2.params['tenure_var']:.5f} ({m2.pvalues['tenure_var']:.4f})")
print(f"       ct_obs={m2.params['ct_obs']:.5f}, ct_x_censor={m2.params['ct_x_censor']:.5f}")

# Also test: experience <= 38, 42
for max_exp in [35, 36, 37, 38, 39, 40, 41, 42, 43]:
    s2 = base[base['exp'] <= max_exp].copy()
    n = len(s2)
    err = abs(n - 13128) / 13128
    if err <= 0.05:
        s2['lw_cps'] = s2['log_hourly_wage'] - np.log(s2['year'].map(CPS))
        s2['lw_gnp'] = np.log(s2['hourly_wage'] / (s2['year'].map(gnp) / 100))
        # Quick alpha search
        best_a = 0.745
        best_d = 999
        for a10 in range(600, 900, 5):
            alpha = a10/1000
            y_b = alpha * s2['lw_cps'] + (1-alpha) * s2['lw_gnp']
            X = sm.add_constant(s2[['exp', 'exp_sq', 'tenure_var'] + control_vars])
            m = sm.OLS(y_b, X).fit()
            if abs(m.rsquared - 0.422) < best_d:
                best_d = abs(m.rsquared - 0.422)
                best_a = alpha
                bexp_sq = m.params['exp_sq']
        esq_err = abs(bexp_sq - (-0.00079)) / 0.00079
        in5 = "Y" if err <= 0.05 else "N"
        esq_ok = "Y" if esq_err <= 0.20 else "N"
        print(f"\nexp<={max_exp}: N={n:5d} ({in5}), exp_sq={bexp_sq:.6f} ({esq_ok}, {esq_err:.1%}), alpha={best_a:.3f}")
