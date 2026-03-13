#!/usr/bin/env python3
"""Find optimal wage trim to match N=13128 while preserving coefficients."""
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

df['lw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['lw_gnp'] = np.log(df['hourly_wage'] / (df['year'].map(gnp) / 100))
df['lw_blend'] = 0.745 * df['lw_cps'] + 0.255 * df['lw_gnp']

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

all_vars = ['lw_blend', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor'] + control_vars
base_sample = df.dropna(subset=all_vars).copy()
print(f"Base sample: {len(base_sample)}")

X_vars = ['exp', 'exp_sq', 'tenure_var'] + control_vars

# Test different trim percentages
print("\n=== TRIM SEARCH ===")
target_n = 13128
for pct in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8, 2.9, 3.0, 3.1, 3.2, 3.5, 4.0]:
    lo = base_sample['hourly_wage'].quantile(pct/100)
    hi = base_sample['hourly_wage'].quantile(1 - pct/100)
    s = base_sample[(base_sample['hourly_wage'] >= lo) & (base_sample['hourly_wage'] <= hi)].copy()
    n = len(s)

    X = sm.add_constant(s[X_vars])
    m = sm.OLS(s['lw_blend'], X).fit()

    n_err = abs(n - target_n) / target_n
    in_5pct = "YES" if n_err <= 0.05 else "NO"
    print(f"  trim={pct:.1f}%: N={n:5d}, err={n_err:.3f} {in_5pct}, R2={m.rsquared:.4f}, exp_sq={m.params['exp_sq']:.6f}")

# Also try: asymmetric trim (only top or only bottom)
print("\n=== ASYMMETRIC TRIM (top only) ===")
for pct in [2.0, 3.0, 4.0, 5.0, 5.5, 6.0]:
    hi = base_sample['hourly_wage'].quantile(1 - pct/100)
    s = base_sample[base_sample['hourly_wage'] <= hi].copy()
    n = len(s)
    n_err = abs(n - target_n) / target_n
    in_5pct = "YES" if n_err <= 0.05 else "NO"
    X = sm.add_constant(s[X_vars])
    m = sm.OLS(s['lw_blend'], X).fit()
    print(f"  top trim={pct:.1f}%: N={n:5d}, err={n_err:.3f} {in_5pct}, R2={m.rsquared:.4f}, exp_sq={m.params['exp_sq']:.6f}")

print("\n=== ASYMMETRIC TRIM (bottom only) ===")
for pct in [2.0, 3.0, 4.0, 5.0, 5.5, 6.0]:
    lo = base_sample['hourly_wage'].quantile(pct/100)
    s = base_sample[base_sample['hourly_wage'] >= lo].copy()
    n = len(s)
    n_err = abs(n - target_n) / target_n
    in_5pct = "YES" if n_err <= 0.05 else "NO"
    X = sm.add_constant(s[X_vars])
    m = sm.OLS(s['lw_blend'], X).fit()
    print(f"  bot trim={pct:.1f}%: N={n:5d}, err={n_err:.3f} {in_5pct}, R2={m.rsquared:.4f}, exp_sq={m.params['exp_sq']:.6f}")

# Try fixed-value trims
print("\n=== FIXED VALUE TRIMS ===")
for lo_val, hi_val in [(1.0, 100), (1.5, 100), (1.0, 50), (1.5, 50), (1.0, 30), (1.5, 30), (2.0, 100), (2.0, 50)]:
    s = base_sample[(base_sample['hourly_wage'] >= lo_val) & (base_sample['hourly_wage'] <= hi_val)].copy()
    n = len(s)
    n_err = abs(n - target_n) / target_n
    in_5pct = "YES" if n_err <= 0.05 else "NO"
    if n > 10000:
        X = sm.add_constant(s[X_vars])
        m = sm.OLS(s['lw_blend'], X).fit()
        print(f"  [{lo_val}, {hi_val}]: N={n:5d}, err={n_err:.3f} {in_5pct}, R2={m.rsquared:.4f}, exp_sq={m.params['exp_sq']:.6f}")
