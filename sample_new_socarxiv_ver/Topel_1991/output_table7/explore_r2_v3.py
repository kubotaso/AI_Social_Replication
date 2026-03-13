#!/usr/bin/env python3
"""Explore R-squared and coefficient values with different specifications."""
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

# Education dummies
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
df['lw_blend'] = 0.5 * df['lw_cps'] + 0.5 * df['lw_gnp']

df['union'] = df['union_member'].fillna(0)
df['disability'] = df['disabled'].fillna(0)
df['smsa'] = df['lives_in_smsa'].fillna(0)
df['married_d'] = df['married'].fillna(0)

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

# Also try continuous education
control_vars_cont = ['ed_yrs'] + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

all_vars = ['exp', 'exp_sq', 'tenure_topel'] + control_vars + ['lw_cps', 'lw_gnp']
sample = df.dropna(subset=all_vars).copy()
print(f"Sample size: {len(sample)}")

print("\n=== With education DUMMIES ===")
X_base = sample[['exp', 'exp_sq', 'tenure_topel'] + control_vars]
X_base = sm.add_constant(X_base)

for name, y_col in [('CPS', 'lw_cps'), ('GNP', 'lw_gnp'), ('nominal', 'log_hourly_wage'), ('blend', 'lw_blend')]:
    m = sm.OLS(sample[y_col], X_base).fit()
    print(f'{name}: R2={m.rsquared:.4f}, exp={m.params["exp"]:.5f}, exp_sq={m.params["exp_sq"]:.6f}, tenure={m.params["tenure_topel"]:.5f}')

print("\n=== With continuous education ===")
X_cont = sample[['exp', 'exp_sq', 'tenure_topel'] + control_vars_cont]
X_cont = sm.add_constant(X_cont)
for name, y_col in [('CPS', 'lw_cps'), ('GNP', 'lw_gnp'), ('nominal', 'log_hourly_wage'), ('blend', 'lw_blend')]:
    m = sm.OLS(sample[y_col], X_cont).fit()
    print(f'{name}: R2={m.rsquared:.4f}, exp={m.params["exp"]:.5f}, exp_sq={m.params["exp_sq"]:.6f}, tenure={m.params["tenure_topel"]:.5f}')

# Try tenure starting at 0
sample['tenure0'] = sample['tenure_topel'] - 1
print("\n=== Ed dummies + tenure starting at 0 ===")
X_t0 = sample[['exp', 'exp_sq', 'tenure0'] + control_vars]
X_t0 = sm.add_constant(X_t0)
for name, y_col in [('CPS', 'lw_cps'), ('GNP', 'lw_gnp')]:
    m = sm.OLS(sample[y_col], X_t0).fit()
    print(f'{name}: R2={m.rsquared:.4f}, exp={m.params["exp"]:.5f}, exp_sq={m.params["exp_sq"]:.6f}, tenure0={m.params["tenure0"]:.5f}')

# Try without year dummies (only deflation)
control_no_yr = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols
print("\n=== No year dummies (deflation only) ===")
X_no_yr = sample[['exp', 'exp_sq', 'tenure_topel'] + control_no_yr]
X_no_yr = sm.add_constant(X_no_yr)
for name, y_col in [('CPS', 'lw_cps'), ('GNP', 'lw_gnp')]:
    m = sm.OLS(sample[y_col], X_no_yr).fit()
    print(f'{name}: R2={m.rsquared:.4f}, exp={m.params["exp"]:.5f}, exp_sq={m.params["exp_sq"]:.6f}')

# Try GNP deflation without year dummies
# This should give a lower R2 since GNP alone doesn't capture year effects well
print("\n=== Try: experience = age - 18 (all) ===")
sample['exp18'] = (sample['age'] - 18).clip(lower=1)
sample['exp18_sq'] = sample['exp18'] ** 2
X_exp18 = sample[['exp18', 'exp18_sq', 'tenure_topel'] + control_vars]
X_exp18 = sm.add_constant(X_exp18)
for name, y_col in [('CPS', 'lw_cps'), ('GNP', 'lw_gnp')]:
    m = sm.OLS(sample[y_col], X_exp18).fit()
    print(f'{name}: R2={m.rsquared:.4f}, exp18={m.params["exp18"]:.5f}, exp18_sq={m.params["exp18_sq"]:.6f}')

# What if we use experience as reported in the data (if available)?
print("\nColumn check for experience-related:")
for c in df.columns:
    if 'exp' in c.lower() or 'labor' in c.lower() or 'work' in c.lower():
        print(f"  {c}: {df[c].dtype}, range [{df[c].min()}, {df[c].max()}]")

# Try with different education bins
print("\n=== Different education dummy cutoffs ===")
for bins, labels in [
    ([-1, 8, 11, 12, 15, 20], ['lt9', '9_11', '12', '13_15', '16plus']),
    ([-1, 11, 12, 13, 15, 20], ['lt12', '12', '13', '14_15', '16plus']),
]:
    sample['ed_cat2'] = pd.cut(sample['ed_yrs'], bins=bins, labels=labels)
    ed_d2 = pd.get_dummies(sample['ed_cat2'], prefix='ed2', drop_first=True, dtype=float)
    ctrl2 = list(ed_d2.columns) + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols
    X2 = pd.concat([sample[['exp', 'exp_sq', 'tenure_topel']], ed_d2, sample[['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols]], axis=1)
    X2 = sm.add_constant(X2)
    m = sm.OLS(sample['lw_cps'], X2).fit()
    print(f'CPS, bins={bins}: R2={m.rsquared:.4f}, exp_sq={m.params["exp_sq"]:.6f}')
