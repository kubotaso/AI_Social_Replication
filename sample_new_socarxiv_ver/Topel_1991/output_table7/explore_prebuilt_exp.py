#!/usr/bin/env python3
"""Explore using the pre-built experience column and other specifications."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/data/psid_panel.csv')

# Check the pre-built experience column
print("Pre-built experience column stats:")
print(df['experience'].describe())
print()

# Compare with age-ed-6
EDUC = {0:0, 1:3, 2:7, 3:10, 4:12, 5:12, 6:14, 7:16, 8:17, 9:17}
df['ed_yrs'] = df['education_clean'].copy()
for yr in df['year'].unique():
    m = df['year'] == yr
    if df.loc[m, 'education_clean'].max() <= 9:
        df.loc[m, 'ed_yrs'] = df.loc[m, 'education_clean'].map(EDUC)

df['exp_calc'] = (df['age'] - df['ed_yrs'] - 6).clip(lower=1)
print("Computed exp (age-ed-6) stats:")
print(df['exp_calc'].describe())
print()
print("Correlation between pre-built and computed:", df['experience'].corr(df['exp_calc']))
print()

# Check d_experience
print("d_experience stats:")
print(df['d_experience'].describe())
print()

# See how they differ
diff = df['experience'] - df['exp_calc']
print("Difference (prebuilt - computed):")
print(diff.describe())
print()

# Try using the pre-built experience column
df['exp_pre'] = df['experience']
df['exp_pre_sq'] = df['exp_pre'] ** 2

# Education dummies
df['ed_cat'] = pd.cut(df['ed_yrs'], bins=[-1, 11, 12, 15, 20], labels=['lt12', '12', '13_15', '16plus'])
ed_dummies = pd.get_dummies(df['ed_cat'], prefix='ed', drop_first=True, dtype=float)
for col in ed_dummies.columns:
    df[col] = ed_dummies[col]
ed_dum_cols = list(ed_dummies.columns)

CPS = {1968:1.0, 1969:1.032, 1970:1.091, 1971:1.115, 1972:1.113,
       1973:1.151, 1974:1.167, 1975:1.188, 1976:1.117, 1977:1.121,
       1978:1.133, 1979:1.128, 1980:1.128, 1981:1.109, 1982:1.103, 1983:1.089}

df['lw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['union'] = df['union_member'].fillna(0)
df['disability'] = df['disabled'].fillna(0)
df['smsa'] = df['lives_in_smsa'].fillna(0)
df['married_d'] = df['married'].fillna(0)

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

# Sample
all_vars = ['lw_cps', 'exp_pre', 'exp_pre_sq', 'tenure_topel'] + control_vars
sample = df.dropna(subset=all_vars).copy()
print(f"Sample size: {len(sample)}")

# With pre-built experience
X1 = sm.add_constant(sample[['exp_pre', 'exp_pre_sq', 'tenure_topel'] + control_vars])
m1 = sm.OLS(sample['lw_cps'], X1).fit()
print(f"\nPre-built exp: R2={m1.rsquared:.4f}, exp={m1.params['exp_pre']:.5f}, exp_sq={m1.params['exp_pre_sq']:.6f}, tenure={m1.params['tenure_topel']:.5f}")

# With computed experience (age-ed-6)
sample['exp_calc'] = (sample['age'] - sample['ed_yrs'] - 6).clip(lower=1)
sample['exp_calc_sq'] = sample['exp_calc'] ** 2
X2 = sm.add_constant(sample[['exp_calc', 'exp_calc_sq', 'tenure_topel'] + control_vars])
m2 = sm.OLS(sample['lw_cps'], X2).fit()
print(f"Computed exp:  R2={m2.rsquared:.4f}, exp={m2.params['exp_calc']:.5f}, exp_sq={m2.params['exp_calc_sq']:.6f}, tenure={m2.params['tenure_topel']:.5f}")

# Try experience = age - ed_yrs - 5 (school starts at 5 not 6)
sample['exp_5'] = (sample['age'] - sample['ed_yrs'] - 5).clip(lower=1)
sample['exp_5_sq'] = sample['exp_5'] ** 2
X3 = sm.add_constant(sample[['exp_5', 'exp_5_sq', 'tenure_topel'] + control_vars])
m3 = sm.OLS(sample['lw_cps'], X3).fit()
print(f"Exp (age-ed-5): R2={m3.rsquared:.4f}, exp={m3.params['exp_5']:.5f}, exp_sq={m3.params['exp_5_sq']:.6f}")

# Also try: what if tenure_topel starts at 0?
sample['t0'] = sample['tenure_topel'] - 1
X4 = sm.add_constant(sample[['exp_calc', 'exp_calc_sq', 't0'] + control_vars])
m4 = sm.OLS(sample['lw_cps'], X4).fit()
print(f"\nTenure from 0: R2={m4.rsquared:.4f}, exp={m4.params['exp_calc']:.5f}, exp_sq={m4.params['exp_calc_sq']:.6f}, tenure0={m4.params['t0']:.5f}")

# KEY: What about using EXPERIENCE_SQ column from the data directly?
print(f"\nPre-built experience_sq stats:")
print(df['experience_sq'].describe())
print(f"Computed exp_sq stats (from sample):")
print(sample['exp_calc_sq'].describe())

# Try: no education dummies, just continuous, with pre-built experience
ctrl_cont = ['ed_yrs', 'married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols
X5 = sm.add_constant(sample[['exp_pre', 'exp_pre_sq', 'tenure_topel'] + ctrl_cont])
m5 = sm.OLS(sample['lw_cps'], X5).fit()
print(f"\nPre-built exp, continuous ed: R2={m5.rsquared:.4f}, exp={m5.params['exp_pre']:.5f}, exp_sq={m5.params['exp_pre_sq']:.6f}")

# What matters: the target R2 is 0.422. Let me try various combos systematically
print("\n=== SYSTEMATIC SEARCH for R2 near 0.42 ===")
# GNP deflation without year dummies
gnp = {1971:44.4, 1972:46.5, 1973:49.5, 1974:54.0, 1975:59.3, 1976:63.1,
       1977:67.3, 1978:72.2, 1979:78.6, 1980:85.7, 1981:94.0, 1982:100.0, 1983:103.9}
sample['lw_gnp'] = np.log(sample['hourly_wage'] / (sample['year'].map(gnp) / 100))

ctrl_no_yr = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols

# GNP, no year dummies, ed dummies
X_a = sm.add_constant(sample[['exp_calc', 'exp_calc_sq', 'tenure_topel'] + ctrl_no_yr])
m_a = sm.OLS(sample['lw_gnp'], X_a).fit()
print(f'GNP, no yr dummies, ed dummies: R2={m_a.rsquared:.4f}, exp_sq={m_a.params["exp_calc_sq"]:.6f}')

# CPS, no year dummies, ed dummies
X_b = sm.add_constant(sample[['exp_calc', 'exp_calc_sq', 'tenure_topel'] + ctrl_no_yr])
m_b = sm.OLS(sample['lw_cps'], X_b).fit()
print(f'CPS, no yr dummies, ed dummies: R2={m_b.rsquared:.4f}, exp_sq={m_b.params["exp_calc_sq"]:.6f}')

# GNP, with year dummies, ed dummies
X_c = sm.add_constant(sample[['exp_calc', 'exp_calc_sq', 'tenure_topel'] + ctrl_no_yr + yr_cols])
m_c = sm.OLS(sample['lw_gnp'], X_c).fit()
print(f'GNP, with yr dummies, ed dummies: R2={m_c.rsquared:.4f}, exp_sq={m_c.params["exp_calc_sq"]:.6f}')

# Blend of CPS and GNP
for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
    sample['lw_mix'] = alpha * sample['lw_cps'] + (1-alpha) * sample['lw_gnp']
    X_m = sm.add_constant(sample[['exp_calc', 'exp_calc_sq', 'tenure_topel'] + ctrl_no_yr + yr_cols])
    m_m = sm.OLS(sample['lw_mix'], X_m).fit()
    print(f'Blend alpha={alpha}, yr dummies: R2={m_m.rsquared:.4f}, exp_sq={m_m.params["exp_calc_sq"]:.6f}')

# What about no year dummies with nominal wage?
X_n = sm.add_constant(sample[['exp_calc', 'exp_calc_sq', 'tenure_topel'] + ctrl_no_yr])
m_n = sm.OLS(sample['log_hourly_wage'], X_n).fit()
print(f'Nominal, no yr dummies, ed dummies: R2={m_n.rsquared:.4f}, exp_sq={m_n.params["exp_calc_sq"]:.6f}')

# Try: what if the paper uses 8 census region dummies (not 4)?
# We only have 4 regions. Let's also check if we have state or division info
print("\nRegion-related columns:")
for c in df.columns:
    if 'region' in c.lower() or 'state' in c.lower() or 'divis' in c.lower():
        print(f"  {c}: nunique={df[c].nunique()}")
