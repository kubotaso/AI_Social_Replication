#!/usr/bin/env python3
"""
Test augmented completed tenure: combine raw initial tenure + topel increments.
For observations missing raw tenure, impute initial tenure from those who have it.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/data/psid_panel.csv')

# Get raw tenure in years from tenure_mos
df['raw_tenure_yrs'] = df['tenure_mos'].replace(999, np.nan).replace(0, np.nan) / 12.0

# For each job, get the first observation's raw tenure
df_sorted = df.sort_values(['person_id', 'job_id', 'year'])
first_obs = df_sorted.groupby('job_id').first()[['raw_tenure_yrs', 'age', 'year']].reset_index()
first_obs.columns = ['job_id', 'init_raw_ten', 'init_age', 'init_year']
df = df.merge(first_obs, on='job_id', how='left')

print(f"Jobs with raw initial tenure: {df['init_raw_ten'].notna().groupby(df['job_id']).first().sum()} / {df['job_id'].nunique()}")

# For jobs WITHOUT raw tenure, impute from those that have it
# Features for imputation: age at job start, year at job start
jobs_with_raw = df.groupby('job_id').first().reset_index()
has_raw = jobs_with_raw['init_raw_ten'].notna()
print(f"Jobs with raw tenure: {has_raw.sum()}")
print(f"Jobs without raw tenure: {(~has_raw).sum()}")

# Build imputation model
EDUC = {0:0, 1:3, 2:7, 3:10, 4:12, 5:12, 6:14, 7:16, 8:17, 9:17}
df['ed_yrs'] = df['education_clean'].copy()
for yr in df['year'].unique():
    m = df['year'] == yr
    if df.loc[m, 'education_clean'].max() <= 9:
        df.loc[m, 'ed_yrs'] = df.loc[m, 'education_clean'].map(EDUC)

df['exp'] = (df['age'] - df['ed_yrs'] - 6).clip(lower=1)

# For imputation, use the first obs of each job
jobs = df.groupby('job_id').first().reset_index()
has_raw_jobs = jobs[jobs['init_raw_ten'].notna()].copy()
no_raw_jobs = jobs[jobs['init_raw_ten'].isna()].copy()

# Impute raw initial tenure
imp_vars = ['init_age', 'init_year', 'ed_yrs']
X_imp = sm.add_constant(has_raw_jobs[imp_vars])
y_imp = has_raw_jobs['init_raw_ten']
imp_model = sm.OLS(y_imp, X_imp).fit()
print(f"\nImputation model R2: {imp_model.rsquared:.3f}")
print(f"Coefficients: {dict(zip(imp_model.params.index, imp_model.params.values))}")

# Predict for missing jobs
X_pred = sm.add_constant(no_raw_jobs[imp_vars])
no_raw_jobs['imp_init_ten'] = imp_model.predict(X_pred).clip(lower=0)

# Combine: use raw where available, imputed where not
init_ten_map = {}
for _, row in has_raw_jobs.iterrows():
    init_ten_map[row['job_id']] = row['init_raw_ten']
for _, row in no_raw_jobs.iterrows():
    init_ten_map[row['job_id']] = row['imp_init_ten']

df['init_tenure_yrs'] = df['job_id'].map(init_ten_map)

# Augmented tenure = initial_tenure + (topel_tenure - 1)
# topel_tenure starts at 1, so topel_tenure-1 is years elapsed since first obs
df['aug_tenure'] = df['init_tenure_yrs'] + (df['tenure_topel'] - 1)
df['aug_ct'] = df.groupby('job_id')['aug_tenure'].transform('max')

print(f"\n=== AUGMENTED TENURE STATS ===")
print(f"aug_tenure: mean={df['aug_tenure'].mean():.2f}, median={df['aug_tenure'].median():.2f}, max={df['aug_tenure'].max():.2f}")
print(f"aug_ct: mean={df['aug_ct'].mean():.2f}, median={df['aug_ct'].median():.2f}, max={df['aug_ct'].max():.2f}")
print(f"topel tenure: mean={df['tenure_topel'].mean():.2f}, max={df['tenure_topel'].max()}")
print(f"topel CT: mean={df.groupby('job_id')['tenure_topel'].transform('max').mean():.2f}")

# Now run the full model with augmented CT
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

ALPHA = 0.750
df['lw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['lw_gnp'] = np.log(df['hourly_wage'] / (df['year'].map(gnp) / 100))
df['lw'] = ALPHA * df['lw_cps'] + (1 - ALPHA) * df['lw_gnp']

df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

# Test with different tenure/CT combinations
configs = [
    ("topel_ten + topel_ct", 'tenure_topel', 'ct_topel'),
    ("topel_ten + aug_ct", 'tenure_topel', 'aug_ct'),
    ("aug_ten + aug_ct", 'aug_tenure', 'aug_ct'),
]

df['ct_topel'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)

for label, ten_col, ct_col in configs:
    all_vars = ['lw', 'exp', 'exp_sq', ten_col, ct_col, 'censor'] + control_vars
    base = df.dropna(subset=all_vars).copy()
    base = base[base['exp'] <= 39].copy()

    base['t'] = base[ten_col].astype(float)
    base['ct'] = base[ct_col].astype(float)
    base['ct_x_cen'] = base['ct'] * (1 - base['censor'])
    base['ct_x_esq'] = base['ct'] * base['exp_sq']
    base['ct_x_t'] = base['ct'] * base['t']

    # Col 1
    X1 = sm.add_constant(base[['exp', 'exp_sq', 't'] + control_vars])
    m1 = sm.OLS(base['lw'], X1).fit()

    # Col 3 (unrestricted with observed CT)
    X3 = sm.add_constant(base[['exp', 'exp_sq', 't', 'ct', 'ct_x_cen', 'ct_x_esq', 'ct_x_t'] + control_vars])
    m3 = sm.OLS(base['lw'], X3).fit()

    print(f"\n{label}: N={len(base)}")
    print(f"  Col1: tenure={m1.params['t']:.6f} (SE={m1.bse['t']:.6f}), R2={m1.rsquared:.4f}")
    print(f"  Col3: tenure={m3.params['t']:.6f}, ct={m3.params['ct']:.6f}")
    print(f"         ct_x_esq={m3.params['ct_x_esq']:.8f}, ct_x_t={m3.params['ct_x_t']:.6f}")
    print(f"         exp_sq={m3.params['exp_sq']:.6f}")
    print(f"  Targets: ct_x_esq=-0.00061, ct_x_t=0.0142, tenure=0.0137")
