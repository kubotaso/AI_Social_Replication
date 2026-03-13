#!/usr/bin/env python3
"""Test using raw (uncorrected) PSID tenure and tenure_mos for completed tenure."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/data/psid_panel.csv')

# Check raw tenure
print("=== RAW TENURE VARIABLE ===")
print(f"tenure: min={df['tenure'].min()}, max={df['tenure'].max()}")
print(f"Non-missing (!=999): {(df['tenure'] != 999).sum()}, missing (999): {(df['tenure'] == 999).sum()}")

# Clean raw tenure
df_raw = df.copy()
df_raw['raw_tenure'] = df_raw['tenure'].replace(999, np.nan)
print(f"\nraw_tenure after cleaning:")
print(df_raw['raw_tenure'].describe())

# Check tenure_mos
print(f"\n=== TENURE_MOS VARIABLE ===")
print(f"tenure_mos: min={df['tenure_mos'].min()}, max={df['tenure_mos'].max()}")
df_raw['raw_tenure_mos'] = df_raw['tenure_mos'].replace(999, np.nan).replace(0, np.nan)
df_raw['raw_tenure_yrs'] = df_raw['raw_tenure_mos'] / 12.0
print(f"raw_tenure_mos/12 (in years):")
print(df_raw['raw_tenure_yrs'].describe())

# Look at the first observation of each job to see initial tenure
df_first = df_raw.sort_values(['person_id', 'job_id', 'year']).groupby('job_id').first().reset_index()
print(f"\n=== FIRST OBS PER JOB ===")
print(f"tenure_topel at first obs: always {df_first['tenure_topel'].unique()}")
print(f"raw_tenure at first obs:")
print(df_first['raw_tenure'].describe())
print(f"\nraw_tenure_yrs at first obs:")
print(df_first['raw_tenure_yrs'].describe())

# For jobs starting in 1968, the raw tenure tells us about pre-panel tenure
jobs_1968 = df_first[df_first['year'] == 1968]
print(f"\n=== JOBS STARTING IN 1968 (pre-panel tenure info) ===")
print(f"N = {len(jobs_1968)}")
print(f"raw_tenure_yrs stats:")
print(jobs_1968['raw_tenure_yrs'].describe())

# What if we compute "true" completed tenure using raw tenure info?
# For a job starting with raw_tenure_yrs = 5 in 1968 and ending in 1975 (tenure_topel max = 8),
# the actual job length would be 5 + 8 - 1 = 12 years (started ~1963)
# The paper's T^L (last observed tenure) would be raw_tenure at end of job

# Let's construct T^L using raw tenure
# For each job, T^L = raw_tenure at the last observation
df_sorted = df_raw.sort_values(['person_id', 'job_id', 'year'])
df_last = df_sorted.groupby('job_id').last().reset_index()
df_last = df_last[['job_id', 'raw_tenure', 'raw_tenure_yrs']].rename(
    columns={'raw_tenure': 'TL_raw', 'raw_tenure_yrs': 'TL_raw_yrs'})

df_raw = df_raw.merge(df_last, on='job_id', how='left')

# Also try: first obs raw_tenure + (tenure_topel - 1) = adjusted tenure
# This gives the cumulative tenure including pre-panel service
df_raw['first_raw_ten'] = df_raw.groupby('job_id')['raw_tenure_yrs'].transform('first')
df_raw['adj_tenure'] = df_raw['first_raw_ten'] + (df_raw['tenure_topel'] - 1)
df_raw['adj_ct'] = df_raw.groupby('job_id')['adj_tenure'].transform('max')

print(f"\n=== ADJUSTED TENURE (raw first + topel increments) ===")
print(f"adj_tenure stats:")
print(df_raw['adj_tenure'].describe())
print(f"\nadj_ct stats:")
print(df_raw['adj_ct'].describe())

# Now test if adj_tenure/adj_ct gives better interaction terms
EDUC = {0:0, 1:3, 2:7, 3:10, 4:12, 5:12, 6:14, 7:16, 8:17, 9:17}
df_raw['ed_yrs'] = df_raw['education_clean'].copy()
for yr in df_raw['year'].unique():
    m = df_raw['year'] == yr
    if df_raw.loc[m, 'education_clean'].max() <= 9:
        df_raw.loc[m, 'ed_yrs'] = df_raw.loc[m, 'education_clean'].map(EDUC)

df_raw['exp'] = (df_raw['age'] - df_raw['ed_yrs'] - 6).clip(lower=1)
df_raw['exp_sq'] = df_raw['exp'] ** 2
df_raw['ed_cat'] = pd.cut(df_raw['ed_yrs'], bins=[-1, 11, 12, 15, 20], labels=['lt12', '12', '13_15', '16plus'])
ed_dummies = pd.get_dummies(df_raw['ed_cat'], prefix='ed', drop_first=True, dtype=float)
for col in ed_dummies.columns:
    df_raw[col] = ed_dummies[col]
ed_dum_cols = list(ed_dummies.columns)

CPS = {1968:1.0, 1969:1.032, 1970:1.091, 1971:1.115, 1972:1.113,
       1973:1.151, 1974:1.167, 1975:1.188, 1976:1.117, 1977:1.121,
       1978:1.133, 1979:1.128, 1980:1.128, 1981:1.109, 1982:1.103, 1983:1.089}
gnp = {1971:44.4, 1972:46.5, 1973:49.5, 1974:54.0, 1975:59.3, 1976:63.1,
       1977:67.3, 1978:72.2, 1979:78.6, 1980:85.7, 1981:94.0, 1982:100.0, 1983:103.9}

df_raw['union'] = df_raw['union_member'].fillna(0)
df_raw['disability'] = df_raw['disabled'].fillna(0)
df_raw['smsa'] = df_raw['lives_in_smsa'].fillna(0)
df_raw['married_d'] = df_raw['married'].fillna(0)

ALPHA = 0.750
df_raw['lw_cps'] = df_raw['log_hourly_wage'] - np.log(df_raw['year'].map(CPS))
df_raw['lw_gnp'] = np.log(df_raw['hourly_wage'] / (df_raw['year'].map(gnp) / 100))
df_raw['lw'] = ALPHA * df_raw['lw_cps'] + (1 - ALPHA) * df_raw['lw_gnp']

df_raw['censor'] = (df_raw.groupby('job_id')['year'].transform('max') >= 1983).astype(float)
df_raw['ct_topel'] = df_raw.groupby('job_id')['tenure_topel'].transform('max').astype(float)

yr_cols = [c for c in df_raw.columns if c.startswith('year_') and c != 'year_1971' and df_raw[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

# Prepare different tenure/CT versions
# Version A: topel tenure + topel CT (current approach)
# Version B: adjusted tenure (raw initial + topel increments) + adjusted CT
# Version C: topel tenure + adjusted CT (mix)

all_vars = ['lw', 'exp', 'exp_sq', 'tenure_topel', 'ct_topel', 'censor',
            'adj_tenure', 'adj_ct'] + control_vars
base = df_raw.dropna(subset=all_vars).copy()
base = base[base['exp'] <= 39].copy()
print(f"\nSample with adj_tenure: N={len(base)}")

# Version A: current approach
base['ct_x_cen_A'] = base['ct_topel'] * base['censor']
base['ct_x_esq_A'] = base['ct_topel'] * base['exp_sq']
base['ct_x_t_A'] = base['ct_topel'] * base['tenure_topel']

X3_A = sm.add_constant(base[['exp', 'exp_sq', 'tenure_topel', 'ct_topel', 'ct_x_cen_A',
                               'ct_x_esq_A', 'ct_x_t_A'] + control_vars])
m3_A = sm.OLS(base['lw'], X3_A).fit()
print(f"\nVersion A (topel/topel): R2={m3_A.rsquared:.4f}")
print(f"  tenure={m3_A.params['tenure_topel']:.6f} SE={m3_A.bse['tenure_topel']:.6f}")
print(f"  ct_x_esq={m3_A.params['ct_x_esq_A']:.8f}")
print(f"  ct_x_t={m3_A.params['ct_x_t_A']:.6f}")

# Version B: adj_tenure + adj_ct
base['ct_x_cen_B'] = base['adj_ct'] * base['censor']
base['ct_x_esq_B'] = base['adj_ct'] * base['exp_sq']
base['ct_x_t_B'] = base['adj_ct'] * base['adj_tenure']

X3_B = sm.add_constant(base[['exp', 'exp_sq', 'adj_tenure', 'adj_ct', 'ct_x_cen_B',
                               'ct_x_esq_B', 'ct_x_t_B'] + control_vars])
m3_B = sm.OLS(base['lw'], X3_B).fit()
print(f"\nVersion B (adj/adj): R2={m3_B.rsquared:.4f}")
print(f"  tenure={m3_B.params['adj_tenure']:.6f} SE={m3_B.bse['adj_tenure']:.6f}")
print(f"  ct={m3_B.params['adj_ct']:.6f}")
print(f"  ct_x_esq={m3_B.params['ct_x_esq_B']:.8f}")
print(f"  ct_x_t={m3_B.params['ct_x_t_B']:.6f}")

# Version C: topel tenure + adj_ct
base['ct_x_cen_C'] = base['adj_ct'] * base['censor']
base['ct_x_esq_C'] = base['adj_ct'] * base['exp_sq']
base['ct_x_t_C'] = base['adj_ct'] * base['tenure_topel']

X3_C = sm.add_constant(base[['exp', 'exp_sq', 'tenure_topel', 'adj_ct', 'ct_x_cen_C',
                               'ct_x_esq_C', 'ct_x_t_C'] + control_vars])
m3_C = sm.OLS(base['lw'], X3_C).fit()
print(f"\nVersion C (topel tenure + adj CT): R2={m3_C.rsquared:.4f}")
print(f"  tenure={m3_C.params['tenure_topel']:.6f} SE={m3_C.bse['tenure_topel']:.6f}")
print(f"  ct={m3_C.params['adj_ct']:.6f}")
print(f"  ct_x_esq={m3_C.params['ct_x_esq_C']:.8f}")
print(f"  ct_x_t={m3_C.params['ct_x_t_C']:.6f}")

# Column 1 with adj_tenure vs topel
X1_A = sm.add_constant(base[['exp', 'exp_sq', 'tenure_topel'] + control_vars])
X1_B = sm.add_constant(base[['exp', 'exp_sq', 'adj_tenure'] + control_vars])
m1_A = sm.OLS(base['lw'], X1_A).fit()
m1_B = sm.OLS(base['lw'], X1_B).fit()
print(f"\nCol 1 topel: tenure={m1_A.params['tenure_topel']:.6f} SE={m1_A.bse['tenure_topel']:.6f} R2={m1_A.rsquared:.4f}")
print(f"Col 1 adj:   tenure={m1_B.params['adj_tenure']:.6f} SE={m1_B.bse['adj_tenure']:.6f} R2={m1_B.rsquared:.4f}")
print(f"Target:      tenure=0.013800 SE=0.005200")
