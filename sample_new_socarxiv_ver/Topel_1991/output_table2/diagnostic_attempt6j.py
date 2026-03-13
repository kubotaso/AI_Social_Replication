#!/usr/bin/env python3
"""
Reconstruct tenure using Topel's method:
- For jobs starting within the panel: tenure starts at 0, increments by 1
- For jobs in progress at panel start: gauge starting tenure from max reported tenure

From the paper (p. 174):
"For jobs that were in progress at the beginning of a person's record,
I gauged starting tenure relative to the period in which the person
achieved his maximum reported tenure on a job."

This means: if reported tenure peaks at, say, 120 months in year 1980,
and the job ends in 1981, then the job started around 1980 - 10 = 1970.
If the person first appears in 1971, their initial tenure in 1971 is 1 year.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

df = pd.read_csv('data/psid_panel.csv')

# Fix education
df['educ_raw'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'educ_raw'] = df.loc[cat_mask, 'education_clean'].map({**EDUC_MAP, 9: np.nan})

def get_fixed_educ(group):
    good = group[group['year'].isin([1975, 1976])]['educ_raw'].dropna()
    if len(good) > 0:
        return good.iloc[0]
    mapped = group['educ_raw'].dropna()
    if len(mapped) > 0:
        modes = mapped.mode()
        return modes.iloc[0] if len(modes) > 0 else mapped.median()
    return np.nan

person_educ = df.groupby('person_id').apply(get_fixed_educ)
df['education_fixed'] = df['person_id'].map(person_educ)
df = df[df['education_fixed'].notna()].copy()
df['experience'] = df['age'] - df['education_fixed'] - 6

# Clean tenure_mos: 999+ is missing, 0 might be valid or missing
df['ten_mos_clean'] = df['tenure_mos'].copy()
df.loc[df['ten_mos_clean'] >= 999, 'ten_mos_clean'] = np.nan
# Keep 0 as valid for now (some workers truly have 0 months tenure)

# Sort by person, job, year
df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)

# For each job spell, reconstruct tenure using Topel's method
# Step 1: For each job, find the year range and max reported tenure
job_info = df.groupby('job_id').agg(
    first_year=('year', 'min'),
    last_year=('year', 'max'),
    n_years=('year', 'count'),
    max_tenure_mos=('ten_mos_clean', 'max'),
    person_id=('person_id', 'first'),
).reset_index()

# Step 2: For each person, find their first year in the panel
person_first_year = df.groupby('person_id')['year'].min()

# Step 3: Determine if job was "in progress" at panel start
# A job is "in progress" if its first observation year equals the person's first panel year
job_info['person_first_year'] = job_info['person_id'].map(person_first_year)
job_info['in_progress'] = job_info['first_year'] == job_info['person_first_year']

print(f"Total jobs: {len(job_info)}")
print(f"Jobs in progress at panel start: {job_info['in_progress'].sum()}")
print(f"Jobs starting within panel: {(~job_info['in_progress']).sum()}")

# Step 4: For in-progress jobs, compute initial tenure from max reported tenure
# max_tenure_mos is reported in months
# If max tenure is reported in year Y, and the job's first panel year is F,
# then the job started at approximately Y - max_tenure_mos/12
# So initial tenure at year F = F - (Y - max_tenure_mos/12) = max_tenure_mos/12 - (Y - F)

# Actually, simpler: if we know the max reported tenure in months at year Y_max,
# then tenure at any year Y is: max_tenure_mos/12 - (Y_max - Y)
# At the first panel year F: tenure_F = max_tenure_mos/12 - (Y_max - F)

# But we need to find Y_max for each job
# Find the year of max reported tenure for each job (only for jobs with reported tenure)
df_with_tenure = df[df['ten_mos_clean'].notna() & (df['ten_mos_clean'] > 0)].copy()
if len(df_with_tenure) > 0:
    job_max_idx = df_with_tenure.groupby('job_id')['ten_mos_clean'].idxmax()
    job_max_year = df_with_tenure.loc[job_max_idx][['job_id', 'year', 'ten_mos_clean']]
    job_max_year.columns = ['job_id', 'year_of_max', 'max_mos']
    job_info = job_info.merge(job_max_year[['job_id', 'year_of_max']], on='job_id', how='left')
else:
    job_info['year_of_max'] = np.nan

# Compute initial tenure for in-progress jobs
job_info['initial_tenure'] = 0.0  # default for jobs starting within panel
mask_ip = job_info['in_progress'] & job_info['max_tenure_mos'].notna() & (job_info['max_tenure_mos'] > 0)
job_info.loc[mask_ip, 'initial_tenure'] = (
    job_info.loc[mask_ip, 'max_tenure_mos'] / 12.0 -
    (job_info.loc[mask_ip, 'year_of_max'] - job_info.loc[mask_ip, 'first_year'])
)
# Clip negative values to 0
job_info['initial_tenure'] = job_info['initial_tenure'].clip(lower=0)

# For in-progress jobs WITHOUT reported tenure, use tenure_topel as proxy
# (assume they started at the beginning of the panel)
mask_no_report = job_info['in_progress'] & (job_info['max_tenure_mos'].isna() | (job_info['max_tenure_mos'] <= 0))
print(f"\nIn-progress jobs with reported tenure: {mask_ip.sum()}")
print(f"In-progress jobs without reported tenure: {mask_no_report.sum()}")

# Round initial tenure to nearest integer
job_info['initial_tenure'] = job_info['initial_tenure'].round()

print(f"\nInitial tenure for in-progress jobs:")
print(job_info.loc[job_info['in_progress'], 'initial_tenure'].describe())

# Step 5: Assign tenure to each observation
df = df.merge(job_info[['job_id', 'initial_tenure', 'first_year', 'in_progress']],
              on='job_id', suffixes=('', '_job'))
df['tenure_recon'] = df['initial_tenure'] + (df['year'] - df['first_year'])

print(f"\nReconstructed tenure stats (all obs):")
print(f"  Mean: {df['tenure_recon'].mean():.2f} (paper: 9.978)")
print(f"  SD: {df['tenure_recon'].std():.2f} (paper: 8.944)")
print(f"  Max: {df['tenure_recon'].max():.0f}")

# Compare with tenure_topel
print(f"\ntenure_topel stats: mean={df['tenure_topel'].mean():.2f}")

# Now do within-job differences
grp = df.groupby(['person_id', 'job_id'])
df['prev_year'] = grp['year'].shift(1)
df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
df['prev_tenure_recon'] = grp['tenure_recon'].shift(1)
df['prev_experience'] = grp['experience'].shift(1)

within = df[
    (df['prev_year'].notna()) &
    (df['year'] - df['prev_year'] == 1)
].copy()
within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']
within['d_exp'] = within['experience'] - within['prev_experience']
within['d_tenure_recon'] = within['tenure_recon'] - within['prev_tenure_recon']

# Check d_tenure should be 1
print(f"\nd_tenure_recon distribution:")
print(within['d_tenure_recon'].value_counts().sort_index())

# Filter d_exp == 1
base = within[within['d_exp'] == 1].copy()
print(f"\nAfter d_exp==1: N={len(base)}")

# 2-SD trim
m0, s0 = base['d_log_wage'].mean(), base['d_log_wage'].std()
w = base[(base['d_log_wage'] >= m0 - 2*s0) & (base['d_log_wage'] <= m0 + 2*s0)].copy()
print(f"After 2-SD trim: N={len(w)}")
print(f"Mean tenure_recon in sample: {w['tenure_recon'].mean():.2f} (paper: ~9.978)")

# Run Model 1 and Model 3
t = w['tenure_recon'].values.astype(float)
pt = w['prev_tenure_recon'].values.astype(float)
e = w['experience'].values.astype(float)
pe = w['prev_experience'].values.astype(float)

w['d_tenure'] = t - pt
w['d_tenure_sq'] = t**2 - pt**2
w['d_tenure_cu'] = t**3 - pt**3
w['d_tenure_qu'] = t**4 - pt**4
w['d_exp_sq'] = e**2 - pe**2
w['d_exp_cu'] = e**3 - pe**3
w['d_exp_qu'] = e**4 - pe**4

yd = pd.get_dummies(w['year'], prefix='yr', dtype=float)
yc = sorted(yd.columns.tolist())[1:]
y = w['d_log_wage'].values

def run_ols(y_vals, var_list):
    X = pd.concat([w[var_list].reset_index(drop=True), yd[yc].reset_index(drop=True)], axis=1)
    valid = np.isfinite(X.values).all(axis=1) & np.isfinite(y_vals)
    model = sm.OLS(y_vals[valid], X.loc[valid].values, hasconst=True).fit()
    return model, var_list + yc

def gc(m, n, v):
    if v in n: return m.params[n.index(v)], m.bse[n.index(v)]
    return None, None

m1, n1 = run_ols(y, ['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])
m3, n3 = run_ols(y, ['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                      'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])

print(f"\nModel 1:")
for var, scale, gt_c, gt_s in [
    ('d_tenure', 1, 0.1242, 0.0161),
    ('d_exp_sq', 100, -0.6051, 0.1430),
    ('d_exp_cu', 1000, 0.1460, 0.0482),
    ('d_exp_qu', 10000, 0.0131, 0.0054),
]:
    c, s = gc(m1, n1, var)
    if c is not None:
        print(f"  {var:>15s}: {c*scale:>10.4f} ({s*scale:.4f}) paper: {gt_c:>10.4f} ({gt_s:.4f})")

print(f"  R^2: {m1.rsquared:.4f} (paper: .022)")
print(f"  SE: {np.sqrt(m1.mse_resid):.4f} (paper: .218)")

print(f"\nModel 3:")
for var, scale, gt_c, gt_s in [
    ('d_tenure', 1, 0.1258, 0.0162),
    ('d_tenure_sq', 100, -0.4592, 0.1080),
    ('d_tenure_cu', 1000, 0.1846, 0.0526),
    ('d_tenure_qu', 10000, -0.0245, 0.0079),
    ('d_exp_sq', 100, -0.4067, 0.1546),
    ('d_exp_cu', 1000, 0.0989, 0.0517),
    ('d_exp_qu', 10000, 0.0089, 0.0058),
]:
    c, s = gc(m3, n3, var)
    if c is not None:
        print(f"  {var:>15s}: {c*scale:>10.4f} ({s*scale:.4f}) paper: {gt_c:>10.4f} ({gt_s:.4f})")

print(f"  R^2: {m3.rsquared:.4f} (paper: .025)")
print(f"  SE: {np.sqrt(m3.mse_resid):.4f} (paper: .218)")
print(f"  N: {int(m3.nobs)}")
