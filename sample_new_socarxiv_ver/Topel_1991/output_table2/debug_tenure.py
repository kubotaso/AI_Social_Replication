#!/usr/bin/env python3
"""Debug script to understand tenure distribution and its effect on coefficients."""
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
df = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'psid_panel_full.csv'))

print("=== TENURE VARIABLE DIAGNOSTICS ===")
print(f"Total observations: {len(df)}")
print(f"Unique persons: {df['person_id'].nunique()}")
print(f"Unique jobs: {df['job_id'].nunique()}")
print()

# tenure_mos availability
valid_t = df[df['tenure_mos'].notna() & (df['tenure_mos'] < 999)]
print(f"tenure_mos available: {len(valid_t)} / {len(df)} ({100*len(valid_t)/len(df):.1f}%)")
print(f"Years with tenure_mos: {sorted(valid_t['year'].unique())}")
print(f"tenure_mos (months) stats: mean={valid_t['tenure_mos'].mean():.1f}, median={valid_t['tenure_mos'].median():.1f}")
print(f"tenure_mos (years) stats: mean={valid_t['tenure_mos'].mean()/12:.1f}, median={valid_t['tenure_mos'].median()/12:.1f}")
print()

# tenure_topel stats
print(f"tenure_topel stats: mean={df['tenure_topel'].mean():.2f}, median={df['tenure_topel'].median():.0f}, max={df['tenure_topel'].max():.0f}")
print()

# How many jobs are in progress at panel start?
person_first = df.groupby('person_id')['year'].min()
job_first = df.groupby('job_id')['year'].min()
job_person = df.groupby('job_id')['person_id'].first()
in_prog = (job_first == job_person.map(person_first))
print(f"Jobs in progress at panel start: {in_prog.sum()} / {len(in_prog)} ({100*in_prog.mean():.1f}%)")

# For in-progress jobs, what's the reported tenure_mos?
ip_jobs = in_prog[in_prog].index
ip_data = df[df['job_id'].isin(ip_jobs)]
ip_tenure = ip_data.loc[ip_data['tenure_mos'].notna() & (ip_data['tenure_mos'] < 999), 'tenure_mos']
if len(ip_tenure) > 0:
    print(f"In-progress jobs tenure_mos (months): mean={ip_tenure.mean():.1f}, median={ip_tenure.median():.1f}")
    print(f"In-progress jobs tenure_mos (years): mean={ip_tenure.mean()/12:.1f}, median={ip_tenure.median()/12:.1f}")
else:
    print("No tenure_mos data for in-progress jobs")

# For new jobs
nip_jobs = in_prog[~in_prog].index
nip_data = df[df['job_id'].isin(nip_jobs)]
nip_tenure = nip_data.loc[nip_data['tenure_mos'].notna() & (nip_data['tenure_mos'] < 999), 'tenure_mos']
if len(nip_tenure) > 0:
    print(f"New jobs tenure_mos (months): mean={nip_tenure.mean():.1f}, median={nip_tenure.median():.1f}")
    print(f"New jobs tenure_mos (years): mean={nip_tenure.mean()/12:.1f}, median={nip_tenure.median()/12:.1f}")
else:
    print("No tenure_mos data for new jobs")

print()

# Now check: what's the mean tenure in the WITHIN-JOB sample?
# Using our best approach
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

df2 = df.copy()
df2['educ_raw'] = df2['education_clean'].copy()
cat_mask = ~df2['year'].isin([1975, 1976])
df2.loc[cat_mask, 'educ_raw'] = df2.loc[cat_mask, 'education_clean'].map({**EDUC_MAP, 9: np.nan})
df2.loc[df2['educ_raw'] > 17, 'educ_raw'] = 17
df2.loc[(df2['year'].isin([1975, 1976])) & (df2['education_clean'] == 9), 'educ_raw'] = np.nan

def get_fixed_educ(group):
    good = group[group['year'].isin([1975, 1976])]['educ_raw'].dropna()
    if len(good) > 0:
        return good.iloc[0]
    mapped = group['educ_raw'].dropna()
    if len(mapped) > 0:
        modes = mapped.mode()
        return modes.iloc[0] if len(modes) > 0 else mapped.median()
    return np.nan

person_educ = df2.groupby('person_id').apply(get_fixed_educ)
df2['education_fixed'] = df2['person_id'].map(person_educ)
df2 = df2[df2['education_fixed'].notna()].copy()
df2['experience'] = df2['age'] - df2['education_fixed'] - 6

# Max exp = 35
df2 = df2[df2['experience'] <= 35].copy()

# Reconstructed tenure
df2['ten_mos_clean'] = df2['tenure_mos'].copy()
df2.loc[df2['ten_mos_clean'] >= 999, 'ten_mos_clean'] = np.nan
df2 = df2.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)

job_info = df2.groupby('job_id').agg(
    first_year=('year', 'min'),
    max_tenure_mos=('ten_mos_clean', 'max'),
    person_id=('person_id', 'first'),
).reset_index()

person_first_year = df2.groupby('person_id')['year'].min()
job_info['person_first_year'] = job_info['person_id'].map(person_first_year)
job_info['in_progress'] = job_info['first_year'] == job_info['person_first_year']

df_with_tenure = df2[df2['ten_mos_clean'].notna() & (df2['ten_mos_clean'] > 0)].copy()
if len(df_with_tenure) > 0:
    job_max_idx = df_with_tenure.groupby('job_id')['ten_mos_clean'].idxmax()
    job_max_year = df_with_tenure.loc[job_max_idx][['job_id', 'year']]
    job_max_year.columns = ['job_id', 'year_of_max']
    job_info = job_info.merge(job_max_year, on='job_id', how='left')
else:
    job_info['year_of_max'] = np.nan

job_info['initial_tenure'] = 0.0
mask_ip = (job_info['in_progress'] &
           job_info['max_tenure_mos'].notna() &
           (job_info['max_tenure_mos'] > 0) &
           job_info['year_of_max'].notna())
job_info.loc[mask_ip, 'initial_tenure'] = (
    job_info.loc[mask_ip, 'max_tenure_mos'] / 12.0 -
    (job_info.loc[mask_ip, 'year_of_max'] - job_info.loc[mask_ip, 'first_year'])
)
job_info['initial_tenure'] = job_info['initial_tenure'].clip(lower=0).round()

df2 = df2.merge(job_info[['job_id', 'initial_tenure', 'first_year']].rename(
    columns={'first_year': 'job_first_year'}), on='job_id')
df2['tenure'] = df2['initial_tenure'] + (df2['year'] - df2['job_first_year'])

# Within-job
df2 = df2.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
grp = df2.groupby(['person_id', 'job_id'])
df2['prev_year'] = grp['year'].shift(1)
df2['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
df2['prev_tenure'] = grp['tenure'].shift(1)
df2['prev_experience'] = grp['experience'].shift(1)

within = df2[(df2['prev_year'].notna()) & (df2['year'] - df2['prev_year'] == 1)].copy()
within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']
within['d_exp'] = within['experience'] - within['prev_experience']
within = within[within['d_exp'] == 1].copy()

# Min spell = 2
spell_counts = within.groupby(['person_id', 'job_id']).size()
valid_spells = spell_counts[spell_counts >= 2].index
within = within.set_index(['person_id', 'job_id'])
within = within.loc[within.index.isin(valid_spells)].reset_index()

# 2-SD trim
mean_dw = within['d_log_wage'].mean()
std_dw = within['d_log_wage'].std()
within = within[(within['d_log_wage'] >= mean_dw - 2*std_dw) & (within['d_log_wage'] <= mean_dw + 2*std_dw)].copy()

print(f"=== WITHIN-JOB SAMPLE (min_spell=2, max_exp=35, 2-SD trim) ===")
print(f"N = {len(within)}")
print(f"Mean tenure: {within['tenure'].mean():.2f}")
print(f"Median tenure: {within['tenure'].median():.1f}")
print(f"Mean prev_tenure: {within['prev_tenure'].mean():.2f}")
print(f"Max tenure: {within['tenure'].max():.0f}")
print(f"Tenure distribution:")
print(within['tenure'].describe())
print()
print(f"Mean experience: {within['experience'].mean():.2f}")
print(f"Mean d_log_wage: {within['d_log_wage'].mean():.4f}")
print()

# Paper reports: mean tenure = 9.978, mean experience = 13.36
# If paper has higher mean tenure, the quadratic terms absorb more,
# and the linear coefficient is lower
print("Paper reports: mean tenure = 9.978, mean experience = 13.36")
print(f"Our data: mean tenure = {within['tenure'].mean():.3f}, mean exp = {within['experience'].mean():.2f}")
print()

# Check: what if we use tenure_mos directly (in years) as tenure?
within_t_mos = within[within['ten_mos_clean'].notna() & (within['ten_mos_clean'] > 0)]
print(f"Rows with valid tenure_mos: {len(within_t_mos)} / {len(within)}")
if len(within_t_mos) > 0:
    print(f"tenure_mos/12 mean: {within_t_mos['ten_mos_clean'].mean()/12:.2f}")
    print(f"tenure (reconstructed) mean for same rows: {within_t_mos['tenure'].mean():.2f}")

# Distribution of initial_tenure
init_t = job_info[job_info['in_progress']]['initial_tenure']
print(f"\nInitial tenure for in-progress jobs: mean={init_t.mean():.2f}, median={init_t.median():.0f}")
print(f"  Max: {init_t.max():.0f}")
print(f"  Count > 0: {(init_t > 0).sum()} / {len(init_t)}")
print(f"  Distribution:")
print(init_t.describe())
