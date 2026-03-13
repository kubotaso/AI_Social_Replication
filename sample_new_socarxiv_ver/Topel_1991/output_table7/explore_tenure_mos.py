#!/usr/bin/env python3
"""Explore tenure_mos as pre-panel tenure data."""
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/data/psid_panel.csv')

# tenure_mos availability by year
print('tenure_mos availability by year:')
for yr in sorted(df['year'].unique()):
    n_total = len(df[df['year'] == yr])
    n_avail = df.loc[df['year'] == yr, 'tenure_mos'].notna().sum()
    print(f'  {yr}: {n_avail}/{n_total} ({100*n_avail/n_total:.0f}%)')

# Convert tenure_mos to years
df['tenure_years_raw'] = df['tenure_mos'] / 12.0

# For each job, what's the max tenure_mos available?
print('\nJob-level analysis:')
job_max_mos = df.groupby('job_id')['tenure_mos'].max()
job_max_years = job_max_mos / 12.0
print(f'Max tenure_mos per job (in years):')
print(job_max_years.dropna().describe())

# Compare ct_obs (panel-based) with max tenure_mos/12
df['ct_obs'] = df.groupby('job_id')['tenure_topel'].transform('max')
job_data = df.groupby('job_id').agg({
    'ct_obs': 'first',
    'tenure_mos': 'max',
}).reset_index()
job_data['ct_mos'] = job_data['tenure_mos'] / 12.0
both = job_data.dropna()
print(f'\nJobs with both ct_obs and tenure_mos: {len(both)}')
print(f'ct_obs range: {both["ct_obs"].min()}-{both["ct_obs"].max()}')
print(f'ct_mos range: {both["ct_mos"].min():.1f}-{both["ct_mos"].max():.1f}')
print(f'Mean ct_obs: {both["ct_obs"].mean():.2f}')
print(f'Mean ct_mos: {both["ct_mos"].mean():.2f}')
print(f'Correlation: {both[["ct_obs", "ct_mos"]].corr().iloc[0,1]:.3f}')

# Valid tenure_mos (exclude 999 top-code)
valid = both[both['tenure_mos'] < 900]
print(f'\nValid tenure_mos (< 900 months):')
print(f'  N jobs: {len(valid)}')
print(f'  Mean ct_mos: {valid["ct_mos"].mean():.2f} years')
print(f'  Max ct_mos: {valid["ct_mos"].max():.1f} years')

# Show distribution
print(f'\nct_mos distribution (valid):')
for cutoff in [5, 10, 15, 20, 25, 30, 40, 50]:
    pct = (valid['ct_mos'] <= cutoff).mean()
    print(f'  <= {cutoff}: {100*pct:.1f}%')

# Can we construct a better tenure variable?
# For each person-year, the true tenure = (year - start_year) or tenure_mos/12
# tenure_topel starts at 1 for all jobs; tenure_mos includes pre-panel time

# Let's see how tenure_mos relates to tenure_topel within a job
print('\n=== Tenure_mos vs tenure_topel within jobs ===')
sample_jobs = df[df['tenure_mos'].notna()].groupby('job_id').first().head(5).index
for jid in sample_jobs:
    job = df[df['job_id'] == jid].sort_values('year')
    print(f'\nJob {jid}:')
    print(job[['year', 'tenure_topel', 'tenure_mos', 'tenure']].to_string(index=False))

# For the paper's method: tenure is total job tenure, not just panel tenure
# We can estimate initial tenure as: tenure_mos(first available) - panel_years_elapsed
# Or more simply: use tenure_mos/12 directly as the tenure variable

# Let me try building a "true tenure" variable
# For each person-year in a job:
# If tenure_mos is available, use tenure_mos/12
# If not, estimate from the first available tenure_mos for that job
print('\n=== Building true tenure variable ===')
df_sorted = df.sort_values(['job_id', 'year'])

# For each job, find the first available tenure_mos
job_first_mos = df_sorted.dropna(subset=['tenure_mos']).groupby('job_id').first()[['year', 'tenure_mos']]
job_first_mos.columns = ['ref_year', 'ref_tenure_mos']
df_sorted = df_sorted.merge(job_first_mos, on='job_id', how='left')

# True tenure = ref_tenure_mos/12 + (year - ref_year)
df_sorted['true_tenure'] = df_sorted['ref_tenure_mos'] / 12.0 + (df_sorted['year'] - df_sorted['ref_year'])

# Handle 999 (top-coded)
df_sorted.loc[df_sorted['ref_tenure_mos'] >= 999, 'true_tenure'] = np.nan

# Also handle 0 tenure_mos - could mean "less than 1 year" or "missing"
# For now, keep as is (0/12 = 0)

print(f'true_tenure stats (where available):')
print(df_sorted['true_tenure'].dropna().describe())

# Check: for jobs where we have true_tenure, how does it compare to tenure_topel?
both2 = df_sorted.dropna(subset=['true_tenure']).copy()
both2['tenure_diff'] = both2['true_tenure'] - both2['tenure_topel']
print(f'\ntrue_tenure - tenure_topel:')
print(both2['tenure_diff'].describe())
print(f'\nExamples with large differences:')
big = both2.nlargest(10, 'tenure_diff')
print(big[['year', 'job_id', 'person_id', 'tenure_topel', 'true_tenure', 'tenure_mos', 'ref_tenure_mos']].to_string())

# What's the new ct_obs (completed tenure using true tenure)?
df_sorted['ct_true'] = df_sorted.groupby('job_id')['true_tenure'].transform('max')
print(f'\nNew ct_true stats:')
print(df_sorted['ct_true'].dropna().describe())

# How many person-years have true_tenure?
print(f'\nPerson-years with true_tenure: {df_sorted["true_tenure"].notna().sum()} / {len(df_sorted)}')
