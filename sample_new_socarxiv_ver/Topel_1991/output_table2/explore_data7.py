#!/usr/bin/env python3
"""Test using the full panel and proper tenure reconstruction."""
import pandas as pd
import numpy as np

# Use FULL panel which starts at 1970
df = pd.read_csv('data/psid_panel_full.csv')
print(f"Full panel: {len(df)} obs, {df['person_id'].nunique()} persons")
print(f"Years: {sorted(df['year'].unique())}")

# The full panel has 2926 persons. Paper has 1540 persons.
# The main panel (psid_panel.csv) has 2407 persons.
# What additional filtering from the full panel to the main panel reduces from 2926 to 2407?

# Check: main panel filtering
df_main = pd.read_csv('data/psid_panel.csv')
main_pids = set(df_main['person_id'].unique())
full_pids = set(df['person_id'].unique())
print(f"\nPersons in full but not main: {len(full_pids - main_pids)}")
print(f"Persons in main but not full: {len(main_pids - full_pids)}")

# People dropped from full to main - what happened?
dropped_pids = full_pids - main_pids
if len(dropped_pids) > 0:
    dropped = df[df['person_id'].isin(dropped_pids)]
    print(f"Dropped persons' data: {len(dropped)} obs")
    print(f"  Years: {sorted(dropped['year'].unique())}")
    print(f"  Mean hourly_wage: {dropped['hourly_wage'].mean():.2f}")
    # Are these from 1970 only?
    print(f"  Year distribution of dropped:")
    print(dropped['year'].value_counts().sort_index())

# Use the raw tenure in 1971-1972 to initialize tenure for pre-existing jobs
# Raw tenure in 1971 ranges 0-9 (appears to be in years)
print("\n=== RAW TENURE IN EARLY YEARS ===")
for y in [1970, 1971, 1972]:
    sub = df[df['year'] == y]
    t = sub['tenure'].dropna()
    t_valid = t[(t < 900) & (t >= 0)]
    print(f"  Year {y}: n_valid={len(t_valid)}, values={sorted(t_valid.unique())[:20]}")

# Can we use the full panel with proper tenure?
# Start with full panel, recode education, compute first differences
# Education recode
EDUC_CAT_TO_YEARS = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
df['education_years'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(
    {**EDUC_CAT_TO_YEARS, 9: np.nan}
)
df = df[df['education_years'].notna()].copy()

# Experience
df['exp_calc'] = df['age'] - df['education_years'] - 6
df['exp_calc'] = df['exp_calc'].clip(lower=0)

# Check tenure_topel in full panel: starts at 0 in 1970
# But for jobs already in progress, we need to use reported tenure
# For year 1970 (first panel year), use raw tenure as the INITIAL tenure
# Then increment by 1 each year

# Reconstruct tenure using reported tenure as starting point
# For years 1970-1972, 'tenure' column has reported tenure (0-9, in years)
# Use this as the initial value, then increment within job

# First, let's see what tenure_topel looks like in the full panel
print("\n=== TENURE_TOPEL IN FULL PANEL ===")
# This starts at 0 for each person's first observation in the panel
# For a person first observed in 1970 with 5 years of actual tenure,
# tenure_topel = 0 in 1970, 1 in 1971, 2 in 1972, etc.
# But actual tenure would be 5, 6, 7...

# We need to add the starting tenure to get the ACTUAL tenure level
# Use the raw 'tenure' column from 1970-1972 which is in years

# For each job spell, compute starting offset
df_sorted = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)

# For each person-job, get the first year's raw tenure
first_obs = df_sorted.groupby(['person_id', 'job_id']).first()

# Raw tenure (in years) is available for 1970-1972
# Convert tenure_mos to years for later years
first_obs_reset = first_obs.reset_index()
print(f"\nJob spells: {len(first_obs_reset)}")

# For years 1970-1972, raw tenure is in years (0-9)
raw_t_early = first_obs_reset[(first_obs_reset['year'].isin([1970, 1971, 1972])) &
                               (first_obs_reset['tenure'] < 900) &
                               (first_obs_reset['tenure'].notna())]
print(f"Job spells starting 1970-72 with valid raw tenure: {len(raw_t_early)}")
print(f"  Mean raw tenure: {raw_t_early['tenure'].mean():.1f}")

# For later years, use tenure_mos / 12
raw_t_later = first_obs_reset[(~first_obs_reset['year'].isin([1970, 1971, 1972])) &
                               (first_obs_reset['tenure_mos'].notna()) &
                               (first_obs_reset['tenure_mos'] < 900) &
                               (first_obs_reset['tenure_mos'] > 0)]
print(f"Job spells starting after 1972 with valid tenure_mos: {len(raw_t_later)}")
if len(raw_t_later) > 0:
    print(f"  Mean tenure_mos/12: {(raw_t_later['tenure_mos']/12).mean():.1f}")

# Now compute: For each obs, actual_tenure = tenure_topel + initial_offset
# where initial_offset = reported_tenure_at_first_obs_of_job - tenure_topel_at_first_obs_of_job

# The key insight: for Table 2 first-differenced model with d_tenure = 1:
# - Model (1) is UNAFFECTED by tenure level (d_tenure = 1 is just the constant)
# - Models (2) and (3) ARE affected because d_tenure_sq, d_tenure_cu depend on level
# - d_tenure_sq = t^2 - (t-1)^2 = 2t - 1
# - If actual t=5 but we think t=1, d_tenure_sq = 2*1-1=1 instead of 2*5-1=9

# Let's try to reconstruct actual tenure and see if it helps
# For each person-job, compute the offset
offsets = {}
for _, row in first_obs_reset.iterrows():
    pid = row['person_id']
    jid = row['job_id']
    yr = row['year']
    tp = row['tenure_topel']

    reported = np.nan
    if yr in [1970, 1971, 1972] and pd.notna(row.get('tenure')) and row['tenure'] < 900:
        reported = row['tenure']  # In years
    elif pd.notna(row.get('tenure_mos')) and row['tenure_mos'] < 900 and row['tenure_mos'] > 0:
        reported = row['tenure_mos'] / 12.0  # Convert months to years

    if pd.notna(reported):
        offset = reported - tp
    else:
        offset = 0  # Can't determine, assume panel tenure is correct

    offsets[(pid, jid)] = max(offset, 0)  # Offset should be non-negative

# Apply offsets
df_sorted['tenure_actual'] = df_sorted.apply(
    lambda r: r['tenure_topel'] + offsets.get((r['person_id'], r['job_id']), 0),
    axis=1
)

print(f"\n=== ACTUAL TENURE STATS ===")
print(f"Mean actual tenure: {df_sorted['tenure_actual'].mean():.1f}")
print(f"Mean tenure_topel: {df_sorted['tenure_topel'].mean():.1f}")
print(f"Paper Table A1 mean tenure: 9.978")

# Now compute within-job first differences with actual tenure
grp = df_sorted.groupby(['person_id', 'job_id'])
df_sorted['prev_year'] = grp['year'].shift(1)
df_sorted['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
df_sorted['prev_tenure_actual'] = grp['tenure_actual'].shift(1)
df_sorted['prev_exp'] = grp['exp_calc'].shift(1)

within = df_sorted[
    (df_sorted['prev_year'].notna()) &
    (df_sorted['year'] - df_sorted['prev_year'] == 1)
].copy()

within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']
within = within[within['d_log_wage'].between(-2, 2)].copy()
within = within[within['exp_calc'].notna() & within['prev_exp'].notna() & (within['exp_calc'] >= 1)].copy()

print(f"\nWithin-job obs (full panel): {len(within)}")
print(f"  Persons: {within['person_id'].nunique()}")
print(f"  Mean d_log_wage: {within['d_log_wage'].mean():.4f}")
print(f"  Mean actual tenure: {within['tenure_actual'].mean():.1f}")
print(f"  d_tenure check: {(within['tenure_actual'] - within['prev_tenure_actual']).unique()[:5]}")
