#!/usr/bin/env python3
"""
Check if the 'experience' column in the data is already properly computed.
Also check what happens with different tenure variables.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')
print("Columns:", list(df.columns))
print(f"\nExperience column stats:")
print(df['experience'].describe())
print(f"\nExperience NaN: {df['experience'].isna().sum()}")

# Compare experience column with age - education_clean - 6
df['exp_calc'] = df['age'] - df['education_clean'] - 6
print(f"\nExperience vs age-educ-6:")
print(f"  Exact match: {(df['experience'] == df['exp_calc']).sum()}")
print(f"  Different: {(df['experience'] != df['exp_calc']).sum()}")
print(f"  Correlation: {df['experience'].corr(df['exp_calc']):.4f}")

# Check d_experience column
print(f"\nd_experience column:")
print(df['d_experience'].value_counts().sort_index().head(10))

# What about tenure_topel?
print(f"\ntenure_topel stats:")
print(df['tenure_topel'].describe())

# Within-job differences using tenure_topel
df_sorted = df.sort_values(['person_id', 'job_id', 'year'])
grp = df_sorted.groupby(['person_id', 'job_id'])
df_sorted['prev_year'] = grp['year'].shift(1)
df_sorted['prev_tenure_topel'] = grp['tenure_topel'].shift(1)
df_sorted['d_tenure_topel'] = df_sorted['tenure_topel'] - df_sorted['prev_tenure_topel']

within = df_sorted[df_sorted['prev_year'].notna() & (df_sorted['year'] - df_sorted['prev_year'] == 1)]
print(f"\nWithin-job d_tenure_topel distribution:")
print(within['d_tenure_topel'].value_counts().sort_index().head(10))

# Check: does tenure_topel start at 0 or 1?
first_obs = df_sorted.groupby('job_id').first()
print(f"\nFirst obs tenure_topel distribution:")
print(first_obs['tenure_topel'].value_counts().sort_values(ascending=False).head(10))

# Check: tenure_topel for jobs in progress (first obs in panel)
person_first_year = df_sorted.groupby('person_id')['year'].transform('min')
first_panel_obs = df_sorted[df_sorted['year'] == person_first_year]
print(f"\nFirst panel obs tenure_topel distribution:")
print(first_panel_obs['tenure_topel'].describe())
print(first_panel_obs['tenure_topel'].value_counts().sort_values(ascending=False).head(10))

# What about tenure_mos?
print(f"\ntenure_mos for first panel observations:")
first_panel_valid = first_panel_obs[first_panel_obs['tenure_mos'].notna() & (first_panel_obs['tenure_mos'] < 999)]
print(first_panel_valid['tenure_mos'].describe())

# The paper says tenure >= 1 year required. If tenure_topel >= 1 is enforced,
# first obs already has at least tenure_topel = 1
# So d_tenure from tenure_topel will always be 1 for consecutive years
print(f"\nChecking: all within-job d_tenure_topel == 1?")
print(f"  d_tenure_topel == 1: {(within['d_tenure_topel'] == 1).sum()}")
print(f"  d_tenure_topel != 1: {(within['d_tenure_topel'] != 1).sum()}")

# Key question: what is the experience^2 LEVEL at which d_exp_sq is computed?
# d(X^2) = X^2 - (X-1)^2 = 2X - 1
# So d_exp_sq depends on the LEVEL of experience
# If our experience distribution is very different from the paper's,
# the polynomial coefficients will be different

print(f"\nExperience level distribution in within-job obs:")
within_exp = within['experience'].dropna()
print(f"  Mean: {within_exp.mean():.1f}")
print(f"  Median: {within_exp.median():.1f}")
print(f"  Std: {within_exp.std():.1f}")
print(f"  Min: {within_exp.min()}")
print(f"  Max: {within_exp.max()}")
print(f"\n  Percentiles:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"    {p}th: {within_exp.quantile(p/100):.1f}")
