#!/usr/bin/env python3
"""Explore data for Table 2 replication."""
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')

# Check tenure distribution
print("Tenure min by year:")
for y in sorted(df['year'].unique()):
    sub = df[df['year']==y]
    print(f"  {y}: n={len(sub)}, tenure min={sub['tenure_topel'].min()}, max={sub['tenure_topel'].max()}, mean={sub['tenure_topel'].mean():.1f}")

# Check education for all years
print("\nEducation coding by year:")
for y in sorted(df['year'].unique()):
    sub = df[df['year']==y]
    vals = sorted(sub['education_clean'].dropna().unique())
    print(f"  Year {y}: range {min(vals):.0f}-{max(vals):.0f}, n_unique={len(vals)}")

# Check age range
print(f"\nAge: min={df['age'].min()}, max={df['age'].max()}, mean={df['age'].mean():.1f}")

# Check hourly_wage
print(f"\nHourly wage: min={df['hourly_wage'].min():.2f}, max={df['hourly_wage'].max():.2f}, mean={df['hourly_wage'].mean():.2f}")

# d_experience column
print(f"\nd_experience: unique={sorted(df['d_experience'].dropna().unique())[:10]}")

# Check same_emp and new_job
print(f"\nsame_emp unique: {sorted(df['same_emp'].dropna().unique())[:10]}")
print(f"new_job unique: {sorted(df['new_job'].dropna().unique())[:10]}")

# Check if there's a full panel file
import os
full_file = 'data/psid_panel_full.csv'
if os.path.exists(full_file):
    df2 = pd.read_csv(full_file)
    print(f"\nFull panel: {len(df2)} obs, {df2['person_id'].nunique()} persons")
    print(f"  Year range: {df2['year'].min()} - {df2['year'].max()}")
    print(f"  Year distribution:")
    print(df2['year'].value_counts().sort_index())
else:
    print(f"\nNo full panel file at {full_file}")

# Number of unique persons
print(f"\nPersons: {df['person_id'].nunique()}")

# Check within-job spells using person_id + job_id grouping
grp = df.groupby(['person_id', 'job_id'])
spell_lengths = grp.size()
print(f"\nJob spell length stats:")
print(spell_lengths.describe())

# Count consecutive-year obs within jobs
df_sorted = df.sort_values(['person_id', 'job_id', 'year'])
df_sorted['prev_year_within_job'] = grp['year'].shift(1)
consec = df_sorted[df_sorted['year'] - df_sorted['prev_year_within_job'] == 1]
print(f"\nConsecutive within-job obs: {len(consec)}")
print(f"  Unique persons: {consec['person_id'].nunique()}")
