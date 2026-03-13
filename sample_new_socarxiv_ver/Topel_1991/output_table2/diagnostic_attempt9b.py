#!/usr/bin/env python3
"""
Check govt_worker and other filters more carefully.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')
print(f"Total: {len(df)} obs, {df['person_id'].nunique()} persons")

# govt_worker column
print(f"\ngovt_worker value counts (including NaN):")
print(df['govt_worker'].value_counts(dropna=False))

# self_employed column
print(f"\nself_employed value counts (including NaN):")
print(df['self_employed'].value_counts(dropna=False))

# What if NaN in govt_worker means they ARE government workers?
# Or what if NaN means we don't know?
# Let's check which years have govt_worker data
print(f"\ngovt_worker availability by year:")
for yr in sorted(df['year'].unique()):
    yr_data = df[df['year'] == yr]
    n_0 = (yr_data['govt_worker'] == 0).sum()
    n_1 = (yr_data['govt_worker'] == 1).sum()
    n_nan = yr_data['govt_worker'].isna().sum()
    print(f"  {yr}: 0={n_0}, 1={n_1}, NaN={n_nan}, total={len(yr_data)}")

# If govt_worker is only available for some years, we need to
# exclude govt workers when we CAN identify them
# and keep the rest (assuming non-govt by default)

# What about excluding govt workers from the sample?
# If we drop obs where govt_worker == 1:
df_no_govt = df[df['govt_worker'] != 1].copy()
print(f"\nAfter dropping govt_worker==1: {len(df_no_govt)} obs, {df_no_govt['person_id'].nunique()} persons")

# If we also drop obs where govt_worker is NaN (unknown):
df_no_govt2 = df[df['govt_worker'] == 0].copy()
print(f"After keeping only govt_worker==0: {len(df_no_govt2)} obs, {df_no_govt2['person_id'].nunique()} persons")

# The paper may have dropped persons who were EVER identified as govt workers
# Let's try that
ever_govt = df[df['govt_worker'] == 1]['person_id'].unique()
print(f"\nPersons ever govt_worker==1: {len(ever_govt)}")
df_never_govt = df[~df['person_id'].isin(ever_govt)].copy()
print(f"After dropping ever-govt persons: {len(df_never_govt)} obs, {df_never_govt['person_id'].nunique()} persons")

# What about disabled?
# Topel doesn't explicitly mention excluding disabled, but it's common
print(f"\ndisabled column:")
print(df['disabled'].value_counts(dropna=False))

# Check: what happens if we try to match 1540 persons?
# Our data has 2407 persons. The paper has 1540.
# Difference: 867 persons
# Possible additional exclusions:
# 1. Government workers (whole career)
# 2. Hawaii/Alaska (already excluded?)
# 3. Different tenure requirement

# Check region
print(f"\nRegion values:")
print(df['region'].value_counts().sort_index())

# Check: what does the 'white' column look like?
print(f"\nwhite column:")
print(df['white'].value_counts(dropna=False))

# Hours restriction: wages = earnings/hours
# Paper says "since wages refer to average hourly earnings in the year
# preceding the survey" -- so wages = labor_inc / hours
print(f"\nhours distribution:")
print(df['hours'].describe())
print(f"Obs with hours < 250: {(df['hours'] < 250).sum()}")
print(f"Obs with hours < 500: {(df['hours'] < 500).sum()}")

# Maybe the restriction is minimum hours worked?
# Let's try various minimums
for min_h in [250, 500, 750, 1000]:
    n = (df['hours'] >= min_h).sum()
    p = df[df['hours'] >= min_h]['person_id'].nunique()
    print(f"  hours >= {min_h}: {n} obs, {p} persons")
