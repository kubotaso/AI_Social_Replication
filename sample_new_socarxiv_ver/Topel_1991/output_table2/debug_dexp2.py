#!/usr/bin/env python3
"""Debug why d_exp != 1 persists with fixed education."""
import pandas as pd
import numpy as np

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

df = pd.read_csv('data/psid_panel.csv')

# Education fix
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
df['educ_fixed'] = df['person_id'].map(person_educ)
df = df[df['educ_fixed'].notna()].copy()

# Experience
df['experience'] = df['age'] - df['educ_fixed'] - 6

# Check age progression
df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
grp = df.groupby(['person_id', 'job_id'])
df['prev_year'] = grp['year'].shift(1)
df['prev_age'] = grp['age'].shift(1)
df['prev_exp'] = grp['experience'].shift(1)

within = df[
    df['prev_year'].notna() &
    (df['year'] - df['prev_year'] == 1)
].copy()

within['d_age'] = within['age'] - within['prev_age']
within['d_exp'] = within['experience'] - within['prev_exp']

# d_exp should equal d_age (since education is fixed)
# So if d_age != 1, d_exp != 1
bad = within[within['d_exp'] != 1]
print(f"Obs with d_exp != 1: {len(bad)}")
print(f"Of these, d_age != 1: {(bad['d_age'] != 1).sum()}")
print(f"d_age values in bad obs: {sorted(bad['d_age'].unique())[:20]}")
print()

# So the issue is that age doesn't always progress by 1!
# Some persons have the same age in consecutive years (d_age = 0)
# or jump by 2 years
all_d_age = within['d_age'].value_counts()
print("d_age distribution in all within-job obs:")
print(all_d_age.sort_index())

# This is a PSID data issue: age is self-reported and can be inconsistent
# Topel likely handles this by defining experience = experience_prior_year + 1
# (i.e., experience increments by 1 each year regardless of reported age)
print()
print("SOLUTION: Define experience to increment by 1 each year within a job,")
print("regardless of reported age changes.")
