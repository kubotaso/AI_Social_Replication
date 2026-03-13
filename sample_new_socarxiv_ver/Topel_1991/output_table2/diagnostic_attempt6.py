#!/usr/bin/env python3
"""Diagnostic: understand data issues to improve Table 2 replication."""
import pandas as pd
import numpy as np

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

# Load both datasets
df = pd.read_csv('data/psid_panel.csv')
print(f"psid_panel.csv: {df.shape}, years={sorted(df['year'].unique())}, persons={df['person_id'].nunique()}")

df_full = pd.read_csv('data/psid_panel_full.csv')
print(f"psid_panel_full.csv: {df_full.shape}, years={sorted(df_full['year'].unique())}, persons={df_full['person_id'].nunique()}")
print(f"Columns in full not in main: {set(df_full.columns) - set(df.columns)}")
print(f"Columns in main not in full: {set(df.columns) - set(df_full.columns)}")

# Use psid_panel.csv (years 1971-1983)
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

# Construct experience using age
df['experience_age'] = df['age'] - df['education_fixed'] - 6

# Construct experience using initial + year offset
df = df.sort_values(['person_id', 'year']).reset_index(drop=True)
first_obs = df.groupby('person_id').first()[['age', 'year']].reset_index()
first_obs.columns = ['person_id', 'age_first', 'year_first']
df = df.merge(first_obs, on='person_id')
df['initial_exp'] = df['age_first'] - df['education_fixed'] - 6
df['experience_constructed'] = df['initial_exp'] + (df['year'] - df['year_first'])

# Tenure
df['tenure'] = df['tenure_topel'] - 1

# Within-job first differences
df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
grp = df.groupby(['person_id', 'job_id'])
df['prev_year'] = grp['year'].shift(1)
df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
df['prev_tenure'] = grp['tenure'].shift(1)
df['prev_exp_age'] = grp['experience_age'].shift(1)
df['prev_exp_con'] = grp['experience_constructed'].shift(1)

within = df[
    (df['prev_year'].notna()) &
    (df['year'] - df['prev_year'] == 1)
].copy()

within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']
within['d_exp_age'] = within['experience_age'] - within['prev_exp_age']
within['d_exp_con'] = within['experience_constructed'] - within['prev_exp_con']
within['d_tenure'] = within['tenure'] - within['prev_tenure']

print(f"\nWithin-job consecutive obs: {len(within)}")
print(f"  d_exp (age-based) == 1: {(within['d_exp_age'] == 1).sum()}")
print(f"  d_exp (age-based) != 1: {(within['d_exp_age'] != 1).sum()}")
print(f"  d_exp (constructed) == 1: {(within['d_exp_con'] == 1).sum()}")
print(f"  d_tenure == 1: {(within['d_tenure'] == 1).sum()}")

# Various N's under different filters
print("\n--- N under various filter combinations ---")

# Base: no outlier removal
n_base = len(within)
print(f"No filters: {n_base}")

# Filter d_exp_age == 1
w1 = within[within['d_exp_age'] == 1]
print(f"d_exp_age == 1: {len(w1)}")

# Filter d_exp_age == 1 + outlier +-2
w1b = w1[w1['d_log_wage'].between(-2, 2)]
print(f"d_exp_age == 1 + outlier +-2: {len(w1b)}")

# Filter d_exp_age == 1 + outlier +-1
w1c = w1[w1['d_log_wage'].between(-1, 1)]
print(f"d_exp_age == 1 + outlier +-1: {len(w1c)}")

# Filter d_exp_age == 1 + experience >= 1
w1d = w1b[w1b['experience_age'] >= 1]
print(f"d_exp_age == 1 + outlier +-2 + exp>=1: {len(w1d)}")

# Constructed experience: no d_exp filter needed, but need outlier removal
w2 = within.copy()
w2b = w2[w2['d_log_wage'].between(-2, 2)]
print(f"All obs (constructed exp) + outlier +-2: {len(w2b)}")

# Constructed experience + exp >= 1
w2c = w2b[w2b['experience_constructed'] >= 1]
print(f"All obs (constructed exp) + outlier +-2 + exp>=1: {len(w2c)}")

# Try using initial_exp >= 1 to reduce sample
w2d = within[within['initial_exp'] >= 1]
w2d = w2d[w2d['d_log_wage'].between(-2, 2)]
print(f"initial_exp >= 1 + outlier +-2: {len(w2d)}")

# Try initial_exp >= 2
w2e = within[within['initial_exp'] >= 2]
w2e = w2e[w2e['d_log_wage'].between(-2, 2)]
print(f"initial_exp >= 2 + outlier +-2: {len(w2e)}")

# Drop persons with any exp < 1 (age-based)
person_min_exp = df.groupby('person_id')['experience_age'].min()
valid_persons = person_min_exp[person_min_exp >= 1].index
w3 = within[within['person_id'].isin(valid_persons)]
w3 = w3[w3['d_exp_age'] == 1]
w3 = w3[w3['d_log_wage'].between(-2, 2)]
print(f"Drop persons exp_age<1 + d_exp==1 + outlier +-2: {len(w3)}")

# Try different outlier thresholds with d_exp_age == 1
for thresh in [0.5, 0.75, 1.0, 1.5, 2.0]:
    wt = within[within['d_exp_age'] == 1]
    wt = wt[wt['d_log_wage'].between(-thresh, thresh)]
    print(f"d_exp_age==1 + outlier +-{thresh}: {len(wt)}")

# What if we use tenure >= 0 (paper starts at 0)
print("\n--- Tenure restrictions ---")
w5 = within[within['d_exp_age'] == 1].copy()
w5 = w5[w5['d_log_wage'].between(-2, 2)]
for min_ten in [0, 1, 2]:
    wt = w5[w5['tenure'] >= min_ten]
    print(f"tenure >= {min_ten}: {len(wt)}")

# Experience range restriction
print("\n--- Experience restrictions ---")
for max_exp in [30, 35, 40, 45]:
    wt = w5[w5['experience_age'] <= max_exp]
    print(f"exp <= {max_exp}: {len(wt)}")

for min_exp in [1, 2, 3]:
    wt = w5[w5['experience_age'] >= min_exp]
    print(f"exp >= {min_exp}: {len(wt)}")

# Check distribution of experience and tenure
print("\n--- Distribution stats ---")
print(f"Experience (age-based) in within-job sample (d_exp==1):")
w_clean = within[within['d_exp_age'] == 1].copy()
w_clean = w_clean[w_clean['d_log_wage'].between(-2, 2)]
print(f"  min={w_clean['experience_age'].min()}, max={w_clean['experience_age'].max()}, mean={w_clean['experience_age'].mean():.1f}")
print(f"  N = {len(w_clean)}")
print(f"  Persons = {w_clean['person_id'].nunique()}")
print(f"  d_log_wage mean = {w_clean['d_log_wage'].mean():.4f}")
print(f"  d_log_wage std = {w_clean['d_log_wage'].std():.4f}")

# Paper says mean real wage change = .026
# GNP deflator approach
GNP_DEFLATOR = {
    1967: 100.00, 1968: 104.28, 1969: 109.13, 1970: 113.94, 1971: 118.92,
    1972: 123.16, 1973: 130.27, 1974: 143.08, 1975: 155.56, 1976: 163.42,
    1977: 173.43, 1978: 186.18, 1979: 201.33, 1980: 220.39, 1981: 241.02,
    1982: 255.09, 1983: 264.00
}
w_clean['gnp_defl'] = w_clean['year'].map(GNP_DEFLATOR)
w_clean['log_real_wage'] = w_clean['log_hourly_wage'] - np.log(w_clean['gnp_defl'] / 100.0)
w_clean['prev_gnp'] = w_clean['year'].map(lambda y: GNP_DEFLATOR.get(y-1, GNP_DEFLATOR.get(y)))
w_clean['prev_log_real_wage'] = w_clean['prev_log_wage'] - np.log(w_clean['prev_gnp'] / 100.0)
w_clean['d_log_real_wage'] = w_clean['log_real_wage'] - w_clean['prev_log_real_wage']
print(f"  d_log_real_wage (GNP) mean = {w_clean['d_log_real_wage'].mean():.4f}")

# CPS approach
CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}
w_clean['cps'] = w_clean['year'].map(CPS_WAGE_INDEX)
w_clean['prev_cps'] = w_clean['year'].map(lambda y: CPS_WAGE_INDEX.get(y-1, CPS_WAGE_INDEX.get(y)))
w_clean['log_real_wage_cps'] = w_clean['log_hourly_wage'] - np.log(w_clean['cps'])
w_clean['prev_lrw_cps'] = w_clean['prev_log_wage'] - np.log(w_clean['prev_cps'])
w_clean['d_lrw_cps'] = w_clean['log_real_wage_cps'] - w_clean['prev_lrw_cps']
print(f"  d_log_real_wage (CPS) mean = {w_clean['d_lrw_cps'].mean():.4f}")

# Number of observations per person
print(f"\n--- Obs per person ---")
obs_per_person = w_clean.groupby('person_id').size()
print(f"  Mean: {obs_per_person.mean():.1f}")
print(f"  Min: {obs_per_person.min()}")
print(f"  Max: {obs_per_person.max()}")
print(f"  Distribution:")
print(obs_per_person.value_counts().sort_index())

# Try min obs per person filter
for min_obs in [2, 3, 4]:
    valid = obs_per_person[obs_per_person >= min_obs].index
    wt = w_clean[w_clean['person_id'].isin(valid)]
    print(f"\n  min_obs >= {min_obs}: N={len(wt)}, persons={wt['person_id'].nunique()}")
