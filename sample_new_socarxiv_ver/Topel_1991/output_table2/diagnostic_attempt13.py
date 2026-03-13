#!/usr/bin/env python3
"""
Check experience distribution more carefully.
Try to understand what the paper's experience distribution looks like.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

# Fix education
df['educ_raw'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'educ_raw'] = df.loc[cat_mask, 'education_clean'].map(
    {**EDUC_MAP, 9: np.nan}
)
df.loc[df['educ_raw'] > 17, 'educ_raw'] = 17
df.loc[(df['year'].isin([1975, 1976])) & (df['education_clean'] == 9), 'educ_raw'] = np.nan

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

# Drop persons with exp < 1
person_min_exp = df.groupby('person_id')['experience'].min()
valid_persons = person_min_exp[person_min_exp >= 1].index
df = df[df['person_id'].isin(valid_persons)].copy()

print(f"Total: {len(df)} obs, {df['person_id'].nunique()} persons")
print(f"\nMean age: {df['age'].mean():.2f}")
print(f"Mean education_fixed: {df['education_fixed'].mean():.2f}")
print(f"Mean experience: {df['experience'].mean():.2f}")

# Paper: mean experience = 20.021, mean education = 12.645
# So paper's mean age = 20.021 + 12.645 + 6 = 38.67

# What if there's a different offset? In some formulations, experience = age - education - 5
# (starting school at 5 instead of 6)
df['exp_alt'] = df['age'] - df['education_fixed'] - 5
print(f"\nWith -5 offset: mean experience = {df['exp_alt'].mean():.2f}")

# What about using experience from the data (age - education_clean - 6)?
# In non-1975/1976 years, education_clean is categorical (0-8)
# But the 'experience' column in the data uses this formula
# Let's check the data's own experience distribution
df2 = pd.read_csv('data/psid_panel.csv')
print(f"\nData's own experience column mean: {df2['experience'].mean():.2f}")
# This is wrong because education_clean is categorical for most years

# What about for within-job observations?
df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
grp = df.groupby(['person_id', 'job_id'])
df['prev_year'] = grp['year'].shift(1)
df['prev_experience'] = grp['experience'].shift(1)
within = df[(df['prev_year'].notna()) & (df['year'] - df['prev_year'] == 1)].copy()
within['d_exp'] = within['experience'] - within['prev_experience']
within = within[within['d_exp'] == 1].copy()

print(f"\nWithin-job (d_exp==1):")
print(f"  N = {len(within)}")
print(f"  Mean experience: {within['experience'].mean():.2f}")
print(f"  Mean age: {within['age'].mean():.2f}")
print(f"  Mean education_fixed: {within['education_fixed'].mean():.2f}")
print(f"  Mean tenure_topel: {within['tenure_topel'].mean():.2f}")

# Paper: N=8,683, mean exp=20.021, mean tenure=9.978
# Our within-job: N~9,000, mean exp~19, mean tenure~4

# The tenure difference is huge: our mean tenure=4 vs paper's 9.978
# This is because we don't reconstruct pre-panel tenure properly
# Tenure reconstruction gives ~6.5 (from earlier diagnostics)

# But even with reconstruction, tenure mean is 6.5 vs paper's 9.978
# The paper's data has MUCH higher tenure because:
# 1. They have 1968-1970 data (jobs observed longer = higher tenure)
# 2. Their sample of 1,540 persons may be selected for longer jobs

# KEY INSIGHT: The paper reports summary stats for the LEVELS data
# (13,128 job-years on 1,540 persons), not the first-differenced data.
# Mean tenure in the levels data = 9.978
# Mean tenure in the first-differenced data would be slightly higher
# (because first obs of each job is dropped)

print(f"\n\nLevels data stats for comparison:")
print(f"  Mean tenure_topel: {df['tenure_topel'].mean():.2f}")
print(f"  Mean experience: {df['experience'].mean():.2f}")
print(f"  Mean education_fixed: {df['education_fixed'].mean():.2f}")

# Try to understand what coefficient differences really mean
# If our exp^2 coefficient is -0.28 (x100) vs paper's -0.60 (x100)
# And both get similar R^2 and similar predictions for d_tenure...
# Then the experience polynomial is capturing DIFFERENT curvature

# The polynomial coefficients are NOT individually meaningful when
# correlated. What matters is the COMBINED prediction.
# Let's compute the implied wage growth curve

print("\n\nImplied wage growth from experience polynomial alone:")
print("(holding tenure constant at 0)")
for exp_level in [5, 10, 15, 20, 25, 30]:
    # d(X^2) = 2X-1 for X->X+1
    # d(X^3) = 3X^2+3X+1
    # d(X^4) = 4X^3+6X^2+4X+1
    dx2 = 2*exp_level - 1
    dx3 = 3*exp_level**2 + 3*exp_level + 1
    dx4 = 4*exp_level**3 + 6*exp_level**2 + 4*exp_level + 1

    # Paper's coefficients (Model 1): -0.006051, 0.000146, 0.00000131
    paper_growth = -0.006051*dx2 + 0.000146*dx3 + 0.00000131*dx4

    # Our coefficients (Model 1): -0.002752, 0.0000494, -0.00000032
    our_growth = -0.002752*dx2 + 0.0000494*dx3 + (-0.0000032)*dx4

    print(f"  Exp={exp_level}: Paper exp_growth={paper_growth:.4f}, Ours={our_growth:.4f}")
