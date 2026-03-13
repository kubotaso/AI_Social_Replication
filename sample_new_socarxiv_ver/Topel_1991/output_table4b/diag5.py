"""
Deeper diagnostic: Check experience distribution and how X_0 relates to wages.
The paper gets positive beta_1 (more experience = higher wages). Getting negative
beta_1 means something is fundamentally wrong.
"""
import pandas as pd, numpy as np, statsmodels.api as sm

df = pd.read_csv('data/psid_panel.csv')
df = df[~df['region'].isin([5, 6])]

EDUC = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17,9:17}
df['ey'] = df['education_clean'].copy()
m = ~df['year'].isin([1975, 1976])
df.loc[m, 'ey'] = df.loc[m, 'education_clean'].map(EDUC)
df = df.dropna(subset=['ey'])
df['exp'] = (df['age'] - df['ey'] - 6).clip(lower=0)

# Basic check: does log wage increase with experience?
print("Correlation of log_hourly_wage with experience:", df[['log_hourly_wage','exp']].corr().iloc[0,1])
print("Correlation of log_hourly_wage with age:", df[['log_hourly_wage','age']].corr().iloc[0,1])
print()

# Check experience distribution
print("Experience distribution:")
print(df['exp'].describe())
print()
print("Education years distribution:")
print(df['ey'].describe())
print()

# Check: for 1975/1976, education is already in years
# For other years, education is categorical and we remap
# Verify that the remapping gives reasonable values
print("Education by year:")
for yr in sorted(df['year'].unique()):
    sub = df[df['year']==yr]
    print(f"  {yr}: mean_ey={sub['ey'].mean():.1f}, mean_exp={sub['exp'].mean():.1f}, mean_age={sub['age'].mean():.1f}, n={len(sub)}")

# The issue: for 1975-1976, education is in actual years (mean ~12.6)
# For other years, it's recoded from categories (mean becomes different)
# Let's check what happens to experience
print()
print("Experience calculation check:")
# For 1975: exp = age - ey - 6, with ey ~ 12.6 and age ~ 37
# For 1977: exp = age - ey - 6, with ey from categorical remap (mean ~?)
# If categorical remap gives systematically different education years,
# experience will be inconsistent across years

# Check: within same person across years
df = df.sort_values(['person_id','year'])
df['prev_exp'] = df.groupby('person_id')['exp'].shift(1)
df['prev_yr'] = df.groupby('person_id')['year'].shift(1)
df['d_exp'] = df['exp'] - df['prev_exp']
consec = df[(df['prev_yr']==df['year']-1)].copy()
print(f"\nConsecutive obs: {len(consec)}")
print("d_exp distribution:")
print(consec['d_exp'].describe())
print()
print("d_exp value counts (top 10):")
print(consec['d_exp'].value_counts().head(15))
print()

# The d_exp should always be 1 for consecutive years (assuming exp = age - edu - 6)
# If education changes between years (due to recoding), d_exp will NOT be 1
# This is a major problem for the polynomial terms in Step 1!
bad = consec[consec['d_exp'] != 1]
print(f"Observations where d_exp != 1: {len(bad)} ({100*len(bad)/len(consec):.1f}%)")
print("Cross-tab of d_exp != 1 by year transition:")
if len(bad) > 0:
    bad['yr_transition'] = bad['prev_yr'].astype(int).astype(str) + '->' + bad['year'].astype(int).astype(str)
    print(bad['yr_transition'].value_counts().head(20))
