#!/usr/bin/env python3
"""Check if there's a better tenure variable using reported tenure months."""
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')
print("Columns:", list(df.columns))
print(f"\ntenure_topel distribution:")
print(df['tenure_topel'].describe())
print(f"\ntenure distribution (original):")
print(df['tenure'].describe())
print(f"\ntenure_mos distribution:")
print(df['tenure_mos'].describe())

# tenure_mos might be the raw reported tenure in months
# Let's see
print(f"\ntenure_mos value counts (top 20):")
print(df['tenure_mos'].value_counts().sort_index().head(30))

# Check correlation
both = df[df['tenure_mos'].notna() & df['tenure_topel'].notna()]
print(f"\nCorrelation tenure_topel vs tenure_mos: {both['tenure_topel'].corr(both['tenure_mos']):.3f}")

# tenure_mos should be reported months with current employer
# Convert to years: tenure_years_reported = tenure_mos / 12
both['ten_years_rep'] = both['tenure_mos'] / 12
print(f"\nMean reported tenure (years): {both['ten_years_rep'].mean():.2f}")
print(f"Mean tenure_topel: {both['tenure_topel'].mean():.2f}")

# Perhaps we should use the REPORTED tenure instead of the constructed one
# Topel says: "Tenure is reconstructed from the employer mobility data"
# But he also says: "The main shortcoming of this method is that initial
# tenure is lost for persons who do not change employers during the panel"

# For Table 2, what matters is the LEVEL of tenure (for the polynomial differences)
# The reported tenure_mos gives us the actual tenure level

# Let's try using reported tenure
print(f"\n=== Using reported tenure (tenure_mos/12) ===")

# But tenure_mos might not be available for all years
avail = df.groupby('year')['tenure_mos'].apply(lambda x: x.notna().mean())
print(f"\nFraction with tenure_mos by year:")
print(avail)

# What about the 'tenure' column (original, not tenure_topel)?
avail2 = df.groupby('year')['tenure'].apply(lambda x: x.notna().mean())
print(f"\nFraction with tenure by year:")
print(avail2)

# Check tenure column
print(f"\ntenure column unique values (sorted):")
print(sorted(df['tenure'].dropna().unique())[:30])
print(f"max: {df['tenure'].max()}")

# Look at a few persons to understand tenure
sample_persons = df['person_id'].unique()[:5]
for pid in sample_persons:
    pdata = df[df['person_id'] == pid][['year', 'age', 'job_id', 'tenure_topel', 'tenure', 'tenure_mos', 'same_emp', 'new_job']].sort_values('year')
    print(f"\nPerson {pid}:")
    print(pdata.to_string(index=False))

# Check: what's the tenure variable used in Table A1?
# Table A1 says mean S (tenure/seniority) = 9.365
# Our tenure_topel mean is 3.18 (level data)
# If we use tenure_mos / 12, what's the mean?
print(f"\n\nMean tenure_mos / 12 (all obs): {(df['tenure_mos']/12).mean():.2f}")

# The 'tenure' column (original) - what does it represent?
# It was built in build_psid_panel.py Step 5 from same_employer
print(f"\nMean tenure (original): {df['tenure'].mean():.2f}")

# Check if tenure_mos reflects actual reported months
# In PSID, tenure is typically asked as "How many years/months have you worked for your present employer?"
# This should give us the ACTUAL tenure, not the panel-reconstructed one

# For the differenced model, we need CONSISTENT tenure progression
# If we use reported tenure (tenure_mos/12), it should increment by ~1 each year
# But self-reported tenure might not increment exactly by 1

# Let's check d_tenure using reported tenure
df_sorted = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
grp = df_sorted.groupby(['person_id', 'job_id'])
df_sorted['prev_year'] = grp['year'].shift(1)
df_sorted['ten_rep'] = df_sorted['tenure_mos'] / 12
df_sorted['prev_ten_rep'] = grp['ten_rep'].shift(1)

within_rep = df_sorted[
    (df_sorted['prev_year'].notna()) &
    (df_sorted['year'] - df_sorted['prev_year'] == 1) &
    (df_sorted['ten_rep'].notna()) &
    (df_sorted['prev_ten_rep'].notna())
].copy()
within_rep['d_ten_rep'] = within_rep['ten_rep'] - within_rep['prev_ten_rep']
print(f"\nd_tenure_reported distribution:")
print(within_rep['d_ten_rep'].describe())
print(f"\nValue counts (rounded):")
print(within_rep['d_ten_rep'].round(0).value_counts().sort_index().head(10))
