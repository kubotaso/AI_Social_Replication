import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')

# How many persons per id_1968?
persons_per_id1968 = df.groupby('id_1968')['person_id'].nunique()
print("Persons per id_1968:")
print(persons_per_id1968.value_counts().sort_index())

# Check if keeping only 1 person per id_1968 helps
# Maybe keep the one with the most observations
person_obs = df.groupby('person_id').size()
df['n_obs'] = df['person_id'].map(person_obs)

# For each id_1968, keep the person_id with most observations
best_person = df.groupby('id_1968').apply(
    lambda x: x.sort_values('n_obs', ascending=False)['person_id'].iloc[0]
).reset_index(name='best_person_id')

head_ids = set(best_person['best_person_id'])
df_heads = df[df['person_id'].isin(head_ids)]
print(f"\nKeeping 1 person per id_1968 (most obs):")
print(f"  Persons: {df_heads['person_id'].nunique()}")
print(f"  Observations: {len(df_heads)}")

# Compute within-job obs for this subset
df_heads = df_heads.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
df_heads['prev_year'] = df_heads.groupby('job_id')['year'].shift(1)
wj = df_heads[(df_heads['prev_year'].notna()) & (df_heads['year'] - df_heads['prev_year'] == 1)]
print(f"  Within-job obs: {len(wj)}")

# Also try: original sample heads (lowest person_id per id_1968)
first_person = df.groupby('id_1968')['person_id'].min()
first_ids = set(first_person)
df_first = df[df['person_id'].isin(first_ids)]
print(f"\nKeeping first person_id per id_1968:")
print(f"  Persons: {df_first['person_id'].nunique()}")
print(f"  Observations: {len(df_first)}")

df_first = df_first.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
df_first['prev_year'] = df_first.groupby('job_id')['year'].shift(1)
wj2 = df_first[(df_first['prev_year'].notna()) & (df_first['year'] - df_first['prev_year'] == 1)]
print(f"  Within-job obs: {len(wj2)}")

# Check if the PSID sequence number can help
# In PSID, person_id = family_id * 1000 + sequence_number
# Sequence 1 = head, 2 = wife
# Let's check if person_id has embedded sequence numbers
sample_ids = df['person_id'].unique()[:20]
print(f"\nSample person_ids: {sorted(sample_ids)[:20]}")

# Check the pattern of person_id relative to id_1968
df['seq_guess'] = df['person_id'] - df['id_1968'] * 1000
print(f"\nSequence number guess stats:")
print(df['seq_guess'].describe())
print(f"Value counts:")
print(df['seq_guess'].value_counts().head(10))
