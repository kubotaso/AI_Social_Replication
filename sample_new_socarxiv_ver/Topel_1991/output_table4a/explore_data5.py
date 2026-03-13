import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')

# PSID sequence numbers:
# 1-19: Original sample members (head=1, wife=2, children etc.)
# 20+: Born-in sample members
# 170+: Moved-in/split-off members

df['seq_num'] = df['person_id'] - df['id_1968'] * 1000

# Unique persons by sequence number range
print("Persons by sequence number:")
for low, high, label in [(1, 1, "seq=1 (head)"),
                          (1, 19, "seq=1-19"),
                          (1, 30, "seq=1-30"),
                          (20, 169, "seq=20-169"),
                          (170, 999, "seq=170+")]:
    mask = (df['seq_num'] >= low) & (df['seq_num'] <= high)
    n_pers = df.loc[mask, 'person_id'].nunique()
    n_obs = mask.sum()
    print(f"  {label}: {n_pers} persons, {n_obs} obs")

# Try seq <= 30 (original sample members and their children born in)
for max_seq in [1, 3, 5, 10, 19, 30, 40, 50]:
    mask = df['seq_num'] <= max_seq
    sub = df[mask].copy()
    sub = sub.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
    sub['prev_year'] = sub.groupby('job_id')['year'].shift(1)
    wj = sub[(sub['prev_year'].notna()) & (sub['year'] - sub['prev_year'] == 1)]
    n_pers = sub['person_id'].nunique()
    print(f"  seq<={max_seq}: {n_pers} persons, {len(wj)} within-job obs")

# Try excluding the "moved-in" persons (seq >= 170) only
mask = df['seq_num'] < 170
sub = df[mask].copy()
sub = sub.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
sub['prev_year'] = sub.groupby('job_id')['year'].shift(1)
wj = sub[(sub['prev_year'].notna()) & (sub['year'] - sub['prev_year'] == 1)]
print(f"\n  seq<170: {sub['person_id'].nunique()} persons, {len(wj)} within-job obs")

# Try keeping only persons whose seq_num is 1 or 3 (original household members)
mask = df['seq_num'].isin([1, 3])
sub = df[mask].copy()
sub = sub.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
sub['prev_year'] = sub.groupby('job_id')['year'].shift(1)
wj = sub[(sub['prev_year'].notna()) & (sub['year'] - sub['prev_year'] == 1)]
print(f"  seq in [1,3]: {sub['person_id'].nunique()} persons, {len(wj)} within-job obs")

# What about unique id_1968 count for different seq filters?
print(f"\n  All data: {df['id_1968'].nunique()} unique id_1968")
mask1 = df['seq_num'] == 1
print(f"  seq=1: {df.loc[mask1, 'id_1968'].nunique()} unique id_1968")
