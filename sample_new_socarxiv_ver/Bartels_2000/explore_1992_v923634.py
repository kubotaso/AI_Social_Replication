import pandas as pd
import numpy as np

df = pd.read_csv('panel_1992.csv')

# Let's look at V923634 more carefully
print("V923634 distribution:")
print(df['V923634'].value_counts().sort_index())
print()

# Compare V923634 to pid_current
print("V923634 vs pid_current mapping:")
for v in sorted(df['V923634'].unique()):
    subset = df[df['V923634'] == v]
    pid_vals = subset['pid_current'].value_counts(dropna=False).sort_index()
    print(f"  V923634={v}: pid_current = {pid_vals.to_dict()}")
print()

# Can we recode V923634=7 to pure independent (pid=4)?
# The person with V923634=7 who has a valid vote:
v7_voters = df[(df['V923634']==7) & df['vote_pres'].isin([1,2])]
print(f"V923634=7 with valid vote: {len(v7_voters)}")
if len(v7_voters) > 0:
    print(v7_voters[['V923634', 'pid_current', 'pid_lagged', 'vote_pres']])
print()

# Check V923634=8
v8_voters = df[(df['V923634']==8) & df['vote_pres'].isin([1,2])]
print(f"V923634=8 with valid vote: {len(v8_voters)}")
if len(v8_voters) > 0:
    print(v8_voters[['V923634', 'pid_current', 'pid_lagged', 'vote_pres']])
print()

# Check V923634=9
v9_voters = df[(df['V923634']==9) & df['vote_pres'].isin([1,2])]
print(f"V923634=9 with valid vote: {len(v9_voters)}")
if len(v9_voters) > 0:
    print(v9_voters[['V923634', 'pid_current', 'pid_lagged', 'vote_pres']])
print()

# If we recode V923634=7 as pid_current=4 (pure independent):
# That means Strong, Weak, Lean all = 0 for those respondents
# Let's see what happens to N
df2 = df.copy()
# Map V923634 to pid_current for those with NaN
mask_v7 = (df2['V923634'] == 7) & df2['pid_current'].isna()
print(f"V923634=7 with NaN pid_current: {mask_v7.sum()}")
if mask_v7.sum() > 0:
    df2.loc[mask_v7, 'pid_current'] = 4  # pure independent

# Also check V923634=8 - these might be "don't know" -> could be treated as independent?
mask_v8 = (df2['V923634'] == 8) & df2['pid_current'].isna()
print(f"V923634=8 with NaN pid_current: {mask_v8.sum()}")

mask_v9 = (df2['V923634'] == 9) & df2['pid_current'].isna()
print(f"V923634=9 with NaN pid_current: {mask_v9.sum()}")

# What N would we get if we include V923634=7 as pure independent?
mask_new = (
    df2['vote_pres'].isin([1, 2]) &
    df2['pid_current'].isin([1, 2, 3, 4, 5, 6, 7]) &
    df2['pid_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
)
print(f"\nN with V923634=7 recoded: {mask_new.sum()}")

# What if we also include V923634=8 as pure independent?
df3 = df2.copy()
df3.loc[mask_v8, 'pid_current'] = 4
mask_new2 = (
    df3['vote_pres'].isin([1, 2]) &
    df3['pid_current'].isin([1, 2, 3, 4, 5, 6, 7]) &
    df3['pid_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
)
print(f"N with V923634=7,8 recoded: {mask_new2.sum()}")

# And V923634=9?
df4 = df3.copy()
df4.loc[mask_v9, 'pid_current'] = 4
mask_new3 = (
    df4['vote_pres'].isin([1, 2]) &
    df4['pid_current'].isin([1, 2, 3, 4, 5, 6, 7]) &
    df4['pid_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
)
print(f"N with V923634=7,8,9 recoded: {mask_new3.sum()}")
