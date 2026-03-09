import pandas as pd
import numpy as np

df = pd.read_csv('panel_1992.csv')

# Check for NaN values
print("NaN counts:")
print(df.isna().sum())
print()

# Those with valid vote but missing current PID
has_valid_vote = df['vote_pres'].isin([1, 2])
valid_cpid = df['pid_current'].isin([1, 2, 3, 4, 5, 6, 7])

missing_cpid = has_valid_vote & ~valid_cpid
print(f"Respondents with valid vote but missing current PID: {missing_cpid.sum()}")
if missing_cpid.sum() > 0:
    print(df.loc[missing_cpid, ['pid_current', 'pid_lagged', 'vote_pres', 'V923634', 'V900320']])
    print()

# Check vote_pres - any value besides 1 and 2 that could be a vote?
print("vote_pres distribution (all values):")
print(df['vote_pres'].value_counts(dropna=False).sort_index())
print()

# Check V925609 for those with non-standard vote_pres
non_std_vote = ~df['vote_pres'].isin([1, 2]) & ~df['vote_pres'].isna()
print(f"Non-standard vote_pres: {non_std_vote.sum()}")
if non_std_vote.sum() > 0:
    print(df.loc[non_std_vote, ['vote_pres', 'V925609', 'pid_current', 'pid_lagged']].head(20))
print()

# The paper says N=729. We get 725. Let's check if pid_current has any
# values that might be valid (e.g., 0 for apolitical mapped differently)
print("pid_current distribution including NaN:")
print(df['pid_current'].value_counts(dropna=False).sort_index())
print()

# What if we look at V923634 for the missing pid_current cases?
print("V923634 distribution:")
print(df['V923634'].value_counts(dropna=False).sort_index())
print()

# Let's see what V923634 values those 3 missing-PID voters have
if missing_cpid.sum() > 0:
    print("V923634 for missing PID voters:")
    print(df.loc[missing_cpid, 'V923634'].value_counts(dropna=False))

# Check: total 1336 rows, are any rows exactly duplicated?
print(f"\nDuplicate rows: {df.duplicated().sum()}")
