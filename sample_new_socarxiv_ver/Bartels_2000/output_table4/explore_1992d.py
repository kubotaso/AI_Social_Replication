import pandas as pd
df = pd.read_csv('panel_1992.csv')

# Check if V923634=7 cases are problematic (only 3 cases)
# V923634=7 might be "other party" or DK, not a valid PID
# In that case, pid_current with value 8 (=V923634+1=8) would be excluded
# But we only include pid_current in [1,2,3,4,5,6,7], so these are already included

# V923634 value 7: what does pid_current become?
mask7 = df['V923634'] == 7
print("V923634=7 cases:")
print(df[mask7][['pid_current', 'pid_lagged', 'vote_pres', 'V923634', 'V900320']].to_string())
print()

# V923634 value 8 and 9 (missing/DK?)
mask89 = df['V923634'].isin([8, 9])
print("V923634=8,9 cases:")
print(df[mask89][['pid_current', 'pid_lagged', 'vote_pres', 'V923634', 'V900320']].head(10).to_string())
print()

# Check the V900320 (lagged PID) coding
print("V900320 value counts:")
print(df['V900320'].value_counts().sort_index())
print()

# Check if there are valid pid_lagged values we're missing
print("pid_lagged value counts (full):")
print(df['pid_lagged'].value_counts().sort_index())
print()

# Maybe V923634=7 is "other" party and should be excluded
# That would change pid_current=8 (not in 1-7 range) so it's already excluded
# But wait - pid_current was created as V923634+1, and V923634=7 -> pid_current=8
# So these 3 cases are already excluded from our analysis since we filter for 1-7

# Maybe the issue is different: perhaps we need to include V923634=7 as Ind
# Actually V923634 values: 0-6 are the 7-point scale, 7,8,9 are other/DK/NA
# pid_current = V923634 + 1 for values 0-6, giving 1-7
# V923634=7,8,9 get pid_current=8,9,10 which are excluded by our 1-7 filter

# So our data filtering is correct. The 4-respondent difference and the
# intercept discrepancy are likely due to having a slightly different sample.

# Let me also check if lagged PID has value 0 that should be handled
mask_lag0 = df['V900320'] == 0
print(f"Cases with V900320=0: {mask_lag0.sum()}")
print(f"What pid_lagged values do they get?")
print(df[mask_lag0]['pid_lagged'].value_counts().sort_index())
