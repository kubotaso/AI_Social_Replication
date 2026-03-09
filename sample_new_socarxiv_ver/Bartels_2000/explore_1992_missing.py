import pandas as pd

df = pd.read_csv('panel_1992.csv')
print('Total rows:', len(df))
print('Columns:', list(df.columns))
print()

# Check vote_pres values
print('vote_pres values:', df['vote_pres'].value_counts().sort_index().to_dict())
print('pid_current values:', df['pid_current'].value_counts().sort_index().to_dict())
print('pid_lagged values:', df['pid_lagged'].value_counts().sort_index().to_dict())
print()

# Strict filter (what we currently use)
mask_strict = (
    df['vote_pres'].isin([1, 2]) &
    df['pid_current'].isin([1, 2, 3, 4, 5, 6, 7]) &
    df['pid_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
)
print(f'Strict filter N: {mask_strict.sum()}')

# Check what values 8 and 9 are in pid_current/pid_lagged
has_valid_vote = df['vote_pres'].isin([1, 2])
valid_cpid = df['pid_current'].isin([1, 2, 3, 4, 5, 6, 7])
valid_lpid = df['pid_lagged'].isin([1, 2, 3, 4, 5, 6, 7])

print(f'Valid vote: {has_valid_vote.sum()}')
print(f'Valid vote + current PID: {(has_valid_vote & valid_cpid).sum()}')
print(f'Valid vote + lagged PID: {(has_valid_vote & valid_lpid).sum()}')
print(f'Valid vote + both PIDs: {(has_valid_vote & valid_cpid & valid_lpid).sum()}')

# Lost due to lagged PID
lost_lagged = has_valid_vote & valid_cpid & ~valid_lpid
print(f'\nLost due to lagged PID: {lost_lagged.sum()}')
print('Their lagged PID values:', df.loc[lost_lagged, 'pid_lagged'].value_counts().to_dict())

# Lost due to current PID
lost_current = has_valid_vote & ~valid_cpid & valid_lpid
print(f'Lost due to current PID: {lost_current.sum()}')
print('Their current PID values:', df.loc[lost_current, 'pid_current'].value_counts().to_dict())

# Lost due to both
lost_both = has_valid_vote & ~valid_cpid & ~valid_lpid
print(f'Lost due to both PIDs: {lost_both.sum()}')

# Check V923634 raw variable for those with pid_current=8
if 'V923634' in df.columns:
    print('\nV923634 for pid_current=8:',
          df.loc[df['pid_current']==8, 'V923634'].value_counts().to_dict())
    print('V923634 for pid_current=9:',
          df.loc[df['pid_current']==9, 'V923634'].value_counts().to_dict())

# What if V900320 (raw lagged pid) has value 8 = apolitical?
if 'V900320' in df.columns:
    print('\nV900320 for pid_lagged=8:',
          df.loc[df['pid_lagged']==8, 'V900320'].value_counts().to_dict())
    print('V900320 for pid_lagged=9:',
          df.loc[df['pid_lagged']==9, 'V900320'].value_counts().to_dict())

# Try including pid_current=8 as pure independent (recode to 4)
# and similarly for lagged
mask_loose = (
    df['vote_pres'].isin([1, 2]) &
    df['pid_current'].isin([1, 2, 3, 4, 5, 6, 7, 8]) &
    df['pid_lagged'].isin([1, 2, 3, 4, 5, 6, 7, 8])
)
print(f'\nWith pid=8 included: {mask_loose.sum()}')
