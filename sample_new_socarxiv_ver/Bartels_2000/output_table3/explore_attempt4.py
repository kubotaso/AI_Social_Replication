import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit

df = pd.read_csv('anes_cumulative.csv', usecols=['VCF0004','VCF0301','VCF0707','VCF0902'], low_memory=False)

# Investigate 1974: what if we require valid PID?
print("=== 1974 ANALYSIS ===")
for approach in ['all_pid_null_open', 'valid_pid_null_open', 'all_pid_exclude_null', 'valid_pid_exclude_null']:
    sub = df[(df['VCF0004']==1974) & (df['VCF0707'].isin([1,2]))].copy()

    if 'valid_pid' in approach:
        sub = sub[sub['VCF0301'].isin([1,2,3,4,5,6,7])]

    if 'exclude_null' in approach:
        sub = sub[sub['VCF0902'].notna()]

    sub['vote_rep'] = (sub['VCF0707'] == 2).astype(int)
    sub['strong'] = 0; sub['weak'] = 0; sub['leaner'] = 0
    sub.loc[sub['VCF0301']==7, 'strong'] = 1; sub.loc[sub['VCF0301']==1, 'strong'] = -1
    sub.loc[sub['VCF0301']==6, 'weak'] = 1; sub.loc[sub['VCF0301']==2, 'weak'] = -1
    sub.loc[sub['VCF0301']==5, 'leaner'] = 1; sub.loc[sub['VCF0301']==3, 'leaner'] = -1
    sub['incumbency'] = 0
    sub.loc[sub['VCF0902'].isin([12,13,14,19]), 'incumbency'] = -1
    sub.loc[sub['VCF0902'].isin([21,23,24,29]), 'incumbency'] = 1

    y = sub['vote_rep']
    X = sm.add_constant(sub[['strong','weak','leaner','incumbency']])
    r = Probit(y, X).fit(disp=0)
    print(f'{approach:30s} N={len(sub):4d}: strong={r.params["strong"]:.3f} weak={r.params["weak"]:.3f} leaner={r.params["leaner"]:.3f} inc={r.params["incumbency"]:.3f} const={r.params["const"]:.3f} LL={r.llf:.1f} R2={r.prsquared:.3f}')

print("\nTrue 1974: N=798 strong=1.138 weak=0.721 leaner=0.722 inc=0.474 const=-0.168 LL=-355.2 R2=0.33")

# Investigate 1996
print("\n=== 1996 ANALYSIS ===")
for approach in ['all_pid', 'valid_pid']:
    sub = df[(df['VCF0004']==1996) & (df['VCF0707'].isin([1,2]))].copy()

    if approach == 'valid_pid':
        sub = sub[sub['VCF0301'].isin([1,2,3,4,5,6,7])]

    sub['vote_rep'] = (sub['VCF0707'] == 2).astype(int)
    sub['strong'] = 0; sub['weak'] = 0; sub['leaner'] = 0
    sub.loc[sub['VCF0301']==7, 'strong'] = 1; sub.loc[sub['VCF0301']==1, 'strong'] = -1
    sub.loc[sub['VCF0301']==6, 'weak'] = 1; sub.loc[sub['VCF0301']==2, 'weak'] = -1
    sub.loc[sub['VCF0301']==5, 'leaner'] = 1; sub.loc[sub['VCF0301']==3, 'leaner'] = -1
    sub['incumbency'] = 0
    sub.loc[sub['VCF0902'].isin([12,13,14,19]), 'incumbency'] = -1
    sub.loc[sub['VCF0902'].isin([21,23,24,29]), 'incumbency'] = 1

    y = sub['vote_rep']
    X = sm.add_constant(sub[['strong','weak','leaner','incumbency']])
    r = Probit(y, X).fit(disp=0)
    print(f'{approach:30s} N={len(sub):4d}: strong={r.params["strong"]:.3f} weak={r.params["weak"]:.3f} leaner={r.params["leaner"]:.3f} inc={r.params["incumbency"]:.3f} const={r.params["const"]:.3f} LL={r.llf:.1f} R2={r.prsquared:.3f}')

print("\nTrue 1996: N=1031 strong=1.503 weak=0.865 leaner=0.874 inc=0.742 const=0.142 LL=-373.4 R2=0.48")

# Check if VCF0902 has additional codes we might be missing
print("\n=== VCF0902 unique values by year ===")
for year in [1974, 1976, 1996]:
    sub = df[(df['VCF0004']==year) & (df['VCF0707'].isin([1,2]))]
    vals = sub['VCF0902'].dropna().unique()
    vals.sort()
    print(f'{year}: {vals} (null: {sub["VCF0902"].isna().sum()})')

# Check what VCF0301 values appear for 1974 that are not 1-7
print("\n=== 1974 VCF0301 values ===")
sub74 = df[(df['VCF0004']==1974) & (df['VCF0707'].isin([1,2]))]
print(sub74['VCF0301'].value_counts(dropna=False).sort_index())

# Check 1996 VCF0301
print("\n=== 1996 VCF0301 values ===")
sub96 = df[(df['VCF0004']==1996) & (df['VCF0707'].isin([1,2]))]
print(sub96['VCF0301'].value_counts(dropna=False).sort_index())

# Also check what columns exist
df2 = pd.read_csv('anes_cumulative.csv', low_memory=False, nrows=2)
inc_cols = [c for c in df2.columns if '090' in c]
print('\nVCF090x columns:', inc_cols)

# Check VCF0900a, VCF0900b, etc. if they exist
for c in inc_cols:
    print(f'\n{c} unique values (1974):')
    df3 = pd.read_csv('anes_cumulative.csv', usecols=['VCF0004','VCF0707',c], low_memory=False)
    sub = df3[(df3['VCF0004']==1974) & (df3['VCF0707'].isin([1,2]))]
    print(sub[c].value_counts(dropna=False).sort_index())
