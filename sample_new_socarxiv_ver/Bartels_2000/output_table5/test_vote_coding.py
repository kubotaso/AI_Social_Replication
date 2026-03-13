"""Test different vote coding approaches."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

def construct_pid_vars(df, pid_col, suffix):
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
    return df

# Check VCF0707 and VCF0706 value distributions for 1960 panel respondents
cdf60 = cdf[cdf['VCF0004']==1960].copy()
cdf58 = cdf[cdf['VCF0004']==1958].copy()
panel60 = cdf60[cdf60['VCF0006a'] < 19600000].copy()
merged60 = panel60.merge(cdf58[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

print('=== 1960 VCF0707 values (post-election House vote) ===')
print(merged60['VCF0707'].value_counts().sort_index())
print('\n=== 1960 VCF0706 values (pre-election House intent) ===')
print(merged60['VCF0706'].value_counts().sort_index())

# VCF0707: 1=Dem, 2=Rep (major party only in CDF)
# VCF0706: 1=Dem, 2=Rep, 4=minor, 7=other/refused

# What if instead of union (use 0706 to fill 0707 NAs), we should
# use ONLY VCF0706 for pre-election years?
# No, that was already tested and gives wrong R2.

# What about using VCF0702 (Congressional vote, Democrat/Republican)
# or VCF0703 (Congressional vote intent)?
vote_cols = [c for c in cdf.columns if c.startswith('VCF070')]
print(f'\nVCF070x columns: {vote_cols}')

for c in vote_cols:
    vals = merged60[c].dropna()
    if len(vals) > 0:
        print(f'\n  {c}: n={len(vals)}, values={sorted(vals.unique())}')

# Check 1976
cdf76 = cdf[cdf['VCF0004']==1976].copy()
cdf74 = cdf[cdf['VCF0004']==1974].copy()
panel76 = cdf76[cdf76['VCF0006a'] < 19760000].copy()
merged76 = panel76.merge(cdf74[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
print('\n\n=== 1976 VCF0707 ===')
print(merged76['VCF0707'].value_counts().sort_index())
print('\n=== 1976 VCF0706 ===')
print(merged76['VCF0706'].value_counts().sort_index())

for c in vote_cols:
    vals = merged76[c].dropna()
    if len(vals) > 0:
        print(f'\n  {c}: n={len(vals)}, values={sorted(vals.unique())}')

# Check 1992
cdf92 = cdf[cdf['VCF0004']==1992].copy()
cdf90 = cdf[cdf['VCF0004']==1990].copy()
panel92 = cdf92[cdf92['VCF0006a'] < 19920000].copy()
merged92 = panel92.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
print('\n\n=== 1992 VCF0707 ===')
print(merged92['VCF0707'].value_counts().sort_index())
print('\n=== 1992 VCF0706 ===')
print(merged92['VCF0706'].value_counts().sort_index())

for c in vote_cols:
    vals = merged92[c].dropna()
    if len(vals) > 0:
        print(f'\n  {c}: n={len(vals)}, values={sorted(vals.unique())}')

# Try using VCF0702 (House vote Democrat) or VCF0703 (House vote Republican)
# Actually check VCF0702 and VCF0703
v702_cols = [c for c in cdf.columns if c.startswith('VCF0702') or c.startswith('VCF0703')]
print(f'\n\nVCF0702/0703 columns: {v702_cols}')
