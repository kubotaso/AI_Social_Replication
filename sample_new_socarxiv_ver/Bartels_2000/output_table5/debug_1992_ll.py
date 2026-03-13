import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

# 1992 panel
cdf92 = cdf[cdf['VCF0004']==1992].copy()
cdf92_panel = cdf92[cdf92['VCF0006a'] < 19920000].copy()
cdf90 = cdf[cdf['VCF0004']==1990].copy()

merged = cdf92_panel.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

valid = merged[
    merged['VCF0707'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()

print(f'N={len(valid)}')
print(f'Dem: {(valid["VCF0707"]==1.0).sum()}, Rep: {(valid["VCF0707"]==2.0).sum()}')
print(f'Dem pct: {(valid["VCF0707"]==1.0).mean():.4f}')

# Check if there's a weight variable
print(f'\nWeight columns: {[c for c in cdf.columns if "weight" in c.lower() or "wt" in c.lower() or "VCF0009" in c]}')

# Try with different filtering
# What if Bartels used VCF0706 instead of VCF0707 for 1992?
valid706 = merged[
    merged['VCF0706'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()

print(f'\nWith VCF0706: N={len(valid706)}')
valid706['house_rep'] = (valid706['VCF0706'] == 2.0).astype(int)
valid706['strong'] = np.where(valid706['VCF0301']==7, 1, np.where(valid706['VCF0301']==1, -1, 0))
valid706['weak'] = np.where(valid706['VCF0301']==6, 1, np.where(valid706['VCF0301']==2, -1, 0))
valid706['lean'] = np.where(valid706['VCF0301']==5, 1, np.where(valid706['VCF0301']==3, -1, 0))
X706 = sm.add_constant(valid706[['strong','weak','lean']])
mod706 = Probit(valid706['house_rep'], X706).fit(disp=0)
print(f'VCF0706: LL={mod706.llf:.1f}, R2={mod706.prsquared:.4f}, Strong={mod706.params["strong"]:.3f}')

# What if we need to use a different PID variable?
# VCF0303 is pre-election PID, VCF0301 is post-election
print(f'\nVCF0303 in data: {"VCF0303" in cdf.columns}')
if 'VCF0303' in cdf.columns:
    valid_pre = merged[
        merged['VCF0707'].isin([1.0, 2.0]) &
        merged['VCF0303'].isin([1,2,3,4,5,6,7]) &
        merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
    ].copy()
    print(f'With VCF0303 (pre-election PID): N={len(valid_pre)}')
    if len(valid_pre) > 0:
        valid_pre['house_rep'] = (valid_pre['VCF0707'] == 2.0).astype(int)
        valid_pre['strong'] = np.where(valid_pre['VCF0303']==7, 1, np.where(valid_pre['VCF0303']==1, -1, 0))
        valid_pre['weak'] = np.where(valid_pre['VCF0303']==6, 1, np.where(valid_pre['VCF0303']==2, -1, 0))
        valid_pre['lean'] = np.where(valid_pre['VCF0303']==5, 1, np.where(valid_pre['VCF0303']==3, -1, 0))
        X_pre = sm.add_constant(valid_pre[['strong','weak','lean']])
        mod_pre = Probit(valid_pre['house_rep'], X_pre).fit(disp=0)
        print(f'VCF0303: LL={mod_pre.llf:.1f}, R2={mod_pre.prsquared:.4f}')
        print(f'Strong={mod_pre.params["strong"]:.3f}({mod_pre.bse["strong"]:.3f})')
        print(f'Weak={mod_pre.params["weak"]:.3f}({mod_pre.bse["weak"]:.3f})')
        print(f'Lean={mod_pre.params["lean"]:.3f}({mod_pre.bse["lean"]:.3f})')
        print(f'Int={mod_pre.params["const"]:.3f}({mod_pre.bse["const"]:.3f})')

# What about VCF0301a (strength of party ID)?
for alt in ['VCF0301a', 'VCF0302', 'VCF0305']:
    if alt in cdf.columns:
        print(f'\n{alt} in CDF: yes')
        print(f'{alt} dist for 1992 panel: {merged[alt].value_counts(dropna=False).sort_index().to_dict()}')

print(f'\nTarget: N=760, LL=-408.2, R2=0.20, Strong=0.975(0.094), Weak=0.627(0.084), Lean=0.472(0.098), Int=-0.211(0.051)')
print(f'Current: N={len(valid)}, LL=-393.9, R2=0.2226')
