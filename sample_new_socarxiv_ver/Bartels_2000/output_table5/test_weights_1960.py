"""Test weighted probit for 1960 - weights range 1-4, possibly frequency weights."""

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

# 1960 panel
cdf60 = cdf[cdf['VCF0004']==1960].copy()
cdf58 = cdf[cdf['VCF0004']==1958].copy()
panel60 = cdf60[cdf60['VCF0006a'] < 19600000].copy()
merged = panel60.merge(cdf58[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

# Union vote
merged['house_vote'] = merged['VCF0707']
mask = merged['house_vote'].isna() & merged['VCF0706'].isin([1.0, 2.0])
merged.loc[mask, 'house_vote'] = merged.loc[mask, 'VCF0706']

valid = merged[
    merged['house_vote'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
valid = construct_pid_vars(valid, 'VCF0301', 'curr')
valid = construct_pid_vars(valid, 'VCF0301_lag', 'lag')

print(f'1960 N={len(valid)}')
print(f'Weight distribution:')
print(valid['VCF0009x'].value_counts().sort_index())
print(f'Sum of weights: {valid["VCF0009x"].sum():.0f}')

# Unweighted
X = sm.add_constant(valid[['strong_curr','weak_curr','lean_curr']])
mod_uw = Probit(valid['house_rep'].astype(float), X).fit(disp=0)
print(f'\nUnweighted: N={len(valid)}, LL={mod_uw.llf:.1f}, R2={mod_uw.prsquared:.4f}')
print(f'  Strong={mod_uw.params["strong_curr"]:.3f}({mod_uw.bse["strong_curr"]:.3f})')
print(f'  Weak={mod_uw.params["weak_curr"]:.3f}({mod_uw.bse["weak_curr"]:.3f})')
print(f'  Lean={mod_uw.params["lean_curr"]:.3f}({mod_uw.bse["lean_curr"]:.3f})')
print(f'  Int={mod_uw.params["const"]:.3f}({mod_uw.bse["const"]:.3f})')

# Weighted with freq_weights
wt = valid['VCF0009x'].values
mod_wt = Probit(valid['house_rep'].astype(float), X).fit(disp=0, freq_weights=wt)
print(f'\nWeighted (freq): N_eff={wt.sum():.0f}, LL={mod_wt.llf:.1f}, R2={mod_wt.prsquared:.4f}')
print(f'  Strong={mod_wt.params["strong_curr"]:.3f}({mod_wt.bse["strong_curr"]:.3f})')
print(f'  Weak={mod_wt.params["weak_curr"]:.3f}({mod_wt.bse["weak_curr"]:.3f})')
print(f'  Lean={mod_wt.params["lean_curr"]:.3f}({mod_wt.bse["lean_curr"]:.3f})')
print(f'  Int={mod_wt.params["const"]:.3f}({mod_wt.bse["const"]:.3f})')

# Also try lagged model weighted
X_lag = sm.add_constant(valid[['strong_lag','weak_lag','lean_lag']])
mod_lag_wt = Probit(valid['house_rep'].astype(float), X_lag).fit(disp=0, freq_weights=wt)
print(f'\nWeighted lagged: LL={mod_lag_wt.llf:.1f}, R2={mod_lag_wt.prsquared:.4f}')
print(f'  Strong={mod_lag_wt.params["strong_lag"]:.3f}({mod_lag_wt.bse["strong_lag"]:.3f})')
print(f'  Weak={mod_lag_wt.params["weak_lag"]:.3f}({mod_lag_wt.bse["weak_lag"]:.3f})')
print(f'  Lean={mod_lag_wt.params["lean_lag"]:.3f}({mod_lag_wt.bse["lean_lag"]:.3f})')
print(f'  Int={mod_lag_wt.params["const"]:.3f}({mod_lag_wt.bse["const"]:.3f})')

print(f'\nTarget current: N=911, LL=-372.7, R2=0.41, Strong=1.358, Weak=1.028, Lean=0.855, Int=0.035')
print(f'Target lagged: LL=-403.9, R2=0.36, Strong=1.363, Weak=0.842, Lean=0.564, Int=0.068')

# Check 1976 weights
print('\n\n=== 1976 ===')
cdf76 = cdf[cdf['VCF0004']==1976].copy()
panel76 = cdf76[cdf76['VCF0006a'] < 19760000].copy()
cdf74 = cdf[cdf['VCF0004']==1974].copy()
m76 = panel76.merge(cdf74[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
m76['house_vote'] = m76['VCF0707']
mask76 = m76['house_vote'].isna() & m76['VCF0706'].isin([1.0, 2.0])
m76.loc[mask76, 'house_vote'] = m76.loc[mask76, 'VCF0706']
v76 = m76[m76['house_vote'].isin([1.0,2.0]) & m76['VCF0301'].isin([1,2,3,4,5,6,7]) & m76['VCF0301_lag'].isin([1,2,3,4,5,6,7])].copy()
v76['house_rep'] = (v76['house_vote'] == 2.0).astype(int)
v76 = construct_pid_vars(v76, 'VCF0301', 'curr')
print(f'1976 weight distribution:')
print(v76['VCF0009x'].value_counts().sort_index())
print(f'Sum: {v76["VCF0009x"].sum():.0f}')
