"""Test VCF0707-only with expanded weights for 1960."""

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

cdf60 = cdf[cdf['VCF0004']==1960].copy()
cdf58 = cdf[cdf['VCF0004']==1958].copy()
panel60 = cdf60[cdf60['VCF0006a'] < 19600000].copy()
merged = panel60.merge(cdf58[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

# Expand by weights
wt = merged['VCF0009x'].fillna(1.0).astype(int)
expanded = merged.loc[merged.index.repeat(wt)].reset_index(drop=True)

# VCF0707 only (no union)
valid = expanded[
    expanded['VCF0707'].isin([1.0, 2.0]) &
    expanded['VCF0301'].isin([1,2,3,4,5,6,7]) &
    expanded['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
valid['house_rep'] = (valid['VCF0707'] == 2.0).astype(int)
valid = construct_pid_vars(valid, 'VCF0301', 'curr')
valid = construct_pid_vars(valid, 'VCF0301_lag', 'lag')

print(f'VCF0707-only expanded: N={len(valid)}')
X = sm.add_constant(valid[['strong_curr','weak_curr','lean_curr']])
mod = Probit(valid['house_rep'].astype(float), X).fit(disp=0)
print(f'Current: LL={mod.llf:.1f}, R2={mod.prsquared:.4f}')
print(f'  Strong={mod.params["strong_curr"]:.3f}({mod.bse["strong_curr"]:.3f}) diff={abs(mod.params["strong_curr"]-1.358):.3f}')
print(f'  Weak={mod.params["weak_curr"]:.3f}({mod.bse["weak_curr"]:.3f}) diff={abs(mod.params["weak_curr"]-1.028):.3f}')
print(f'  Lean={mod.params["lean_curr"]:.3f}({mod.bse["lean_curr"]:.3f}) diff={abs(mod.params["lean_curr"]-0.855):.3f}')
print(f'  Int={mod.params["const"]:.3f}({mod.bse["const"]:.3f}) diff={abs(mod.params["const"]-0.035):.3f}')

X_lag = sm.add_constant(valid[['strong_lag','weak_lag','lean_lag']])
mod_lag = Probit(valid['house_rep'].astype(float), X_lag).fit(disp=0)
print(f'\nLagged: LL={mod_lag.llf:.1f}, R2={mod_lag.prsquared:.4f}')
print(f'  Strong={mod_lag.params["strong_lag"]:.3f}({mod_lag.bse["strong_lag"]:.3f}) diff={abs(mod_lag.params["strong_lag"]-1.363):.3f}')
print(f'  Weak={mod_lag.params["weak_lag"]:.3f}({mod_lag.bse["weak_lag"]:.3f}) diff={abs(mod_lag.params["weak_lag"]-0.842):.3f}')
print(f'  Lean={mod_lag.params["lean_lag"]:.3f}({mod_lag.bse["lean_lag"]:.3f}) diff={abs(mod_lag.params["lean_lag"]-0.564):.3f}')
print(f'  Int={mod_lag.params["const"]:.3f}({mod_lag.bse["const"]:.3f}) diff={abs(mod_lag.params["const"]-0.068):.3f}')

# Compare with union expanded (attempt 6 approach)
print('\n--- Union expanded (attempt 6): ---')
expanded2 = merged.loc[merged.index.repeat(wt)].reset_index(drop=True)
expanded2['house_vote'] = expanded2['VCF0707']
mask = expanded2['house_vote'].isna() & expanded2['VCF0706'].isin([1.0, 2.0])
expanded2.loc[mask, 'house_vote'] = expanded2.loc[mask, 'VCF0706']
valid2 = expanded2[
    expanded2['house_vote'].isin([1.0, 2.0]) &
    expanded2['VCF0301'].isin([1,2,3,4,5,6,7]) &
    expanded2['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
valid2['house_rep'] = (valid2['house_vote'] == 2.0).astype(int)
valid2 = construct_pid_vars(valid2, 'VCF0301', 'curr')
X2 = sm.add_constant(valid2[['strong_curr','weak_curr','lean_curr']])
mod2 = Probit(valid2['house_rep'].astype(float), X2).fit(disp=0)
print(f'N={len(valid2)}, LL={mod2.llf:.1f}, R2={mod2.prsquared:.4f}')
print(f'  Strong={mod2.params["strong_curr"]:.3f} diff={abs(mod2.params["strong_curr"]-1.358):.3f}')
print(f'  Weak={mod2.params["weak_curr"]:.3f} diff={abs(mod2.params["weak_curr"]-1.028):.3f}')

print('\nTarget: N=911, LL=-372.7, R2=0.41')
print('  Strong=1.358, Weak=1.028, Lean=0.855, Int=0.035')
