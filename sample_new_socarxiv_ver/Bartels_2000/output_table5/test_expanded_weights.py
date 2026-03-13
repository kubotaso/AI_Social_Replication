"""Test expanding rows by frequency weights for 1960 panel."""

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

# 1960 panel with expanded weights
cdf60 = cdf[cdf['VCF0004']==1960].copy()
cdf58 = cdf[cdf['VCF0004']==1958].copy()
panel60 = cdf60[cdf60['VCF0006a'] < 19600000].copy()
merged = panel60.merge(cdf58[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

# Expand by weights BEFORE filtering
wt = merged['VCF0009x'].fillna(1.0).astype(int)
expanded = merged.loc[merged.index.repeat(wt)].reset_index(drop=True)
print(f'Before expansion: {len(merged)} rows')
print(f'After expansion: {len(expanded)} rows')

# Union vote
expanded['house_vote'] = expanded['VCF0707']
mask = expanded['house_vote'].isna() & expanded['VCF0706'].isin([1.0, 2.0])
expanded.loc[mask, 'house_vote'] = expanded.loc[mask, 'VCF0706']

valid = expanded[
    expanded['house_vote'].isin([1.0, 2.0]) &
    expanded['VCF0301'].isin([1,2,3,4,5,6,7]) &
    expanded['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
valid = construct_pid_vars(valid, 'VCF0301', 'curr')
valid = construct_pid_vars(valid, 'VCF0301_lag', 'lag')

print(f'Expanded valid N: {len(valid)} (target 911)')

# Current PID probit
X = sm.add_constant(valid[['strong_curr','weak_curr','lean_curr']])
mod = Probit(valid['house_rep'].astype(float), X).fit(disp=0)
print(f'\nExpanded current: N={len(valid)}, LL={mod.llf:.1f}, R2={mod.prsquared:.4f}')
print(f'  Strong={mod.params["strong_curr"]:.3f}({mod.bse["strong_curr"]:.3f})')
print(f'  Weak={mod.params["weak_curr"]:.3f}({mod.bse["weak_curr"]:.3f})')
print(f'  Lean={mod.params["lean_curr"]:.3f}({mod.bse["lean_curr"]:.3f})')
print(f'  Int={mod.params["const"]:.3f}({mod.bse["const"]:.3f})')

# Lagged PID probit
X_lag = sm.add_constant(valid[['strong_lag','weak_lag','lean_lag']])
mod_lag = Probit(valid['house_rep'].astype(float), X_lag).fit(disp=0)
print(f'\nExpanded lagged: LL={mod_lag.llf:.1f}, R2={mod_lag.prsquared:.4f}')
print(f'  Strong={mod_lag.params["strong_lag"]:.3f}({mod_lag.bse["strong_lag"]:.3f})')
print(f'  Weak={mod_lag.params["weak_lag"]:.3f}({mod_lag.bse["weak_lag"]:.3f})')
print(f'  Lean={mod_lag.params["lean_lag"]:.3f}({mod_lag.bse["lean_lag"]:.3f})')
print(f'  Int={mod_lag.params["const"]:.3f}({mod_lag.bse["const"]:.3f})')

# IV probit with 6 PID dummies
for val in [1, 2, 3, 5, 6, 7]:
    valid[f'lag_pid_d{val}'] = (valid['VCF0301_lag'] == val).astype(float)
instr = [f'lag_pid_d{v}' for v in [1,2,3,5,6,7]]

predicted = pd.DataFrame(index=valid.index)
curr_vars = ['strong_curr', 'weak_curr', 'lean_curr']
for var in curr_vars:
    X_first = sm.add_constant(valid[instr].astype(float))
    ols = sm.OLS(valid[var].astype(float), X_first).fit()
    predicted[var] = ols.predict(X_first)
X_second = sm.add_constant(predicted[curr_vars].astype(float))
mod_iv = Probit(valid['house_rep'].astype(float), X_second).fit(disp=0, maxiter=1000)
print(f'\nExpanded IV: LL={mod_iv.llf:.1f}, R2={mod_iv.prsquared:.4f}')
print(f'  Strong={mod_iv.params["strong_curr"]:.3f}({mod_iv.bse["strong_curr"]:.3f})')
print(f'  Weak={mod_iv.params["weak_curr"]:.3f}({mod_iv.bse["weak_curr"]:.3f})')
print(f'  Lean={mod_iv.params["lean_curr"]:.3f}({mod_iv.bse["lean_curr"]:.3f})')
print(f'  Int={mod_iv.params["const"]:.3f}({mod_iv.bse["const"]:.3f})')

print(f'\nTarget current: N=911, LL=-372.7, R2=0.41, Strong=1.358(0.094), Weak=1.028(0.083), Lean=0.855(0.131), Int=0.035(0.053)')
print(f'Target lagged: LL=-403.9, R2=0.36, Strong=1.363(0.092), Weak=0.842(0.078), Lean=0.564(0.125), Int=0.068(0.051)')
print(f'Target IV: LL=-403.9, R2=0.36, Strong=1.715(0.173), Weak=0.728(0.239), Lean=1.081(0.696), Int=0.032(0.057)')

# Also try VCF0707 only (not union) with expanded weights
merged2 = panel60.merge(cdf58[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
wt2 = merged2['VCF0009x'].fillna(1.0).astype(int)
expanded2 = merged2.loc[merged2.index.repeat(wt2)].reset_index(drop=True)
valid2 = expanded2[
    expanded2['VCF0707'].isin([1.0, 2.0]) &
    expanded2['VCF0301'].isin([1,2,3,4,5,6,7]) &
    expanded2['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
valid2['house_rep'] = (valid2['VCF0707'] == 2.0).astype(int)
valid2 = construct_pid_vars(valid2, 'VCF0301', 'curr')
X2 = sm.add_constant(valid2[['strong_curr','weak_curr','lean_curr']])
mod2 = Probit(valid2['house_rep'].astype(float), X2).fit(disp=0)
print(f'\nExpanded VCF0707-only: N={len(valid2)}, LL={mod2.llf:.1f}, R2={mod2.prsquared:.4f}')
print(f'  Strong={mod2.params["strong_curr"]:.3f}({mod2.bse["strong_curr"]:.3f})')
print(f'  Weak={mod2.params["weak_curr"]:.3f}({mod2.bse["weak_curr"]:.3f})')
print(f'  Lean={mod2.params["lean_curr"]:.3f}({mod2.bse["lean_curr"]:.3f})')
print(f'  Int={mod2.params["const"]:.3f}({mod2.bse["const"]:.3f})')
