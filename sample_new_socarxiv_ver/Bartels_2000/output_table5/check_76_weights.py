import pandas as pd
cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
cdf76 = cdf[cdf['VCF0004']==1976]
panel76 = cdf76[cdf76['VCF0006a'] < 19760000]
print('1976 panel VCF0009x dist:')
print(panel76['VCF0009x'].value_counts().sort_index())
print(f'Sum: {panel76["VCF0009x"].sum():.0f}, N: {len(panel76)}')

# Also test hybrid IV approach: 3 directional dummies as instruments
# to verify IV LL = lagged LL
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

def construct_pid_vars(df, pid_col, suffix):
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
    return df

cdf60 = cdf[cdf['VCF0004']==1960].copy()
cdf58 = cdf[cdf['VCF0004']==1958].copy()
panel60 = cdf60[cdf60['VCF0006a'] < 19600000].copy()
merged = panel60.merge(cdf58[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
wt = merged['VCF0009x'].fillna(1.0).astype(int)
expanded = merged.loc[merged.index.repeat(wt)].reset_index(drop=True)
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

curr_vars = ['strong_curr', 'weak_curr', 'lean_curr']
lag_vars = ['strong_lag', 'weak_lag', 'lean_lag']

# IV with 3 directional dummies
predicted_3d = pd.DataFrame(index=valid.index)
for var in curr_vars:
    X_first = sm.add_constant(valid[lag_vars].astype(float))
    ols = sm.OLS(valid[var].astype(float), X_first).fit()
    predicted_3d[var] = ols.predict(X_first)
X_iv_3d = sm.add_constant(predicted_3d[curr_vars].astype(float))
mod_iv_3d = Probit(valid['house_rep'].astype(float), X_iv_3d).fit(disp=0)

# Lagged model
X_lag = sm.add_constant(valid[lag_vars].astype(float))
mod_lag = Probit(valid['house_rep'].astype(float), X_lag).fit(disp=0)

print(f'\n=== 1960 expanded ===')
print(f'Lagged: LL={mod_lag.llf:.1f}, R2={mod_lag.prsquared:.4f}')
print(f'IV 3-dummy: LL={mod_iv_3d.llf:.1f}, R2={mod_iv_3d.prsquared:.4f}')
print(f'IV 3-dummy Strong={mod_iv_3d.params["strong_curr"]:.3f}({mod_iv_3d.bse["strong_curr"]:.3f})')
print(f'IV 3-dummy Weak={mod_iv_3d.params["weak_curr"]:.3f}({mod_iv_3d.bse["weak_curr"]:.3f})')
print(f'IV 3-dummy Lean={mod_iv_3d.params["lean_curr"]:.3f}({mod_iv_3d.bse["lean_curr"]:.3f})')
print(f'IV 3-dummy Int={mod_iv_3d.params["const"]:.3f}({mod_iv_3d.bse["const"]:.3f})')

# IV with 6 PID dummies
for val in [1, 2, 3, 5, 6, 7]:
    valid[f'lag_pid_d{val}'] = (valid['VCF0301_lag'] == val).astype(float)
instr6 = [f'lag_pid_d{v}' for v in [1,2,3,5,6,7]]
predicted_6d = pd.DataFrame(index=valid.index)
for var in curr_vars:
    X_first = sm.add_constant(valid[instr6].astype(float))
    ols = sm.OLS(valid[var].astype(float), X_first).fit()
    predicted_6d[var] = ols.predict(X_first)
X_iv_6d = sm.add_constant(predicted_6d[curr_vars].astype(float))
mod_iv_6d = Probit(valid['house_rep'].astype(float), X_iv_6d).fit(disp=0)
print(f'IV 6-dummy: LL={mod_iv_6d.llf:.1f}, R2={mod_iv_6d.prsquared:.4f}')
print(f'IV 6-dummy Strong={mod_iv_6d.params["strong_curr"]:.3f}({mod_iv_6d.bse["strong_curr"]:.3f})')
print(f'IV 6-dummy Weak={mod_iv_6d.params["weak_curr"]:.3f}({mod_iv_6d.bse["weak_curr"]:.3f})')
print(f'IV 6-dummy Lean={mod_iv_6d.params["lean_curr"]:.3f}({mod_iv_6d.bse["lean_curr"]:.3f})')

print(f'\nTarget IV: Strong=1.715(0.173), Weak=0.728(0.239), Lean=1.081(0.696), Int=0.032(0.057)')
print(f'Target lagged LL=-403.9, R2=0.36')
