"""Test 1992 panel data file for Table 5 replication."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

p92 = pd.read_csv('panel_1992.csv', low_memory=False)

print(f'Total rows: {len(p92)}')
print(f'Columns: {list(p92.columns)}')
print(f'pid_current values: {sorted(p92["pid_current"].dropna().unique())}')
print(f'pid_lagged values: {sorted(p92["pid_lagged"].dropna().unique())}')
print(f'vote_house values: {sorted(p92["vote_house"].dropna().unique())}')

# pid_current is 1-7 scale (matching CDF VCF0301)
# vote_house: 1=Dem, 2=Rep (matching VCF0707)
# pid_lagged is also 1-7 but as int

valid = p92[
    p92['vote_house'].isin([1.0, 2.0]) &
    p92['pid_current'].isin([1,2,3,4,5,6,7]) &
    p92['pid_lagged'].isin([1,2,3,4,5,6,7])
].copy()

print(f'\nValid (vote_house in [1,2], both PIDs in [1-7]): {len(valid)} (target 760)')

valid['house_rep'] = (valid['vote_house'] == 2.0).astype(int)

# Construct PID vars
valid['strong_curr'] = np.where(valid['pid_current'] == 7, 1, np.where(valid['pid_current'] == 1, -1, 0))
valid['weak_curr'] = np.where(valid['pid_current'] == 6, 1, np.where(valid['pid_current'] == 2, -1, 0))
valid['lean_curr'] = np.where(valid['pid_current'] == 5, 1, np.where(valid['pid_current'] == 3, -1, 0))
valid['strong_lag'] = np.where(valid['pid_lagged'] == 7, 1, np.where(valid['pid_lagged'] == 1, -1, 0))
valid['weak_lag'] = np.where(valid['pid_lagged'] == 6, 1, np.where(valid['pid_lagged'] == 2, -1, 0))
valid['lean_lag'] = np.where(valid['pid_lagged'] == 5, 1, np.where(valid['pid_lagged'] == 3, -1, 0))

# Current PID
X = sm.add_constant(valid[['strong_curr','weak_curr','lean_curr']])
mod = Probit(valid['house_rep'].astype(float), X).fit(disp=0)
print(f'\nCurrent PID: LL={mod.llf:.1f}, R2={mod.prsquared:.4f}')
print(f'  Strong={mod.params["strong_curr"]:.3f}({mod.bse["strong_curr"]:.3f}) target=0.975(0.094)')
print(f'  Weak={mod.params["weak_curr"]:.3f}({mod.bse["weak_curr"]:.3f}) target=0.627(0.084)')
print(f'  Lean={mod.params["lean_curr"]:.3f}({mod.bse["lean_curr"]:.3f}) target=0.472(0.098)')
print(f'  Int={mod.params["const"]:.3f}({mod.bse["const"]:.3f}) target=-0.211(0.051)')

# Lagged PID
Xl = sm.add_constant(valid[['strong_lag','weak_lag','lean_lag']])
modl = Probit(valid['house_rep'].astype(float), Xl).fit(disp=0)
print(f'\nLagged PID: LL={modl.llf:.1f}, R2={modl.prsquared:.4f}')
print(f'  Strong={modl.params["strong_lag"]:.3f}({modl.bse["strong_lag"]:.3f}) target=1.061(0.100)')
print(f'  Weak={modl.params["weak_lag"]:.3f}({modl.bse["weak_lag"]:.3f}) target=0.404(0.077)')
print(f'  Lean={modl.params["lean_lag"]:.3f}({modl.bse["lean_lag"]:.3f}) target=0.519(0.101)')
print(f'  Int={modl.params["const"]:.3f}({modl.bse["const"]:.3f}) target=-0.168(0.051)')

# IV probit - 6 dummy approach
instrument_dummies = []
for val in [1, 2, 3, 5, 6, 7]:
    col_name = f'lag_pid_d{val}'
    valid[col_name] = (valid['pid_lagged'] == val).astype(float)
    instrument_dummies.append(col_name)

endog_vars = ['strong_curr','weak_curr','lean_curr']
predicted = pd.DataFrame(index=valid.index)
for var in endog_vars:
    X_first = sm.add_constant(valid[instrument_dummies].astype(float))
    ols_model = sm.OLS(valid[var].astype(float), X_first).fit()
    predicted[var] = ols_model.predict(X_first)
X_second = sm.add_constant(predicted[endog_vars].astype(float))
iv_model = Probit(valid['house_rep'].astype(float), X_second).fit(disp=0, maxiter=1000)
print(f'\nIV (6-dummy): LL={iv_model.llf:.1f}, R2={iv_model.prsquared:.4f}')
print(f'  Strong={iv_model.params["strong_curr"]:.3f}({iv_model.bse["strong_curr"]:.3f}) target=1.516(0.180)')
print(f'  Weak={iv_model.params["weak_curr"]:.3f}({iv_model.bse["weak_curr"]:.3f}) target=-0.225(0.268)')
print(f'  Lean={iv_model.params["lean_curr"]:.3f}({iv_model.bse["lean_curr"]:.3f}) target=1.824(0.513)')
print(f'  Int={iv_model.params["const"]:.3f}({iv_model.bse["const"]:.3f}) target=-0.125(0.053)')

# IV probit - 3 dummy approach
valid['s_lag'] = np.where(valid['pid_lagged'] == 7, 1, np.where(valid['pid_lagged'] == 1, -1, 0))
valid['w_lag'] = np.where(valid['pid_lagged'] == 6, 1, np.where(valid['pid_lagged'] == 2, -1, 0))
valid['l_lag'] = np.where(valid['pid_lagged'] == 5, 1, np.where(valid['pid_lagged'] == 3, -1, 0))
iv_instruments_3 = ['s_lag', 'w_lag', 'l_lag']
predicted3 = pd.DataFrame(index=valid.index)
for var in endog_vars:
    X_first3 = sm.add_constant(valid[iv_instruments_3].astype(float))
    ols_model3 = sm.OLS(valid[var].astype(float), X_first3).fit()
    predicted3[var] = ols_model3.predict(X_first3)
X_second3 = sm.add_constant(predicted3[endog_vars].astype(float))
iv_model3 = Probit(valid['house_rep'].astype(float), X_second3).fit(disp=0, maxiter=1000)
print(f'\nIV (3-dummy): LL={iv_model3.llf:.1f}, R2={iv_model3.prsquared:.4f}')
print(f'  Strong={iv_model3.params["strong_curr"]:.3f}({iv_model3.bse["strong_curr"]:.3f}) target=1.516(0.180)')
print(f'  Weak={iv_model3.params["weak_curr"]:.3f}({iv_model3.bse["weak_curr"]:.3f}) target=-0.225(0.268)')
print(f'  Lean={iv_model3.params["lean_curr"]:.3f}({iv_model3.bse["lean_curr"]:.3f}) target=1.824(0.513)')
print(f'  Int={iv_model3.params["const"]:.3f}({iv_model3.bse["const"]:.3f}) target=-0.125(0.053)')

print(f'\nTarget LL for lagged and IV should be same: -416.2, R2=0.19')
print(f'IV LL with 3-dummy: {iv_model3.llf:.1f} (should = lagged LL {modl.llf:.1f})')
print(f'IV LL with 6-dummy: {iv_model.llf:.1f}')

# Check if pid_lagged might be from original 1990 study (not CDF)
# Compare with CDF approach
cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
cdf92 = cdf[cdf['VCF0004']==1992].copy()
cdf90 = cdf[cdf['VCF0004']==1990].copy()
panel92 = cdf92[cdf92['VCF0006a'] < 19920000].copy()
merged92 = panel92.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
valid_cdf = merged92[
    merged92['VCF0707'].isin([1.0, 2.0]) &
    merged92['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged92['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
print(f'\nCDF approach N: {len(valid_cdf)} vs Panel file N: {len(valid)}')
print(f'CDF total panel: {len(merged92)} vs Panel file total: {len(p92)}')
