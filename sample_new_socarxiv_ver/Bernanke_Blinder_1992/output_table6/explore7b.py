import pandas as pd, numpy as np
from statsmodels.tsa.api import VAR
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

macro_a = ['log_industrial_production', 'log_capacity_utilization', 'log_employment']
nbr = 'log_nonborrowed_reserves_real'
pol = 'funds_rate'
cols = macro_a + [nbr, pol]

# 1. 6-var VAR with CPI
print("=== 6-var VAR with CPI (Set A) ===")
for pol_name, pol_col in [('FUNDS', 'funds_rate'), ('FFBOND', 'ffbond')]:
    cols6 = macro_a + ['log_cpi', nbr, pol_col]
    vd = df.loc['1959-08':'1979-09', cols6].dropna()
    m = VAR(vd)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res = r.resid
    y = res[pol_col]
    x = res[[nbr]]
    z = res[macro_a]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    print(f'  {pol_name}: scaled={iv.params.iloc[-1]*0.01:.6f}, SE={iv.std_errors.iloc[-1]*0.01:.6f}')

# 2. Try with capacity utilization in levels (not log)
print("\n=== Cap util in levels ===")
macro_a_lev = ['log_industrial_production', 'capacity_utilization', 'log_employment']
for pol_name, pol_col in [('FUNDS', 'funds_rate'), ('FFBOND', 'ffbond')]:
    cols_lev = macro_a_lev + [nbr, pol_col]
    vd = df.loc['1959-08':'1979-09', cols_lev].dropna()
    m = VAR(vd)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res = r.resid
    y = res[pol_col]
    x = res[[nbr]]
    z = res[macro_a_lev]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    print(f'  {pol_name}: scaled={iv.params.iloc[-1]*0.01:.6f}, SE={iv.std_errors.iloc[-1]*0.01:.6f}')

# 3. What if unemployment is in levels (not log)? The paper says
# "unemployment rate" for Set B - this is already in levels
print("\n=== Set B results check ===")
macro_b = ['unemp_male_2554', 'log_housing_starts', 'log_personal_income_real']
for pol_name, pol_col in [('FUNDS', 'funds_rate'), ('FFBOND', 'ffbond')]:
    cols_b = macro_b + [nbr, pol_col]
    vd = df.loc['1959-08':'1979-09', cols_b].dropna()
    m = VAR(vd)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res = r.resid
    y = res[pol_col]
    x = res[[nbr]]
    z = res[macro_b]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    fs = sm.OLS(res[nbr].values, sm.add_constant(res[macro_b].values)).fit()
    print(f'  {pol_name}: scaled={iv.params.iloc[-1]*0.01:.6f}, SE={iv.std_errors.iloc[-1]*0.01:.6f}, FS_F={fs.fvalue:.2f}')

# 4. What if the paper includes a constant AND trend in the VAR?
print("\n=== VAR with constant + trend (Set A) ===")
for pol_name, pol_col in [('FUNDS', 'funds_rate'), ('FFBOND', 'ffbond')]:
    cols_ct = macro_a + [nbr, pol_col]
    vd = df.loc['1959-08':'1979-09', cols_ct].dropna()
    m = VAR(vd)
    r = m.fit(maxlags=6, ic=None, trend='ct')
    res = r.resid
    y = res[pol_col]
    x = res[[nbr]]
    z = res[macro_a]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    fs = sm.OLS(res[nbr].values, sm.add_constant(res[macro_a].values)).fit()
    print(f'  {pol_name}: scaled={iv.params.iloc[-1]*0.01:.6f}, SE={iv.std_errors.iloc[-1]*0.01:.6f}, FS_F={fs.fvalue:.2f}')

# 5. Try: use unemp_rate instead of unemp_male_2554 for Set B
print("\n=== Set B with unemp_rate instead of unemp_male ===")
macro_b2 = ['unemp_rate', 'log_housing_starts', 'log_personal_income_real']
for pol_name, pol_col in [('FUNDS', 'funds_rate'), ('FFBOND', 'ffbond')]:
    cols_b2 = macro_b2 + [nbr, pol_col]
    vd = df.loc['1959-08':'1979-09', cols_b2].dropna()
    m = VAR(vd)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res = r.resid
    y = res[pol_col]
    x = res[[nbr]]
    z = res[macro_b2]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    print(f'  {pol_name}: scaled={iv.params.iloc[-1]*0.01:.6f}, SE={iv.std_errors.iloc[-1]*0.01:.6f}')

# 6. Try: what if we DON'T include CPI lags in the VAR
# but the paper's Table 1 uses "six lags of the log of CPI, six lags of the
# forecasted variable, and six lags of each of [the interest rate variables]"
# For Table 6, the spec is simpler: 5-var VAR with 6 lags
# But what if "six lags" means something different? Like: the VAR starts at lag 1?
# In statsmodels, VAR(maxlags=6) means lags 1 through 6.
# That should be correct.

# 7. Final check: does statsmodels VAR include constant by default?
print("\n=== Statsmodels VAR details ===")
vd = df.loc['1959-08':'1979-09', macro_a + [nbr, 'funds_rate']].dropna()
m = VAR(vd)
r = m.fit(maxlags=6, ic=None, trend='c')
print(f'nobs: {r.nobs}')
print(f'k_ar: {r.k_ar}')
print(f'k_trend: {r.k_trend}')
print(f'names: {r.names}')
print(f'dates: {r.dates[0]} to {r.dates[-1]}')
print(f'resid shape: {r.resid.shape}')
print(f'resid dates: {r.resid.index[0]} to {r.resid.index[-1]}')
