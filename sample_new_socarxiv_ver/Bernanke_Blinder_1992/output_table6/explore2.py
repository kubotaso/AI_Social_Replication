import pandas as pd, numpy as np
from statsmodels.tsa.api import VAR
from linearmodels.iv import IV2SLS
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

macro_a = ['log_industrial_production', 'log_capacity_utilization', 'log_employment']
pol = 'funds_rate'

# Try nominal and real NBR
for nbr_label, nbr_var in [('nominal', 'log_nonborrowed_reserves'), ('real', 'log_nonborrowed_reserves_real')]:
    cols = macro_a + [nbr_var, pol]
    var_data = df.loc['1959-08':'1979-09', cols].dropna()
    m = VAR(var_data)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res = r.resid
    y = res[pol]
    x = res[[nbr_var]]
    z = res[macro_a]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    raw_beta = iv.params.iloc[-1]
    raw_se = iv.std_errors.iloc[-1]
    print(f'{nbr_label} NBR: raw_beta={raw_beta:.4f}, raw_se={raw_se:.4f}')
    print(f'  scaled (x0.01): beta={raw_beta*0.01:.6f}, se={raw_se*0.01:.6f}')

print()
# Now try: scale NBR innovations by 100 (convert to percent) before IV
print("=== With NBR innovations scaled to percent ===")
for nbr_label, nbr_var in [('real', 'log_nonborrowed_reserves_real')]:
    cols = macro_a + [nbr_var, pol]
    var_data = df.loc['1959-08':'1979-09', cols].dropna()
    m = VAR(var_data)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res = r.resid
    # Scale NBR innovations by 100
    res_scaled = res.copy()
    res_scaled[nbr_var] = res_scaled[nbr_var] * 100
    y = res_scaled[pol]
    x = res_scaled[[nbr_var]]
    z = res_scaled[macro_a]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    print(f'beta={iv.params.iloc[-1]:.6f}, SE={iv.std_errors.iloc[-1]:.6f}')

print()
# Try: maybe the paper uses the variable in 100*log form
print("=== Using 100*log for all log variables ===")
for nbr_label, nbr_var in [('real', 'log_nonborrowed_reserves_real')]:
    cols = macro_a + [nbr_var, pol]
    var_data = df.loc['1959-08':'1979-09', cols].copy().dropna()
    # Scale all log variables by 100
    for c in macro_a + [nbr_var]:
        var_data[c] = var_data[c] * 100
    m = VAR(var_data)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res = r.resid
    y = res[pol]
    x = res[[nbr_var]]
    z = res[macro_a]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    print(f'beta={iv.params.iloc[-1]:.6f}, SE={iv.std_errors.iloc[-1]:.6f}')

print()
# Check: what is the OLS coefficient (not IV)?
print("=== OLS (not IV) for comparison ===")
for nbr_label, nbr_var in [('real', 'log_nonborrowed_reserves_real')]:
    cols = macro_a + [nbr_var, pol]
    var_data = df.loc['1959-08':'1979-09', cols].dropna()
    m = VAR(var_data)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res = r.resid
    import statsmodels.api as sm
    ols = sm.OLS(res[pol].values, sm.add_constant(res[nbr_var].values)).fit()
    print(f'OLS: beta={ols.params[1]:.6f}, SE={ols.bse[1]:.6f}')
    # scaled
    print(f'OLS scaled: beta={ols.params[1]*0.01:.6f}, SE={ols.bse[1]*0.01:.6f}')

print()
# Maybe the correct interpretation: run VAR on the actual data
# where NBR is already multiplied by 100
# (so innovations are in percent, and the coefficient is directly the response
# of FUNDS in percentage points to a 1-percent NBR innovation)
print("=== VAR with NBR * 100, then IV directly ===")
for nbr_label, nbr_var in [('real', 'log_nonborrowed_reserves_real')]:
    cols_orig = macro_a + [nbr_var, pol]
    var_data = df.loc['1959-08':'1979-09', cols_orig].copy().dropna()
    # Only scale NBR by 100
    var_data[nbr_var] = var_data[nbr_var] * 100
    m = VAR(var_data)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res = r.resid
    y = res[pol]
    x = res[[nbr_var]]
    z = res[macro_a]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    print(f'beta={iv.params.iloc[-1]:.6f}, SE={iv.std_errors.iloc[-1]:.6f}')
