import pandas as pd, numpy as np
from statsmodels.tsa.api import VAR
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

macro_a = ['log_industrial_production', 'log_capacity_utilization', 'log_employment']
nbr_var = 'log_nonborrowed_reserves_real'
pol = 'funds_rate'
cols = macro_a + [nbr_var, pol]

# Try different trend specifications
for trend in ['c', 'ct', 'n', 'ctt']:
    try:
        var_data = df.loc['1959-08':'1979-09', cols].dropna()
        m = VAR(var_data)
        r = m.fit(maxlags=6, ic=None, trend=trend)
        res = r.resid
        y = res[pol]
        x = res[[nbr_var]]
        z = res[macro_a]
        exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
        iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
        raw_beta = iv.params.iloc[-1]
        raw_se = iv.std_errors.iloc[-1]
        fs = sm.OLS(res[nbr_var].values, sm.add_constant(res[macro_a].values)).fit()
        print(f'trend={trend:4s}: raw_beta={raw_beta:8.4f}, SE={raw_se:8.4f}, '
              f'scaled_beta={raw_beta*0.01:.6f}, scaled_se={raw_se*0.01:.6f}, '
              f'FS_F={fs.fvalue:.3f}, FS_R2={fs.rsquared:.4f}')
    except Exception as e:
        print(f'trend={trend}: ERROR {e}')

print()
# Try different lag lengths
for lags in [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]:
    var_data = df.loc['1959-08':'1979-09', cols].dropna()
    m = VAR(var_data)
    r = m.fit(maxlags=lags, ic=None, trend='c')
    res = r.resid
    y = res[pol]
    x = res[[nbr_var]]
    z = res[macro_a]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    raw_beta = iv.params.iloc[-1]
    raw_se = iv.std_errors.iloc[-1]
    fs = sm.OLS(res[nbr_var].values, sm.add_constant(res[macro_a].values)).fit()
    print(f'lags={lags:2d}: raw_beta={raw_beta:8.4f}, SE={raw_se:8.4f}, '
          f'scaled_beta={raw_beta*0.01:.6f}, scaled_se={raw_se*0.01:.6f}, '
          f'FS_F={fs.fvalue:.3f}, FS_R2={fs.rsquared:.4f}, N={len(y)}')

print()
# Try: what if we include time trend in VAR AND use fewer lags?
# Also try: include seasonal dummies
print("=== With seasonal dummies in VAR ===")
var_data = df.loc['1959-08':'1979-09', cols].dropna()
# Add month dummies
for m_i in range(1, 12):
    var_data[f'month_{m_i}'] = (var_data.index.month == m_i).astype(float)
exog_cols = [f'month_{m_i}' for m_i in range(1, 12)]

m2 = VAR(var_data[cols], exog=var_data[exog_cols])
r2 = m2.fit(maxlags=6, ic=None, trend='c')
res2 = r2.resid
y2 = res2[pol]
x2 = res2[[nbr_var]]
z2 = res2[macro_a]
exog2 = pd.DataFrame(np.ones(len(y2)), index=y2.index, columns=['const'])
iv2 = IV2SLS(dependent=y2, exog=exog2, endog=x2, instruments=z2).fit()
print(f'beta={iv2.params.iloc[-1]:.4f}, SE={iv2.std_errors.iloc[-1]:.4f}')
fs2 = sm.OLS(res2[nbr_var].values, sm.add_constant(res2[macro_a].values)).fit()
print(f'FS_F={fs2.fvalue:.3f}, FS_R2={fs2.rsquared:.4f}')

print()
print("=== Correlation of innovations (Set A, 6 lags, trend=c) ===")
var_data = df.loc['1959-08':'1979-09', cols].dropna()
m = VAR(var_data)
r = m.fit(maxlags=6, ic=None, trend='c')
res = r.resid
# Check partial correlations between NBR and macro innovations
# after controlling for FUNDS
print("Correlation NBR innovation with macro innovations:")
for c in macro_a:
    print(f"  {c}: {res[nbr_var].corr(res[c]):.4f}")
print(f"Correlation NBR innovation with FUNDS: {res[nbr_var].corr(res[pol]):.4f}")

# Try: first difference all variables before VAR
print()
print("=== First-differenced variables in VAR ===")
var_data_d = df.loc['1959-08':'1979-09', cols].diff().dropna()
m_d = VAR(var_data_d)
r_d = m_d.fit(maxlags=6, ic=None, trend='c')
res_d = r_d.resid
y_d = res_d[pol]
x_d = res_d[[nbr_var]]
z_d = res_d[macro_a]
exog_d = pd.DataFrame(np.ones(len(y_d)), index=y_d.index, columns=['const'])
iv_d = IV2SLS(dependent=y_d, exog=exog_d, endog=x_d, instruments=z_d).fit()
print(f'beta={iv_d.params.iloc[-1]:.6f}, SE={iv_d.std_errors.iloc[-1]:.6f}')
fs_d = sm.OLS(res_d[nbr_var].values, sm.add_constant(res_d[macro_a].values)).fit()
print(f'FS_F={fs_d.fvalue:.3f}, FS_R2={fs_d.rsquared:.4f}')
