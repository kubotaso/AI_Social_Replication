"""
Explore: What if we use LAGGED macro variables as additional instruments
in the IV regression (beyond just the contemporary innovations)?
Or use current and lagged innovations?
"""
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
var_data = df.loc['1959-08':'1979-09', cols].dropna()
m = VAR(var_data)
r = m.fit(maxlags=6, ic=None, trend='c')
res = r.resid

# 1. What if we add lagged innovations as instruments?
print("=== Adding lagged innovations as instruments ===")
for n_extra_lags in [0, 1, 2, 3, 6]:
    # Build instrument matrix
    z_list = [res[macro_a]]  # current innovations
    for lag in range(1, n_extra_lags + 1):
        z_list.append(res[macro_a].shift(lag))
    z_combined = pd.concat(z_list, axis=1).dropna()
    common = z_combined.index

    y = res.loc[common, pol]
    x = res.loc[common, [nbr]]
    z = z_combined
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    scaled_b = iv.params.iloc[-1] * 0.01
    scaled_se = iv.std_errors.iloc[-1] * 0.01
    fs_z = sm.add_constant(z.values)
    fs = sm.OLS(res.loc[common, nbr].values, fs_z).fit()
    print(f'extra_lags={n_extra_lags}: scaled_beta={scaled_b:8.4f}, SE={scaled_se:8.4f}, N={len(y)}, FS_F={fs.fvalue:.2f}')

# 2. What if we include lag 1 of RAW macro variables as instruments?
print("\n=== Using lag 1 of raw macro variables as instruments ===")
z_raw_lags = pd.concat([var_data[mc].shift(1) for mc in macro_a], axis=1).iloc[7:]
y2 = var_data.loc[z_raw_lags.index, pol]
x2 = var_data.loc[z_raw_lags.index, [nbr]]
exog2 = pd.DataFrame(np.ones(len(y2)), index=y2.index, columns=['const'])
iv2 = IV2SLS(dependent=y2, exog=exog2, endog=x2, instruments=z_raw_lags).fit()
print(f'beta={iv2.params.iloc[-1]:.6f}')

# 3. Try: what if the innovations are from equations estimated SEPARATELY
# (not from a full VAR)? I.e., each variable is regressed on its own lags
# plus lags of the other 4 variables.
# Wait - that IS the VAR. But what if each variable is only regressed on
# its own lags? (univariate AR)
print("\n=== Univariate AR(6) innovations ===")
ar_resids = {}
for c in cols:
    s = var_data[c]
    X_ar = pd.concat([s.shift(i) for i in range(1, 7)], axis=1).dropna()
    X_ar = sm.add_constant(X_ar)
    y_ar = s.loc[X_ar.index]
    ar_fit = sm.OLS(y_ar.values, X_ar.values).fit()
    ar_resids[c] = pd.Series(ar_fit.resid, index=X_ar.index)

common_ar = ar_resids[pol].index
for c in cols:
    common_ar = common_ar.intersection(ar_resids[c].index)

y_ar = ar_resids[pol].loc[common_ar]
x_ar = pd.DataFrame(ar_resids[nbr].loc[common_ar], columns=[nbr])
z_ar = pd.DataFrame({c: ar_resids[c].loc[common_ar] for c in macro_a})
exog_ar = pd.DataFrame(np.ones(len(y_ar)), index=y_ar.index, columns=['const'])
iv_ar = IV2SLS(dependent=y_ar, exog=exog_ar, endog=x_ar, instruments=z_ar).fit()
print(f'AR(6) innovations: raw={iv_ar.params.iloc[-1]:.4f}, scaled={iv_ar.params.iloc[-1]*0.01:.6f}')
fs_ar = sm.OLS(ar_resids[nbr].loc[common_ar].values,
               sm.add_constant(z_ar.values)).fit()
print(f'FS_F={fs_ar.fvalue:.2f}, R2={fs_ar.rsquared:.4f}')

# 4. Try: exclude capacity utilization (which is very collinear with IP)
# and use only IP and employment as instruments
print("\n=== Just 2 instruments: IP and employment ===")
macro_2 = ['log_industrial_production', 'log_employment']
z_2 = res[macro_2]
exog_2 = pd.DataFrame(np.ones(len(res)), index=res.index, columns=['const'])
iv_2 = IV2SLS(dependent=res[pol], exog=exog_2, endog=res[[nbr]], instruments=z_2).fit()
print(f'2 instr: raw={iv_2.params.iloc[-1]:.4f}, scaled={iv_2.params.iloc[-1]*0.01:.6f}')

# 5. Try: what if the nine-variable VAR from Table 1 is used to extract innovations,
# and then we pick 3 macro innovations from that VAR?
print("\n=== 6-variable VAR (all Set A macro + CPI + NBR + FUNDS) ===")
cols6 = macro_a + ['log_cpi', nbr, pol]
var_data6 = df.loc['1959-08':'1979-09', cols6].dropna()
m6 = VAR(var_data6)
r6 = m6.fit(maxlags=6, ic=None, trend='c')
res6 = r6.resid
y6 = res6[pol]
x6 = res6[[nbr]]
z6 = res6[macro_a]
exog6 = pd.DataFrame(np.ones(len(y6)), index=y6.index, columns=['const'])
iv6 = IV2SLS(dependent=y6, exog=exog6, endog=x6, instruments=z6).fit()
print(f'6-var: raw={iv6.params.iloc[-1]:.4f}, scaled={iv6.params.iloc[-1]*0.01:.6f}')
fs6 = sm.OLS(res6[nbr].values, sm.add_constant(res6[macro_a].values)).fit()
print(f'FS_F={fs6.fvalue:.2f}')

# Also try with CPI as additional instrument
z6_with_cpi = res6[macro_a + ['log_cpi']]
iv6b = IV2SLS(dependent=y6, exog=exog6, endog=x6, instruments=z6_with_cpi).fit()
print(f'6-var + CPI instr: raw={iv6b.params.iloc[-1]:.4f}, scaled={iv6b.params.iloc[-1]*0.01:.6f}')
fs6b = sm.OLS(res6[nbr].values, sm.add_constant(res6[macro_a + ['log_cpi']].values)).fit()
print(f'FS_F={fs6b.fvalue:.2f}')

# 6. Try with VERY short sample (after 1965 when funds rate was more active)
print("\n=== Shorter sample 1965-01 to 1979-09 ===")
var_data_short = df.loc['1965-01':'1979-09', cols].dropna()
m_s = VAR(var_data_short)
r_s = m_s.fit(maxlags=6, ic=None, trend='c')
res_s = r_s.resid
y_s = res_s[pol]
x_s = res_s[[nbr]]
z_s = res_s[macro_a]
exog_s = pd.DataFrame(np.ones(len(y_s)), index=y_s.index, columns=['const'])
iv_s = IV2SLS(dependent=y_s, exog=exog_s, endog=x_s, instruments=z_s).fit()
print(f'Short: raw={iv_s.params.iloc[-1]:.4f}, scaled={iv_s.params.iloc[-1]*0.01:.6f}')
fs_s = sm.OLS(res_s[nbr].values, sm.add_constant(res_s[macro_a].values)).fit()
print(f'FS_F={fs_s.fvalue:.2f}')
