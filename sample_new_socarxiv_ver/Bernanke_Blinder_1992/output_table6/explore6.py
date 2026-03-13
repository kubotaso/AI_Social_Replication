"""
Explore: Try different sample periods, different starting points,
and see if any configuration gives results closer to the paper.
Also try: what if the VAR is estimated on a DIFFERENT sample than the IV?
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

# 1. Try different sample start dates
print("=== Different sample starts (end=1979-09) ===")
for start in ['1955-01', '1957-01', '1958-01', '1959-01', '1959-07', '1959-08',
              '1960-01', '1961-01', '1962-01', '1965-01']:
    var_data = df.loc[start:'1979-09', cols].dropna()
    if len(var_data) < 20:
        continue
    m = VAR(var_data)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res = r.resid
    y = res[pol]
    x = res[[nbr]]
    z = res[macro_a]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    beta_sc = iv.params.iloc[-1] * 0.01
    se_sc = iv.std_errors.iloc[-1] * 0.01
    fs = sm.OLS(res[nbr].values, sm.add_constant(res[macro_a].values)).fit()
    print(f'start={start}: beta={beta_sc:8.4f}, se={se_sc:8.4f}, N={len(y)}, FS_F={fs.fvalue:.2f}')

# 2. Try different sample end dates
print("\n=== Different sample ends (start=1959-08) ===")
for end in ['1978-06', '1978-12', '1979-06', '1979-09', '1979-10', '1979-12', '1980-06']:
    var_data = df.loc['1959-08':end, cols].dropna()
    if len(var_data) < 20:
        continue
    m = VAR(var_data)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res = r.resid
    y = res[pol]
    x = res[[nbr]]
    z = res[macro_a]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    beta_sc = iv.params.iloc[-1] * 0.01
    se_sc = iv.std_errors.iloc[-1] * 0.01
    fs = sm.OLS(res[nbr].values, sm.add_constant(res[macro_a].values)).fit()
    print(f'end={end}: beta={beta_sc:8.4f}, se={se_sc:8.4f}, N={len(y)}, FS_F={fs.fvalue:.2f}')

# 3. What if the paper uses level of NBR, not log?
print("\n=== Using NBR in LEVELS (nominal and real) ===")
for nbr_var, label in [('nonborrowed_reserves', 'NBR_nom'),
                        ('nonborrowed_reserves_real', 'NBR_real')]:
    cols2 = macro_a + [nbr_var, pol]
    var_data = df.loc['1959-08':'1979-09', cols2].dropna()
    m = VAR(var_data)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res = r.resid
    y = res[pol]
    x = res[[nbr_var]]
    z = res[macro_a]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    raw = iv.params.iloc[-1]
    raw_se = iv.std_errors.iloc[-1]
    # For levels, 1% of mean
    mean_val = var_data[nbr_var].mean()
    pct1 = 0.01 * mean_val
    scaled = raw * pct1
    scaled_se = raw_se * pct1
    print(f'{label}: raw={raw:.6f}, se={raw_se:.6f}')
    print(f'  mean={mean_val:.2f}, 1%={pct1:.4f}')
    print(f'  scaled (raw*1%)={scaled:.6f}, se={scaled_se:.6f}')

# 4. What if we use log NBR but in a VAR with ALL variables in 100*log?
print("\n=== All variables in 100*log (percent) ===")
var_data = df.loc['1959-08':'1979-09', cols].copy().dropna()
for c in macro_a + [nbr]:
    var_data[c] = var_data[c] * 100  # convert to percent
m = VAR(var_data)
r = m.fit(maxlags=6, ic=None, trend='c')
res = r.resid
y = res[pol]
x = res[[nbr]]
z = res[macro_a]
exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
print(f'100*log: beta={iv.params.iloc[-1]:.6f}, se={iv.std_errors.iloc[-1]:.6f}')
# Now 1 unit of NBR = 1 percent, so coefficient is directly interpretable
print('(coefficient is already in pp per percent)')

# 5. Try: what if capacity utilization is in LEVELS not LOG?
print("\n=== Cap util in levels (not log) ===")
macro_a2 = ['log_industrial_production', 'capacity_utilization', 'log_employment']
cols2 = macro_a2 + [nbr, pol]
var_data = df.loc['1959-08':'1979-09', cols2].dropna()
m = VAR(var_data)
r = m.fit(maxlags=6, ic=None, trend='c')
res = r.resid
y = res[pol]
x = res[[nbr]]
z = res[macro_a2]
exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
print(f'raw beta={iv.params.iloc[-1]:.6f}, scaled={iv.params.iloc[-1]*0.01:.6f}')

# 6. What if we DON'T use REAL NBR but NOMINAL?
print("\n=== Nominal log NBR ===")
nbr_nom = 'log_nonborrowed_reserves'
cols2 = macro_a + [nbr_nom, pol]
var_data = df.loc['1959-08':'1979-09', cols2].dropna()
m = VAR(var_data)
r = m.fit(maxlags=6, ic=None, trend='c')
res = r.resid
y = res[pol]
x = res[[nbr_nom]]
z = res[macro_a]
exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
print(f'raw beta={iv.params.iloc[-1]:.6f}, scaled={iv.params.iloc[-1]*0.01:.6f}')

# 7. What if we use CPI in the VAR instead of deflating?
# Paper's Table 1 includes "six lags of the log of the consumer price index"
# Maybe the Table 6 VARs also include CPI as part of the system?
print("\n=== 6-var VAR with CPI ===")
cols3 = macro_a + ['log_cpi', nbr, pol]
var_data = df.loc['1959-08':'1979-09', cols3].dropna()
m = VAR(var_data)
r = m.fit(maxlags=6, ic=None, trend='c')
res = r.resid
y = res[pol]
x = res[[nbr]]
z = res[macro_a]
exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
print(f'6-var w/ CPI: raw={iv.params.iloc[-1]:.6f}, scaled={iv.params.iloc[-1]*0.01:.6f}')

# 8. What about using the growth rates (first differences of log)?
print("\n=== First differences of log variables ===")
var_data = df.loc['1959-08':'1979-09', cols].diff().dropna()
m = VAR(var_data)
r = m.fit(maxlags=6, ic=None, trend='c')
res = r.resid
y = res[pol]
x = res[[nbr]]
z = res[macro_a]
exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
print(f'Diff: raw={iv.params.iloc[-1]:.6f}, scaled={iv.params.iloc[-1]*0.01:.6f}')
fs = sm.OLS(res[nbr].values, sm.add_constant(res[macro_a].values)).fit()
print(f'FS_F={fs.fvalue:.2f}')
