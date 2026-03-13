"""Explore different approaches for Table 6 replication."""
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

# Standard approach: joint 5-var VAR
cols = macro_a + [nbr, pol]
var_data = df.loc['1959-08':'1979-09', cols].dropna()
m = VAR(var_data)
r = m.fit(maxlags=6, ic=None, trend='c')
res = r.resid

print("=== Approach 1: Joint VAR, linearmodels IV2SLS ===")
y = res[pol]
x = res[[nbr]]
z = res[macro_a]
exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
print(f"slope={iv.params.iloc[-1]:.6f}, SE={iv.std_errors.iloc[-1]:.6f}")

print("\n=== Approach 2: Joint VAR, no constant in IV ===")
iv2 = IV2SLS(dependent=y, exog=None, endog=x, instruments=z).fit()
print(f"slope={iv2.params.iloc[0]:.6f}, SE={iv2.std_errors.iloc[0]:.6f}")

print("\n=== Approach 3: Separate VARs ===")
m_macro = VAR(var_data[macro_a])
r_macro = m_macro.fit(maxlags=6, ic=None, trend='c')
res_macro = r_macro.resid

m_pol = VAR(var_data[[nbr, pol]])
r_pol = m_pol.fit(maxlags=6, ic=None, trend='c')
res_pol = r_pol.resid

common = res_macro.index.intersection(res_pol.index)
y3 = res_pol.loc[common, pol]
x3 = res_pol.loc[common, [nbr]]
z3 = res_macro.loc[common, macro_a]
exog3 = pd.DataFrame(np.ones(len(y3)), index=y3.index, columns=['const'])
iv3 = IV2SLS(dependent=y3, exog=exog3, endog=x3, instruments=z3).fit()
print(f"slope={iv3.params.iloc[-1]:.6f}, SE={iv3.std_errors.iloc[-1]:.6f}")
fs3 = sm.OLS(x3.values.ravel(), sm.add_constant(z3.values)).fit()
print(f"First stage R2={fs3.rsquared:.6f}, F={fs3.fvalue:.4f}")

print("\n=== Approach 4: Single-equation approach ===")
# Maybe the paper runs individual AR(6) for each variable, not a full VAR
all_resids = {}
for c in cols:
    s = var_data[c]
    # Create lags
    X_lags = pd.concat([s.shift(i) for i in range(1, 7)], axis=1)
    X_lags.columns = [f'lag{i}' for i in range(1, 7)]
    X_lags = sm.add_constant(X_lags)
    X_lags = X_lags.dropna()
    s_trim = s.loc[X_lags.index]
    ols_res = sm.OLS(s_trim.values, X_lags.values).fit()
    all_resids[c] = pd.Series(ols_res.resid, index=X_lags.index)

common4 = all_resids[pol].index
for c in cols:
    common4 = common4.intersection(all_resids[c].index)

y4 = all_resids[pol].loc[common4]
x4 = pd.DataFrame(all_resids[nbr].loc[common4])
z4 = pd.DataFrame({c: all_resids[c].loc[common4] for c in macro_a})
exog4 = pd.DataFrame(np.ones(len(y4)), index=y4.index, columns=['const'])
iv4 = IV2SLS(dependent=y4, exog=exog4, endog=x4, instruments=z4).fit()
print(f"slope={iv4.params.iloc[-1]:.6f}, SE={iv4.std_errors.iloc[-1]:.6f}")
fs4 = sm.OLS(x4.values.ravel(), sm.add_constant(z4.values)).fit()
print(f"First stage R2={fs4.rsquared:.6f}, F={fs4.fvalue:.4f}")

print("\n=== Approach 5: Full VAR but using lags of ALL 5 vars as instruments ===")
# The VAR innovations should have correlations because the VAR captures serial but not contemporaneous
# The paper's approach: use innovations from the 5-var VAR
# But maybe the instruments should include lags of macro vars, not just innovations
# Actually, re-reading: maybe the instruments in the IV are the LAGGED macro variables, not their innovations
# Let's try using lagged macro values as instruments for the IV on innovations

print("\n=== Approach 6: Use lags of macro vars as instruments in IV on raw data ===")
# This is a completely different interpretation: maybe the IV is run on levels, not innovations
y6 = var_data[pol].iloc[6:]
x6 = var_data[[nbr]].iloc[6:]
# Build lagged instruments
z_list = []
for lag in range(1, 7):
    for c in macro_a:
        z_list.append(var_data[c].shift(lag).iloc[6:])
z6 = pd.concat(z_list, axis=1)
z6.columns = [f'{c}_lag{lag}' for lag in range(1, 7) for c in macro_a]

exog6 = pd.DataFrame(np.ones(len(y6)), index=y6.index, columns=['const'])
iv6 = IV2SLS(dependent=y6, exog=exog6, endog=x6, instruments=z6).fit()
print(f"slope={iv6.params.iloc[-1]:.6f}, SE={iv6.std_errors.iloc[-1]:.6f}")

print("\n=== Approach 7: Bivariate VAR [NBR, FUNDS] then IV with macro innovations ===")
# Maybe the paper runs separate VARs for different purposes
# VAR 1: just [NBR, FUNDS] to get their innovations
# Instruments: innovations from separate VAR on macro vars
m7a = VAR(var_data[[nbr, pol]])
r7a = m7a.fit(maxlags=6, ic=None, trend='c')
res7a = r7a.resid

m7b = VAR(var_data[macro_a])
r7b = m7b.fit(maxlags=6, ic=None, trend='c')
res7b = r7b.resid

common7 = res7a.index.intersection(res7b.index)
y7 = res7a.loc[common7, pol]
x7 = res7a.loc[common7, [nbr]]
z7 = res7b.loc[common7, macro_a]
exog7 = pd.DataFrame(np.ones(len(y7)), index=y7.index, columns=['const'])
iv7 = IV2SLS(dependent=y7, exog=exog7, endog=x7, instruments=z7).fit()
print(f"slope={iv7.params.iloc[-1]:.6f}, SE={iv7.std_errors.iloc[-1]:.6f}")
fs7 = sm.OLS(x7.values.ravel(), sm.add_constant(z7.values)).fit()
print(f"First stage R2={fs7.rsquared:.6f}, F={fs7.fvalue:.4f}")

print("\n=== Approach 8: 3-var macro VAR innovations as IV for OLS on raw ===")
# Maybe: get innovations from a 3-var macro VAR
# Then use those as instruments in an IV on the RAW FUNDS ~ RAW NBR
# This doesn't really make sense economically but let's check
m8 = VAR(var_data[macro_a])
r8 = m8.fit(maxlags=6, ic=None, trend='c')
res8 = r8.resid

# Align raw data with innovations
y8 = var_data.loc[res8.index, pol]
x8 = var_data.loc[res8.index, [nbr]]
z8 = res8[macro_a]
exog8 = pd.DataFrame(np.ones(len(y8)), index=y8.index, columns=['const'])
iv8 = IV2SLS(dependent=y8, exog=exog8, endog=x8, instruments=z8).fit()
print(f"slope={iv8.params.iloc[-1]:.6f}, SE={iv8.std_errors.iloc[-1]:.6f}")
