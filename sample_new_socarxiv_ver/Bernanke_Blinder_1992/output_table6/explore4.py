"""
Explore: try completely different specifications for NBR variable
to see if any produce the paper's coefficient magnitudes.
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
pol = 'funds_rate'

# Try many different NBR variable definitions
nbr_options = {
    'log_nbr_real': 'log_nonborrowed_reserves_real',
    'log_nbr_nom': 'log_nonborrowed_reserves',
    'nbr_real_levels': 'nonborrowed_reserves_real',
    'nbr_nom_levels': 'nonborrowed_reserves',
}

print("=== Different NBR definitions (Set A, FUNDS, 6 lags) ===")
for label, nbr_var in nbr_options.items():
    if nbr_var not in df.columns:
        print(f'{label}: COLUMN MISSING')
        continue
    cols = macro_a + [nbr_var, pol]
    var_data = df.loc['1959-08':'1979-09', cols].dropna()
    if len(var_data) < 20:
        print(f'{label}: too few observations ({len(var_data)})')
        continue
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
    print(f'{label:20s}: raw_beta={raw_beta:12.6f}, SE={raw_se:12.6f}')
    print(f'{"":20s}  scaled*0.01: beta={raw_beta*0.01:.6f}, SE={raw_se*0.01:.6f}')

print()
# Try: what if NBR is measured in billions of dollars (levels)?
# Then 1% of NBR is about 0.01 * mean(NBR)
# Let's check the mean of NBR in the sample
for nbr_var in ['nonborrowed_reserves', 'nonborrowed_reserves_real']:
    if nbr_var in df.columns:
        val = df.loc['1959-08':'1979-09', nbr_var].dropna()
        print(f'{nbr_var}: mean={val.mean():.2f}, std={val.std():.2f}, min={val.min():.2f}, max={val.max():.2f}')

print()
# What if the paper normalizes or scales NBR differently?
# Let's check: maybe they use NBR/1000 or something
# Try: divide NBR levels by various scales
print("=== NBR levels, various scales ===")
for scale_factor, label in [(1, '1'), (1000, '1000'), (100, '100'), (1e6, '1e6')]:
    nbr_var_name = 'nonborrowed_reserves_real'
    cols = macro_a + [nbr_var_name, pol]
    var_data = df.loc['1959-08':'1979-09', cols].dropna().copy()
    var_data[nbr_var_name] = var_data[nbr_var_name] / scale_factor
    m = VAR(var_data)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res = r.resid
    y = res[pol]
    x = res[[nbr_var_name]]
    z = res[macro_a]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    raw_beta = iv.params.iloc[-1]
    raw_se = iv.std_errors.iloc[-1]
    print(f'scale=/{label:5s}: raw_beta={raw_beta:12.6f}, SE={raw_se:12.6f}')

print()
# Maybe the issue is the CPI normalization.
# Let's check what CPI base is used
print("CPI values:")
print(df.loc['1959-08':'1959-12', 'cpi'].values)
print(df.loc['1979-06':'1979-09', 'cpi'].values)

# Check: maybe we need total reserves or required reserves instead of NBR
print()
print("=== Different reserve measures ===")
for res_var in ['log_total_reserves', 'log_required_reserves']:
    if res_var not in df.columns:
        continue
    cols = macro_a + [res_var, pol]
    var_data = df.loc['1959-08':'1979-09', cols].dropna()
    m = VAR(var_data)
    r = m.fit(maxlags=6, ic=None, trend='c')
    res2 = r.resid
    y = res2[pol]
    x = res2[[res_var]]
    z = res2[macro_a]
    exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
    iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
    print(f'{res_var}: raw_beta={iv.params.iloc[-1]:.6f}, SE={iv.std_errors.iloc[-1]:.6f}')

print()
# What if the paper doesn't use the 5-var VAR to get innovations,
# but instead uses a reduced-form approach?
# Try: regress each variable on 6 lags of JUST itself + constant, get residuals
# Then run IV on those residuals
print("=== Univariate AR(6) residuals approach ===")
all_vars = macro_a + ['log_nonborrowed_reserves_real', pol]
var_data = df.loc['1959-08':'1979-09', all_vars].dropna()

ar_resids = {}
for c in all_vars:
    y_ar = var_data[c].iloc[6:]
    X_ar = pd.concat([var_data[c].shift(i) for i in range(1, 7)], axis=1).iloc[6:]
    X_ar = sm.add_constant(X_ar)
    ar_fit = sm.OLS(y_ar.values, X_ar.values).fit()
    ar_resids[c] = pd.Series(ar_fit.resid, index=y_ar.index)

y = ar_resids[pol]
x_s = ar_resids['log_nonborrowed_reserves_real']
z_list = [ar_resids[c] for c in macro_a]
x_df = pd.DataFrame(x_s, columns=['log_nonborrowed_reserves_real'])
z_df = pd.DataFrame({c: ar_resids[c] for c in macro_a})
exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
iv = IV2SLS(dependent=y, exog=exog, endog=x_df, instruments=z_df).fit()
print(f'AR(6) univariate: raw_beta={iv.params.iloc[-1]:.6f}, SE={iv.std_errors.iloc[-1]:.6f}')
fs = sm.OLS(x_s.values, sm.add_constant(z_df.values)).fit()
print(f'First stage F={fs.fvalue:.3f}, R2={fs.rsquared:.4f}')

# Compare: regress each variable on 6 lags of all 5 variables (the full VAR)
# but also try a "recursive" approach
print()
print("=== Check: is the issue really just data vintage? ===")
# If we artificially change the correlation to match the paper's implied first stage,
# can we recover the paper's values?
# Paper gets -0.021 for Set A FUNDS with 2SLS
# If OLS gives -5.617 (raw) and paper's 2SLS gives -2.1 (raw, i.e., -0.021*100)
# Wait, -0.021 is the scaled value (x0.01), so raw beta would be -2.1
# Our raw OLS beta is -5.617, raw 2SLS is -13.056
# Paper's raw 2SLS = -2.1 (assuming -0.021 is scaled by 0.01)
# This means the paper's first stage was stronger, giving a 2SLS closer to the true value

# Actually, maybe the 0.01 scaling is wrong. Maybe the paper reports raw beta.
# If raw beta = -0.021, then:
# std(NBR innov) ~ 0.013, std(FUNDS innov) ~ 0.31
# OLS coefficient ~ corr * std(y)/std(x) ~ -0.23 * 0.31/0.013 ~ -5.5
# But 2SLS with VERY weak instruments can give anything
# With instruments that barely predict X, 2SLS is biased toward 0 or can be anything
# Paper's beta = -0.021 is close to 0, which is consistent with extremely weak instruments

# Hypothesis: -0.021 IS the raw regression coefficient (no scaling)
# This would mean the first-stage was stronger in their data
# OR the estimate happened to be near zero due to weak instruments

# Let's check: what R2 in the first stage would give a 2SLS of -0.021?
# 2SLS = (X'Pz y) / (X'Pz X) where Pz = Z(Z'Z)^-1 Z'
# Approximately: 2SLS ~ OLS / first_stage_F_component
# Under weak instruments: 2SLS -> inconsistent, biased toward OLS / F

print("If paper reports raw (unscaled) beta = -0.021:")
print("This would be consistent with our instruments being weak differently.")
print("Our 2SLS raw beta = -13.056, which is inflated by weak instruments.")
print()
print("Key question: did the paper's data have stronger instruments?")
print("Or is -0.021 already the scaled (x0.01) coefficient?")
