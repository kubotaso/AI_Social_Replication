"""
Explore: different scalings and normalizations to match paper's coefficient magnitudes.
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

vd = df.loc['1959-08':'1979-09', cols].dropna()
m = VAR(vd)
r = m.fit(maxlags=6, ic=None, trend='c')
res = r.resid

y = res[pol]
x = res[[nbr]]
z = res[macro_a]
exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
raw_beta = iv.params.iloc[-1]
raw_se = iv.std_errors.iloc[-1]

# Various scalings
mean_funds = df.loc['1959-08':'1979-09', 'funds_rate'].mean()
std_nbr = res[nbr].std()
std_funds = res[pol].std()

print(f"Raw beta: {raw_beta:.4f}")
print(f"Raw SE: {raw_se:.4f}")
print(f"Mean funds rate in sample: {mean_funds:.4f}")
print(f"Std of NBR innovation: {std_nbr:.6f}")
print(f"Std of FUNDS innovation: {std_funds:.6f}")
print()

# Scaling options
print("=== Different scalings ===")
print(f"raw * 0.01 = {raw_beta * 0.01:.6f} (paper Set A FUNDS: -0.021)")
print(f"raw * 0.001 = {raw_beta * 0.001:.6f}")
print(f"raw * std(NBR) = {raw_beta * std_nbr:.6f}")
print(f"raw * std(NBR)^2 = {raw_beta * std_nbr**2:.6f}")
print(f"raw / mean(FUNDS) = {raw_beta / mean_funds:.6f}")
print(f"raw * 0.01 / mean(FUNDS) = {raw_beta * 0.01 / mean_funds:.6f}")
print(f"raw * std(NBR) / std(FUNDS) = {raw_beta * std_nbr / std_funds:.6f}")

# What scaling gives exactly -0.021?
target = -0.021
implied_scale = target / raw_beta
print(f"\nImplied scale factor to get -0.021: {implied_scale:.8f}")
print(f"1/std(NBR) = {1/std_nbr:.4f}")
print(f"Is implied scale = std(NBR)^2? {std_nbr**2:.8f}")
print(f"Is implied scale = std(NBR)? {std_nbr:.8f}")
print(f"Is implied scale = variance(NBR)? {res[nbr].var():.8f}")

# What if the paper reports beta * var(innovation_NBR)?
beta_times_var = raw_beta * res[nbr].var()
print(f"\nbeta * var(NBR_innov) = {beta_times_var:.6f}")

# What if the paper reports the reduced-form covariance divided by something?
cov_funds_nbr = np.cov(res[pol].values, res[nbr].values)[0, 1]
var_nbr = np.var(res[nbr].values)
print(f"\ncov(FUNDS, NBR) / var(NBR) = {cov_funds_nbr/var_nbr:.4f} (this is OLS)")
print(f"cov scaled = {cov_funds_nbr:.8f}")

# What if the paper normalizes by 100*var(NBR)?
print(f"\nbeta * 100 * var(NBR) = {raw_beta * 100 * var_nbr:.6f}")

# Check: what is the std of innovations in percentage terms?
print(f"\nstd(NBR innov) * 100 = {std_nbr * 100:.4f} (percent)")
print(f"std(FUNDS innov) = {std_funds:.4f} (percentage points)")

# So a 1 std dev shock to NBR = 1.29% change
# The effect on FUNDS = beta * std(NBR) = -13.06 * 0.0129 = -0.168 pp
# This is large - the paper gets much smaller effects

# Let me try: what if the FUNDS variable is also in log?
# Then the coefficient would be an elasticity
print("\n=== Try with log(FUNDS) ===")
vd2 = vd.copy()
vd2['log_funds'] = np.log(vd2['funds_rate'])
cols2 = macro_a + [nbr, 'log_funds']
m2 = VAR(vd2[cols2])
r2 = m2.fit(maxlags=6, ic=None, trend='c')
res2 = r2.resid
y2 = res2['log_funds']
x2 = res2[[nbr]]
z2 = res2[macro_a]
exog2 = pd.DataFrame(np.ones(len(y2)), index=y2.index, columns=['const'])
iv2 = IV2SLS(dependent=y2, exog=exog2, endog=x2, instruments=z2).fit()
print(f"log(FUNDS): raw={iv2.params.iloc[-1]:.6f}, scaled={iv2.params.iloc[-1]*0.01:.6f}")

# The paper says the dependent variable IS the funds rate level
# "the innovation in the policy measure" where policy measure is FUNDS = fed funds rate

# What if there's a divide by 100 somewhere in the DRI data that we don't have?
# E.g., if the original DRI data had reserves in 100s of billions...
# NBR nominal is about 13 billion in 1970. If DRI reported it in 100 billions,
# it would be 0.13, and log(0.13) = -2.04, vs our log(13) = 2.57
# The innovations would be the same in either case (log shift absorbed by VAR constant)

# Actually wait - if the original data has NBR in DIFFERENT units, the LOG INNOVATIONS
# would be the SAME because log(c*NBR) = log(c) + log(NBR), and the constant
# is absorbed. So units don't matter for log innovations.

# FINAL TEST: what if we use unadjusted (non-break-adjusted) reserves?
# The paper says RESFRBNBANS = "adjusted" reserves
# Our series may or may not be adjusted
print("\n=== Key question: are we using break-adjusted reserves? ===")
print("NBR nominal values around definitional changes:")
for date in ['1969-01-01', '1969-06-01', '1969-07-01', '1969-12-01', '1970-01-01']:
    if date in df.index:
        print(f"  {date}: {df.loc[date, 'nonborrowed_reserves']:.2f}")
