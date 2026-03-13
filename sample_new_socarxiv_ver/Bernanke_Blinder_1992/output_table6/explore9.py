"""
Test two hypotheses:
H1: Paper reports raw_beta * 0.01 / mean(policy_var)
H2: Paper uses log(FUNDS) instead of FUNDS in the VAR
"""
import pandas as pd, numpy as np
from statsmodels.tsa.api import VAR
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

instrument_sets = {
    'Set A': ['log_industrial_production', 'log_capacity_utilization', 'log_employment'],
    'Set B': ['unemp_male_2554', 'log_housing_starts', 'log_personal_income_real'],
    'Set C': ['log_retail_sales_real', 'log_consumption_real'],
}

policy_vars = {'FUNDS': 'funds_rate', 'FFBOND': 'ffbond'}
nbr = 'log_nonborrowed_reserves_real'

ground_truth = {
    'Set A': {'FUNDS': -0.021, 'FFBOND': -0.011},
    'Set B': {'FUNDS': -0.0068, 'FFBOND': -0.0072},
    'Set C': {'FUNDS': -0.014, 'FFBOND': -0.014},
}

# Hypothesis 1: raw_beta * 0.01 / mean(policy_var)
print("=== H1: raw_beta * 0.01 / mean(policy) ===")
for sn, mvars in instrument_sets.items():
    for pn, pcol in policy_vars.items():
        cols = mvars + [nbr, pcol]
        vd = df.loc['1959-08':'1979-09', cols].dropna()
        m = VAR(vd)
        r = m.fit(maxlags=6, ic=None, trend='c')
        res = r.resid
        y = res[pcol]
        x = res[[nbr]]
        z = res[mvars]
        exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
        iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
        raw_b = iv.params.iloc[-1]
        raw_se = iv.std_errors.iloc[-1]
        mean_pol = vd[pcol].mean()
        h1_coeff = raw_b * 0.01 / mean_pol
        h1_se = raw_se * 0.01 / mean_pol
        gt = ground_truth[sn][pn]
        abs_err = abs(h1_coeff - gt)
        rel_err = abs_err / abs(gt) * 100
        print(f"  {sn} {pn}: H1={h1_coeff:8.4f} (gt={gt:8.4f}), rel_err={rel_err:.0f}%, mean_pol={mean_pol:.2f}")

# Hypothesis 2: Use log(FUNDS) in the VAR
print("\n=== H2: log(FUNDS) in VAR ===")
for sn, mvars in instrument_sets.items():
    for pn, pcol in policy_vars.items():
        cols = mvars + [nbr, pcol]
        vd = df.loc['1959-08':'1979-09', cols].dropna().copy()
        # For FFBOND, it can be negative, so log doesn't work
        if pn == 'FFBOND':
            # Skip log for FFBOND
            print(f"  {sn} FFBOND: N/A (can be negative)")
            continue
        vd['log_' + pcol] = np.log(vd[pcol])
        log_pcol = 'log_' + pcol
        cols2 = mvars + [nbr, log_pcol]
        m = VAR(vd[cols2])
        r = m.fit(maxlags=6, ic=None, trend='c')
        res = r.resid
        y = res[log_pcol]
        x = res[[nbr]]
        z = res[mvars]
        exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
        iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
        raw_b = iv.params.iloc[-1]
        # This is d(log FUNDS) / d(log NBR) = elasticity
        # Paper reports pp change per 1%, so multiply by mean(FUNDS)?
        # Actually if both are in log, it's an elasticity:
        # d(logF)/d(logNBR) means a 1% change in NBR -> raw_b% change in FUNDS
        # In percentage points: raw_b * mean(FUNDS) / 100
        mean_f = vd[pcol].mean()
        h2a = raw_b * 0.01  # direct scaling: 1% NBR -> raw_b*0.01 in log(FUNDS)
        h2b = raw_b * 0.01 * mean_f  # in pp of funds rate
        gt = ground_truth[sn][pn]
        print(f"  {sn} FUNDS: raw={raw_b:.4f}, h2a(log)={h2a:.6f}, h2b(pp)={h2b:.6f} (gt={gt})")

# Hypothesis 3: The paper normalizes differently.
# What if it reports: d(FUNDS)/d(NBR%) where NBR% = 100*log(NBR)?
# That would be raw_beta * 0.01 (which we already tried)
# OR what if it reports: d(FUNDS%)/d(NBR%) where FUNDS% = 100*FUNDS/mean(FUNDS)?
# Let's test: coeff * 0.01 / mean(FUNDS) * 100 = coeff / mean(FUNDS)
print("\n=== H3: Various mean-scaling combinations ===")
for sn, mvars in instrument_sets.items():
    for pn, pcol in policy_vars.items():
        cols = mvars + [nbr, pcol]
        vd = df.loc['1959-08':'1979-09', cols].dropna()
        m = VAR(vd)
        r = m.fit(maxlags=6, ic=None, trend='c')
        res = r.resid
        y = res[pcol]
        x = res[[nbr]]
        z = res[mvars]
        exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
        iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
        raw_b = iv.params.iloc[-1]
        mean_pol = abs(vd[pcol].mean())  # use abs for FFBOND

        gt = ground_truth[sn][pn]
        # Option A: raw * 0.01
        a = raw_b * 0.01
        # Option B: raw * 0.01 / mean_pol
        b = raw_b * 0.01 / mean_pol
        # Option C: raw * variance of NBR innovation
        var_nbr = res[nbr].var()
        c = raw_b * var_nbr
        # Option D: raw / (100 * mean_pol)
        d = raw_b / (100 * mean_pol)

        best_opt = min([(abs(a-gt), 'A', a), (abs(b-gt), 'B', b),
                       (abs(c-gt), 'C', c), (abs(d-gt), 'D', d)])
        print(f"  {sn} {pn}: A={a:.6f}, B={b:.6f}, C={c:.6f}, D={d:.6f} (gt={gt}) best={best_opt[1]}({best_opt[2]:.6f})")
