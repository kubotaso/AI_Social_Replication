"""
Try: nominal NBR (not deflated) with all instrument sets.
Also try: log(NBR_nominal) with various approaches.
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

ground_truth = {
    'Set A': {'FUNDS': -0.021, 'FFBOND': -0.011},
    'Set B': {'FUNDS': -0.0068, 'FFBOND': -0.0072},
    'Set C': {'FUNDS': -0.014, 'FFBOND': -0.014},
}

# Test with different NBR variables
for nbr_label, nbr_var in [
    ('log_NBR_real', 'log_nonborrowed_reserves_real'),
    ('log_NBR_nom', 'log_nonborrowed_reserves'),
]:
    print(f"\n=== {nbr_label} ===")
    total_matches = 0
    for sn, mvars in instrument_sets.items():
        for pn, pcol in policy_vars.items():
            cols = mvars + [nbr_var, pcol]
            vd = df.loc['1959-08':'1979-09', cols].dropna()
            m = VAR(vd)
            r = m.fit(maxlags=6, ic=None, trend='c')
            res = r.resid
            y = res[pcol]
            x = res[[nbr_var]]
            z = res[mvars]
            exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
            iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
            b = iv.params.iloc[-1] * 0.01
            se = iv.std_errors.iloc[-1] * 0.01
            gt = ground_truth[sn][pn]
            abs_e = abs(b - gt)
            rel_e = abs_e / abs(gt) * 100
            match = abs_e <= 0.005 or rel_e <= 20
            if match:
                total_matches += 1
            sign_ok = 'OK' if (b < 0) == (gt < 0) else 'WRONG'
            print(f"  {sn} {pn}: b={b:10.6f} (gt={gt:8.4f}), "
                  f"rel={rel_e:6.1f}%, sign={sign_ok}, {'PASS' if match else 'fail'}")
    print(f"  TOTAL MATCHES: {total_matches}/6")

# Also try: log_total_reserves and log_required_reserves
for nbr_label, nbr_var in [
    ('log_total_reserves', 'log_total_reserves'),
    ('log_required_reserves', 'log_required_reserves'),
]:
    if nbr_var not in df.columns:
        continue
    print(f"\n=== {nbr_label} ===")
    sn = 'Set A'
    mvars = instrument_sets[sn]
    for pn, pcol in policy_vars.items():
        cols = mvars + [nbr_var, pcol]
        vd = df.loc['1959-08':'1979-09', cols].dropna()
        m = VAR(vd)
        r = m.fit(maxlags=6, ic=None, trend='c')
        res = r.resid
        y = res[pcol]
        x = res[[nbr_var]]
        z = res[mvars]
        exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
        iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
        b = iv.params.iloc[-1] * 0.01
        gt = ground_truth[sn][pn]
        rel_e = abs(b - gt) / abs(gt) * 100
        print(f"  {sn} {pn}: b={b:10.6f} (gt={gt:8.4f}), rel={rel_e:6.1f}%")

# Now try a completely different approach: maybe the paper just uses
# the Wald/2SLS estimate from the structural VAR directly
# In a structural VAR with Choleski ordering [macro, NBR, FUNDS],
# the coefficient of NBR in the FUNDS equation IS the slope
# of the supply function
print("\n=== Structural VAR (Choleski) coefficient ===")
for sn, mvars in instrument_sets.items():
    for pn, pcol in policy_vars.items():
        cols = mvars + [nbr_var, pcol]
        nbr_var2 = 'log_nonborrowed_reserves_real'
        cols = mvars + [nbr_var2, pcol]
        vd = df.loc['1959-08':'1979-09', cols].dropna()
        m = VAR(vd)
        r = m.fit(maxlags=6, ic=None, trend='c')

        # Choleski decomposition of residual covariance
        Sigma = r.sigma_u
        L = np.linalg.cholesky(Sigma)
        # L[i,j] for i > j gives the contemporaneous effect of shock j on variable i
        # Variable ordering: macro1, macro2, macro3, NBR, FUNDS
        # L[4, 3] = effect of NBR shock on FUNDS (this IS the supply slope)
        n_macro = len(mvars)
        nbr_idx = n_macro
        pol_idx = n_macro + 1
        b0 = L[pol_idx, nbr_idx]
        # This is in levels (not scaled to percent)
        # Scaled: a 1% NBR shock has std of 0.01 in log units
        # The Choleski shock is normalized to unit variance
        # std of NBR innovation = sqrt(Sigma[nbr,nbr])
        std_nbr = np.sqrt(Sigma[nbr_idx, nbr_idx])
        # 1% = 0.01 in log, so we scale the shock by 0.01/std_nbr
        scaled_b0 = b0 * 0.01 / std_nbr
        # Actually wait: the Choleski decomposition normalizes differently
        # L[i,j] * e_j gives the response of variable i to shock j
        # where e_j has unit variance
        # A 1% shock to NBR in log = 0.01 in log
        # std of structural shock to NBR = L[nbr,nbr]
        # So a 1% innovation = 0.01/L[nbr,nbr] standard deviations
        # Effect on FUNDS = L[pol,nbr] * (0.01/L[nbr,nbr])
        L_nbr_nbr = L[nbr_idx, nbr_idx]
        scaled_b0_v2 = L[pol_idx, nbr_idx] * 0.01 / L_nbr_nbr

        gt = ground_truth[sn][pn]
        print(f"  {sn} {pn}: L[F,N]={L[pol_idx, nbr_idx]:.6f}, "
              f"scaled_v2={scaled_b0_v2:.6f} (gt={gt:.4f}), "
              f"rel_err={abs(scaled_b0_v2-gt)/abs(gt)*100:.1f}%")
