"""Structural VAR coefficient approach."""
import pandas as pd, numpy as np
from statsmodels.tsa.api import VAR
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
nbr_var = 'log_nonborrowed_reserves_real'

ground_truth = {
    'Set A': {'FUNDS': -0.021, 'FFBOND': -0.011},
    'Set B': {'FUNDS': -0.0068, 'FFBOND': -0.0072},
    'Set C': {'FUNDS': -0.014, 'FFBOND': -0.014},
}

print("=== Structural VAR (Choleski) coefficient approach ===")
print("Order: [macro1, macro2, macro3, NBR, FUNDS/FFBOND]")
print("L[FUNDS, NBR] = contemporaneous effect of structural NBR shock on FUNDS")
print()

for sn, mvars in instrument_sets.items():
    for pn, pcol in policy_vars.items():
        cols = mvars + [nbr_var, pcol]
        vd = df.loc['1959-08':'1979-09', cols].dropna()
        m = VAR(vd)
        r = m.fit(maxlags=6, ic=None, trend='c')

        # Get covariance matrix as numpy array
        Sigma = np.array(r.sigma_u)
        L = np.linalg.cholesky(Sigma)

        n_macro = len(mvars)
        nbr_idx = n_macro
        pol_idx = n_macro + 1

        # L[pol, nbr] is the contemporaneous response of FUNDS to a 1-std-dev NBR shock
        # To get the response to a 1% NBR innovation:
        # 1% in log = 0.01
        # In terms of structural shocks: 0.01 / L[nbr, nbr] standard deviations
        # Response of FUNDS = L[pol, nbr] * (0.01 / L[nbr, nbr])
        L_pol_nbr = L[pol_idx, nbr_idx]
        L_nbr_nbr = L[nbr_idx, nbr_idx]
        scaled = L_pol_nbr * 0.01 / L_nbr_nbr

        gt = ground_truth[sn][pn]
        rel_e = abs(scaled - gt) / abs(gt) * 100
        sign_ok = (scaled < 0) == (gt < 0)
        print(f"{sn} {pn}: L[F,N]={L_pol_nbr:.6f}, L[N,N]={L_nbr_nbr:.6f}, "
              f"scaled={scaled:.6f} (gt={gt:.4f}), rel={rel_e:.1f}%, sign={'OK' if sign_ok else 'WRONG'}")

# Also try the reverse Choleski ordering: [FUNDS, NBR, macro]
print("\n=== Reverse ordering: [FUNDS/FFBOND, NBR, macro1, macro2, macro3] ===")
for sn, mvars in instrument_sets.items():
    for pn, pcol in policy_vars.items():
        cols = [pcol, nbr_var] + mvars
        vd = df.loc['1959-08':'1979-09', cols].dropna()
        m = VAR(vd)
        r = m.fit(maxlags=6, ic=None, trend='c')
        Sigma = np.array(r.sigma_u)
        L = np.linalg.cholesky(Sigma)
        # Now FUNDS is index 0, NBR is index 1
        # L[0, 1] = effect of NBR shock on FUNDS
        # But in Choleski, L is lower triangular, so L[0,1] = 0 (FUNDS comes first)
        # L[1, 0] = effect of FUNDS shock on NBR
        # That's not what we want.
        # Actually, for supply slope, we want effect of NBR demand shock on FUNDS
        # In the ordering [macro, NBR, FUNDS], FUNDS is residual after macro and NBR
        # So L[FUNDS, NBR] captures the supply slope
        # Let's try ordering: [macro, FUNDS, NBR]
        # Then L[NBR, FUNDS] would be the demand slope
        pass

# Try ordering: [macro, FUNDS/FFBOND, NBR]
print("\n=== Ordering: [macro, FUNDS/FFBOND, NBR] ===")
for sn, mvars in instrument_sets.items():
    for pn, pcol in policy_vars.items():
        cols = mvars + [pcol, nbr_var]
        vd = df.loc['1959-08':'1979-09', cols].dropna()
        m = VAR(vd)
        r = m.fit(maxlags=6, ic=None, trend='c')
        Sigma = np.array(r.sigma_u)
        L = np.linalg.cholesky(Sigma)
        n_macro = len(mvars)
        pol_idx = n_macro
        nbr_idx = n_macro + 1
        # L[nbr, pol] = effect of FUNDS shock on NBR
        # L[pol, nbr] = 0 (upper triangular = 0 in Choleski lower)
        # We want: L[pol, nbr] but that's 0 in this ordering
        # Actually L is lower triangular: L[i,j] = 0 for j > i
        # So L[pol_idx, nbr_idx] = L[n_macro, n_macro+1] = 0 (since pol_idx < nbr_idx)
        # And L[nbr_idx, pol_idx] = L[n_macro+1, n_macro] which IS non-zero
        L_nbr_pol = L[nbr_idx, pol_idx]
        L_pol_pol = L[pol_idx, pol_idx]
        # This is the effect of a FUNDS shock on NBR, which is the DEMAND slope
        # scaled: response of NBR to a 1pp FUNDS shock
        scaled_demand = L_nbr_pol * 1.0 / L_pol_pol
        # For supply: we need the reverse
        # The supply slope beta satisfies: dFUNDS/dNBR = beta
        # In the [macro, NBR, FUNDS] ordering: L[FUNDS, NBR]/L[NBR, NBR] * 0.01
        # This is what we already computed above
        pass

# The key insight: try the IMPULSE RESPONSE at horizon 0
# The IRF from NBR to FUNDS at horizon 0 IS the Choleski coefficient
print("\n=== IRF at horizon 0 (Choleski) ===")
for sn, mvars in instrument_sets.items():
    for pn, pcol in policy_vars.items():
        cols = mvars + [nbr_var, pcol]
        vd = df.loc['1959-08':'1979-09', cols].dropna()
        m = VAR(vd)
        r = m.fit(maxlags=6, ic=None, trend='c')
        irf = r.irf(1)
        n_macro = len(mvars)
        nbr_idx = n_macro
        pol_idx = n_macro + 1
        # IRF of FUNDS to NBR shock at horizon 0
        irfval = irf.irfs[0, pol_idx, nbr_idx]
        # This is response to 1-std-dev shock
        # For 1% shock: multiply by (0.01/std)
        std_nbr = np.sqrt(np.array(r.sigma_u)[nbr_idx, nbr_idx])
        # Wait, irf.irfs already accounts for Choleski scaling
        # The orthogonalized IRF at h=0 is just the Choleski L matrix row
        # irfs[0, i, j] = L[i, j]
        scaled_irf = irfval * 0.01 / L_nbr_nbr  # need L_nbr_nbr from this VAR
        Sigma = np.array(r.sigma_u)
        L = np.linalg.cholesky(Sigma)
        L_nbr_nbr_local = L[nbr_idx, nbr_idx]
        scaled_irf2 = irfval * 0.01 / L_nbr_nbr_local

        gt = ground_truth[sn][pn]
        rel_e = abs(scaled_irf2 - gt) / abs(gt) * 100
        print(f"{sn} {pn}: IRF[0]={irfval:.6f}, scaled={scaled_irf2:.6f} (gt={gt:.4f}), rel={rel_e:.1f}%")
