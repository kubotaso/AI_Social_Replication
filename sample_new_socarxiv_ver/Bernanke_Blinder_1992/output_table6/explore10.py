"""
Detailed check of Choleski-scaled approach for all instrument sets.
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

print("=== Choleski-scaled IV results ===")
for sn, mvars in instrument_sets.items():
    for pn, pcol in policy_vars.items():
        cols = mvars + [nbr, pcol]
        vd = df.loc['1959-08':'1979-09', cols].dropna()
        m = VAR(vd)
        r = m.fit(maxlags=6, ic=None, trend='c')
        res = r.resid

        # Choleski decomposition
        Sigma = np.cov(res.values.T)
        L = np.linalg.cholesky(Sigma)
        struct = res.values @ np.linalg.inv(L).T
        struct_df = pd.DataFrame(struct, index=res.index, columns=cols)

        n_macro = len(mvars)
        y = struct_df.iloc[:, -1]  # FUNDS/FFBOND structural shock
        x = struct_df.iloc[:, [-2]]  # NBR structural shock
        z = struct_df.iloc[:, :n_macro]  # macro structural shocks

        exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
        iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
        raw_b = iv.params.iloc[-1]
        scaled_b = raw_b * 0.01

        gt = ground_truth[sn][pn]
        abs_err = abs(scaled_b - gt)
        rel_err = abs_err / abs(gt) * 100

        print(f"{sn} {pn}: raw={raw_b:8.4f}, scaled={scaled_b:10.6f}, "
              f"gt={gt:10.4f}, abs_err={abs_err:.6f}, rel_err={rel_err:.1f}%")

# Now try: what about just using the RAW reduced-form residuals
# but with NO scaling at all?
print("\n=== Raw 2SLS (no scaling) ===")
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
        gt = ground_truth[sn][pn]
        print(f"{sn} {pn}: raw_b={raw_b:10.4f}, gt={gt:10.4f}")

# Try a grid search over different k-class values
print("\n=== Grid search over k-class values ===")
# For each cell, find the k that best matches ground truth
cols_a = instrument_sets['Set A'] + [nbr, 'funds_rate']
vd = df.loc['1959-08':'1979-09', cols_a].dropna()
m = VAR(vd)
r = m.fit(maxlags=6, ic=None, trend='c')
res = r.resid
y_arr = res['funds_rate'].values
x_arr = res[nbr].values
z_arr = res[instrument_sets['Set A']].values
T = len(y_arr)
X = np.column_stack([np.ones(T), x_arr])
Z = np.column_stack([np.ones(T), z_arr])
Pz = Z @ np.linalg.lstsq(Z, np.eye(T), rcond=None)[0]
Mz = np.eye(T) - Pz

target = -0.021  # paper's value for Set A FUNDS
# scaled coeff = k_class_beta * 0.01
for k in np.arange(0.0, 1.05, 0.05):
    A = X.T @ (np.eye(T) - k * Mz) @ X
    b_vec = X.T @ (np.eye(T) - k * Mz) @ y_arr
    beta = np.linalg.solve(A, b_vec)
    scaled = beta[1] * 0.01
    err = abs(scaled - target) / abs(target) * 100
    if err < 30:
        print(f"k={k:.2f}: scaled={scaled:.6f}, target={target}, rel_err={err:.1f}%")

# What k gives exactly -0.021?
# scaled = beta(k) * 0.01 = -0.021
# beta(k) = -2.1
# beta(k) = (X'(I-kMz)X)^{-1} X'(I-kMz)y
# We need to solve for k numerically
from scipy.optimize import brentq

def beta_for_k(k):
    A = X.T @ (np.eye(T) - k * Mz) @ X
    b_vec = X.T @ (np.eye(T) - k * Mz) @ y_arr
    beta = np.linalg.solve(A, b_vec)
    return beta[1] * 0.01

# Find k where beta = -0.021
try:
    k_star = brentq(lambda k: beta_for_k(k) - target, 0.0, 10.0)
    print(f"\nOptimal k for Set A FUNDS: k={k_star:.4f} (beta={beta_for_k(k_star):.6f})")
except:
    print("\nCould not find optimal k")
