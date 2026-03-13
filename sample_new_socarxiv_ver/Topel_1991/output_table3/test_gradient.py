"""Test gradient computation for two-step SE"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}

CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

PAPER_S1 = {
    'b1b2': 0.1258, 'b1b2_se': 0.0161,
    'g2': -0.004592, 'g3': 0.0001846, 'g4': -0.00000245,
    'd2': -0.006051, 'd3': 0.0002067, 'd4': -0.00000238
}


def prepare_data(data_source, pnum_filter=None):
    df = pd.read_csv(data_source)
    if 'pnum' not in df.columns:
        df['pnum'] = df['person_id'] % 1000
    if pnum_filter is not None:
        df = df[df['pnum'].isin(pnum_filter)].copy()

    df['education_years'] = np.nan
    for yr in df['year'].unique():
        mask = df['year'] == yr
        if yr in [1975, 1976]:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean']
        else:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_MAP)
    df = df.dropna(subset=['education_years']).copy()

    df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
    df['tenure'] = df['tenure_topel'].copy()
    df['gnp_def'] = df['year'].map(lambda y: GNP_DEFLATOR.get(y - 1, np.nan))
    df['log_real_gnp'] = df['log_hourly_wage'] - np.log(df['gnp_def'] / 100.0)

    df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
    df = df[df['hourly_wage'] > 0].copy()
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
    df = df[df['tenure'] >= 1].copy()
    df = df.dropna(subset=['log_real_gnp', 'experience', 'tenure']).copy()

    df['union_member'] = df['union_member'].fillna(0)
    df['disabled'] = df['disabled'].fillna(0)
    df['married'] = df['married'].fillna(0)

    for k in [2, 3, 4]:
        df[f'tenure_{k}'] = df['tenure'] ** k
        df[f'exp_{k}'] = df['experience'] ** k

    for yr in range(1971, 1984):
        col = f'year_{yr}'
        if col not in df.columns:
            df[col] = (df['year'] == yr).astype(int)

    return df


def run_iv_fixed(df, step1, ctrl, yr_dums, wage_var='log_real_gnp'):
    """Run IV with fixed data, returns beta_1"""
    b1b2 = step1['b1b2']
    g2, g3, g4 = step1['g2'], step1['g3'], step1['g4']
    d2, d3, d4 = step1['d2'], step1['d3'], step1['d4']

    levels = df.copy()
    levels['w_star'] = (levels[wage_var]
                        - b1b2 * levels['tenure']
                        - g2 * levels['tenure_2']
                        - g3 * levels['tenure_3']
                        - g4 * levels['tenure_4']
                        - d2 * levels['exp_2']
                        - d3 * levels['exp_3']
                        - d4 * levels['exp_4'])

    levels['init_exp'] = (levels['experience'] - levels['tenure']).clip(lower=0)

    all_ctrl = ctrl + yr_dums
    levels = levels.dropna(subset=['w_star', 'experience', 'init_exp'] + all_ctrl).copy()

    exog_df = sm.add_constant(levels[all_ctrl])
    rank = np.linalg.matrix_rank(exog_df.values)
    active_yr = yr_dums.copy()
    while rank < exog_df.shape[1] and active_yr:
        active_yr.pop()
        all_ctrl_temp = ctrl + active_yr
        exog_df = sm.add_constant(levels[all_ctrl_temp])
        rank = np.linalg.matrix_rank(exog_df.values)
    all_ctrl = ctrl + active_yr

    dep = levels['w_star']
    exog = sm.add_constant(levels[all_ctrl])
    endog = levels[['experience']]
    instruments = levels[['init_exp']]

    iv = IV2SLS(dep, exog, endog, instruments).fit(cov_type='unadjusted')
    return iv.params['experience']


# Prepare data
df = prepare_data("data/psid_panel_full.csv", pnum_filter=[1, 170, 3])
ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
yr_dums = yr_dums[1:]

# Base beta_1
beta_1_base = run_iv_fixed(df, PAPER_S1, ctrl, yr_dums)
print(f"Base beta_1: {beta_1_base:.6f}")

# Test gradients with different delta values
print("\n--- Gradient d(beta_1)/d(b1b2) ---")
for delta in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
    step1_plus = PAPER_S1.copy()
    step1_plus['b1b2'] = PAPER_S1['b1b2'] + delta
    b1_plus = run_iv_fixed(df, step1_plus, ctrl, yr_dums)

    step1_minus = PAPER_S1.copy()
    step1_minus['b1b2'] = PAPER_S1['b1b2'] - delta
    b1_minus = run_iv_fixed(df, step1_minus, ctrl, yr_dums)

    grad_forward = (b1_plus - beta_1_base) / delta
    grad_central = (b1_plus - b1_minus) / (2 * delta)
    print(f"  delta={delta:.4f}: forward={grad_forward:.4f}, central={grad_central:.4f}")

# Now think about the analytical gradient.
# w* = log_wage - b1b2*T - g2*T^2 - ... - d2*X^2 - ...
# The IV regression of w* on X (instrumented by X0) gives:
# beta_1 = coefficient on X
# When b1b2 increases by delta, w* decreases by delta*T for each obs.
# So beta_1 changes by -delta * (IV coefficient of T on X) ≈ -delta * corr(T, X_hat)
# where X_hat is the fitted value from the first stage (X on X0)

print("\n--- Understanding the gradient analytically ---")
# In IV: beta_1 = (Z'PX * w*) / (Z'PX * X) where PX is the residual maker
# When w* changes by -delta*T, beta_1 changes by -delta * (Z'PX * T) / (Z'PX * X)
# This is essentially the IV coefficient of T in the same regression
# Run IV with T as dependent variable to get this ratio

levels = df.copy()
levels['w_star'] = (levels['log_real_gnp']
                    - PAPER_S1['b1b2'] * levels['tenure']
                    - PAPER_S1['g2'] * levels['tenure_2']
                    - PAPER_S1['g3'] * levels['tenure_3']
                    - PAPER_S1['g4'] * levels['tenure_4']
                    - PAPER_S1['d2'] * levels['exp_2']
                    - PAPER_S1['d3'] * levels['exp_3']
                    - PAPER_S1['d4'] * levels['exp_4'])
levels['init_exp'] = (levels['experience'] - levels['tenure']).clip(lower=0)

all_ctrl = ctrl + yr_dums
levels = levels.dropna(subset=['w_star', 'experience', 'init_exp'] + all_ctrl).copy()

exog_df = sm.add_constant(levels[all_ctrl])
rank = np.linalg.matrix_rank(exog_df.values)
active_yr = yr_dums.copy()
while rank < exog_df.shape[1] and active_yr:
    active_yr.pop()
    all_ctrl_temp = ctrl + active_yr
    exog_df = sm.add_constant(levels[all_ctrl_temp])
    rank = np.linalg.matrix_rank(exog_df.values)
all_ctrl_used = ctrl + active_yr

# Run IV of tenure on experience, instrumented by init_exp
dep_t = levels['tenure']
exog_t = sm.add_constant(levels[all_ctrl_used])
endog_t = levels[['experience']]
instruments_t = levels[['init_exp']]

iv_t = IV2SLS(dep_t, exog_t, endog_t, instruments_t).fit(cov_type='unadjusted')
print(f"IV coef of experience on tenure: {iv_t.params['experience']:.6f}")
print(f"Expected gradient: {-iv_t.params['experience']:.6f}")
print(f"This is the 'pure' gradient if T were the only thing changing")

# But actually, the gradient computation is:
# d(beta_1)/d(b1b2) where b1b2 affects w* through -b1b2 * T
# Since the IV of w* on X gives: beta_1 = Cov(Z_resid, w*) / Cov(Z_resid, X)
# where Z_resid is the residual of the instrument (init_exp) after partialling out controls
# d(beta_1)/d(b1b2) = -Cov(Z_resid, T) / Cov(Z_resid, X)

# Partial out controls from init_exp, experience, and tenure
from numpy.linalg import lstsq
X_ctrl = sm.add_constant(levels[all_ctrl_used]).values
Z = levels['init_exp'].values
X_exp = levels['experience'].values
T = levels['tenure'].values

# Residualize
Z_hat = X_ctrl @ lstsq(X_ctrl, Z, rcond=None)[0]
Z_resid = Z - Z_hat
X_hat = X_ctrl @ lstsq(X_ctrl, X_exp, rcond=None)[0]
X_resid = X_exp - X_hat
T_hat = X_ctrl @ lstsq(X_ctrl, T, rcond=None)[0]
T_resid = T - T_hat

cov_ZX = np.sum(Z_resid * X_resid)
cov_ZT = np.sum(Z_resid * T_resid)
analytical_grad = -cov_ZT / cov_ZX
print(f"\nAnalytical gradient: -Cov(Z_resid, T)/Cov(Z_resid, X) = {analytical_grad:.6f}")

# Now compute total SE
# Var(beta_1) = Var_step2(beta_1) + grad^2 * Var(b1b2)
# But the step 2 variance needs to account for clustering
# Let's use clustered SE from the IV

exog = sm.add_constant(levels[all_ctrl_used])
dep = levels['w_star']
endog = levels[['experience']]
instruments = levels[['init_exp']]

iv_clustered = IV2SLS(dep, exog, endog, instruments).fit(
    cov_type='clustered', clusters=levels['person_id'])
se_clustered = iv_clustered.std_errors['experience']
print(f"\nClustered SE (step 2 only): {se_clustered:.6f}")

# Total SE with Murphy-Topel correction
grad = analytical_grad
b1b2_se = PAPER_S1['b1b2_se']
var_total = se_clustered**2 + grad**2 * b1b2_se**2
se_total = np.sqrt(var_total)
print(f"grad = {grad:.4f}")
print(f"Total SE = sqrt({se_clustered:.4f}^2 + {grad:.4f}^2 * {b1b2_se:.4f}^2) = {se_total:.4f}")
print(f"Paper SE: 0.0181")

# What if we include gradients for all step 1 params?
# We need SEs for g2, g3, g4, d2, d3, d4
# Paper doesn't report SEs for these separately, but Table 2 col 3 has t-stats
# Let's estimate gradients for the higher-order terms too
print("\n--- Gradients for all step 1 params ---")
for pname in ['g2', 'g3', 'g4', 'd2', 'd3', 'd4']:
    delta = abs(PAPER_S1[pname]) * 0.01 if abs(PAPER_S1[pname]) > 0 else 0.001
    sp_plus = PAPER_S1.copy()
    sp_plus[pname] = PAPER_S1[pname] + delta
    sp_minus = PAPER_S1.copy()
    sp_minus[pname] = PAPER_S1[pname] - delta

    b1_plus = run_iv_fixed(df, sp_plus, ctrl, yr_dums)
    b1_minus = run_iv_fixed(df, sp_minus, ctrl, yr_dums)
    grad_p = (b1_plus - b1_minus) / (2 * delta)

    # What variable is involved?
    if pname.startswith('g'):
        k = int(pname[1])
        var_name = f"tenure_{k}"
        cov_Zv = np.sum(Z_resid * (levels[var_name].values - X_ctrl @ lstsq(X_ctrl, levels[var_name].values, rcond=None)[0]))
        analytical_g = -cov_Zv / cov_ZX
    else:
        k = int(pname[1])
        var_name = f"exp_{k}"
        cov_Zv = np.sum(Z_resid * (levels[var_name].values - X_ctrl @ lstsq(X_ctrl, levels[var_name].values, rcond=None)[0]))
        analytical_g = -cov_Zv / cov_ZX

    print(f"  {pname}: numerical={grad_p:.4f}, analytical={analytical_g:.4f}")

# The Murphy-Topel formula is:
# V_total = V_2 + G * V_1 * G' - [G * C * V_2 + (G * C * V_2)']
# where G is the gradient, V_1 is step 1 variance, V_2 is step 2 variance,
# and C accounts for cross-equation dependence
# For simplicity, the correction term reduces SE, but let's use the simple version:
# V_total = V_2 + G * V_1 * G'

print("\n\n=== SE APPROACH: Clustered step-2 SE + gradient correction ===")
print(f"Step 2 clustered SE: {se_clustered:.4f}")
print(f"Gradient (b1b2): {analytical_grad:.4f}")
print(f"b1b2 SE: {b1b2_se:.4f}")
print(f"Contribution from step 1: |{analytical_grad:.4f}| * {b1b2_se:.4f} = {abs(analytical_grad * b1b2_se):.4f}")
print(f"Total SE: sqrt({se_clustered:.4f}^2 + ({analytical_grad:.4f} * {b1b2_se:.4f})^2) = {se_total:.4f}")

# What if the Murphy-Topel formula gives LARGER SE?
# The full Murphy-Topel correction is:
# V_corrected = V_2 + D * V_1 * D' where D = d(beta_2_hat)/d(theta_1)
# In our case, theta_1 = (b1b2, g2, g3, g4, d2, d3, d4) and beta_2_hat = beta_1 (step 2 param)
# We need V_1 = covariance matrix of step 1 estimates
# The paper reports b1b2_se = 0.0161 but not SEs for higher-order terms

# From Table 2, we can read off the t-statistics:
# b1b2 (coef on d_tenure): t = 0.1258/0.0161 = 7.81
# g2 (tenure^2): coef = -0.4592/100, t = 2.53 -> SE = 0.004592/2.53 = 0.001815
# g3 (tenure^3): coef = 0.1846/1000, t = 1.98 -> SE = 0.0001846/1.98 = 0.0000932
# g4 (tenure^4): coef = -0.0245/10000, t = 0.63 -> SE = 0.00000245/0.63 = 0.00000389
# d2 (exp^2): coef = -0.6051/100, t = 3.77 -> SE = 0.006051/3.77 = 0.001605
# d3 (exp^3): coef = 0.2067/1000, t = 2.82 -> SE = 0.0002067/2.82 = 0.0000733
# d4 (exp^4): coef = -0.0238/10000, t = 2.25 -> SE = 0.00000238/2.25 = 0.00000106
# Wait, let me recheck Table 2 values...
