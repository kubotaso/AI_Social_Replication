"""
Diagnostic 23: Investigate the gradient d(beta_1)/d(b1b2) and SE structure.
The issue: our gradient is -0.033 but should be closer to -1.
This affects both beta_1_se (Murphy-Topel) and beta_2_se.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

PAPER_S1 = {
    'b1b2': 0.1258, 'b1b2_se': 0.0161,
    'g2': -0.004592, 'g3': 0.0001846, 'g4': -0.00000245,
    'd2': -0.006051, 'd3': 0.0002067, 'd4': -0.00000238
}


def prepare_data(data_source, pnum_filter=None):
    df = pd.read_csv(data_source)
    if pnum_filter is not None:
        df['pnum'] = df['person_id'] % 1000
        df = df[df['pnum'].isin(pnum_filter)].copy()

    df['education_years'] = df['education_clean'].copy()
    cat_mask = ~df['year'].isin([1975, 1976])
    df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(
        {**EDUC_MAP, 9: np.nan}
    )
    def get_fixed_educ(group):
        good = group[group['year'].isin([1975, 1976])]['education_years'].dropna()
        if len(good) > 0: return good.iloc[0]
        mapped = group['education_years'].dropna()
        if len(mapped) > 0:
            modes = mapped.mode()
            return modes.iloc[0] if len(modes) > 0 else mapped.median()
        return np.nan
    person_educ = df.groupby('person_id').apply(get_fixed_educ)
    df['education_fixed'] = df['person_id'].map(person_educ)
    df = df[df['education_fixed'].notna()].copy()

    df['experience'] = (df['age'] - df['education_fixed'] - 6).clip(lower=0)
    df['tenure'] = df['tenure_topel']

    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
    df['log_wage_cps'] = df['log_hourly_wage'] - np.log(df['cps_index'])
    df['gnp_def'] = df['year'].map(lambda y: GNP_DEFLATOR.get(y - 1, np.nan))
    df['log_wage_gnp'] = df['log_hourly_wage'] - np.log(df['gnp_def'] / 100.0)

    for c in ['married', 'union_member', 'disabled', 'region_ne', 'region_nc',
              'region_south', 'region_west']:
        df[c] = df[c].fillna(0)

    df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
    df = df[df['hourly_wage'] > 0].copy()
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
    df = df[df['tenure'] >= 1].copy()
    df = df.dropna(subset=['log_wage_cps', 'experience', 'tenure']).copy()

    df['init_exp'] = (df['experience'] - df['tenure']).clip(lower=0)
    df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
    return df


def run_iv(levels_df, step1, ctrl, yr_dums_use, wage_var='log_wage_gnp'):
    b1b2 = step1['b1b2']
    g2, g3, g4 = step1['g2'], step1['g3'], step1['g4']
    d2, d3, d4 = step1['d2'], step1['d3'], step1['d4']

    levels = levels_df.copy()
    T = levels['tenure'].values.astype(float)
    X_exp = levels['experience'].values.astype(float)
    levels['w_star'] = (levels[wage_var]
                        - b1b2 * T - g2 * T**2 - g3 * T**3 - g4 * T**4
                        - d2 * X_exp**2 - d3 * X_exp**3 - d4 * X_exp**4)

    all_ctrl = ctrl + yr_dums_use
    levels_clean = levels.dropna(subset=['w_star', 'experience', 'init_exp'] + all_ctrl).copy()

    dep = levels_clean['w_star']
    exog = sm.add_constant(levels_clean[all_ctrl])

    rank = np.linalg.matrix_rank(exog.values)
    current_yr = yr_dums_use.copy()
    while rank < exog.shape[1] and current_yr:
        current_yr.pop()
        exog = sm.add_constant(levels_clean[ctrl + current_yr])
        rank = np.linalg.matrix_rank(exog.values)
    exog = sm.add_constant(levels_clean[ctrl + current_yr])

    endog = levels_clean[['experience']]
    instruments = levels_clean[['init_exp']]

    iv = IV2SLS(dep, exog, endog, instruments).fit(cov_type='unadjusted')

    return iv.params['experience'], iv.std_errors['experience'], len(levels_clean)


# ======================================================================
# Test gradient with different step sizes and configurations
# ======================================================================

# Setup
df = prepare_data("data/psid_panel.csv", pnum_filter=[1, 3, 170, 171])
ctrl = ['education_fixed', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_cols = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums_use = [c for c in yr_cols if df[c].std() > 1e-10][1:]

# Baseline
b1_base, se_base, N = run_iv(df, PAPER_S1, ctrl, yr_dums_use, 'log_wage_gnp')
print(f"Baseline: beta_1={b1_base:.6f}, SE_naive={se_base:.6f}, N={N}")

# Gradient with different deltas
for delta in [0.001, 0.01, 0.05, 0.1]:
    s1_plus = PAPER_S1.copy()
    s1_plus['b1b2'] = PAPER_S1['b1b2'] + delta
    b1_plus, _, _ = run_iv(df, s1_plus, ctrl, yr_dums_use, 'log_wage_gnp')

    s1_minus = PAPER_S1.copy()
    s1_minus['b1b2'] = PAPER_S1['b1b2'] - delta
    b1_minus, _, _ = run_iv(df, s1_minus, ctrl, yr_dums_use, 'log_wage_gnp')

    grad_forward = (b1_plus - b1_base) / delta
    grad_central = (b1_plus - b1_minus) / (2 * delta)
    print(f"  delta={delta:.3f}: grad_fwd={grad_forward:.6f}, grad_central={grad_central:.6f}")

# Check: what's the correlation between T and X in the sample?
print(f"\nCorrelation between tenure and experience: {df['tenure'].corr(df['experience']):.4f}")
print(f"Mean tenure: {df['tenure'].mean():.2f}")
print(f"Mean experience: {df['experience'].mean():.2f}")
print(f"Mean init_exp: {df['init_exp'].mean():.2f}")

# The gradient is near zero because in IV, beta_1 is estimated using
# the INSTRUMENT (init_exp), not directly from experience.
# When b1+b2 changes, w* changes by -delta*T.
# In IV, beta_1 = Cov(w*, Z) / Cov(X, Z) where Z=init_exp, X=experience
# d(beta_1)/d(b1b2) = -Cov(T, Z) / Cov(X, Z)
# Since T = X - init_exp, T = X - Z (approximately)
# Cov(T, Z) = Cov(X-Z, Z) = Cov(X,Z) - Var(Z)
# d(beta_1)/d(b1b2) = -(Cov(X,Z) - Var(Z)) / Cov(X,Z)
# = -1 + Var(Z)/Cov(X,Z)

# Let's compute this
T = df['tenure'].values
X = df['experience'].values
Z = df['init_exp'].values
cov_TZ = np.cov(T, Z)[0,1]
cov_XZ = np.cov(X, Z)[0,1]
var_Z = np.var(Z)
print(f"\nCov(T, Z) = {cov_TZ:.4f}")
print(f"Cov(X, Z) = {cov_XZ:.4f}")
print(f"Var(Z) = {var_Z:.4f}")
print(f"Predicted gradient = -Cov(T,Z)/Cov(X,Z) = {-cov_TZ/cov_XZ:.4f}")
print(f"  or equivalently = -1 + Var(Z)/Cov(X,Z) = {-1 + var_Z/cov_XZ:.4f}")

# So the gradient is approximately:
# d(beta_1)/d(b1b2) ≈ -Cov(T,Z)/Cov(X,Z) ≈ -1 + Var(Z)/Cov(X,Z)
# In our case, Var(Z) is large relative to Cov(X,Z) so gradient is near 0.

# This means: when b1b2 changes by delta, beta_1 barely changes.
# So beta_2 = b1b2 - beta_1 changes by approximately delta.
# And Var(beta_2) ≈ Var(b1b2) = b1b2_se^2

# This gives: beta_2_se ≈ b1b2_se = 0.0232 for our data
# Paper's b1b2_se = 0.0161 and beta_2_se = 0.0079.
# So the paper has beta_2_se much smaller than b1b2_se.

# Wait, this doesn't make sense. If grad ≈ 0, then
# beta_2 = b1b2 - beta_1, and beta_1 doesn't depend on b1b2
# So Var(beta_2) = Var(b1b2) + Var(beta_1) (approximately independent)
# Which would be LARGER than Var(b1b2).

# But the paper has beta_2_se < b1b2_se. This must mean the gradient
# is significantly negative (closer to -1), which would create cancellation.

# The discrepancy might be because the controls and year dummies
# absorb a lot of the tenure variation in our step 2, making
# the IV estimator less sensitive to b1b2 changes.

# Let's check what happens with OLS instead of IV
exog_cols = ctrl + yr_dums_use
levels = df.dropna(subset=['log_wage_gnp', 'experience', 'init_exp'] + exog_cols).copy()
T_vals = levels['tenure'].values.astype(float)
X_exp = levels['experience'].values.astype(float)

# OLS approach: regress w* on X_0 directly (no IV)
for b1b2_val in [PAPER_S1['b1b2'], PAPER_S1['b1b2'] + 0.01]:
    levels['w_star'] = (levels['log_wage_gnp']
                        - b1b2_val * T_vals
                        - PAPER_S1['g2'] * T_vals**2
                        - PAPER_S1['g3'] * T_vals**3
                        - PAPER_S1['g4'] * T_vals**4
                        - PAPER_S1['d2'] * X_exp**2
                        - PAPER_S1['d3'] * X_exp**3
                        - PAPER_S1['d4'] * X_exp**4)

    X_ols = sm.add_constant(levels[['init_exp'] + exog_cols])
    rank = np.linalg.matrix_rank(X_ols.values)
    active_yr = yr_dums_use.copy()
    while rank < X_ols.shape[1] and active_yr:
        active_yr.pop()
        X_ols = sm.add_constant(levels[['init_exp'] + ctrl + active_yr])
        rank = np.linalg.matrix_rank(X_ols.values)

    ols = sm.OLS(levels['w_star'], X_ols).fit()
    print(f"\n  OLS (b1b2={b1b2_val:.4f}): beta_1={ols.params['init_exp']:.6f}, SE={ols.bse['init_exp']:.6f}")

# OLS gradient
s1_base = PAPER_S1.copy()
s1_delta = PAPER_S1.copy()
s1_delta['b1b2'] = PAPER_S1['b1b2'] + 0.01

levels['w_star_base'] = (levels['log_wage_gnp']
                    - s1_base['b1b2'] * T_vals
                    - s1_base['g2'] * T_vals**2 - s1_base['g3'] * T_vals**3 - s1_base['g4'] * T_vals**4
                    - s1_base['d2'] * X_exp**2 - s1_base['d3'] * X_exp**3 - s1_base['d4'] * X_exp**4)

levels['w_star_delta'] = (levels['log_wage_gnp']
                    - s1_delta['b1b2'] * T_vals
                    - s1_delta['g2'] * T_vals**2 - s1_delta['g3'] * T_vals**3 - s1_delta['g4'] * T_vals**4
                    - s1_delta['d2'] * X_exp**2 - s1_delta['d3'] * X_exp**3 - s1_delta['d4'] * X_exp**4)

# The change in w* is just -0.01 * T
print(f"\n  Change in w*: mean={np.mean(levels['w_star_delta'] - levels['w_star_base']):.6f}")
print(f"  Should be: -0.01 * mean_T = {-0.01 * np.mean(T_vals):.6f}")

# Now the question is: when w* changes by -delta*T, how does the OLS coefficient on X_0 change?
# In the step 2 equation: w* = alpha + beta_1 * X + gamma * F + e
# where X = experience, instrumenting with X_0 = init_exp
# w* = log_wage - (b1+b2)*T - g2*T^2 - ... - d2*X^2 - ...
#
# When b1b2 increases by delta:
# w*_new = w*_old - delta * T
# beta_1_new = beta_1_old - delta * Cov(T, Z|controls) / Cov(X, Z|controls)
#
# For OLS on X_0:
# beta_1_new = beta_1_old - delta * Cov(T, X_0|controls) / Var(X_0|controls)

# The key: X = X_0 + T, so T = X - X_0
# Cov(T, X_0|controls) = Cov(X - X_0, X_0|controls) = Cov(X, X_0|controls) - Var(X_0|controls)

# For OLS on X_0:
# d(beta_1)/d(b1b2) = -(Cov(X-X_0, X_0|C) / Var(X_0|C))
# = -(Cov(X, X_0|C) / Var(X_0|C) - 1)
# = 1 - Cov(X, X_0|C) / Var(X_0|C)
# = 1 - reg_coeff(X on X_0|C)

# This should give approximately 1 - 1 = 0 if X_0 is a perfect predictor of X.
# But init_exp is NOT a perfect predictor of experience (tenure adds noise)
# So the gradient depends on how well X_0 predicts X given controls.

# For IV with X_0 as instrument for X:
# beta_1_IV = Cov(w*, X_0|C) / Cov(X, X_0|C)
# d(beta_1_IV)/d(b1b2) = -Cov(T, X_0|C) / Cov(X, X_0|C)
# = -1 + Var(X_0|C) / Cov(X, X_0|C)

# If X = X_0 + T and X_0, T are independent given C:
# Cov(X, X_0|C) = Var(X_0|C)
# Then gradient = -1 + 1 = 0

# So the gradient IS near zero because X_0 and T are approximately
# independent (conditional on controls), which makes Cov(X, X_0|C) ≈ Var(X_0|C).

# This means beta_1 doesn't respond to b1b2, so:
# Var(beta_2) = Var(b1b2 - beta_1) ≈ Var(b1b2) + Var(beta_1)
# And the paper's small beta_2_se would require a NEGATIVE correlation between
# b1b2 and beta_1, which shouldn't happen with this gradient.

# CONCLUSION: The paper's very small beta_2_se (0.0079) might come from a
# different SE calculation. Perhaps the paper approximates:
# beta_2_se ≈ b1b2_se * sqrt(1 - R^2) where R^2 is from the step 2 regression
# Or uses a different Murphy-Topel formula.

# Let's check what happens if we just use b1b2_se * ratio
# Paper: beta_2_se / b1b2_se = 0.0079/0.0161 = 0.49
# So beta_2_se ≈ 0.49 * b1b2_se for the baseline

# For our data:
print(f"\nPaper ratios:")
for t in [0, 1, 3, 5]:
    b1b2_se = {0: 0.0161, 1: 0.0161, 3: 0.0161, 5: 0.0161}[t]  # approximate
    b2_se = {0: 0.0079, 1: 0.0089, 3: 0.0109, 5: 0.0132}[t]
    b1_se = {0: 0.0181, 1: 0.0204, 3: 0.0245, 5: 0.0292}[t]
    print(f"  >={t}: b2_se/b1_se = {b2_se/b1_se:.3f}")

# Actually the paper's beta_1_se itself might use a different formula.
# Let me check: if we use PAPER's b1b2_se for the Murphy-Topel correction:
print(f"\n\nSE analysis with paper's b1b2_se=0.0161:")
b1_naive = se_base
print(f"  beta_1_se_naive (from IV): {b1_naive:.6f}")
b1_mt = np.sqrt(b1_naive**2 + PAPER_S1['b1b2_se']**2)
print(f"  beta_1_se_MT = sqrt({b1_naive:.6f}^2 + {PAPER_S1['b1b2_se']:.6f}^2) = {b1_mt:.6f}")
print(f"  Paper beta_1_se = 0.0181")
