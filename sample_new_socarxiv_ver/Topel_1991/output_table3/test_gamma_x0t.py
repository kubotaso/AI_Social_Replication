"""Compute gamma_{X0,T} - OLS regression of T on X_0"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
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
    for yr in range(1970, 1984):
        col = f'year_{yr}'
        if col not in df.columns:
            df[col] = (df['year'] == yr).astype(int)
    return df


df = prepare_data("data/psid_panel.csv", pnum_filter=[1, 170, 3])
df['init_exp'] = (df['experience'] - df['tenure']).clip(lower=0)

# Simple regression of T on X_0
ols_simple = sm.OLS(df['tenure'], sm.add_constant(df['init_exp'])).fit()
print(f"Simple OLS: T on X_0")
print(f"  gamma_X0T = {ols_simple.params['init_exp']:.4f}")
print(f"  Paper says gamma_X0T = -0.25")

# With controls
ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
yr_dums = yr_dums[1:]

X_ctrl = sm.add_constant(df[['init_exp'] + ctrl + yr_dums])
ols_ctrl = sm.OLS(df['tenure'], X_ctrl).fit()
print(f"\nWith controls: T on X_0 + controls")
print(f"  gamma_X0T = {ols_ctrl.params['init_exp']:.4f}")

# Now, the bias formula from (8a) is:
# E(beta_1_hat) = beta_1 + b_1 + gamma_X0T * (b_1 + b_2)
# where b_1, b_2 are the true biases
# This tells us that the SENSITIVITY of beta_1_hat to (b1+b2) is gamma_X0T
#
# But for the SE calculation, we need:
# Var(beta_1_hat) includes the component from estimating B_hat = b1+b2 in step 1
# The relevant gradient is d(beta_1_hat)/d(B_hat)
#
# In the IV regression: y - T*B_hat = X_0*beta_1 + e
# When B_hat changes by delta, y-TB changes by -delta*T
# beta_1_hat = (X_0'M * (y-TB)) / (X_0'M * X)
# d(beta_1_hat)/d(B) = -(X_0'M * T) / (X_0'M * X)
#
# This is the Wald/IV formula. Let me compute it properly
# by partialling out controls

from numpy.linalg import lstsq

all_ctrl = ctrl + yr_dums
X_c = sm.add_constant(df[all_ctrl]).values
Z = df['init_exp'].values  # instrument
X_exp = df['experience'].values  # endogenous
T = df['tenure'].values

# Partial out controls
Z_hat = X_c @ lstsq(X_c, Z, rcond=None)[0]
Z_r = Z - Z_hat
X_hat = X_c @ lstsq(X_c, X_exp, rcond=None)[0]
X_r = X_exp - X_hat
T_hat = X_c @ lstsq(X_c, T, rcond=None)[0]
T_r = T - T_hat

# The IV gradient
iv_grad = -np.sum(Z_r * T_r) / np.sum(Z_r * X_r)
print(f"\nIV gradient d(beta1)/d(B_hat) = {iv_grad:.4f}")

# The reduced-form gradient (which is what enters the bias formula)
# = -Cov(Z_r, T_r) / Cov(Z_r, X_r)
# This should be related to gamma_X0T but not identical

# gamma_X0T as defined by the paper: LS coefficient of T on X_0
# This is Cov(X_0, T) / Var(X_0) (simple bivariate)
gamma_simple = np.cov(Z, T)[0,1] / np.var(Z)
print(f"\ngamma_X0T (bivariate) = {gamma_simple:.4f}")

# With partial controls
gamma_partial = np.sum(Z_r * T_r) / np.sum(Z_r * Z_r)
print(f"gamma_X0T (partial) = {gamma_partial:.4f}")

# The SE formula from Murphy-Topel for this specific case:
# In equation (7): y - T*B_hat = X_0*beta_1 + e
# where e = epsilon + T*(B - B_hat)
# If we use IV (instrument X with X_0), the coefficient is:
#   beta_1_hat = (Z'M*Y_tilde) / (Z'M*X)
# where Y_tilde = y - T*B_hat
# = (Z'M*(X_0*beta_1 + epsilon + T*(B-B_hat))) / (Z'M*X)
# = beta_1 * (Z'M*X_0)/(Z'M*X) + (Z'M*epsilon)/(Z'M*X) + (B-B_hat)*(Z'M*T)/(Z'M*X)
#
# The first term = beta_1 (since X = X_0 + T, and this is the IV)
# The second term has variance = sigma^2 / (Z'M*X) which is the standard IV SE
# The third term has variance = Var(B_hat) * [(Z'M*T)/(Z'M*X)]^2
#
# So: Var(beta_1_hat) = sigma^2/(Z'MZ * first_stage_F) + [gamma]^2 * Var(B_hat)
# where gamma = (Z'M*T)/(Z'M*X) = iv_grad (= -0.028)

# But the paper says SE = 0.0181. With gamma = -0.028:
# sqrt(0.0009^2 + 0.028^2 * 0.0161^2) = sqrt(8.1e-7 + 2e-7) = sqrt(1e-6) = 0.001
# This gives only 0.001, way too small.

# BUT WAIT. The paper uses equation (7) which is the SIMPLIFIED version:
# y - T*B_hat = X_0*beta_1 + e
# In this version, X_0 appears DIRECTLY (not current experience X).
# The IV is instrumenting X with X_0, but in the simplified model,
# X_0 IS the regressor!

# Let me re-read equation (10):
# y - chi_hat*Gamma_hat = X_0*beta_1 + F*gamma + e    (equation 10)
# Here chi*Gamma includes ALL higher-order terms (tenure and experience polynomials)
# and X_0 is initial experience.

# WAIT - the paper says "the estimated value of beta_1 from implementing (10) is about 7 percent"
# So equation (10) regresses the adjusted wage on X_0 (initial experience) directly,
# not on X (current experience)!

# But that can't be right. If X_0 is the regressor AND the instrument,
# it's just OLS of adjusted wage on X_0.

# Actually, re-reading equation (7): y - T*B_hat = X_0*beta_1 + e
# This IS OLS because X_0 is directly the regressor.
# Since X = X_0 + T, we have: y = X_0*beta_1 + T*B + epsilon
#   y - T*B_hat = X_0*beta_1 + T*(B - B_hat) + epsilon
# So this is an OLS regression of (y - T*B_hat) on X_0.

# If it's OLS on X_0, then there's no first-stage / IV issue!
# The SE would just be the OLS SE of X_0, inflated by the Murphy-Topel correction.

# Let me try OLS with X_0 as regressor instead of IV with X instrumented by X_0
print("\n\n=== OLS of adjusted wage on X_0 (initial experience) ===")

levels = df.copy()
levels['w_star'] = (levels['log_real_gnp']
                    - PAPER_S1['b1b2'] * levels['tenure']
                    - PAPER_S1['g2'] * levels['tenure_2']
                    - PAPER_S1['g3'] * levels['tenure_3']
                    - PAPER_S1['g4'] * levels['tenure_4']
                    - PAPER_S1['d2'] * levels['exp_2']
                    - PAPER_S1['d3'] * levels['exp_3']
                    - PAPER_S1['d4'] * levels['exp_4'])

all_x = ['init_exp'] + ctrl + yr_dums
X_ols = sm.add_constant(levels[all_x])
ols_x0 = sm.OLS(levels['w_star'], X_ols).fit()
print(f"OLS coefficient on X_0: {ols_x0.params['init_exp']:.4f}")
print(f"OLS SE on X_0: {ols_x0.bse['init_exp']:.4f}")
print(f"Paper: beta_1 = 0.0713, SE = 0.0181")

# Now, what if we use clustered SE
from linearmodels.iv import IV2SLS

# OLS is just IV where the instrument IS the regressor
dep = levels['w_star']
exog = sm.add_constant(levels[ctrl + yr_dums])
endog = levels[['init_exp']]
instruments = levels[['init_exp']]

# Actually, let's just use statsmodels OLS with clusters
from statsmodels.stats.sandwich_covariance import cov_cluster
ols_fit = sm.OLS(levels['w_star'], X_ols).fit()
cluster_cov = cov_cluster(ols_fit, levels['person_id'])
ols_fit_clustered = ols_fit.get_robustcov_results(cov_type='cluster',
                                                    groups=levels['person_id'])
print(f"\nOLS coefficient on X_0 (clustered): {ols_fit_clustered.params[all_x.index('init_exp')+1]:.4f}")
print(f"OLS SE on X_0 (clustered): {ols_fit_clustered.bse[all_x.index('init_exp')+1]:.4f}")

# Also try with current experience instrumented by X_0 (the standard IV approach)
print("\n\n=== IV: experience instrumented by X_0 ===")
from linearmodels.iv import IV2SLS
dep_iv = levels['w_star']
exog_iv = sm.add_constant(levels[ctrl + yr_dums])
endog_iv = levels[['experience']]
instruments_iv = levels[['init_exp']]

for cov in ['unadjusted', 'robust', 'clustered']:
    if cov == 'clustered':
        iv_fit = IV2SLS(dep_iv, exog_iv, endog_iv, instruments_iv).fit(
            cov_type='clustered', clusters=levels['person_id'])
    else:
        iv_fit = IV2SLS(dep_iv, exog_iv, endog_iv, instruments_iv).fit(cov_type=cov)
    print(f"  IV {cov}: coef={iv_fit.params['experience']:.4f}, SE={iv_fit.std_errors['experience']:.4f}")

print(f"\n  Paper: beta_1=0.0713, SE=0.0181")
