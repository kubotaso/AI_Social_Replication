"""Test OLS on X_0 approach and SE calculation"""
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

ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
yr_dums = yr_dums[1:]

# Drop NaNs for all analysis vars
levels = df.dropna(subset=['log_real_gnp', 'experience', 'tenure', 'init_exp'] + ctrl).copy()

# Construct w*
levels['w_star'] = (levels['log_real_gnp']
                    - PAPER_S1['b1b2'] * levels['tenure']
                    - PAPER_S1['g2'] * levels['tenure_2']
                    - PAPER_S1['g3'] * levels['tenure_3']
                    - PAPER_S1['g4'] * levels['tenure_4']
                    - PAPER_S1['d2'] * levels['exp_2']
                    - PAPER_S1['d3'] * levels['exp_3']
                    - PAPER_S1['d4'] * levels['exp_4'])

# gamma_X0T: regression of T on X_0
ols_gamma = sm.OLS(levels['tenure'], sm.add_constant(levels['init_exp'])).fit()
print(f"gamma_X0T (simple): {ols_gamma.params['init_exp']:.4f}")

# with controls
all_ctrl = ctrl + yr_dums
exog_g = sm.add_constant(levels[['init_exp'] + all_ctrl])
ols_gamma2 = sm.OLS(levels['tenure'], exog_g).fit()
print(f"gamma_X0T (with controls): {ols_gamma2.params['init_exp']:.4f}")

print(f"\nMean tenure: {levels['tenure'].mean():.2f}")
print(f"Mean init_exp: {levels['init_exp'].mean():.2f}")
print(f"Mean experience: {levels['experience'].mean():.2f}")
print(f"Corr(T, X0): {levels['tenure'].corr(levels['init_exp']):.4f}")

# ===============================================
# OLS of w* on X_0 (direct, no IV)
# ===============================================
print("\n=== OLS of w* on X_0 directly ===")
X_ols_x0 = sm.add_constant(levels[['init_exp'] + all_ctrl])
ols_x0 = sm.OLS(levels['w_star'], X_ols_x0).fit()
print(f"beta_1 (OLS on X_0): {ols_x0.params['init_exp']:.4f}")
print(f"SE (OLS on X_0): {ols_x0.bse['init_exp']:.4f}")

# Clustered
ols_x0_c = ols_x0.get_robustcov_results(cov_type='cluster', groups=levels['person_id'])
print(f"beta_1 (OLS on X_0, clustered): {ols_x0_c.params[1]:.4f}")
print(f"SE (OLS on X_0, clustered): {ols_x0_c.bse[1]:.4f}")

# ===============================================
# IV: w* ~ experience, instrument = X_0
# ===============================================
print("\n=== IV: w* ~ experience, instrument = X_0 ===")
exog_iv = sm.add_constant(levels[all_ctrl])

# Fix rank
rank = np.linalg.matrix_rank(exog_iv.values)
active_yr = yr_dums.copy()
while rank < exog_iv.shape[1] and active_yr:
    active_yr.pop()
    all_ctrl_temp = ctrl + active_yr
    exog_iv = sm.add_constant(levels[all_ctrl_temp])
    rank = np.linalg.matrix_rank(exog_iv.values)
final_ctrl = ctrl + active_yr

exog_iv = sm.add_constant(levels[final_ctrl])
endog_iv = levels[['experience']]
instruments_iv = levels[['init_exp']]
dep_iv = levels['w_star']

for cov in ['unadjusted', 'robust', 'clustered']:
    if cov == 'clustered':
        iv_fit = IV2SLS(dep_iv, exog_iv, endog_iv, instruments_iv).fit(
            cov_type='clustered', clusters=levels['person_id'])
    else:
        iv_fit = IV2SLS(dep_iv, exog_iv, endog_iv, instruments_iv).fit(cov_type=cov)
    print(f"  IV {cov}: coef={iv_fit.params['experience']:.4f}, SE={iv_fit.std_errors['experience']:.4f}")

print(f"\n  Paper: beta_1=0.0713, SE=0.0181")

# ===============================================
# Key question: what is the correct specification?
# ===============================================
# The paper says equation (10): y - chi_hat*Gamma_hat = X_0*beta_1 + F*gamma + e
# This uses X_0 (initial experience) as the REGRESSOR directly, not as an instrument.
# But our implementation uses X (current experience) as the endogenous variable
# and X_0 as the instrument.
# Since X = X_0 + T, the IV coefficient on X should equal the OLS coefficient on X_0
# if the model is correctly specified.

# Let's verify: the IV coefficient on X should be the same as:
# reduced form: regress w* on X_0 + controls -> get pi
# first stage: regress X on X_0 + controls -> get delta
# IV = pi / delta

# First stage
X_fs = sm.add_constant(levels[['init_exp'] + final_ctrl])
fs = sm.OLS(levels['experience'], X_fs).fit()
print(f"\nFirst stage: coef on X_0 = {fs.params['init_exp']:.4f}")
print(f"  (should be ~1 since X = X_0 + T and T varies)")

# Reduced form
X_rf = sm.add_constant(levels[['init_exp'] + final_ctrl])
rf = sm.OLS(levels['w_star'], X_rf).fit()
print(f"Reduced form: coef on X_0 = {rf.params['init_exp']:.4f}")

# IV = reduced form / first stage
print(f"IV = RF/FS = {rf.params['init_exp'] / fs.params['init_exp']:.4f}")

# The SE of the reduced form (OLS on X_0) is the proper SE for the coefficient
# when X_0 is the regressor directly. But this needs Murphy-Topel correction.
print(f"\nReduced form SE: {rf.bse['init_exp']:.4f}")
rf_c = rf.get_robustcov_results(cov_type='cluster', groups=levels['person_id'])
print(f"Reduced form clustered SE: {rf_c.bse[1]:.4f}")

# Murphy-Topel correction: add step 1 uncertainty
# The gradient is: d(beta_1)/d(B_hat) for the OLS-on-X_0 specification
# When B_hat changes, w* = y - B*T - ... changes, and the OLS coef on X_0 changes.
# The gradient = -Cov(X_0_r, T_r) / Var(X_0_r) where _r means residualized from controls
from numpy.linalg import lstsq
X_c = sm.add_constant(levels[final_ctrl]).values
X0 = levels['init_exp'].values
T = levels['tenure'].values

X0_hat = X_c @ lstsq(X_c, X0, rcond=None)[0]
X0_r = X0 - X0_hat
T_hat = X_c @ lstsq(X_c, T, rcond=None)[0]
T_r = T - T_hat

gradient_ols = -np.sum(X0_r * T_r) / np.sum(X0_r * X0_r)
print(f"\nGradient (OLS on X_0): d(beta1)/d(B_hat) = {gradient_ols:.4f}")
print(f"  This is -Cov(X0_r, T_r)/Var(X0_r) = gamma_X0T with controls")

# Total SE with Murphy-Topel
se_step2 = rf_c.bse[1]  # clustered SE from step 2
var_total = se_step2**2 + gradient_ols**2 * PAPER_S1['b1b2_se']**2
se_total = np.sqrt(var_total)
print(f"\nMurphy-Topel corrected SE:")
print(f"  Step 2 clustered SE: {se_step2:.4f}")
print(f"  Gradient: {gradient_ols:.4f}")
print(f"  Step 1 SE (b1b2): {PAPER_S1['b1b2_se']:.4f}")
print(f"  Total SE: sqrt({se_step2:.4f}^2 + {gradient_ols:.4f}^2 * {PAPER_S1['b1b2_se']:.4f}^2) = {se_total:.4f}")
print(f"  Paper SE: 0.0181")
