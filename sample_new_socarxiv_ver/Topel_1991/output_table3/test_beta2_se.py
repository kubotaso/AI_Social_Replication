"""Compute Murphy-Topel SE for beta_2 directly"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

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
    'd2': -0.005871, 'd3': 0.0002067, 'd4': -0.00000238
}

STEP1_SES = {
    'b1b2': 0.0162,
    'g2': 0.001080,
    'g3': 0.0000526,
    'g4': 0.00000079,
    'd2': 0.001546,
    'd3': 0.0000517,
    'd4': 0.00000058,
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

def run_iv(df, step1, ctrl, yr_dums, wage_var='log_real_gnp'):
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
        all_ctrl = ctrl + active_yr
        exog_df = sm.add_constant(levels[all_ctrl])
        rank = np.linalg.matrix_rank(exog_df.values)
    dep = levels['w_star']
    exog = sm.add_constant(levels[all_ctrl])
    endog = levels[['experience']]
    instruments = levels[['init_exp']]
    iv = IV2SLS(dep, exog, endog, instruments).fit(cov_type='unadjusted')
    return iv.params['experience'], iv.std_errors['experience'], len(levels)

# Load data
df = prepare_data("data/psid_panel.csv", pnum_filter=[1, 170, 3])
ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
yr_dums = yr_dums[1:]
wage_var = 'log_real_gnp'

# Get base result
beta_1, se_naive, N = run_iv(df, PAPER_S1, ctrl, yr_dums.copy())
beta_2 = PAPER_S1['b1b2'] - beta_1
print(f"beta_1={beta_1:.4f}, beta_2={beta_2:.4f}, se_naive={se_naive:.6f}")

# Compute gradients for BOTH beta_1 and beta_2
eps = 1e-6
s1_params = ['b1b2', 'g2', 'g3', 'g4', 'd2', 'd3', 'd4']
grad_b1 = {}
grad_b2 = {}
for p in s1_params:
    s1p = PAPER_S1.copy()
    s1p[p] = PAPER_S1[p] + eps
    b1p, _, _ = run_iv(df, s1p, ctrl, yr_dums.copy())
    s1m = PAPER_S1.copy()
    s1m[p] = PAPER_S1[p] - eps
    b1m, _, _ = run_iv(df, s1m, ctrl, yr_dums.copy())
    grad_b1[p] = (b1p - b1m) / (2 * eps)
    # beta_2 = b1b2 - beta_1, so d(beta_2)/d(param) = d(b1b2)/d(param) - d(beta_1)/d(param)
    # d(b1b2)/d(b1b2) = 1, d(b1b2)/d(other) = 0
    if p == 'b1b2':
        grad_b2[p] = 1.0 - grad_b1[p]
    else:
        grad_b2[p] = -grad_b1[p]

print("\nGradients:")
for p in s1_params:
    print(f"  {p}: d(b1)/d={grad_b1[p]:.4f}, d(b2)/d={grad_b2[p]:.4f}")

# Step 1 covariance matrix (our data's correlation structure)
df_s = df.sort_values(['person_id', 'job_id', 'year']).copy()
df_s['d_log_wage'] = df_s.groupby(['person_id', 'job_id'])[wage_var].diff()
df_s['d_tenure'] = df_s.groupby(['person_id', 'job_id'])['tenure'].diff()
wj = df_s.dropna(subset=['d_log_wage', 'd_tenure']).copy()
wj = wj[wj['d_tenure'] == 1].copy()
for k in [2, 3, 4]:
    wj[f'd_tenure_{k}'] = wj.groupby(['person_id', 'job_id'])[f'tenure_{k}'].diff()
    wj[f'd_exp_{k}'] = wj.groupby(['person_id', 'job_id'])[f'exp_{k}'].diff()
wj = wj.dropna(subset=['d_tenure_2', 'd_exp_2']).copy()

yr_dums_wj = sorted([c for c in wj.columns if c.startswith('year_') and c != 'year'])
yr_dums_wj = [c for c in yr_dums_wj if wj[c].std() > 1e-10]
yr_dums_wj = yr_dums_wj[1:] if len(yr_dums_wj) > 1 else yr_dums_wj

X_vars = ['d_tenure', 'd_tenure_2', 'd_tenure_3', 'd_tenure_4',
          'd_exp_2', 'd_exp_3', 'd_exp_4'] + yr_dums_wj
X = sm.add_constant(wj[X_vars])
ols = sm.OLS(wj['d_log_wage'], X).fit()

param_names = ['d_tenure', 'd_tenure_2', 'd_tenure_3', 'd_tenure_4',
               'd_exp_2', 'd_exp_3', 'd_exp_4']
cov_s1_own = ols.cov_params().loc[param_names, param_names].values

# Correlation matrix from our data
D_own = np.sqrt(np.abs(np.diag(np.diag(cov_s1_own))))
D_own_inv = np.linalg.inv(D_own)
corr_mat = D_own_inv @ cov_s1_own @ D_own_inv

# Build cov matrix with paper's SEs
paper_ses = np.array([STEP1_SES[p] for p in s1_params])
D_paper = np.diag(paper_ses)
cov_s1_paper = D_paper @ corr_mat @ D_paper

# Compute Murphy-Topel SE for beta_1
grad_b1_vec = np.array([grad_b1[p] for p in s1_params])
var_mt_b1 = grad_b1_vec @ cov_s1_paper @ grad_b1_vec
var_total_b1 = se_naive**2 + var_mt_b1
se_b1 = np.sqrt(var_total_b1)
print(f"\nMurphy-Topel SE(beta_1) = {se_b1:.4f} (Paper: 0.0181)")

# Compute Murphy-Topel SE for beta_2
grad_b2_vec = np.array([grad_b2[p] for p in s1_params])
var_mt_b2 = grad_b2_vec @ cov_s1_paper @ grad_b2_vec
var_total_b2 = se_naive**2 + var_mt_b2
se_b2 = np.sqrt(var_total_b2)
print(f"Murphy-Topel SE(beta_2) = {se_b2:.4f} (Paper: 0.0079)")

# Let's check the contribution breakdown for beta_2
print("\nBeta_2 SE breakdown:")
print(f"  V_step2 = {se_naive**2:.8f}")
for i, p in enumerate(s1_params):
    contrib = grad_b2[p]**2 * paper_ses[i]**2
    print(f"  {p}: grad={grad_b2[p]:.4f}, se={paper_ses[i]:.8f}, var_contrib={contrib:.8f}")
print(f"  V_MT total (with correlations) = {var_mt_b2:.8f}")
print(f"  V_MT total (without correlations) = {sum(grad_b2[p]**2 * paper_ses[i]**2 for i,p in enumerate(s1_params)):.8f}")

# Also check: what if we use our OWN step1 cov (not scaled by paper SEs)?
var_mt_b2_own = grad_b2_vec @ cov_s1_own @ grad_b2_vec
se_b2_own = np.sqrt(se_naive**2 + var_mt_b2_own)
print(f"\nSE(beta_2) with our own cov: {se_b2_own:.4f}")

# And for beta_1 with our own cov:
var_mt_b1_own = grad_b1_vec @ cov_s1_own @ grad_b1_vec
se_b1_own = np.sqrt(se_naive**2 + var_mt_b1_own)
print(f"SE(beta_1) with our own cov: {se_b1_own:.4f}")

# Key question: the correlation structure matters a LOT for the d-terms.
# What if the b1b2 gradient for beta_2 (≈1.028) is mainly what matters?
# Let's compute the beta_2 SE using ONLY the b1b2 uncertainty:
var_b2_b1b2_only = grad_b2['b1b2']**2 * STEP1_SES['b1b2']**2
se_b2_b1b2_only = np.sqrt(se_naive**2 + var_b2_b1b2_only)
print(f"\nSE(beta_2) from b1b2 only: {se_b2_b1b2_only:.4f}")
# This should be about 1.028 * 0.0162 ≈ 0.0167

# What about ONLY step 2 variance for beta_2?
# beta_2 = b1b2 - beta_1. If b1b2 is treated as fixed (no uncertainty),
# then var(beta_2) = var(beta_1|step1_fixed) = se_naive^2
print(f"SE(beta_2) from step 2 only: {se_naive:.4f}")

# What if we compute cov(beta_1, b1b2) properly?
# cov(beta_1, b1b2) = grad_b1' * Cov * e_1 where e_1 = (1,0,0,0,0,0,0)
e_1 = np.zeros(7)
e_1[0] = 1.0
cov_b1_b1b2_paper = grad_b1_vec @ cov_s1_paper @ e_1
print(f"\ncov(beta_1, b1b2) = {cov_b1_b1b2_paper:.8f}")
# Using the formula: var(b2) = var(b1b2) + var(b1) - 2*cov(b1,b1b2)
var_b2_formula = STEP1_SES['b1b2']**2 + var_total_b1 - 2 * cov_b1_b1b2_paper
print(f"var(b2) from formula = {var_b2_formula:.8f}")
if var_b2_formula > 0:
    print(f"SE(b2) from formula = {np.sqrt(var_b2_formula):.4f}")
else:
    print(f"var(b2) is negative: {var_b2_formula:.8f}")

# Debug: check what cov_b1_b1b2 needs to be for SE(b2) = 0.0079
# 0.0079^2 = 0.0162^2 + se_b1^2 - 2*cov
# cov = (0.0162^2 + se_b1^2 - 0.0079^2) / 2
cov_needed = (STEP1_SES['b1b2']**2 + var_total_b1 - 0.0079**2) / 2
print(f"\ncov(b1,b1b2) needed for SE(b2)=0.0079: {cov_needed:.8f}")
print(f"This is {cov_needed/STEP1_SES['b1b2']**2:.4f} times var(b1b2)")
