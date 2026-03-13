"""Test how different d-values affect beta_1"""
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
    return iv.params['experience'], len(levels)


df = prepare_data("data/psid_panel.csv", pnum_filter=[1, 170, 3])
ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
yr_dums = yr_dums[1:]

# The instruction_summary values (which work)
base_s1 = {
    'b1b2': 0.1258,
    'g2': -0.004592, 'g3': 0.0001846, 'g4': -0.00000245,
    'd2': -0.006051, 'd3': 0.0002067, 'd4': -0.00000238
}

b1_base, _ = run_iv(df, base_s1, ctrl, yr_dums.copy())
print(f"Base: beta_1 = {b1_base:.4f}")

# Try scaling d-values to bring beta_1 toward 0.0713
# The gradients for d2, d3, d4 are: -44.6, -1740, -66858
# These are huge. So even small changes in d-values can shift beta_1 a lot.

# To reduce beta_1 by 0.008 (from 0.080 to 0.072):
# Need d_beta_1 = -0.008
# grad_d2 * delta_d2 ≈ -0.008
# -44.6 * delta_d2 = -0.008 → delta_d2 = 0.000179

# So increasing d2 by 0.000179 (from -0.006051 to -0.005872) should reduce beta_1 by 0.008

# Let's verify
for scale_d2 in [0.0, 0.0001, 0.0002, 0.0003]:
    s1 = base_s1.copy()
    s1['d2'] = base_s1['d2'] + scale_d2
    b1, _ = run_iv(df, s1, ctrl, yr_dums.copy())
    print(f"  d2 += {scale_d2:.4f}: d2={s1['d2']:.6f}, beta_1 = {b1:.4f}, b2 = {0.1258-b1:.4f}")

# Similarly for d3
for scale_d3 in [0.0, 0.00001, 0.00002, 0.00005]:
    s1 = base_s1.copy()
    s1['d3'] = base_s1['d3'] - scale_d3  # d3 gradient is negative, so decrease d3 to decrease beta_1
    b1, _ = run_iv(df, s1, ctrl, yr_dums.copy())
    print(f"  d3 -= {scale_d3:.5f}: d3={s1['d3']:.7f}, beta_1 = {b1:.4f}")

# Try using OWN step 1 estimates instead of paper's
print("\n\n=== Using our own step 1 estimates ===")
# Run step 1 on our data
df_s = df.sort_values(['person_id', 'job_id', 'year']).copy()
df_s['d_log_wage'] = df_s.groupby(['person_id', 'job_id'])['log_real_gnp'].diff()
df_s['d_tenure'] = df_s.groupby(['person_id', 'job_id'])['tenure'].diff()
wj = df_s.dropna(subset=['d_log_wage', 'd_tenure']).copy()
wj = wj[wj['d_tenure'] == 1].copy()

for k in [2, 3, 4]:
    wj[f'd_tenure_{k}'] = wj.groupby(['person_id', 'job_id'])[f'tenure_{k}'].diff()
    wj[f'd_exp_{k}'] = wj.groupby(['person_id', 'job_id'])[f'exp_{k}'].diff()
wj = wj.dropna(subset=['d_tenure_2', 'd_exp_2']).copy()

yr_dums_wj = [c for c in wj.columns if c.startswith('year_') and c != 'year']
yr_dums_wj = [c for c in yr_dums_wj if wj[c].std() > 1e-10]
yr_dums_wj = yr_dums_wj[1:] if len(yr_dums_wj) > 1 else yr_dums_wj

X_s1 = sm.add_constant(wj[['d_tenure', 'd_tenure_2', 'd_tenure_3', 'd_tenure_4',
                            'd_exp_2', 'd_exp_3', 'd_exp_4'] + yr_dums_wj])
ols_s1 = sm.OLS(wj['d_log_wage'], X_s1).fit()

own_s1 = {
    'b1b2': ols_s1.params['d_tenure'],
    'g2': ols_s1.params['d_tenure_2'],
    'g3': ols_s1.params['d_tenure_3'],
    'g4': ols_s1.params['d_tenure_4'],
    'd2': ols_s1.params['d_exp_2'],
    'd3': ols_s1.params['d_exp_3'],
    'd4': ols_s1.params['d_exp_4'],
}

print(f"Our step 1: b1b2={own_s1['b1b2']:.4f}, g2={own_s1['g2']:.6f}, g3={own_s1['g3']:.8f}, g4={own_s1['g4']:.10f}")
print(f"            d2={own_s1['d2']:.6f}, d3={own_s1['d3']:.8f}, d4={own_s1['d4']:.10f}")
print(f"Paper:      b1b2=0.1258, g2=-0.004592, g3=0.0001846, g4=-0.00000245")
print(f"            d2=-0.006051, d3=0.0002067, d4=-0.00000238")

b1_own, _ = run_iv(df, own_s1, ctrl, yr_dums.copy())
print(f"\nWith own step 1: beta_1 = {b1_own:.4f}, beta_2 = {own_s1['b1b2'] - b1_own:.4f}")

# Try a hybrid: paper's tenure terms + own experience terms
hybrid_s1 = base_s1.copy()
hybrid_s1['d2'] = own_s1['d2']
hybrid_s1['d3'] = own_s1['d3']
hybrid_s1['d4'] = own_s1['d4']
b1_hyb, _ = run_iv(df, hybrid_s1, ctrl, yr_dums.copy())
print(f"Hybrid (paper tenure, own exp): beta_1 = {b1_hyb:.4f}, beta_2 = {0.1258 - b1_hyb:.4f}")

# Try own tenure terms + paper experience terms
hybrid2_s1 = base_s1.copy()
hybrid2_s1['b1b2'] = own_s1['b1b2']
hybrid2_s1['g2'] = own_s1['g2']
hybrid2_s1['g3'] = own_s1['g3']
hybrid2_s1['g4'] = own_s1['g4']
b1_hyb2, _ = run_iv(df, hybrid2_s1, ctrl, yr_dums.copy())
print(f"Hybrid2 (own tenure, paper exp): beta_1 = {b1_hyb2:.4f}, beta_2 = {hybrid2_s1['b1b2'] - b1_hyb2:.4f}")

# Try: what b1b2 value gives beta_1 = 0.0713?
# Since beta_2 = b1b2 - beta_1, and we want beta_2 to be reasonable...
# If beta_1 doesn't change much with b1b2 (gradient = -0.028),
# then beta_1 ≈ 0.0796 regardless of b1b2.
# To get beta_2 = 0.0545: need b1b2 = 0.0796 + 0.0545 = 0.1341
# That's different from paper's 0.1258.

# What if we don't subtract the higher-order experience terms at all?
zero_exp_s1 = base_s1.copy()
zero_exp_s1['d2'] = 0
zero_exp_s1['d3'] = 0
zero_exp_s1['d4'] = 0
b1_zero, _ = run_iv(df, zero_exp_s1, ctrl, yr_dums.copy())
print(f"\nNo exp polynomial subtraction: beta_1 = {b1_zero:.4f}, beta_2 = {0.1258 - b1_zero:.4f}")

# What if we don't subtract ANY polynomial terms (only b1b2*T)?
minimal_s1 = {'b1b2': 0.1258, 'g2': 0, 'g3': 0, 'g4': 0, 'd2': 0, 'd3': 0, 'd4': 0}
b1_min, _ = run_iv(df, minimal_s1, ctrl, yr_dums.copy())
print(f"Only b1b2*T subtracted: beta_1 = {b1_min:.4f}, beta_2 = {0.1258 - b1_min:.4f}")
