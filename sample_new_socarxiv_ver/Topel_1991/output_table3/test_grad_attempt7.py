"""Test: try to reproduce attempt 7's gradient of 0.512"""
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

CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
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

    df['cps_idx'] = df['year'].map(CPS_WAGE_INDEX)
    df['gnp_def'] = df['year'].map(lambda y: GNP_DEFLATOR.get(y - 1, np.nan))
    df['log_real_gnp'] = df['log_hourly_wage'] - np.log(df['gnp_def'] / 100.0)
    df['log_real_cps'] = df['log_hourly_wage'] - np.log(df['cps_idx'])
    df['log_real_both'] = df['log_hourly_wage'] - np.log(df['gnp_def'] / 100.0) - np.log(df['cps_idx'])

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

    return df


def run_iv(levels, step1, ctrl, yr_dums_use, wage_var='log_real_gnp'):
    b1b2 = step1['b1b2']
    g2, g3, g4 = step1['g2'], step1['g3'], step1['g4']
    d2, d3, d4 = step1['d2'], step1['d3'], step1['d4']

    levels = levels.copy()
    levels['w_star'] = (levels[wage_var]
                        - b1b2 * levels['tenure']
                        - g2 * levels['tenure_2']
                        - g3 * levels['tenure_3']
                        - g4 * levels['tenure_4']
                        - d2 * levels['exp_2']
                        - d3 * levels['exp_3']
                        - d4 * levels['exp_4'])
    levels['init_exp'] = (levels['experience'] - levels['tenure']).clip(lower=0)

    all_ctrl = ctrl + yr_dums_use
    levels = levels.dropna(subset=['w_star', 'experience', 'init_exp'] + all_ctrl).copy()

    exog_df = sm.add_constant(levels[all_ctrl])
    rank = np.linalg.matrix_rank(exog_df.values)
    active_yr = yr_dums_use.copy()
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
    return iv.params['experience'], iv.std_errors['experience'], len(levels), levels, all_ctrl


# Reproduce attempt 7's setup:
# Best spec was "heads+sons_GNP" which used pnum_filter=[1,3,170,171]
# and wage_var='log_real_gnp'

df = prepare_data("data/psid_panel.csv", pnum_filter=[1, 3, 170, 171])
ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
yr_dums = yr_dums[1:]

wage_var = 'log_real_gnp'

# Main beta_1
beta_1, se, N, levels, ctrl_used = run_iv(df, PAPER_S1, ctrl, yr_dums.copy(), wage_var)
print(f"Main beta_1 = {beta_1:.6f}, N = {N}")

# Attempt 7's gradient computation:
# It used df.dropna(subset=[wage_var]).copy() for the gradient, NOT levels
delta = 0.001
step1_plus = PAPER_S1.copy()
step1_plus['b1b2'] = PAPER_S1['b1b2'] + delta

# Path 1: use df (the full prepared data, as attempt 7 did)
b1_plus_df, _, _, _, _ = run_iv(
    df.dropna(subset=[wage_var]).copy(), step1_plus, ctrl, yr_dums.copy(), wage_var)
grad_df = (b1_plus_df - beta_1) / delta
print(f"\nGradient using df: {grad_df:.4f}")

# Path 2: use levels (the filtered data after IV)
b1_plus_levels, _, _, _, _ = run_iv(
    levels.copy(), step1_plus, ctrl, ctrl_used[len(ctrl):], wage_var)
grad_levels = (b1_plus_levels - beta_1) / delta
print(f"Gradient using levels: {grad_levels:.4f}")

# The question is: does using df vs levels give different gradients?
# In attempt 7, the main beta_1 was from the "best specification" search,
# which may have been computed on a DIFFERENT data subset
# (e.g., heads_GNP, heads_CPS, etc.) than what the gradient was computed on.

# Let me replicate the exact search from attempt 7
for pfilter, plabel in [([1, 170], 'heads'), ([1, 3, 170, 171], 'heads+sons')]:
    df2 = prepare_data("data/psid_panel.csv", pnum_filter=pfilter)
    ctrl2 = ['education_years', 'married', 'union_member', 'disabled',
             'region_ne', 'region_nc', 'region_south']
    yr2 = sorted([c for c in df2.columns if c.startswith('year_') and c != 'year'])
    yr2 = [c for c in yr2 if df2[c].std() > 1e-10]
    yr2 = yr2[1:]

    for wv, wlabel in [('log_real_gnp', 'GNP'), ('log_real_cps', 'CPS'), ('log_real_both', 'Both')]:
        levels2 = df2.dropna(subset=[wv]).copy()
        try:
            b1, _, n, _, _ = run_iv(levels2, PAPER_S1, ctrl2, yr2.copy(), wv)
            b2 = PAPER_S1['b1b2'] - b1
            print(f"  {plabel}_{wlabel}: N={n}, b1={b1:.4f}")
        except Exception as e:
            print(f"  {plabel}_{wlabel}: FAILED")

# The best specification was heads+sons_GNP
# Then the gradient was computed using:
# df = prepare_data(..., pnum_filter=[1, 3, 170, 171])  # full heads+sons
# b1_plus, _, _, _, _ = run_iv(
#     df.dropna(subset=[wage_var]).copy(), step1_plus, ctrl_base, yr_dums_base.copy(), wage_var)
# grad_b1b2 = (b1_plus - beta_1) / delta
# where beta_1 was the result from the search loop's best spec

# The issue: in the search loop, beta_1 was computed from levels=df2.dropna(subset=[wv])
# But for the gradient, it used df = prepare_data(pnum_filter=[1, 3, 170, 171])
# These should be the same data... unless there's a subtle difference.

# Let me check attempt 7's exact gradient code path:
# Line 206-207: df = prepare_data(data_source, pnum_filter=[1, 170] if 'heads' in best_key else [1, 3, 170, 171]...)
# Line 207: ctrl_base, yr_dums_base = get_controls_and_year_dummies(df)
# ...
# Line 252: b1_plus, _, _, _, _ = run_iv(
#     df.dropna(subset=[wage_var]).copy(), step1_plus, ctrl_base, yr_dums_base.copy(), wage_var)
# Line 254: grad_b1b2 = (b1_plus - beta_1) / delta

# beta_1 was from line 163-165: the results dict was built from run_iv() on different data paths
# r = results[best_key]
# beta_1 = r['beta_1']

# So beta_1 comes from ONE call to run_iv, and the gradient uses ANOTHER call.
# If these were truly on the same data, gradient should be -0.028.
# Getting 0.512 would require a fundamentally different data path.

# WAIT - I just realized: in attempt 7, after the search loop, the code does:
# Line 206: df = prepare_data(data_source, pnum_filter=...)
# This is a NEW call to prepare_data. If 'sons' in best_key, it uses [1,3,170,171].
# But beta_1 was computed in the search loop which also used prepare_data.
# These should give the same data...

# Unless the run_iv for the gradient drops different year dummies due to rank issues.
# Let me check what year dummies survive
print(f"\nYear dummies surviving: {ctrl_used[len(ctrl):]}")
print(f"Full yr_dums: {yr_dums}")

# The gradient 0.512 in attempt 7 is suspicious. It may have been due to
# randomness in which year dummies were dropped. Let me check a few deltas.
print("\n\nGradient exploration:")
for d in [0.0001, 0.001, 0.01, 0.1]:
    sp = PAPER_S1.copy()
    sp['b1b2'] = PAPER_S1['b1b2'] + d
    b1p, _, _, _, _ = run_iv(df.dropna(subset=[wage_var]).copy(), sp, ctrl, yr_dums.copy(), wage_var)
    g = (b1p - beta_1) / d
    print(f"  delta={d}: b1_plus={b1p:.6f}, grad={(b1p-beta_1)/d:.6f}")
