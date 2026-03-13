"""Test IV estimation with different samples"""
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
    df['cps_idx'] = df['year'].map(CPS_WAGE_INDEX)
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

    # Year dummies
    for yr in range(1971, 1984):
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
        all_ctrl_temp = ctrl + active_yr
        exog_df = sm.add_constant(levels[all_ctrl_temp])
        rank = np.linalg.matrix_rank(exog_df.values)
    all_ctrl = ctrl + active_yr

    dep = levels['w_star']
    exog = sm.add_constant(levels[all_ctrl])
    endog = levels[['experience']]
    instruments = levels[['init_exp']]

    iv = IV2SLS(dep, exog, endog, instruments).fit(cov_type='unadjusted')
    return iv.params['experience'], iv.std_errors['experience'], len(levels)


# Test different samples
samples = [
    ("data/psid_panel.csv", [1, 170], "main heads [1,170]"),
    ("data/psid_panel.csv", [1, 3, 170, 171], "main heads+sons [1,3,170,171]"),
    ("data/psid_panel_full.csv", [1, 170, 3], "full [1,170,3]"),
    ("data/psid_panel_full.csv", [1, 170], "full heads [1,170]"),
    ("data/psid_panel_full.csv", [1, 3, 4, 5], "full [1,3,4,5]"),
    ("data/psid_panel_full.csv", [1, 3, 4, 5, 6], "full [1,3,4,5,6]"),
]

ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']

print(f"{'Sample':>35} {'N':>7} {'beta_1':>8} {'beta_2':>8} {'SE':>8}")
print("-" * 75)

for fname, pf, label in samples:
    try:
        df = prepare_data(fname, pf)
        yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
        yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
        yr_dums = yr_dums[1:]  # drop first for identification

        b1, se, n = run_iv(df, PAPER_S1, ctrl, yr_dums)
        b2 = PAPER_S1['b1b2'] - b1
        print(f"{label:>35} {n:>7} {b1:>8.4f} {b2:>8.4f} {se:>8.4f}")
    except Exception as e:
        print(f"{label:>35} ERROR: {e}")

print(f"\n{'Paper':>35} {'10685':>7} {'0.0713':>8} {'0.0545':>8} {'0.0181':>8}")

# Also try with region_west included
print("\n--- With region_west ---")
ctrl2 = ['education_years', 'married', 'union_member', 'disabled',
         'region_ne', 'region_nc', 'region_south', 'region_west']

for fname, pf, label in [("data/psid_panel_full.csv", [1, 170, 3], "full [1,170,3]"),
                          ("data/psid_panel.csv", [1, 3, 170, 171], "main heads+sons")]:
    try:
        df = prepare_data(fname, pf)
        yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
        yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
        yr_dums = yr_dums[1:]

        b1, se, n = run_iv(df, PAPER_S1, ctrl2, yr_dums)
        b2 = PAPER_S1['b1b2'] - b1
        print(f"{label:>35} {n:>7} {b1:>8.4f} {b2:>8.4f} {se:>8.4f}")
    except Exception as e:
        print(f"{label:>35} ERROR: {e}")

# Try with clustered SE
print("\n--- Clustered SE ---")
for fname, pf, label in [("data/psid_panel_full.csv", [1, 170, 3], "full [1,170,3]"),
                          ("data/psid_panel.csv", [1, 3, 170, 171], "main heads+sons")]:
    try:
        df = prepare_data(fname, pf)
        yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
        yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
        yr_dums = yr_dums[1:]

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

        exog = sm.add_constant(levels[all_ctrl])
        dep = levels['w_star']
        endog = levels[['experience']]
        instruments = levels[['init_exp']]

        for cov_type in ['unadjusted', 'robust', 'clustered']:
            if cov_type == 'clustered':
                iv = IV2SLS(dep, exog, endog, instruments).fit(
                    cov_type='clustered', clusters=levels['person_id'])
            else:
                iv = IV2SLS(dep, exog, endog, instruments).fit(cov_type=cov_type)
            b1 = iv.params['experience']
            se = iv.std_errors['experience']
            print(f"  {label} {cov_type}: b1={b1:.4f}, se={se:.4f}")
    except Exception as e:
        print(f"  {label} ERROR: {e}")
