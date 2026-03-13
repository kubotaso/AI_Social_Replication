"""
Diagnostic 25: Test OLS on X_0 (initial experience) instead of IV.
Paper's equation (10): y - x'gamma = beta_1 * X_0 + F'delta + e
This suggests OLS on X_0 directly.

Also explore: what if w* is computed differently?
The paper's step 2 uses: w*_i = log_wage_i - (b1+b2)*T_i - g2*T^2 - g3*T^3 - g4*T^4
                                               - d2*X^2 - d3*X^3 - d4*X^4
Then regress w* on X_0, controls, year dummies using OLS.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm

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

PAPER_S1_BY_THRESH = {
    0: {'b1b2': 0.1258,
        'g2': -0.0045905, 'g3': 0.0001845, 'g4': -0.0000024512,
        'd2': -0.006051, 'd3': 0.0002067, 'd4': -0.00000238},
    1: {'b1b2': 0.1338,
        'g2': -0.0049426, 'g3': 0.0001971, 'g4': -0.0000025326,
        'd2': -0.006051, 'd3': 0.0002067, 'd4': -0.00000238},
    3: {'b1b2': 0.1275,
        'g2': -0.0054984, 'g3': 0.0002424, 'g4': -0.0000033406,
        'd2': -0.006051, 'd3': 0.0002067, 'd4': -0.00000238},
    5: {'b1b2': 0.1191,
        'g2': -0.0054908, 'g3': 0.0002294, 'g4': -0.0000031256,
        'd2': -0.006051, 'd3': 0.0002067, 'd4': -0.00000238},
}

GROUND_TRUTH = {
    'beta_1': {0: 0.0713, 1: 0.0792, 3: 0.0716, 5: 0.0607},
    'beta_2': {0: 0.0545, 1: 0.0546, 3: 0.0559, 5: 0.0584},
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
    df = df.dropna(subset=['log_wage_cps', 'log_wage_gnp', 'experience', 'tenure']).copy()

    df['init_exp'] = (df['experience'] - df['tenure']).clip(lower=0)
    df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
    return df


def run_ols_x0(levels_df, step1, ctrl, yr_dums_use, wage_var='log_wage_gnp'):
    """OLS on X_0 (initial experience)."""
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
    levels_clean = levels.dropna(subset=['w_star', 'init_exp'] + all_ctrl).copy()

    # OLS: w* = const + beta_1 * X_0 + controls + year_dummies + e
    X_ols = sm.add_constant(levels_clean[['init_exp'] + all_ctrl])

    # Fix rank
    rank = np.linalg.matrix_rank(X_ols.values)
    current_yr = yr_dums_use.copy()
    while rank < X_ols.shape[1] and current_yr:
        current_yr.pop()
        X_ols = sm.add_constant(levels_clean[['init_exp'] + ctrl + current_yr])
        rank = np.linalg.matrix_rank(X_ols.values)

    m = sm.OLS(levels_clean['w_star'], X_ols).fit()

    return m.params['init_exp'], m.bse['init_exp'], len(levels_clean)


# Test OLS on X_0 with paper's step 1 for each threshold
for pfilter_label, pfilter in [("HoH", [1, 170]), ("HoH+sons", [1, 3, 170, 171])]:
    df = prepare_data("data/psid_panel.csv", pnum_filter=pfilter)
    ctrl = ['education_fixed', 'married', 'union_member', 'disabled',
            'region_ne', 'region_nc', 'region_south']
    yr_cols = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
    yr_dums_use = [c for c in yr_cols if df[c].std() > 1e-10][1:]

    for wage_var, wlabel in [('log_wage_gnp', 'GNP'), ('log_wage_cps', 'CPS')]:
        print(f"\n{'='*80}")
        print(f"OLS on X_0: {pfilter_label}, {wlabel} (N={len(df)})")
        print(f"{'='*80}")

        for thresh in [0, 1, 3, 5]:
            s1 = PAPER_S1_BY_THRESH[thresh]
            b1, se, N = run_ols_x0(df, s1, ctrl, yr_dums_use.copy(), wage_var)
            b2 = s1['b1b2'] - b1

            b1_err = abs(b1 - GROUND_TRUTH['beta_1'][thresh])
            b2_err = abs(b2 - GROUND_TRUTH['beta_2'][thresh])
            print(f"  >={thresh}: N={N}, b1={b1:.4f}(e{b1_err:.4f}), b2={b2:.4f}(e{b2_err:.4f}), se={se:.6f}")

        # Also test gradient
        print(f"\n  Gradient test (OLS X_0):")
        for delta in [0.01]:
            s1_plus = PAPER_S1.copy()
            s1_plus['b1b2'] = PAPER_S1['b1b2'] + delta
            b1_plus, _, _ = run_ols_x0(df, s1_plus, ctrl, yr_dums_use.copy(), wage_var)
            s1_minus = PAPER_S1.copy()
            s1_minus['b1b2'] = PAPER_S1['b1b2'] - delta
            b1_minus, _, _ = run_ols_x0(df, s1_minus, ctrl, yr_dums_use.copy(), wage_var)
            b1_base, _, _ = run_ols_x0(df, PAPER_S1, ctrl, yr_dums_use.copy(), wage_var)
            grad = (b1_plus - b1_minus) / (2 * delta)
            print(f"    delta={delta}: grad={grad:.6f}")
