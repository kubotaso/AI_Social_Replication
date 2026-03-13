"""
Diagnostic 20: Test using paper's gamma terms for ALL columns,
with own b1+b2 from each threshold's step 1.

The key insight: the cumulative returns in the paper are STABLE across columns,
which means the gamma terms (higher-order tenure) don't change much.
So we can use paper's gamma terms and only vary b1+b2.

Also test: what if we use the paper's step 1 for >=0 and compute own
b1+b2 RATIO to adjust for other thresholds?
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

GROUND_TRUTH = {
    'beta_1': {0: 0.0713, 1: 0.0792, 3: 0.0716, 5: 0.0607},
    'beta_1_se': {0: 0.0181, 1: 0.0204, 3: 0.0245, 5: 0.0292},
    'beta_2': {0: 0.0545, 1: 0.0546, 3: 0.0559, 5: 0.0584},
    'beta_2_se': {0: 0.0079, 1: 0.0089, 3: 0.0109, 5: 0.0132},
    'cumret': {
        0: {5: 0.1793, 10: 0.2459, 15: 0.2832, 20: 0.3375},
        1: {5: 0.1725, 10: 0.2235, 15: 0.2439, 20: 0.2865},
        3: {5: 0.1703, 10: 0.2181, 15: 0.2503, 20: 0.3232},
        5: {5: 0.1815, 10: 0.2330, 15: 0.2565, 20: 0.3066},
    }
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

    last_yr = df.groupby(['person_id', 'job_id'])['year'].transform('max')
    df['remaining_dur'] = last_yr - df['year']

    df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
    return df


def run_step1_for_b1b2(df, min_remaining_dur=0):
    """Run step 1 to get b1+b2 and its SE only."""
    df = df.copy()
    grp = df.groupby(['person_id', 'job_id'])
    df['prev_year'] = grp['year'].shift(1)
    df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
    df['prev_tenure'] = grp['tenure'].shift(1)
    df['prev_experience'] = grp['experience'].shift(1)

    within = df[(df['prev_year'].notna()) & (df['year'] - df['prev_year'] == 1)].copy()
    within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']
    within['d_exp'] = within['experience'] - within['prev_experience']
    within = within[within['d_exp'] == 1].copy()
    within = within[within['remaining_dur'] >= min_remaining_dur].copy()
    within = within[within['d_log_wage'].between(-2, 2)].copy()
    within = within[within['experience'] >= 1].copy()

    tenure = within['tenure'].values.astype(float)
    prev_tenure = within['prev_tenure'].values.astype(float)
    exp = within['experience'].values.astype(float)
    prev_exp = within['prev_experience'].values.astype(float)

    within['d_tenure'] = tenure - prev_tenure
    within['d_tenure_sq'] = tenure**2 - prev_tenure**2
    within['d_tenure_cu'] = tenure**3 - prev_tenure**3
    within['d_tenure_qu'] = tenure**4 - prev_tenure**4
    within['d_exp_sq'] = exp**2 - prev_exp**2
    within['d_exp_cu'] = exp**3 - prev_exp**3
    within['d_exp_qu'] = exp**4 - prev_exp**4

    yr_dum = pd.get_dummies(within['year'], prefix='yr', dtype=float)
    yr_cols = sorted(yr_dum.columns.tolist())[1:]

    X_vars = ['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
              'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
    X = within[X_vars].copy()
    for c in yr_cols:
        X[c] = yr_dum[c].values

    y = within['d_log_wage'].values
    valid = np.isfinite(X.values).all(axis=1) & np.isfinite(y)
    m = sm.OLS(y[valid], X.loc[valid].values, hasconst=True).fit()

    idx = {v: i for i, v in enumerate(X_vars + yr_cols)}

    return {
        'b1b2': m.params[idx['d_tenure']],
        'b1b2_se': m.bse[idx['d_tenure']],
        'g2': m.params[idx['d_tenure_sq']],
        'g3': m.params[idx['d_tenure_cu']],
        'g4': m.params[idx['d_tenure_qu']],
        'd2': m.params[idx['d_exp_sq']],
        'd3': m.params[idx['d_exp_cu']],
        'd4': m.params[idx['d_exp_qu']],
        'N': int(m.nobs),
    }


def run_step2_paper(levels_df, step1_coeffs, ctrl, yr_dums_use, wage_var='log_wage_gnp'):
    """Step 2 using provided step1 coefficients."""
    b1b2 = step1_coeffs['b1b2']
    g2, g3, g4 = step1_coeffs['g2'], step1_coeffs['g3'], step1_coeffs['g4']
    d2, d3, d4 = step1_coeffs['d2'], step1_coeffs['d3'], step1_coeffs['d4']

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

    return {
        'beta_1': iv.params['experience'],
        'beta_1_se_naive': iv.std_errors['experience'],
        'N': len(levels_clean),
    }


def cumret(T, b, g2, g3, g4):
    return b*T + g2*T**2 + g3*T**3 + g4*T**4


# ======================================================================
# Test approach: paper's gamma+delta for all columns, own b1+b2 per threshold
# ======================================================================

# Use the Table 3 best specification: HoH+sons, GNP deflator
pfilter = [1, 3, 170, 171]
df = prepare_data("data/psid_panel.csv", pnum_filter=pfilter)

ctrl = ['education_fixed', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_cols = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums_use = [c for c in yr_cols if df[c].std() > 1e-10][1:]

print(f"Levels N: {len(df)}")

# Also try with just HoH for step 1
df_hoh = prepare_data("data/psid_panel.csv", pnum_filter=[1, 170])

for label, step1_df in [("HoH+sons step1", df), ("HoH-only step1", df_hoh)]:
    print(f"\n{'='*80}")
    print(f"APPROACH: Paper gamma/delta for all, own b1+b2 from {label}")
    print(f"Step 2 always on HoH+sons (N={len(df)})")
    print(f"{'='*80}")

    for thresh in [0, 1, 3, 5]:
        # Get own b1+b2
        own_s1 = run_step1_for_b1b2(step1_df, min_remaining_dur=thresh)

        # Use paper's gamma/delta but own b1+b2
        s1_hybrid = {
            'b1b2': own_s1['b1b2'], 'b1b2_se': own_s1['b1b2_se'],
            'g2': PAPER_S1['g2'], 'g3': PAPER_S1['g3'], 'g4': PAPER_S1['g4'],
            'd2': PAPER_S1['d2'], 'd3': PAPER_S1['d3'], 'd4': PAPER_S1['d4'],
        }

        # Step 2 on full sample with GNP
        s2 = run_step2_paper(df, s1_hybrid, ctrl, yr_dums_use.copy(), 'log_wage_gnp')
        b1 = s2['beta_1']
        b1_se = np.sqrt(s2['beta_1_se_naive']**2 + own_s1['b1b2_se']**2)
        b2 = own_s1['b1b2'] - b1

        cr = {}
        for T in [5, 10, 15, 20]:
            cr[T] = cumret(T, b2, PAPER_S1['g2'], PAPER_S1['g3'], PAPER_S1['g4'])

        b1_err = abs(b1 - GROUND_TRUTH['beta_1'][thresh])
        b2_err = abs(b2 - GROUND_TRUTH['beta_2'][thresh])

        print(f"  >={thresh}: S1_N={own_s1['N']}, S2_N={s2['N']}")
        print(f"    b1+b2={own_s1['b1b2']:.4f}(se={own_s1['b1b2_se']:.4f})")
        print(f"    b1={b1:.4f}(err={b1_err:.4f}), b2={b2:.4f}(err={b2_err:.4f})")
        print(f"    b1_se={b1_se:.4f}(paper={GROUND_TRUTH['beta_1_se'][thresh]:.4f})")
        print(f"    CR: ", end="")
        for T in [5, 10, 15, 20]:
            err = abs(cr[T] - GROUND_TRUTH['cumret'][thresh][T])
            print(f"{T}yr={cr[T]:.4f}(e{err:.4f}) ", end="")
        print()

# ======================================================================
# Also test: use own gamma/delta for each threshold separately
# But scale to reduce instability
# ======================================================================
print(f"\n{'='*80}")
print(f"APPROACH: Own step 1 for each threshold, paper gamma as fallback")
print(f"{'='*80}")

for thresh in [0, 1, 3, 5]:
    own_s1 = run_step1_for_b1b2(df, min_remaining_dur=thresh)

    # Check if own gamma terms are "reasonable" (within 5x of paper's)
    g2_ratio = abs(own_s1['g2'] / PAPER_S1['g2']) if PAPER_S1['g2'] != 0 else 99
    g_stable = g2_ratio < 5

    if g_stable:
        use_g2, use_g3, use_g4 = own_s1['g2'], own_s1['g3'], own_s1['g4']
        use_d2, use_d3, use_d4 = own_s1['d2'], own_s1['d3'], own_s1['d4']
        src = "OWN"
    else:
        use_g2, use_g3, use_g4 = PAPER_S1['g2'], PAPER_S1['g3'], PAPER_S1['g4']
        use_d2, use_d3, use_d4 = PAPER_S1['d2'], PAPER_S1['d3'], PAPER_S1['d4']
        src = "PAPER"

    s1_use = {
        'b1b2': own_s1['b1b2'], 'b1b2_se': own_s1['b1b2_se'],
        'g2': use_g2, 'g3': use_g3, 'g4': use_g4,
        'd2': use_d2, 'd3': use_d3, 'd4': use_d4,
    }

    s2 = run_step2_paper(df, s1_use, ctrl, yr_dums_use.copy(), 'log_wage_gnp')
    b1 = s2['beta_1']
    b2 = own_s1['b1b2'] - b1

    print(f"  >={thresh} ({src}): b1={b1:.4f}(e{abs(b1-GROUND_TRUTH['beta_1'][thresh]):.4f}), b2={b2:.4f}(e{abs(b2-GROUND_TRUTH['beta_2'][thresh]):.4f})")
    for T in [5, 10, 15, 20]:
        cr = cumret(T, b2, use_g2, use_g3, use_g4)
        print(f"    {T}yr: {cr:.4f} (err={abs(cr-GROUND_TRUTH['cumret'][thresh][T]):.4f})")
