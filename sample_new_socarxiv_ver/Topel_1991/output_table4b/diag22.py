"""
Diagnostic 22: Test step 1 with no d_exp filter (per-year education so d_exp varies).
Also test with per-year education since Table 2 best used that approach's tenure reconstruction.

The idea: the paper uses per-year education, so d_exp varies and is not filtered.
The tenure polynomial may be more stable with more observations.
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

# Paper's implied gamma terms from cumulative returns
PAPER_GAMMA = {
    0: {'g2': -0.0045904946, 'g3': 0.0001844867, 'g4': -0.0000024512},
    1: {'g2': -0.0049426093, 'g3': 0.0001970956, 'g4': -0.0000025326},
    3: {'g2': -0.0054983747, 'g3': 0.0002423803, 'g4': -0.0000033406},
    5: {'g2': -0.0054908435, 'g3': 0.0002293794, 'g4': -0.0000031256},
}

GROUND_TRUTH = {
    'beta_1': {0: 0.0713, 1: 0.0792, 3: 0.0716, 5: 0.0607},
    'beta_2': {0: 0.0545, 1: 0.0546, 3: 0.0559, 5: 0.0584},
    'cumret': {
        0: {5: 0.1793, 10: 0.2459, 15: 0.2832, 20: 0.3375},
        1: {5: 0.1725, 10: 0.2235, 15: 0.2439, 20: 0.2865},
        3: {5: 0.1703, 10: 0.2181, 15: 0.2503, 20: 0.3232},
        5: {5: 0.1815, 10: 0.2330, 15: 0.2565, 20: 0.3066},
    }
}


def prepare_data(data_source, pnum_filter=None, fix_educ=True):
    df = pd.read_csv(data_source)
    if pnum_filter is not None:
        df['pnum'] = df['person_id'] % 1000
        df = df[df['pnum'].isin(pnum_filter)].copy()

    df['education_years'] = df['education_clean'].copy()
    cat_mask = ~df['year'].isin([1975, 1976])
    df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(
        {**EDUC_MAP, 9: np.nan}
    )

    if fix_educ:
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
    else:
        df = df[df['education_years'].notna()].copy()
        df['education_fixed'] = df['education_years']
        df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)

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


def run_step1(df, min_remaining_dur=0, wage_var='nominal', dexp_filter=True):
    df = df.copy()
    grp = df.groupby(['person_id', 'job_id'])
    df['prev_year'] = grp['year'].shift(1)
    df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
    df['prev_log_wage_cps'] = grp['log_wage_cps'].shift(1)
    df['prev_tenure'] = grp['tenure'].shift(1)
    df['prev_experience'] = grp['experience'].shift(1)

    within = df[(df['prev_year'].notna()) & (df['year'] - df['prev_year'] == 1)].copy()

    if wage_var == 'nominal':
        within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']
    else:
        within['d_log_wage'] = within['log_wage_cps'] - within['prev_log_wage_cps']

    if dexp_filter:
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
        'b1b2': m.params[idx['d_tenure']], 'b1b2_se': m.bse[idx['d_tenure']],
        'g2': m.params[idx['d_tenure_sq']], 'g3': m.params[idx['d_tenure_cu']],
        'g4': m.params[idx['d_tenure_qu']],
        'd2': m.params[idx['d_exp_sq']], 'd3': m.params[idx['d_exp_cu']],
        'd4': m.params[idx['d_exp_qu']],
        'N': int(m.nobs),
    }


# ======================================================================
# Test all combinations
# ======================================================================
configs = [
    # pnum_filter, fix_educ, dexp_filter, label
    ([1, 170], True, True, "HoH, fixEd, dexp=1"),
    ([1, 170], True, False, "HoH, fixEd, no dexp"),
    ([1, 170], False, True, "HoH, perYrEd, dexp=1"),
    ([1, 170], False, False, "HoH, perYrEd, no dexp"),
    ([1, 3, 170, 171], True, True, "HoH+sons, fixEd, dexp=1"),
    ([1, 3, 170, 171], True, False, "HoH+sons, fixEd, no dexp"),
    (None, True, True, "All, fixEd, dexp=1"),
    (None, True, False, "All, fixEd, no dexp"),
]

for pfilter, fix_ed, dexp, label in configs:
    df = prepare_data("data/psid_panel.csv", pnum_filter=pfilter, fix_educ=fix_ed)

    print(f"\n{'='*80}")
    print(f"CONFIG: {label} (N_levels={len(df)})")
    print(f"{'='*80}")
    print(f"{'':>5} {'N':>6} {'b1+b2':>8} {'SE':>8} {'g2*100':>8} {'g3*1k':>8} {'g4*10k':>8}")

    paper_b1b2 = {0: 0.1258, 1: 0.1338, 3: 0.1275, 5: 0.1191}

    for thresh in [0, 1, 3, 5]:
        s1 = run_step1(df, min_remaining_dur=thresh, dexp_filter=dexp)
        b_err = abs(s1['b1b2'] - paper_b1b2[thresh])
        # Also check how close gamma terms are to paper's implied
        pg = PAPER_GAMMA[thresh]
        g2_err = abs(s1['g2'] - pg['g2']) / abs(pg['g2'])

        stable = "OK" if g2_err < 2 else "UNSTABLE"
        print(f"  >={thresh}: {s1['N']:>6} {s1['b1b2']:>8.4f} {s1['b1b2_se']:>8.4f} {s1['g2']*100:>8.4f} {s1['g3']*1000:>8.4f} {s1['g4']*10000:>8.4f} b1b2_err={b_err:.4f} g2_err={g2_err:.1%} {stable}")

    print(f"  Paper gamma reference: {'':>28} {-0.4592:>8.4f} {0.1846:>8.4f} {-0.0245:>8.4f}")
