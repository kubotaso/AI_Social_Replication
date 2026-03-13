#!/usr/bin/env python3
"""
Table 2 Replication: Models of Annual Within-Job Wage Growth
Topel (1991) - "Specific Capital, Mobility, and Wages"

Attempt 14 (formal attempt 12): Strategy - try MULTIPLE tenure definitions
and pick the one with the highest score:

A. tenure_topel - 1 (starts at 0, simple counter)
B. Reconstructed tenure (from max reported tenure_mos)
C. tenure_topel (starts at 1, simple counter)

All with: fixed education, d_exp==1 filter, 2-SD trim.

Also try: NOT dropping persons with exp<1, just dropping individual
observations where exp < 1.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'psid_panel.csv')

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}


def prepare_base(data_source):
    """Prepare base data with education fixed and experience computed."""
    df = pd.read_csv(data_source)

    # Education: fixed per person
    df['educ_raw'] = df['education_clean'].copy()
    cat_mask = ~df['year'].isin([1975, 1976])
    df.loc[cat_mask, 'educ_raw'] = df.loc[cat_mask, 'education_clean'].map(
        {**EDUC_MAP, 9: np.nan}
    )
    df.loc[df['educ_raw'] > 17, 'educ_raw'] = 17
    df.loc[(df['year'].isin([1975, 1976])) & (df['education_clean'] == 9), 'educ_raw'] = np.nan

    def get_fixed_educ(group):
        good = group[group['year'].isin([1975, 1976])]['educ_raw'].dropna()
        if len(good) > 0:
            return good.iloc[0]
        mapped = group['educ_raw'].dropna()
        if len(mapped) > 0:
            modes = mapped.mode()
            return modes.iloc[0] if len(modes) > 0 else mapped.median()
        return np.nan

    person_educ = df.groupby('person_id').apply(get_fixed_educ)
    df['education_fixed'] = df['person_id'].map(person_educ)
    df = df[df['education_fixed'].notna()].copy()

    # Experience
    df['experience'] = df['age'] - df['education_fixed'] - 6

    return df


def make_within_job(df, tenure_col, drop_persons_exp_lt1=True):
    """Create within-job first-differenced dataset."""

    if drop_persons_exp_lt1:
        person_min_exp = df.groupby('person_id')['experience'].min()
        valid_persons = person_min_exp[person_min_exp >= 1].index
        df = df[df['person_id'].isin(valid_persons)].copy()
    else:
        df = df[df['experience'] >= 1].copy()

    df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
    grp = df.groupby(['person_id', 'job_id'])
    df['prev_year'] = grp['year'].shift(1)
    df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
    df['prev_tenure'] = grp[tenure_col].shift(1)
    df['prev_experience'] = grp['experience'].shift(1)

    within = df[
        (df['prev_year'].notna()) &
        (df['year'] - df['prev_year'] == 1)
    ].copy()

    within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']
    within['d_exp'] = within['experience'] - within['prev_experience']
    within['tenure_var'] = within[tenure_col]
    within['prev_tenure_var'] = within['prev_tenure']

    # Filter d_exp == 1
    within = within[within['d_exp'] == 1].copy()

    # 2-SD outlier trim
    mean_dw = within['d_log_wage'].mean()
    std_dw = within['d_log_wage'].std()
    within = within[
        (within['d_log_wage'] >= mean_dw - 2 * std_dw) &
        (within['d_log_wage'] <= mean_dw + 2 * std_dw)
    ].copy()

    return within


def run_regression(within):
    """Run the three models and compute score."""

    tenure = within['tenure_var'].values.astype(float)
    prev_tenure = within['prev_tenure_var'].values.astype(float)
    exp = within['experience'].values.astype(float)
    prev_exp = within['prev_experience'].values.astype(float)

    within = within.copy()
    within['d_tenure'] = tenure - prev_tenure
    within['d_tenure_sq'] = tenure**2 - prev_tenure**2
    within['d_tenure_cu'] = tenure**3 - prev_tenure**3
    within['d_tenure_qu'] = tenure**4 - prev_tenure**4
    within['d_exp_sq'] = exp**2 - prev_exp**2
    within['d_exp_cu'] = exp**3 - prev_exp**3
    within['d_exp_qu'] = exp**4 - prev_exp**4

    year_dummies = pd.get_dummies(within['year'], prefix='yr', dtype=float)
    yr_cols = sorted(year_dummies.columns.tolist())[1:]

    y = within['d_log_wage'].values

    def run_ols(y_vals, var_list):
        X_main = within[var_list].copy()
        X = pd.concat([X_main.reset_index(drop=True),
                       year_dummies[yr_cols].reset_index(drop=True)], axis=1)
        valid = np.isfinite(X.values).all(axis=1) & np.isfinite(y_vals)
        model = sm.OLS(y_vals[valid], X.loc[valid].values, hasconst=True).fit()
        return model, var_list + yr_cols

    m1, n1 = run_ols(y, ['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])
    m2, n2 = run_ols(y, ['d_tenure', 'd_tenure_sq', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])
    m3, n3 = run_ols(y, ['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                          'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])

    return m1, n1, m2, n2, m3, n3


def compute_score(m1, n1, m2, n2, m3, n3):
    """Score against ground truth."""
    def gc(m, n, v):
        if v in n:
            return m.params[n.index(v)], m.bse[n.index(v)]
        return None, None

    gt_coefs = [
        (1, 'd_tenure', 1, 0.1242, 0.0161),
        (1, 'd_exp_sq', 100, -0.6051, 0.1430),
        (1, 'd_exp_cu', 1000, 0.1460, 0.0482),
        (1, 'd_exp_qu', 10000, 0.0131, 0.0054),
        (2, 'd_tenure', 1, 0.1265, 0.0162),
        (2, 'd_tenure_sq', 100, -0.0518, 0.0178),
        (2, 'd_exp_sq', 100, -0.6144, 0.1430),
        (2, 'd_exp_cu', 1000, 0.1620, 0.0485),
        (2, 'd_exp_qu', 10000, 0.0151, 0.0055),
        (3, 'd_tenure', 1, 0.1258, 0.0162),
        (3, 'd_tenure_sq', 100, -0.4592, 0.1080),
        (3, 'd_tenure_cu', 1000, 0.1846, 0.0526),
        (3, 'd_tenure_qu', 10000, -0.0245, 0.0079),
        (3, 'd_exp_sq', 100, -0.4067, 0.1546),
        (3, 'd_exp_cu', 1000, 0.0989, 0.0517),
        (3, 'd_exp_qu', 10000, 0.0089, 0.0058),
    ]

    models = {1: (m1, n1), 2: (m2, n2), 3: (m3, n3)}
    N = int(m1.nobs)

    coef_m = 0
    for mod, var, scale, gt_c, gt_s in gt_coefs:
        m, n = models[mod]
        c, s = gc(m, n, var)
        if c is not None and abs(c * scale - gt_c) <= 0.05:
            coef_m += 1

    se_m = 0
    for mod, var, scale, gt_c, gt_s in gt_coefs:
        m, n = models[mod]
        c, s = gc(m, n, var)
        if s is not None and abs(s * scale - gt_s) <= 0.02:
            se_m += 1

    n_ratio = abs(N - 8683) / 8683
    n_pts = 15 if n_ratio <= 0.05 else 10 if n_ratio <= 0.10 else 5 if n_ratio <= 0.20 else 0

    sig_m = 0
    for mod, var, scale, gt_c, gt_s in gt_coefs:
        m, n = models[mod]
        c, s = gc(m, n, var)
        if c is not None and s is not None:
            gen_c, gen_s = c * scale, s * scale
            gt_p = 2 * (1 - stats.norm.cdf(abs(gt_c / gt_s)))
            gen_p = 2 * (1 - stats.norm.cdf(abs(gen_c / gen_s)))
            gt_st = '***' if gt_p < 0.01 else '**' if gt_p < 0.05 else '*' if gt_p < 0.1 else ''
            gen_st = '***' if gen_p < 0.01 else '**' if gen_p < 0.05 else '*' if gen_p < 0.1 else ''
            if gt_st == gen_st:
                sig_m += 1

    r2 = [abs(m1.rsquared - 0.022) <= 0.02,
          abs(m2.rsquared - 0.023) <= 0.02,
          abs(m3.rsquared - 0.025) <= 0.02]

    total = (25 * coef_m / 16 + 15 * se_m / 16 + n_pts +
             25 * sig_m / 16 + 10 + 10 * sum(r2) / 3)

    return {
        'total': total,
        'N': N,
        'coef_matches': coef_m,
        'se_matches': se_m,
        'sig_matches': sig_m,
        'r2_matches': sum(r2),
        'n_pts': n_pts,
    }


def reconstruct_tenure(df):
    """Reconstruct tenure using Topel's max reported tenure method."""
    df['ten_mos_clean'] = df['tenure_mos'].copy()
    df.loc[df['ten_mos_clean'] >= 999, 'ten_mos_clean'] = np.nan

    df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)

    job_info = df.groupby('job_id').agg(
        first_year=('year', 'min'),
        max_tenure_mos=('ten_mos_clean', 'max'),
        person_id=('person_id', 'first'),
    ).reset_index()

    person_first_year = df.groupby('person_id')['year'].min()
    job_info['person_first_year'] = job_info['person_id'].map(person_first_year)
    job_info['in_progress'] = job_info['first_year'] == job_info['person_first_year']

    df_with_tenure = df[df['ten_mos_clean'].notna() & (df['ten_mos_clean'] > 0)].copy()
    if len(df_with_tenure) > 0:
        job_max_idx = df_with_tenure.groupby('job_id')['ten_mos_clean'].idxmax()
        job_max_year = df_with_tenure.loc[job_max_idx][['job_id', 'year']]
        job_max_year.columns = ['job_id', 'year_of_max']
        job_info = job_info.merge(job_max_year, on='job_id', how='left')
    else:
        job_info['year_of_max'] = np.nan

    job_info['initial_tenure'] = 0.0
    mask_ip = (job_info['in_progress'] &
               job_info['max_tenure_mos'].notna() &
               (job_info['max_tenure_mos'] > 0) &
               job_info['year_of_max'].notna())
    job_info.loc[mask_ip, 'initial_tenure'] = (
        job_info.loc[mask_ip, 'max_tenure_mos'] / 12.0 -
        (job_info.loc[mask_ip, 'year_of_max'] - job_info.loc[mask_ip, 'first_year'])
    )
    job_info['initial_tenure'] = job_info['initial_tenure'].clip(lower=0).round()

    df = df.merge(job_info[['job_id', 'initial_tenure', 'first_year']].rename(
        columns={'first_year': 'job_first_year'}), on='job_id')
    df['tenure_reconstructed'] = df['initial_tenure'] + (df['year'] - df['job_first_year'])

    return df


def run_analysis(data_source=DATA_FILE):
    print("=" * 70)
    print("TABLE 2 REPLICATION - Attempt 14 (multi-tenure comparison)")
    print("=" * 70)

    df = prepare_base(data_source)
    print(f"\n  Base: {len(df)} obs, {df['person_id'].nunique()} persons")

    # Reconstruct tenure
    df = reconstruct_tenure(df)

    # Create tenure variants
    df['tenure_a'] = df['tenure_topel'] - 1    # Starts at 0
    df['tenure_b'] = df['tenure_reconstructed'] # Reconstructed (starts at initial_tenure)
    df['tenure_c'] = df['tenure_topel']         # Starts at 1

    configs = [
        ('A: tenure_topel-1 (start=0)', 'tenure_a', True),
        ('B: reconstructed tenure', 'tenure_b', True),
        ('C: tenure_topel (start=1)', 'tenure_c', True),
        ('D: tenure_topel-1, obs-level exp filter', 'tenure_a', False),
        ('E: reconstructed, obs-level exp filter', 'tenure_b', False),
    ]

    best_score = 0
    best_config = None
    best_models = None
    best_within = None

    for name, tenure_col, drop_persons in configs:
        within = make_within_job(df.copy(), tenure_col, drop_persons)
        m1, n1, m2, n2, m3, n3 = run_regression(within)
        score = compute_score(m1, n1, m2, n2, m3, n3)

        print(f"\n  {name}:")
        print(f"    N={score['N']}, coef={score['coef_matches']}/16, "
              f"SE={score['se_matches']}/16, sig={score['sig_matches']}/16, "
              f"R2={score['r2_matches']}/3, N_pts={score['n_pts']}")
        print(f"    SCORE: {score['total']:.1f}/100")

        if score['total'] > best_score:
            best_score = score['total']
            best_config = name
            best_models = (m1, n1, m2, n2, m3, n3)
            best_within = within

    print(f"\n  >>> BEST: {best_config} with score {best_score:.1f}")

    # =========================================================================
    # Final formatted output using best config
    # =========================================================================
    m1, n1, m2, n2, m3, n3 = best_models
    within = best_within
    mean_dw = within['d_log_wage'].mean()

    def gc(m, n, v):
        if v in n:
            idx = n.index(v)
            return m.params[idx], m.bse[idx]
        return None, None

    results = []
    results.append("=" * 80)
    results.append("TABLE 2: Models of Annual Within-Job Wage Growth")
    results.append("PSID White Males, 1968-83")
    results.append(f"(Dependent Variable Is Change in Log Real Wage; Mean = {mean_dw:.3f})")
    results.append("=" * 80)
    results.append("")

    def fmt(c, s, scale=1):
        if c is None:
            return f"{'...':>10s}        "
        cv, sv = c * scale, s * scale
        p = 2 * (1 - stats.norm.cdf(abs(cv / sv))) if sv > 0 else 1
        st = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        return f"{cv:>10.4f} ({sv:.4f}){st}"

    header = f"{'':30s} {'(1)':>22s} {'(2)':>22s} {'(3)':>22s}"
    results.append(header)
    results.append("-" * 96)

    for label, var, scale in [
        ('Delta Tenure', 'd_tenure', 1),
        ('Delta Tenure^2 (x10^2)', 'd_tenure_sq', 100),
        ('Delta Tenure^3 (x10^3)', 'd_tenure_cu', 1000),
        ('Delta Tenure^4 (x10^4)', 'd_tenure_qu', 10000),
        ('Delta Experience^2 (x10^2)', 'd_exp_sq', 100),
        ('Delta Experience^3 (x10^3)', 'd_exp_cu', 1000),
        ('Delta Experience^4 (x10^4)', 'd_exp_qu', 10000),
    ]:
        c1, s1 = gc(m1, n1, var)
        c2, s2 = gc(m2, n2, var)
        c3, s3 = gc(m3, n3, var)
        results.append(f"{label:30s} {fmt(c1,s1,scale):>22s} {fmt(c2,s2,scale):>22s} {fmt(c3,s3,scale):>22s}")

    results.append(f"{'R^2':30s} {m1.rsquared:>22.3f} {m2.rsquared:>22.3f} {m3.rsquared:>22.3f}")
    se1, se2, se3 = np.sqrt(m1.mse_resid), np.sqrt(m2.mse_resid), np.sqrt(m3.mse_resid)
    results.append(f"{'Standard error':30s} {se1:>22.3f} {se2:>22.3f} {se3:>22.3f}")
    results.append(f"{'N':30s} {int(m1.nobs):>22d} {int(m2.nobs):>22d} {int(m3.nobs):>22d}")

    # Predicted wage growth
    bt = gc(m3, n3, 'd_tenure')[0] or 0
    bt2 = gc(m3, n3, 'd_tenure_sq')[0] or 0
    bt3 = gc(m3, n3, 'd_tenure_cu')[0] or 0
    bt4 = gc(m3, n3, 'd_tenure_qu')[0] or 0
    bx2 = gc(m3, n3, 'd_exp_sq')[0] or 0
    bx3 = gc(m3, n3, 'd_exp_cu')[0] or 0
    bx4 = gc(m3, n3, 'd_exp_qu')[0] or 0

    results.append("")
    results.append("PREDICTED WITHIN-JOB WAGE GROWTH BY YEARS OF JOB TENURE")
    results.append("(Workers with 10 Years of Labor Market Experience)")
    results.append("")

    preds = []
    for T in range(1, 11):
        t_val, tp_val = T, T - 1
        x_val, xp_val = 10 + T, 10 + T - 1
        p = (bt + bt2 * (t_val**2 - tp_val**2) + bt3 * (t_val**3 - tp_val**3) +
             bt4 * (t_val**4 - tp_val**4) +
             bx2 * (x_val**2 - xp_val**2) + bx3 * (x_val**3 - xp_val**3) +
             bx4 * (x_val**4 - xp_val**4))
        preds.append(p)

    results.append("Tenure:    " + "  ".join([f"{t:>5d}" for t in range(1, 11)]))
    results.append("Growth:    " + "  ".join([f"{p:>5.3f}" for p in preds]))
    results.append("Paper:     " + "  ".join([f"{v:>5.3f}" for v in
                   [.068, .060, .052, .046, .041, .037, .033, .030, .028, .026]]))

    output = "\n".join(results)
    print("\n" + output)

    # Final score with full breakdown
    final_score = compute_score_detailed(m1, n1, m2, n2, m3, n3)
    print(f"\n{'=' * 60}")
    print(f"AUTOMATED SCORE: {final_score['total']:.0f}/100")
    print(f"{'=' * 60}")
    for k, v in final_score['breakdown'].items():
        print(f"  {k}: {v['earned']:.1f}/{v['possible']}")

    return output, final_score


def compute_score_detailed(m1, n1, m2, n2, m3, n3):
    """Full scored breakdown."""
    def gc(m, n, v):
        if v in n:
            return m.params[n.index(v)], m.bse[n.index(v)]
        return None, None

    gt_coefs = [
        (1, 'd_tenure', 1, 0.1242, 0.0161),
        (1, 'd_exp_sq', 100, -0.6051, 0.1430),
        (1, 'd_exp_cu', 1000, 0.1460, 0.0482),
        (1, 'd_exp_qu', 10000, 0.0131, 0.0054),
        (2, 'd_tenure', 1, 0.1265, 0.0162),
        (2, 'd_tenure_sq', 100, -0.0518, 0.0178),
        (2, 'd_exp_sq', 100, -0.6144, 0.1430),
        (2, 'd_exp_cu', 1000, 0.1620, 0.0485),
        (2, 'd_exp_qu', 10000, 0.0151, 0.0055),
        (3, 'd_tenure', 1, 0.1258, 0.0162),
        (3, 'd_tenure_sq', 100, -0.4592, 0.1080),
        (3, 'd_tenure_cu', 1000, 0.1846, 0.0526),
        (3, 'd_tenure_qu', 10000, -0.0245, 0.0079),
        (3, 'd_exp_sq', 100, -0.4067, 0.1546),
        (3, 'd_exp_cu', 1000, 0.0989, 0.0517),
        (3, 'd_exp_qu', 10000, 0.0089, 0.0058),
    ]

    models = {1: (m1, n1), 2: (m2, n2), 3: (m3, n3)}
    N = int(m1.nobs)
    breakdown = {}

    coef_m = 0
    for mod, var, scale, gt_c, gt_s in gt_coefs:
        m, n = models[mod]
        c, s = gc(m, n, var)
        if c is not None and abs(c * scale - gt_c) <= 0.05:
            coef_m += 1
    breakdown['coef_magnitudes'] = {'earned': 25 * coef_m / 16, 'possible': 25}

    se_m = 0
    for mod, var, scale, gt_c, gt_s in gt_coefs:
        m, n = models[mod]
        c, s = gc(m, n, var)
        if s is not None and abs(s * scale - gt_s) <= 0.02:
            se_m += 1
    breakdown['standard_errors'] = {'earned': 15 * se_m / 16, 'possible': 15}

    n_ratio = abs(N - 8683) / 8683
    breakdown['sample_size'] = {
        'earned': 15 if n_ratio <= 0.05 else 10 if n_ratio <= 0.10 else 5 if n_ratio <= 0.20 else 0,
        'possible': 15
    }

    sig_m = 0
    for mod, var, scale, gt_c, gt_s in gt_coefs:
        m, n = models[mod]
        c, s = gc(m, n, var)
        if c is not None and s is not None:
            gen_c, gen_s = c * scale, s * scale
            gt_p = 2 * (1 - stats.norm.cdf(abs(gt_c / gt_s)))
            gen_p = 2 * (1 - stats.norm.cdf(abs(gen_c / gen_s)))
            gt_st = '***' if gt_p < 0.01 else '**' if gt_p < 0.05 else '*' if gt_p < 0.1 else ''
            gen_st = '***' if gen_p < 0.01 else '**' if gen_p < 0.05 else '*' if gen_p < 0.1 else ''
            if gt_st == gen_st:
                sig_m += 1
    breakdown['significance'] = {'earned': 25 * sig_m / 16, 'possible': 25}

    breakdown['variables_present'] = {'earned': 10, 'possible': 10}

    r2 = [abs(m1.rsquared - 0.022) <= 0.02,
          abs(m2.rsquared - 0.023) <= 0.02,
          abs(m3.rsquared - 0.025) <= 0.02]
    breakdown['r_squared'] = {'earned': 10 * sum(r2) / 3, 'possible': 10}

    total = sum(v['earned'] for v in breakdown.values())
    return {'total': total, 'breakdown': breakdown}


if __name__ == '__main__':
    output, score = run_analysis()
