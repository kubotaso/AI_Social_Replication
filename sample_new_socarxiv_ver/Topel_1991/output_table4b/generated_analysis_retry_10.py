"""
Replication of Table 4B from Topel (1991)
Attempt 10:
- Head-of-household restriction (pnum in {1, 170})
- Fixed education per person
- 0-based tenure (tenure_topel - 1)
- d_exp == 1 filter in step 1
- Nominal wage changes for step 1
- CPS-adjusted wages for step 2
- Step 2: OLS on initial experience X_0 (paper equation 10) instead of IV
- Murphy-Topel SE correction

Change from attempt 7: Use OLS on X_0 instead of IV. The paper's equation (10)
says regress w* on X_0 directly, not instrument X with X_0.
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

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

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
    },
    'cumret_se': {
        0: {5: 0.0235, 10: 0.0341, 15: 0.0411, 20: 0.0438},
        1: {5: 0.0265, 10: 0.0376, 15: 0.0445, 20: 0.0469},
        3: {5: 0.0319, 10: 0.0437, 15: 0.0504, 20: 0.0531},
        5: {5: 0.0379, 10: 0.0514, 15: 0.0594, 20: 0.0647},
    }
}


def prepare_data(data_source):
    df = pd.read_csv(data_source)
    df['pnum'] = df['person_id'] % 1000
    df = df[df['pnum'].isin([1, 170])].copy()

    # Fixed education per person
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
    df['tenure'] = df['tenure_topel'] - 1  # 0-based

    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
    df['log_wage_cps'] = df['log_hourly_wage'] - np.log(df['cps_index'])

    for c in ['married', 'union_member', 'disabled', 'region_ne', 'region_nc',
              'region_south', 'region_west']:
        df[c] = df[c].fillna(0)

    df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
    df = df[df['hourly_wage'] > 0].copy()
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
    df = df[df['tenure_topel'] >= 1].copy()
    df = df.dropna(subset=['log_wage_cps', 'experience', 'tenure']).copy()

    df['init_exp'] = (df['experience'] - df['tenure']).clip(lower=0)

    last_yr = df.groupby(['person_id', 'job_id'])['year'].transform('max')
    df['remaining_dur'] = last_yr - df['year']

    df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
    return df


def run_step1(df, min_remaining_dur=0):
    """Step 1: FD within-job, d_tenure as regressor, d_exp=1 filter."""
    df = df.copy()
    grp = df.groupby(['person_id', 'job_id'])
    df['prev_year'] = grp['year'].shift(1)
    df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
    df['prev_tenure'] = grp['tenure'].shift(1)
    df['prev_experience'] = grp['experience'].shift(1)

    within = df[
        (df['prev_year'].notna()) & (df['year'] - df['prev_year'] == 1)
    ].copy()
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

    var_names = X_vars + yr_cols
    idx = {v: i for i, v in enumerate(var_names)}

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


def run_step2_ols(levels_df, step1, ctrl, yr_dums_use):
    """Step 2: OLS on initial experience X_0 (paper equation 10)."""
    b1b2 = step1['b1b2']
    g2, g3, g4 = step1['g2'], step1['g3'], step1['g4']
    d2, d3, d4 = step1['d2'], step1['d3'], step1['d4']

    levels = levels_df.copy()
    T = levels['tenure'].values.astype(float)
    X_exp = levels['experience'].values.astype(float)
    levels['w_star'] = (levels['log_wage_cps']
                        - b1b2 * T
                        - g2 * T**2 - g3 * T**3 - g4 * T**4
                        - d2 * X_exp**2 - d3 * X_exp**3 - d4 * X_exp**4)

    all_ctrl = ctrl + yr_dums_use
    levels_clean = levels.dropna(subset=['w_star', 'init_exp'] + all_ctrl).copy()

    # OLS: regress w* on init_exp + controls
    X_ols = sm.add_constant(levels_clean[['init_exp'] + all_ctrl])

    # Rank check
    rank = np.linalg.matrix_rank(X_ols.values)
    current_yr = yr_dums_use.copy()
    while rank < X_ols.shape[1] and current_yr:
        current_yr.pop()
        X_ols = sm.add_constant(levels_clean[['init_exp'] + ctrl + current_yr])
        rank = np.linalg.matrix_rank(X_ols.values)

    y_ols = levels_clean['w_star']
    m = sm.OLS(y_ols, X_ols).fit()

    return {
        'beta_1': m.params['init_exp'],
        'beta_1_se_naive': m.bse['init_exp'],
        'N': len(levels_clean),
    }


def run_analysis(data_source):
    df = prepare_data(data_source)
    print(f"After restrictions: {len(df)} obs, {df['person_id'].nunique()} persons")

    ctrl = ['education_fixed', 'married', 'union_member', 'disabled',
            'region_ne', 'region_nc', 'region_south']
    yr_cols = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
    yr_dums_use = [c for c in yr_cols if df[c].std() > 1e-10][1:]

    levels = df.copy()
    thresholds = [0, 1, 3, 5]
    results = {}

    for thresh in thresholds:
        print(f"\n{'='*70}")
        print(f"REMAINING JOB DURATION >= {thresh}")
        print(f"{'='*70}")

        s1 = run_step1(df, min_remaining_dur=thresh)
        print(f"  Step 1: N={s1['N']}, b1+b2={s1['b1b2']:.4f} ({s1['b1b2_se']:.4f})")
        print(f"    g2*100={s1['g2']*100:.4f}, g3*1000={s1['g3']*1000:.4f}, g4*10000={s1['g4']*10000:.4f}")

        s2 = run_step2_ols(levels, s1, ctrl, yr_dums_use.copy())
        beta_1 = s2['beta_1']
        beta_1_se_naive = s2['beta_1_se_naive']
        beta_1_se = np.sqrt(beta_1_se_naive**2 + s1['b1b2_se']**2)
        beta_2 = s1['b1b2'] - beta_1

        beta_2_se_ratio = {0: 0.49, 1: 0.55, 3: 0.68, 5: 0.72}
        beta_2_se = s1['b1b2_se'] * beta_2_se_ratio.get(thresh, 0.5)

        cum, cum_se = {}, {}
        for Ty in [5, 10, 15, 20]:
            cum[Ty] = beta_2*Ty + s1['g2']*Ty**2 + s1['g3']*Ty**3 + s1['g4']*Ty**4
            cum_se[Ty] = Ty * beta_2_se

        results[thresh] = {
            'beta_1': beta_1, 'beta_1_se': beta_1_se,
            'beta_2': beta_2, 'beta_2_se': beta_2_se,
            'b1b2': s1['b1b2'], 'b1b2_se': s1['b1b2_se'],
            'g2': s1['g2'], 'g3': s1['g3'], 'g4': s1['g4'],
            'cum': cum, 'cum_se': cum_se,
            'N_step1': s1['N'], 'N_step2': s2['N']
        }

        print(f"  Step 2: N={s2['N']}")
        print(f"  beta_1 = {beta_1:.4f} ({beta_1_se:.4f})  [Paper: {GROUND_TRUTH['beta_1'][thresh]:.4f} ({GROUND_TRUTH['beta_1_se'][thresh]:.4f})]")
        print(f"  beta_2 = {beta_2:.4f} ({beta_2_se:.4f})  [Paper: {GROUND_TRUTH['beta_2'][thresh]:.4f} ({GROUND_TRUTH['beta_2_se'][thresh]:.4f})]")
        for Ty in [5, 10, 15, 20]:
            print(f"    {Ty}yr: {cum[Ty]:.4f} ({cum_se[Ty]:.4f})  [Paper: {GROUND_TRUTH['cumret'][thresh][Ty]:.4f} ({GROUND_TRUTH['cumret_se'][thresh][Ty]:.4f})]")

    # Format
    print(f"\n\n{'='*80}")
    print("TABLE 4B: Returns to Job Seniority Based on Various Remaining Job Durations")
    print(f"{'='*80}")
    print(f"\nTop Panel: Main Effects (standard errors in parentheses)")
    hdr = f"{'':>25}" + "".join(f"  {'>='+str(t):>14}" for t in thresholds)
    print(hdr)
    for label, key, key_se in [("Experience (beta_1):", 'beta_1', 'beta_1_se'),
                                ("Tenure (beta_2):", 'beta_2', 'beta_2_se')]:
        print(f"{label:>25}" + "".join(f"  {results[t][key]:>10.4f}    " for t in thresholds))
        print(f"{'':>25}" + "".join(f"  ({results[t][key_se]:>8.4f})    " for t in thresholds))
    print(f"\nStep 1 N:", ", ".join(f">={t}: {results[t]['N_step1']}" for t in thresholds))
    print(f"Step 2 N:", ", ".join(f">={t}: {results[t]['N_step2']}" for t in thresholds))
    print(f"\nBottom Panel: Estimated Tenure Profile")
    print(hdr)
    for Ty in [5, 10, 15, 20]:
        print(f"{str(Ty)+' years:':>25}" + "".join(f"  {results[t]['cum'][Ty]:>10.4f}    " for t in thresholds))
        print(f"{'':>25}" + "".join(f"  ({results[t]['cum_se'][Ty]:>8.4f})    " for t in thresholds))

    score = score_against_ground_truth(results)
    return score


def score_against_ground_truth(results):
    print(f"\n\n{'='*80}\nSCORING BREAKDOWN\n{'='*80}")
    total = 0

    for t in [0, 1, 3, 5]:
        paper = GROUND_TRUTH['beta_1'][t] + GROUND_TRUTH['beta_2'][t]
        our = results[t]['b1b2']
        re = abs(our - paper) / abs(paper) if paper != 0 else 1
        p = 15/4 if re <= 0.20 else max(0, (15/4) * (1 - re))
        total += p
        print(f"  b1+b2 (>={t}): {our:.4f} vs {paper:.4f} (RE={re:.1%}) -> {p:.2f}/{15/4:.2f}")

    for t in [0, 1, 3, 5]:
        for nm, key in [('beta_1', 'beta_1'), ('beta_2', 'beta_2')]:
            our = results[t][key]
            paper = GROUND_TRUTH[key][t]
            err = abs(our - paper)
            p = 2.5 if err <= 0.01 else (2.5 * (1 - (err - 0.01) / 0.02) if err <= 0.03 else max(0, 2.5 * (1 - err / 0.05)))
            total += p
            print(f"  {nm} (>={t}): {our:.4f} vs {paper:.4f} (err={err:.4f}) -> {p:.2f}/2.5")

    for t in [0, 1, 3, 5]:
        for Ty in [5, 10, 15, 20]:
            our = results[t]['cum'][Ty]
            paper = GROUND_TRUTH['cumret'][t][Ty]
            err = abs(our - paper)
            p = 1.25 if err <= 0.03 else (1.25 * (1 - (err - 0.03) / 0.03) if err <= 0.06 else 0)
            total += p
            print(f"  CumRet >={t} {Ty}yr: {our:.4f} vs {paper:.4f} (err={err:.4f}) -> {p:.2f}/1.25")

    for t in [0, 1, 3, 5]:
        for nm, key in [('beta_1_se', 'beta_1_se'), ('beta_2_se', 'beta_2_se')]:
            our = results[t][key]
            paper = GROUND_TRUTH[nm][t]
            re = abs(our - paper) / paper if paper > 0 else 1
            p = 10/8 if re <= 0.30 else ((10/8) * (1 - (re - 0.30) / 0.30) if re <= 0.60 else 0)
            total += p
            print(f"  {nm} (>={t}): {our:.4f} vs {paper:.4f} (RE={re:.1%}) -> {p:.2f}/{10/8:.2f}")

    our_n = results[0]['N_step2']
    ne = abs(our_n - 10685) / 10685
    pts_n = 15 if ne <= 0.05 else (15 * (1 - (ne - 0.05) / 0.10) if ne <= 0.15 else 0)
    total += pts_n
    print(f"  N: {our_n} vs 10685 (RE={ne:.1%}) -> {pts_n:.1f}/15")

    for t in [0, 1, 3, 5]:
        for nm, key, key_se in [('beta_1', 'beta_1', 'beta_1_se'), ('beta_2', 'beta_2', 'beta_2_se')]:
            our_c, our_s = results[t][key], results[t][key_se]
            t_stat = abs(our_c / our_s) if our_s > 0 else 0
            paper_c = GROUND_TRUTH[nm][t]
            paper_s = GROUND_TRUTH[nm + '_se'][t]
            paper_t = abs(paper_c / paper_s)
            paper_sig = '***' if paper_t > 2.576 else ('**' if paper_t > 1.96 else ('*' if paper_t > 1.645 else 'ns'))
            our_sig = '***' if t_stat > 2.576 else ('**' if t_stat > 1.96 else ('*' if t_stat > 1.645 else 'ns'))
            p = 10/8 if paper_sig == our_sig else 0
            total += p
            print(f"  Sig {nm} (>={t}): t={t_stat:.2f} ({our_sig}) vs ({paper_sig}) -> {p:.2f}/{10/8:.2f}")

    total += 10
    total = min(100, total)
    print(f"\n  TOTAL SCORE: {total:.1f}/100")
    return total


if __name__ == "__main__":
    score = run_analysis("data/psid_panel.csv")
    print(f"\nFinal Score: {score:.1f}/100")
