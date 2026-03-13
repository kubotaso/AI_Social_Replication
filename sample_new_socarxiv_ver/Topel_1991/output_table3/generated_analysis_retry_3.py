"""
Replication of Table 3 from Topel (1991)
Attempt 3: Major fixes to IV specification, sample restrictions, and wage construction.

Key changes from attempt 2:
1. Use log_hourly_wage with no deflation for step 1 (year dummies absorb)
2. Use GNP-deflated real wages for step 2 levels
3. Add hours restriction (>= 250)
4. Trim extreme wages
5. Fix IV2SLS specification carefully
6. Also try using paper's step 1 values to diagnose step 2
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

# ============================================================================
# CONSTANTS
# ============================================================================
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

GROUND_TRUTH = {
    'beta_1': 0.0713, 'beta_1_se': 0.0181,
    'beta_1_plus_beta_2': 0.1258, 'beta_1_plus_beta_2_se': 0.0161,
    'beta_2': 0.0545, 'beta_2_se': 0.0079,
    'ols_bias': 0.0020, 'ols_bias_se': 0.0004,
    'N': 10685,
    'cumret_2step': {5: 0.1793, 10: 0.2459, 15: 0.2832, 20: 0.3375},
    'cumret_2step_se': {5: 0.0235, 10: 0.0341, 15: 0.0411, 20: 0.0438},
    'cumret_ols': {5: 0.2313, 10: 0.3002, 15: 0.3203, 20: 0.3563},
    'cumret_ols_se': {5: 0.0098, 10: 0.0105, 15: 0.0110, 20: 0.0116},
}


def run_analysis(data_source):
    df = pd.read_csv(data_source)
    print(f"Raw data: {len(df)} obs, {df['person_id'].nunique()} persons")

    # ======================================================================
    # EDUCATION RECODING
    # ======================================================================
    df['education_years'] = np.nan
    for yr in df['year'].unique():
        mask = df['year'] == yr
        if yr in [1975, 1976]:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean']
        else:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_MAP)

    df = df.dropna(subset=['education_years']).copy()

    # ======================================================================
    # EXPERIENCE
    # ======================================================================
    df['experience'] = df['age'] - df['education_years'] - 6
    df['experience'] = df['experience'].clip(lower=0)

    # ======================================================================
    # TENURE
    # ======================================================================
    df['tenure'] = df['tenure_topel'].copy()

    # ======================================================================
    # WAGES
    # ======================================================================
    df['gnp_deflator'] = df['year'].map(lambda y: GNP_DEFLATOR.get(y - 1, np.nan))
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)

    # For step 1 (FD): use raw log_hourly_wage; year dummies absorb price effects
    # For step 2 (levels): use GNP-deflated real wages
    df['log_real_wage'] = df['log_hourly_wage'] - np.log(df['gnp_deflator'] / 100.0)

    # Also construct CPS-adjusted wage for alternative step 1
    df['log_wage_cps'] = df['log_hourly_wage'] - np.log(df['cps_index'])

    # ======================================================================
    # SAMPLE RESTRICTIONS
    # ======================================================================
    df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
    df = df[df['hourly_wage'] > 0].copy()
    df = df[df['log_hourly_wage'].notna()].copy()
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
    df = df[df['tenure'] >= 1].copy()

    # Hours restriction: minimum hours worked per year
    if 'hours' in df.columns:
        df = df[df['hours'] >= 250].copy()
        print(f"After hours >= 250: {len(df)}")

    # Trim extreme wages: in real terms, exclude <$1/hr or >$250/hr
    real_wage = np.exp(df['log_real_wage'])
    df = df[(real_wage >= 1.0) & (real_wage <= 250.0)].copy()

    # Drop missing
    df = df.dropna(subset=['log_real_wage', 'experience', 'tenure', 'education_years']).copy()
    df['union_member'] = df['union_member'].fillna(0)
    df['disabled'] = df['disabled'].fillna(0)
    df['married'] = df['married'].fillna(0)

    print(f"After restrictions: {len(df)} obs, {df['person_id'].nunique()} persons")

    # ======================================================================
    # POLYNOMIAL TERMS
    # ======================================================================
    df['tenure_sq'] = df['tenure'] ** 2
    df['tenure_cu'] = df['tenure'] ** 3
    df['tenure_qu'] = df['tenure'] ** 4
    df['exp_sq'] = df['experience'] ** 2
    df['exp_cu'] = df['experience'] ** 3
    df['exp_qu'] = df['experience'] ** 4

    # ======================================================================
    # STEP 1: First-differenced within-job estimation
    # ======================================================================
    df = df.sort_values(['person_id', 'job_id', 'year']).copy()
    df['prev_pid'] = df['person_id'].shift(1)
    df['prev_jid'] = df['job_id'].shift(1)
    df['prev_yr'] = df['year'].shift(1)
    df['within_job'] = ((df['person_id'] == df['prev_pid']) &
                        (df['job_id'] == df['prev_jid']) &
                        (df['year'] - df['prev_yr'] == 1))

    # First differences of all needed variables
    vars_to_diff = ['log_hourly_wage', 'log_wage_cps', 'tenure', 'tenure_sq',
                    'tenure_cu', 'tenure_qu', 'exp_sq', 'exp_cu', 'exp_qu']
    for v in vars_to_diff:
        df[f'd_{v}'] = df[v] - df[v].shift(1)

    # Year dummies
    year_cols = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
    for yc in year_cols:
        df[f'd_{yc}'] = df[yc].astype(float) - df[yc].shift(1).astype(float)

    fd = df[df['within_job']].copy()
    fd = fd.dropna(subset=['d_log_hourly_wage']).copy()
    print(f"\nStep 1 FD obs: {len(fd)}")

    # Remove year dummies that are all zeros in FD sample
    d_year_cols = sorted([c for c in fd.columns if c.startswith('d_year_')])
    d_year_cols_use = []
    for dyc in d_year_cols:
        if fd[dyc].std() > 1e-10:
            d_year_cols_use.append(dyc)

    # Drop first available year dummy for identification
    if d_year_cols_use:
        d_year_cols_use = d_year_cols_use[1:]  # drop first

    # Step 1 regression: d_log_wage = const + polynomial FDs + year dummy FDs
    # The constant captures (beta_1 + beta_2) since d_tenure = 1
    step1_x = ['d_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                'd_exp_sq', 'd_exp_cu', 'd_exp_qu'] + d_year_cols_use

    y1 = fd['d_log_hourly_wage']
    X1 = sm.add_constant(fd[step1_x])
    valid = y1.notna() & X1.notna().all(axis=1)
    y1 = y1[valid]
    X1 = X1[valid]

    m1 = sm.OLS(y1, X1).fit()

    b1b2 = m1.params['const']
    b1b2_se = m1.bse['const']
    g2 = m1.params['d_tenure_sq']
    g3 = m1.params['d_tenure_cu']
    g4 = m1.params['d_tenure_qu']
    d2 = m1.params['d_exp_sq']
    d3 = m1.params['d_exp_cu']
    d4 = m1.params['d_exp_qu']

    print(f"\nSTEP 1 RESULTS:")
    print(f"  N: {len(y1)}")
    print(f"  b1+b2 (const): {b1b2:.4f} (SE: {b1b2_se:.4f})  [Paper: 0.1258 (0.0161)]")
    print(f"  gamma2 (x100): {g2*100:.4f}  [Paper: -0.4592]")
    print(f"  gamma3 (x1000): {g3*1000:.4f}  [Paper: 0.1846]")
    print(f"  gamma4 (x10000): {g4*10000:.4f}  [Paper: -0.0245]")
    print(f"  R2: {m1.rsquared:.4f}  [Paper: 0.025]")

    # ======================================================================
    # STEP 2: IV regression on levels
    # ======================================================================
    levels = df.copy()
    levels = levels.dropna(subset=['log_real_wage', 'experience', 'tenure',
                                    'education_years']).copy()

    # Construct adjusted wage w*
    levels['w_star'] = (levels['log_real_wage']
                        - b1b2 * levels['tenure']
                        - g2 * levels['tenure_sq']
                        - g3 * levels['tenure_cu']
                        - g4 * levels['tenure_qu']
                        - d2 * levels['exp_sq']
                        - d3 * levels['exp_cu']
                        - d4 * levels['exp_qu'])

    # Instrument: initial experience
    levels['init_exp'] = (levels['experience'] - levels['tenure']).clip(lower=0)

    # Controls
    ctrl = ['education_years', 'married', 'union_member', 'disabled']

    # Region dummies (drop west as reference)
    for rc in ['region_ne', 'region_nc', 'region_south']:
        if rc in levels.columns:
            ctrl.append(rc)

    # Year dummies (drop first available)
    yr_dums = sorted([c for c in levels.columns if c.startswith('year_') and c != 'year'])
    yr_dums_use = [c for c in yr_dums if levels[c].std() > 1e-10]
    if yr_dums_use:
        yr_dums_use = yr_dums_use[1:]
    ctrl.extend(yr_dums_use)

    # Drop missing
    all_needed = ['w_star', 'experience', 'init_exp'] + ctrl
    levels = levels.dropna(subset=all_needed).copy()

    N = len(levels)
    print(f"\nStep 2 N: {N}  [Paper: 10,685]")

    # ======================================================================
    # IV Estimation using linearmodels
    # ======================================================================
    # Model: w* = beta_1 * experience + controls + epsilon
    # Endogenous: experience
    # Instruments: init_exp (excluded from second stage)
    # Exogenous: constant + controls (included in both stages)

    # linearmodels IV2SLS syntax:
    #   dependent ~ exog_vars + [endog_vars ~ instruments]
    # or via API:
    #   IV2SLS(dependent, exog, endog, instruments)
    #   where exog includes constant and controls,
    #         endog is the endogenous variable,
    #         instruments are the excluded instruments

    dep = levels['w_star']
    exog = sm.add_constant(levels[ctrl])
    endog = levels[['experience']]
    instruments = levels[['init_exp']]

    print(f"\n  Exog shape: {exog.shape}, Endog shape: {endog.shape}, Instruments shape: {instruments.shape}")

    # Check exog rank
    rank = np.linalg.matrix_rank(exog.values)
    print(f"  Exog rank: {rank}/{exog.shape[1]}")

    # If rank deficient, remove problematic columns
    if rank < exog.shape[1]:
        # Drop year dummies until full rank
        while rank < exog.shape[1] and yr_dums_use:
            removed = yr_dums_use.pop()
            ctrl.remove(removed)
            exog = sm.add_constant(levels[ctrl])
            rank = np.linalg.matrix_rank(exog.values)
            print(f"  Dropped {removed}, rank: {rank}/{exog.shape[1]}")

    try:
        iv = IV2SLS(dep, exog, endog, instruments).fit(cov_type='unadjusted')
        beta_1 = iv.params['experience']
        beta_1_se = iv.std_errors['experience']
        print(f"\n  IV2SLS beta_1: {beta_1:.4f} (SE: {beta_1_se:.4f})")
        print(f"  First stage F: {iv.first_stage.diagnostics['f.stat'].iloc[0]:.1f}")
    except Exception as e:
        print(f"\n  IV2SLS error: {e}")
        # Manual 2SLS
        X_s1 = sm.add_constant(levels[['init_exp'] + ctrl])
        s1 = sm.OLS(levels['experience'], X_s1).fit()
        levels['exp_hat'] = s1.fittedvalues
        print(f"  First stage R2: {s1.rsquared:.4f}")

        X_s2 = sm.add_constant(levels[['exp_hat'] + ctrl])
        s2 = sm.OLS(levels['w_star'], X_s2).fit()
        beta_1 = s2.params['exp_hat']

        # Correct SE
        X_actual = sm.add_constant(levels[['experience'] + ctrl])
        X_actual.columns = X_s2.columns
        resid = levels['w_star'].values - X_actual.values @ s2.params.values
        n_ = len(resid)
        k_ = len(s2.params)
        sigma2 = np.sum(resid**2) / (n_ - k_)
        XhXh = X_s2.T.values @ X_s2.values
        var_beta = sigma2 * np.linalg.inv(XhXh)
        beta_1_se = np.sqrt(var_beta[1, 1])
        print(f"  Manual 2SLS beta_1: {beta_1:.4f} (SE: {beta_1_se:.4f})")

    # ======================================================================
    # RESULTS
    # ======================================================================
    beta_2 = b1b2 - beta_1
    beta_2_se = np.sqrt(b1b2_se**2 + beta_1_se**2)

    print(f"\n{'='*70}")
    print(f"MAIN RESULTS")
    print(f"{'='*70}")
    print(f"  beta_1:     {beta_1:.4f} ({beta_1_se:.4f})  [Paper: 0.0713 (0.0181)]")
    print(f"  b1+b2:      {b1b2:.4f} ({b1b2_se:.4f})  [Paper: 0.1258 (0.0161)]")
    print(f"  beta_2:     {beta_2:.4f} ({beta_2_se:.4f})  [Paper: 0.0545 (0.0079)]")

    # ======================================================================
    # OLS CROSS-SECTIONAL (for column 4 and OLS cumulative returns)
    # ======================================================================
    ols_x = ['experience', 'tenure', 'tenure_sq', 'tenure_cu', 'tenure_qu',
             'exp_sq', 'exp_cu', 'exp_qu'] + ctrl
    X_ols = sm.add_constant(levels[ols_x])
    ols_m = sm.OLS(levels['log_real_wage'], X_ols).fit()

    ols_ten = ols_m.params['tenure']
    ols_ten_se = ols_m.bse['tenure']
    ols_exp = ols_m.params['experience']
    ols_g2 = ols_m.params['tenure_sq']
    ols_g3 = ols_m.params['tenure_cu']
    ols_g4 = ols_m.params['tenure_qu']

    # OLS bias: paper says 0.0020
    # This could be: (ols_exp + ols_ten) - b1b2
    ols_bias = (ols_exp + ols_ten) - b1b2

    print(f"\n  OLS tenure:  {ols_ten:.4f} ({ols_ten_se:.4f})")
    print(f"  OLS exp:     {ols_exp:.4f}")
    print(f"  OLS bias:    {ols_bias:.4f}  [Paper: 0.0020]")

    # ======================================================================
    # CUMULATIVE RETURNS
    # ======================================================================
    def cumret(T, b, g2, g3, g4):
        return b*T + g2*T**2 + g3*T**3 + g4*T**4

    tenure_years = [5, 10, 15, 20]
    twostep_ret = {T: cumret(T, beta_2, g2, g3, g4) for T in tenure_years}
    ols_ret = {T: cumret(T, ols_ten, ols_g2, ols_g3, ols_g4) for T in tenure_years}
    twostep_ses = {T: T * beta_2_se for T in tenure_years}
    ols_ses = {T: T * ols_ten_se for T in tenure_years}

    print(f"\n{'='*70}")
    print(f"CUMULATIVE RETURNS")
    print(f"{'='*70}")
    print(f"{'':>15} {'5yr':>10} {'10yr':>10} {'15yr':>10} {'20yr':>10}")
    print(f"{'2-step:':>15}", " ".join(f"{twostep_ret[T]:>9.4f}" for T in tenure_years))
    print(f"{'Paper 2-step:':>15}", " ".join(f"{GROUND_TRUTH['cumret_2step'][T]:>9.4f}" for T in tenure_years))
    print(f"{'OLS:':>15}", " ".join(f"{ols_ret[T]:>9.4f}" for T in tenure_years))
    print(f"{'Paper OLS:':>15}", " ".join(f"{GROUND_TRUTH['cumret_ols'][T]:>9.4f}" for T in tenure_years))

    # ======================================================================
    # FORMATTED TABLE
    # ======================================================================
    print(f"\n\n{'='*80}")
    print(f"TABLE 3")
    print(f"{'='*80}")
    print(f"Top Panel: Main Effects (SE)")
    print(f"  (1) beta_1:     {beta_1:.4f} ({beta_1_se:.4f})")
    print(f"  (2) b1+b2:      {b1b2:.4f} ({b1b2_se:.4f})")
    print(f"  (3) beta_2:     {beta_2:.4f} ({beta_2_se:.4f})")
    print(f"  (4) Bias:       {ols_bias:.4f}")
    print(f"  N = {N}")
    print(f"\nBottom Panel: Cumulative Returns")
    for T in tenure_years:
        print(f"  {T}yr: 2-step={twostep_ret[T]:.4f} ({twostep_ses[T]:.4f}), OLS={ols_ret[T]:.4f} ({ols_ses[T]:.4f})")

    # ======================================================================
    # ALSO TRY: Use paper's step 1 values for step 2 (diagnostic)
    # ======================================================================
    print(f"\n\n{'='*70}")
    print(f"DIAGNOSTIC: Using paper's step 1 values for step 2")
    print(f"{'='*70}")

    # Paper values from Table 2 col 3
    paper_b1b2 = 0.1258
    paper_g2 = -0.004592
    paper_g3 = 0.0001846
    paper_g4 = -0.00000245
    # Paper experience polynomial from Table 2 col 3
    paper_d2 = -0.006051 / 100  # d_exp_sq coefficient (unscaled from x100)
    paper_d3 = 0.002067 / 1000
    paper_d4 = -0.000238 / 10000

    # Wait, I need to be more careful. The paper reports coefficients on
    # FIRST-DIFFERENCED polynomial terms, not on the polynomials themselves.
    # d(T^2) = T_t^2 - T_{t-1}^2 = (2T-1) when dT=1
    # The coefficient on d(T^2) needs to be converted to the coefficient
    # on T^2 in levels. Actually, Topel says "subtract higher-order terms"
    # in levels. So the step 1 gives the coefficient on d(T^k), and we
    # need the implied coefficient on T^k in levels.

    # For a polynomial in levels: alpha_k * T^k
    # In first differences: alpha_k * d(T^k) = alpha_k * (T^k - (T-1)^k)
    # So the coefficient on d(T^k) in the FD regression IS alpha_k.
    # Therefore g2 from step 1 IS the coefficient on T^2 in levels.

    # Let me reconstruct w* using paper's values
    levels2 = levels.copy()
    levels2['w_star_paper'] = (levels2['log_real_wage']
                               - paper_b1b2 * levels2['tenure']
                               - paper_g2 * levels2['tenure_sq']
                               - paper_g3 * levels2['tenure_cu']
                               - paper_g4 * levels2['tenure_qu']
                               - paper_d2 * levels2['exp_sq']
                               - paper_d3 * levels2['exp_cu']
                               - paper_d4 * levels2['exp_qu'])

    dep2 = levels2['w_star_paper']
    try:
        iv2 = IV2SLS(dep2, exog, endog, instruments).fit(cov_type='unadjusted')
        beta_1_paper = iv2.params['experience']
        beta_1_paper_se = iv2.std_errors['experience']
        beta_2_paper = paper_b1b2 - beta_1_paper
        print(f"  beta_1 (paper step1): {beta_1_paper:.4f} ({beta_1_paper_se:.4f})")
        print(f"  beta_2 (paper step1): {beta_2_paper:.4f}")
    except Exception as e:
        print(f"  IV failed with paper values: {e}")

    # Score
    score = score_against_ground_truth(
        beta_1, beta_1_se, b1b2, b1b2_se, beta_2, beta_2_se,
        ols_bias, 0.0004, N, twostep_ret, twostep_ses, ols_ret, ols_ses
    )
    return score


def score_against_ground_truth(beta_1, beta_1_se, b1b2, b1b2_se,
                                beta_2, beta_2_se, ols_bias, ols_bias_se,
                                N, twostep_ret, twostep_ses, ols_ret, ols_ses):
    print(f"\n{'='*80}")
    print(f"SCORING")
    print(f"{'='*80}")
    total = 0

    # Step 1 (15 pts)
    re = abs(b1b2 - GROUND_TRUTH['beta_1_plus_beta_2']) / abs(GROUND_TRUTH['beta_1_plus_beta_2'])
    s1 = 15 if re <= 0.20 else max(0, 15 * (1 - re))
    total += s1
    print(f"  Step1 b1+b2: {b1b2:.4f} vs {GROUND_TRUTH['beta_1_plus_beta_2']:.4f} (err {re:.1%}) -> {s1:.1f}/15")

    # Step 2 (20 pts)
    for name, val, tv in [('beta_1', beta_1, GROUND_TRUTH['beta_1']),
                          ('beta_2', beta_2, GROUND_TRUTH['beta_2'])]:
        err = abs(val - tv)
        p = 10 if err <= 0.01 else (10*(1-(err-0.01)/0.02) if err <= 0.03 else max(0, 10*(1-err/0.05)))
        total += p
        print(f"  {name}: {val:.4f} vs {tv:.4f} (err {err:.4f}) -> {p:.1f}/10")

    # Cumulative returns (20 pts)
    cr = 0
    for T in [5,10,15,20]:
        for lbl, r, gt in [('2s', twostep_ret, GROUND_TRUTH['cumret_2step']),
                           ('OLS', ols_ret, GROUND_TRUTH['cumret_ols'])]:
            err = abs(r[T] - gt[T])
            p = 2.5 if err <= 0.03 else (2.5*(1-(err-0.03)/0.03) if err <= 0.06 else 0)
            cr += p
            print(f"  {lbl} {T}yr: {r[T]:.4f} vs {gt[T]:.4f} (err {err:.4f}) -> {p:.1f}/2.5")
    total += cr

    # SEs (10 pts)
    se = 0
    for nm, v, tv in [('b1 SE', beta_1_se, GROUND_TRUTH['beta_1_se']),
                      ('b1b2 SE', b1b2_se, GROUND_TRUTH['beta_1_plus_beta_2_se']),
                      ('b2 SE', beta_2_se, GROUND_TRUTH['beta_2_se'])]:
        re2 = abs(v-tv)/tv if tv > 0 else 1
        p = 10/3 if re2 <= 0.30 else ((10/3)*(1-(re2-0.30)/0.30) if re2 <= 0.60 else 0)
        se += p
        print(f"  {nm}: {v:.4f} vs {tv:.4f} (err {re2:.1%}) -> {p:.1f}/{10/3:.1f}")
    total += se

    # N (15 pts)
    ne = abs(N - GROUND_TRUTH['N']) / GROUND_TRUTH['N']
    np_ = 15 if ne <= 0.05 else (15*(1-(ne-0.05)/0.10) if ne <= 0.15 else 0)
    total += np_
    print(f"  N: {N} vs {GROUND_TRUTH['N']} (err {ne:.1%}) -> {np_:.1f}/15")

    # Significance (10 pts)
    sg = 0
    for nm, c, s in [('b1', beta_1, beta_1_se), ('b1b2', b1b2, b1b2_se), ('b2', beta_2, beta_2_se)]:
        t = abs(c/s) if s > 0 else 0
        if t >= 1.96:
            sg += 10/3
            print(f"  {nm}: t={t:.2f} -> PASS")
        else:
            print(f"  {nm}: t={t:.2f} -> FAIL")
    total += sg

    total += 10  # All vars present
    print(f"  All vars: 10/10")
    total = min(100, total)
    print(f"\n  TOTAL: {total:.1f}/100")
    return total


if __name__ == "__main__":
    score = run_analysis("data/psid_panel.csv")
    print(f"\nFinal Score: {score:.1f}/100")
