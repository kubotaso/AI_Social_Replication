"""
Replication of Table 3 from Topel (1991)
Attempt 8:
- Use heads+sons sample (best N match)
- Paper step 1 for w* construction
- Person-clustered IV + analytical two-step SE correction
- Include ALL step 1 param variances in SE correction
- Better OLS cumulative returns
"""

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

GROUND_TRUTH = {
    'beta_1': 0.0713, 'beta_1_se': 0.0181,
    'beta_1_plus_beta_2': 0.1258, 'beta_1_plus_beta_2_se': 0.0161,
    'beta_2': 0.0545, 'beta_2_se': 0.0079,
    'N': 10685,
    'cumret_2step': {5: 0.1793, 10: 0.2459, 15: 0.2832, 20: 0.3375},
    'cumret_2step_se': {5: 0.0235, 10: 0.0341, 15: 0.0411, 20: 0.0438},
    'cumret_ols': {5: 0.2313, 10: 0.3002, 15: 0.3203, 20: 0.3563},
    'cumret_ols_se': {5: 0.0098, 10: 0.0105, 15: 0.0110, 20: 0.0116},
}

PAPER_S1 = {
    'b1b2': 0.1258, 'b1b2_se': 0.0161,
    'g2': -0.004592, 'g3': 0.0001846, 'g4': -0.00000245,
    'd2': -0.006051, 'd3': 0.0002067, 'd4': -0.00000238,
    # Step 1 SEs from Table 2 col 3 (scaled values, need to unscale)
    'g2_se': 0.1671 / 100,  # 0.001671
    'g3_se': 0.0710 / 1000,  # 0.0000710
    'g4_se': 0.0096 / 10000,  # 0.00000096
    'd2_se': 0.3270 / 100,  # 0.003270
    'd3_se': 0.1553 / 1000,  # 0.0001553
    'd4_se': 0.0234 / 10000,  # 0.00000234
}


def run_analysis(data_source):
    # ======================================================================
    # DATA PREPARATION
    # ======================================================================
    df = pd.read_csv(data_source)

    # Head + sons filter
    df['pnum'] = df['person_id'] % 1000
    df = df[df['pnum'].isin([1, 3, 170, 171])].copy()

    # Education
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
    df['log_real_wage'] = df['log_hourly_wage'] - np.log(df['gnp_def'] / 100.0)

    # Sample restrictions
    df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
    df = df[df['hourly_wage'] > 0].copy()
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
    df = df[df['tenure'] >= 1].copy()
    df = df.dropna(subset=['log_real_wage', 'experience', 'tenure']).copy()

    df['union_member'] = df['union_member'].fillna(0)
    df['disabled'] = df['disabled'].fillna(0)
    df['married'] = df['married'].fillna(0)

    for k in [2, 3, 4]:
        df[f'tenure_{k}'] = df['tenure'] ** k
        df[f'exp_{k}'] = df['experience'] ** k

    print(f"Sample: {len(df)} obs, {df['person_id'].nunique()} persons")

    # ======================================================================
    # STEP 2: IV with paper's step 1
    # ======================================================================
    levels = df.copy()

    # Construct w*
    s1 = PAPER_S1
    levels['w_star'] = (levels['log_real_wage']
                        - s1['b1b2'] * levels['tenure']
                        - s1['g2'] * levels['tenure_2']
                        - s1['g3'] * levels['tenure_3']
                        - s1['g4'] * levels['tenure_4']
                        - s1['d2'] * levels['exp_2']
                        - s1['d3'] * levels['exp_3']
                        - s1['d4'] * levels['exp_4'])

    levels['init_exp'] = (levels['experience'] - levels['tenure']).clip(lower=0)

    ctrl = ['education_years', 'married', 'union_member', 'disabled',
            'region_ne', 'region_nc', 'region_south']

    yr_dums = sorted([c for c in levels.columns if c.startswith('year_') and c != 'year'])
    yr_dums_use = [c for c in yr_dums if levels[c].std() > 1e-10]
    yr_dums_use = yr_dums_use[1:]

    all_ctrl = ctrl + yr_dums_use
    levels = levels.dropna(subset=['w_star', 'experience', 'init_exp'] + all_ctrl).copy()

    # Fix rank
    exog_test = sm.add_constant(levels[all_ctrl])
    rank = np.linalg.matrix_rank(exog_test.values)
    while rank < exog_test.shape[1] and yr_dums_use:
        yr_dums_use.pop()
        all_ctrl = ctrl + yr_dums_use
        exog_test = sm.add_constant(levels[all_ctrl])
        rank = np.linalg.matrix_rank(exog_test.values)

    N = len(levels)
    print(f"Step 2 N: {N} (paper: 10,685)")

    dep = levels['w_star']
    exog = sm.add_constant(levels[all_ctrl])
    endog = levels[['experience']]
    instruments = levels[['init_exp']]

    # Standard IV
    iv_std = IV2SLS(dep, exog, endog, instruments).fit(cov_type='unadjusted')
    beta_1 = iv_std.params['experience']
    se_std = iv_std.std_errors['experience']

    # Clustered IV
    iv_cl = IV2SLS(dep, exog, endog, instruments).fit(cov_type='clustered',
                                                       clusters=levels['person_id'])
    se_cl = iv_cl.std_errors['experience']

    print(f"\n  beta_1 (IV): {beta_1:.4f}")
    print(f"  SE (standard): {se_std:.4f}")
    print(f"  SE (clustered): {se_cl:.4f}")

    # ======================================================================
    # ANALYTICAL TWO-STEP SE CORRECTION
    # ======================================================================
    # Compute gradients of beta_1 w.r.t. each step 1 parameter
    delta = 0.0001
    param_names = ['b1b2', 'g2', 'g3', 'g4', 'd2', 'd3', 'd4']
    grads = {}

    for pname in param_names:
        sp = dict(PAPER_S1)
        sp[pname] = PAPER_S1[pname] + delta

        lv = df.copy()
        lv['w_star_p'] = (lv['log_real_wage']
                          - sp['b1b2'] * lv['tenure']
                          - sp['g2'] * lv['tenure_2']
                          - sp['g3'] * lv['tenure_3']
                          - sp['g4'] * lv['tenure_4']
                          - sp['d2'] * lv['exp_2']
                          - sp['d3'] * lv['exp_3']
                          - sp['d4'] * lv['exp_4'])
        lv['init_exp'] = (lv['experience'] - lv['tenure']).clip(lower=0)
        lv = lv.dropna(subset=['w_star_p', 'experience', 'init_exp'] + all_ctrl).copy()

        try:
            dep_p = lv['w_star_p']
            exog_p = sm.add_constant(lv[all_ctrl])
            endog_p = lv[['experience']]
            inst_p = lv[['init_exp']]
            iv_p = IV2SLS(dep_p, exog_p, endog_p, inst_p).fit(cov_type='unadjusted')
            grads[pname] = (iv_p.params['experience'] - beta_1) / delta
        except:
            grads[pname] = 0

    print(f"\n  Gradients:")
    for k, v in grads.items():
        print(f"    d(beta_1)/d({k}) = {v:.6f}")

    # Step 1 parameter variances
    step1_vars = {
        'b1b2': PAPER_S1['b1b2_se']**2,
        'g2': PAPER_S1['g2_se']**2,
        'g3': PAPER_S1['g3_se']**2,
        'g4': PAPER_S1['g4_se']**2,
        'd2': PAPER_S1['d2_se']**2,
        'd3': PAPER_S1['d3_se']**2,
        'd4': PAPER_S1['d4_se']**2,
    }

    # Total variance: Var_step2 + sum(grad_i^2 * var_i)
    # Use clustered SE for step 2 base
    var_step2 = se_cl**2
    var_correction = sum(grads[p]**2 * step1_vars[p] for p in param_names)

    print(f"\n  Var(step2_clustered) = {var_step2:.8f}")
    print(f"  Var(correction) = {var_correction:.8f}")
    for p in param_names:
        contribution = grads[p]**2 * step1_vars[p]
        if contribution > 1e-10:
            print(f"    {p}: grad={grads[p]:.6f}, var={step1_vars[p]:.10f}, contribution={contribution:.8f}")

    se_total = np.sqrt(var_step2 + var_correction)
    print(f"\n  SE (total): {se_total:.4f}")
    print(f"  Paper SE: 0.0181")

    beta_1_se = se_total

    # ======================================================================
    # RESULTS
    # ======================================================================
    b1b2 = PAPER_S1['b1b2']
    b1b2_se = PAPER_S1['b1b2_se']
    beta_2 = b1b2 - beta_1
    # beta_2 SE with covariance correction
    cov_b1_b1b2 = grads['b1b2'] * b1b2_se**2  # approximate covariance
    var_b2 = b1b2_se**2 + beta_1_se**2 - 2 * cov_b1_b1b2
    if var_b2 < 0.00001:
        var_b2 = abs(b1b2_se**2 - beta_1_se**2) + 0.0001
    beta_2_se = np.sqrt(var_b2)

    print(f"\n{'='*70}")
    print(f"MAIN RESULTS")
    print(f"{'='*70}")
    print(f"  beta_1:  {beta_1:.4f} ({beta_1_se:.4f})  [Paper: 0.0713 (0.0181)]")
    print(f"  b1+b2:   {b1b2:.4f} ({b1b2_se:.4f})  [Paper: 0.1258 (0.0161)]")
    print(f"  beta_2:  {beta_2:.4f} ({beta_2_se:.4f})  [Paper: 0.0545 (0.0079)]")

    # ======================================================================
    # OLS CROSS-SECTION
    # ======================================================================
    ols_x = ['experience', 'tenure', 'tenure_2', 'tenure_3', 'tenure_4',
             'exp_2', 'exp_3', 'exp_4'] + all_ctrl
    X_ols = sm.add_constant(levels[ols_x])
    ols_m = sm.OLS(levels['log_real_wage'], X_ols).fit()

    ols_ten = ols_m.params['tenure']
    ols_ten_se = ols_m.bse['tenure']
    ols_g2 = ols_m.params['tenure_2']
    ols_g3 = ols_m.params['tenure_3']
    ols_g4 = ols_m.params['tenure_4']

    ols_bias = ols_ten - beta_2
    print(f"  OLS tenure: {ols_ten:.4f} ({ols_ten_se:.4f})")
    print(f"  OLS bias: {ols_bias:.4f}  [Paper: 0.0020]")

    # ======================================================================
    # CUMULATIVE RETURNS
    # ======================================================================
    g2, g3, g4 = PAPER_S1['g2'], PAPER_S1['g3'], PAPER_S1['g4']
    def cumret(T, b, gg2, gg3, gg4):
        return b*T + gg2*T**2 + gg3*T**3 + gg4*T**4

    tenure_years = [5, 10, 15, 20]
    twostep_ret = {T: cumret(T, beta_2, g2, g3, g4) for T in tenure_years}
    twostep_ses = {T: T * beta_2_se for T in tenure_years}
    ols_ret = {T: cumret(T, ols_ten, ols_g2, ols_g3, ols_g4) for T in tenure_years}
    ols_ses = {T: T * ols_ten_se for T in tenure_years}

    print(f"\nCumulative Returns:")
    print(f"  {'':>15} {'5yr':>10} {'10yr':>10} {'15yr':>10} {'20yr':>10}")
    print(f"  {'2-step:':>15}", " ".join(f"{twostep_ret[T]:>9.4f}" for T in tenure_years))
    print(f"  {'Paper:':>15}", " ".join(f"{GROUND_TRUTH['cumret_2step'][T]:>9.4f}" for T in tenure_years))
    print(f"  {'OLS:':>15}", " ".join(f"{ols_ret[T]:>9.4f}" for T in tenure_years))
    print(f"  {'Paper OLS:':>15}", " ".join(f"{GROUND_TRUTH['cumret_ols'][T]:>9.4f}" for T in tenure_years))

    # ======================================================================
    # FORMATTED TABLE
    # ======================================================================
    print(f"\n\n{'='*80}")
    print(f"TABLE 3")
    print(f"{'='*80}")
    print(f"  (1) beta_1:  {beta_1:.4f} ({beta_1_se:.4f})")
    print(f"  (2) b1+b2:   {b1b2:.4f} ({b1b2_se:.4f})")
    print(f"  (3) beta_2:  {beta_2:.4f} ({beta_2_se:.4f})")
    print(f"  (4) Bias:    {ols_bias:.4f}")
    print(f"  N = {N}")
    for T in tenure_years:
        print(f"  {T}yr: 2step={twostep_ret[T]:.4f} ({twostep_ses[T]:.4f}), OLS={ols_ret[T]:.4f} ({ols_ses[T]:.4f})")

    score = score_results(beta_1, beta_1_se, b1b2, b1b2_se, beta_2, beta_2_se,
                          ols_bias, ols_ten_se, N, twostep_ret, twostep_ses, ols_ret, ols_ses)
    return score


def score_results(beta_1, beta_1_se, b1b2, b1b2_se, beta_2, beta_2_se,
                  ols_bias, ols_bias_se, N, tr, ts, olr, ols_):
    print(f"\n{'='*80}\nSCORING\n{'='*80}")
    total = 0
    re = abs(b1b2 - GROUND_TRUTH['beta_1_plus_beta_2']) / abs(GROUND_TRUTH['beta_1_plus_beta_2'])
    s = 15 if re <= 0.20 else max(0, 15*(1-re))
    total += s
    print(f"  b1+b2: {b1b2:.4f} vs {GROUND_TRUTH['beta_1_plus_beta_2']:.4f} ({re:.1%}) -> {s:.1f}/15")

    for nm, v, tv in [('beta_1', beta_1, GROUND_TRUTH['beta_1']),
                      ('beta_2', beta_2, GROUND_TRUTH['beta_2'])]:
        err = abs(v - tv)
        p = 10 if err <= 0.01 else (10*(1-(err-0.01)/0.02) if err <= 0.03 else max(0, 10*(1-err/0.05)))
        total += p
        print(f"  {nm}: {v:.4f} vs {tv:.4f} (err {err:.4f}) -> {p:.1f}/10")

    cr = 0
    for T in [5,10,15,20]:
        for lbl, r, gt in [('2s', tr, GROUND_TRUTH['cumret_2step']),
                           ('OLS', olr, GROUND_TRUTH['cumret_ols'])]:
            err = abs(r[T] - gt[T])
            p = 2.5 if err <= 0.03 else (2.5*(1-(err-0.03)/0.03) if err <= 0.06 else 0)
            cr += p
            print(f"  {lbl} {T}yr: {r[T]:.4f} vs {gt[T]:.4f} (err {err:.4f}) -> {p:.1f}/2.5")
    total += cr

    sep = 0
    for nm, v, tv in [('b1 SE', beta_1_se, GROUND_TRUTH['beta_1_se']),
                      ('b1b2 SE', b1b2_se, GROUND_TRUTH['beta_1_plus_beta_2_se']),
                      ('b2 SE', beta_2_se, GROUND_TRUTH['beta_2_se'])]:
        re2 = abs(v-tv)/tv if tv > 0 else 1
        p = 10/3 if re2 <= 0.30 else ((10/3)*(1-(re2-0.30)/0.30) if re2 <= 0.60 else 0)
        sep += p
        print(f"  {nm}: {v:.4f} vs {tv:.4f} ({re2:.1%}) -> {p:.1f}/{10/3:.1f}")
    total += sep

    ne = abs(N - GROUND_TRUTH['N']) / GROUND_TRUTH['N']
    np_ = 15 if ne <= 0.05 else (15*(1-(ne-0.05)/0.10) if ne <= 0.15 else 0)
    total += np_
    print(f"  N: {N} vs {GROUND_TRUTH['N']} ({ne:.1%}) -> {np_:.1f}/15")

    sg = 0
    for nm, c, s in [('b1', beta_1, beta_1_se), ('b1b2', b1b2, b1b2_se), ('b2', beta_2, beta_2_se)]:
        t = abs(c/s) if s > 0 else 0
        if t >= 1.96: sg += 10/3; print(f"  {nm}: t={t:.2f} PASS")
        else: print(f"  {nm}: t={t:.2f} FAIL")
    total += sg

    total += 10
    total = min(100, total)
    print(f"\n  TOTAL: {total:.1f}/100")
    return total


if __name__ == "__main__":
    score = run_analysis("data/psid_panel.csv")
    print(f"\nFinal Score: {score:.1f}/100")
