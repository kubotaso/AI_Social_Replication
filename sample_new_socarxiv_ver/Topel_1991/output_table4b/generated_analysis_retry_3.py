#!/usr/bin/env python3
"""
Table 4B Replication (Attempt 3): Returns to Job Seniority Based on Various
Remaining Job Durations in First-Step Model

Topel (1991) - "Specific Capital, Mobility, and Wages"

Key fixes from attempt 2:
- Use linearmodels IV2SLS for proper IV estimation
- Fix beta_1 estimation (was getting ~0.02, should be ~0.07)
- The issue was likely in the manual 2SLS computation
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'psid_panel.csv')

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

EDUC_CAT_TO_YEARS = {
    0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17, 9: 17
}

GROUND_TRUTH = {
    'beta_1': {0: 0.0713, 1: 0.0792, 3: 0.0716, 5: 0.0607},
    'beta_1_se': {0: 0.0181, 1: 0.0204, 3: 0.0245, 5: 0.0292},
    'beta_2': {0: 0.0545, 1: 0.0546, 3: 0.0559, 5: 0.0584},
    'beta_2_se': {0: 0.0079, 1: 0.0089, 3: 0.0109, 5: 0.0132},
    'cum_5': {0: 0.1793, 1: 0.1725, 3: 0.1703, 5: 0.1815},
    'cum_5_se': {0: 0.0235, 1: 0.0265, 3: 0.0319, 5: 0.0379},
    'cum_10': {0: 0.2459, 1: 0.2235, 3: 0.2181, 5: 0.2330},
    'cum_10_se': {0: 0.0341, 1: 0.0376, 3: 0.0437, 5: 0.0514},
    'cum_15': {0: 0.2832, 1: 0.2439, 3: 0.2503, 5: 0.2565},
    'cum_15_se': {0: 0.0411, 1: 0.0445, 3: 0.0504, 5: 0.0594},
    'cum_20': {0: 0.3375, 1: 0.2865, 3: 0.3232, 5: 0.3066},
    'cum_20_se': {0: 0.0438, 1: 0.0469, 3: 0.0531, 5: 0.0647},
}


def run_analysis(data_source=DATA_FILE):
    """Run Table 4B replication."""
    print("=" * 60)
    print("TABLE 4B REPLICATION (Attempt 3)")
    print("=" * 60)

    # =========================================================================
    # Load and prepare data
    # =========================================================================
    df = pd.read_csv(data_source)

    # Exclude Alaska/Hawaii
    df = df[~df['region'].isin([5, 6])].copy()

    # Education recoding
    df['education_years'] = df['education_clean'].copy()
    cat_mask = ~df['year'].isin([1975, 1976])
    df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT_TO_YEARS)
    df = df.dropna(subset=['education_years'])

    # Recompute experience
    df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)

    # Real wages
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
    df['log_real_wage_cps'] = df['log_hourly_wage'] - np.log(df['cps_index'])

    # For step 2 (levels): deflate by GNP deflator and CPS index
    df['gnp_deflator'] = (df['year'] - 1).map(GNP_DEFLATOR)
    df['log_real_wage_full'] = (df['log_hourly_wage']
                                 - np.log(df['gnp_deflator'] / 100.0)
                                 - np.log(df['cps_index']))

    # Sort
    df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)

    # Job structure
    df['last_year_in_job'] = df.groupby(['person_id', 'job_id'])['year'].transform('max')
    df['remaining_duration'] = df['last_year_in_job'] - df['year']

    # Initial experience
    df['initial_experience'] = (df['experience'] - df['tenure_topel']).clip(lower=0)

    # Fill missing controls with 0
    for c in ['married', 'union_member', 'disabled', 'lives_in_smsa',
              'region_ne', 'region_nc', 'region_south', 'region_west']:
        df[c] = df[c].fillna(0)

    # =========================================================================
    # Within-job first differences
    # =========================================================================
    df['prev_year'] = df.groupby(['person_id', 'job_id'])['year'].shift(1)
    df['prev_log_real_wage_cps'] = df.groupby(['person_id', 'job_id'])['log_real_wage_cps'].shift(1)
    df['prev_tenure'] = df.groupby(['person_id', 'job_id'])['tenure_topel'].shift(1)
    df['prev_experience'] = df.groupby(['person_id', 'job_id'])['experience'].shift(1)

    within_job = df[
        (df['prev_year'].notna()) &
        (df['year'] - df['prev_year'] == 1)
    ].copy()

    within_job['d_log_wage'] = within_job['log_real_wage_cps'] - within_job['prev_log_real_wage_cps']
    within_job = within_job[within_job['d_log_wage'].between(-2, 2)].copy()

    # Polynomial first differences
    t = within_job['tenure_topel']
    pt = within_job['prev_tenure']
    x = within_job['experience']
    px = within_job['prev_experience']

    within_job['d_tenure'] = t - pt
    within_job['d_tenure_sq'] = t**2 - pt**2
    within_job['d_tenure_cu'] = t**3 - pt**3
    within_job['d_tenure_qu'] = t**4 - pt**4
    within_job['d_exp_sq'] = x**2 - px**2
    within_job['d_exp_cu'] = x**3 - px**3
    within_job['d_exp_qu'] = x**4 - px**4

    print(f"Within-job obs: {len(within_job)}")
    print(f"Mean d_log_wage: {within_job['d_log_wage'].mean():.4f}")

    # =========================================================================
    # Step 2 sample (full person-year observations)
    # =========================================================================
    step2_df = df.dropna(subset=['log_real_wage_full', 'experience',
                                  'education_years', 'tenure_topel',
                                  'initial_experience']).copy()
    step2_df = step2_df.reset_index(drop=True)
    print(f"Step 2 sample: {len(step2_df)} obs")

    # =========================================================================
    # Two-step estimation for each threshold
    # =========================================================================
    thresholds = [0, 1, 3, 5]
    results = {}

    for threshold in thresholds:
        print(f"\n{'='*50}")
        print(f"  Remaining duration >= {threshold}")
        print(f"{'='*50}")

        # =============================================
        # Step 1: First-differenced regression
        # =============================================
        if threshold == 0:
            s1 = within_job.copy()
        else:
            s1 = within_job[within_job['remaining_duration'] >= threshold].copy()

        y1 = s1['d_log_wage']
        X1_vars = ['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                    'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
        X1 = s1[X1_vars].copy()

        # Year dummies
        yr_dum1 = pd.get_dummies(s1['year'], prefix='yr', dtype=float)
        yr_cols1 = sorted(yr_dum1.columns.tolist())
        if yr_cols1:
            yr_cols1 = yr_cols1[1:]
        for c in yr_cols1:
            X1[c] = yr_dum1[c].values

        X1 = sm.add_constant(X1)
        valid1 = X1.notna().all(axis=1) & y1.notna()
        model1 = sm.OLS(y1[valid1], X1[valid1]).fit()

        beta_hat = model1.params['d_tenure']
        beta_hat_se = model1.bse['d_tenure']
        gamma2 = model1.params['d_tenure_sq']
        gamma3 = model1.params['d_tenure_cu']
        gamma4 = model1.params['d_tenure_qu']
        delta2 = model1.params['d_exp_sq']
        delta3 = model1.params['d_exp_cu']
        delta4 = model1.params['d_exp_qu']

        print(f"  Step 1 N: {int(model1.nobs)}")
        print(f"  beta_hat (b1+b2): {beta_hat:.4f} ({beta_hat_se:.4f})")

        # =============================================
        # Step 2: IV regression of adjusted wage
        # =============================================
        s2 = step2_df.copy()
        T_vals = s2['tenure_topel'].values.astype(float)
        X_vals = s2['experience'].values.astype(float)

        # Construct adjusted wage: w* = log_wage - (b1+b2)*T - gamma*T^2 - ... - delta*X^2 - ...
        s2['w_star'] = (s2['log_real_wage_full'].values
                        - beta_hat * T_vals
                        - gamma2 * T_vals**2
                        - gamma3 * T_vals**3
                        - gamma4 * T_vals**4
                        - delta2 * X_vals**2
                        - delta3 * X_vals**3
                        - delta4 * X_vals**4)

        # Use linearmodels IV2SLS
        # dependent: w_star
        # endogenous: experience (instrumented by initial_experience)
        # exogenous: education, married, union, disabled, smsa, region dummies, year dummies, constant

        # Drop lives_in_smsa (all zeros) and region_west (reference category)
        # to avoid rank deficiency with the constant
        controls = ['education_years', 'married', 'union_member', 'disabled',
                     'region_ne', 'region_nc', 'region_south']

        # Year dummies for step 2
        yr_dum2 = pd.get_dummies(s2['year'], prefix='yr', dtype=float)
        yr_cols2 = sorted(yr_dum2.columns.tolist())
        if yr_cols2:
            yr_cols2 = yr_cols2[1:]

        # Build exogenous controls dataframe
        exog_df = s2[controls].copy()
        for c in yr_cols2:
            exog_df[c] = yr_dum2[c].values
        exog_df = sm.add_constant(exog_df)

        # Endogenous variable
        endog = s2[['experience']]

        # Instrument
        instruments = s2[['initial_experience']]

        # Drop any NaN rows
        all_data = pd.concat([pd.DataFrame({'w_star': s2['w_star']}),
                               endog, exog_df, instruments], axis=1)
        valid2 = all_data.notna().all(axis=1)
        n2 = valid2.sum()

        print(f"  Step 2 N: {n2}")

        # Run IV2SLS
        iv_model = IV2SLS(
            dependent=s2.loc[valid2, 'w_star'],
            exog=exog_df[valid2],
            endog=endog[valid2],
            instruments=instruments[valid2]
        ).fit()

        beta_1 = iv_model.params['experience']
        beta_1_se = iv_model.std_errors['experience']

        beta_2 = beta_hat - beta_1
        beta_2_se = np.sqrt(beta_hat_se**2 + beta_1_se**2)

        print(f"  beta_1 (experience): {beta_1:.4f} ({beta_1_se:.4f})")
        print(f"  beta_2 (tenure): {beta_2:.4f} ({beta_2_se:.4f})")

        # First stage diagnostics
        print(f"  IV model summary (first few params):")
        for pname in ['experience', 'education_years', 'married']:
            if pname in iv_model.params.index:
                print(f"    {pname}: {iv_model.params[pname]:.4f} ({iv_model.std_errors[pname]:.4f})")

        # Cumulative returns
        cum_returns = {}
        cum_se_dict = {}
        cov1 = model1.cov_params()

        for T_yrs in [5, 10, 15, 20]:
            cum_ret = (beta_2 * T_yrs
                       + gamma2 * T_yrs**2
                       + gamma3 * T_yrs**3
                       + gamma4 * T_yrs**4)
            cum_returns[T_yrs] = cum_ret

            # Delta method SE
            var_b2 = beta_2_se**2
            var_g2 = cov1.loc['d_tenure_sq', 'd_tenure_sq']
            var_g3 = cov1.loc['d_tenure_cu', 'd_tenure_cu']
            var_g4 = cov1.loc['d_tenure_qu', 'd_tenure_qu']
            cov_g2g3 = cov1.loc['d_tenure_sq', 'd_tenure_cu']
            cov_g2g4 = cov1.loc['d_tenure_sq', 'd_tenure_qu']
            cov_g3g4 = cov1.loc['d_tenure_cu', 'd_tenure_qu']

            var_cum = (T_yrs**2 * var_b2
                       + T_yrs**4 * var_g2
                       + T_yrs**6 * var_g3
                       + T_yrs**8 * var_g4
                       + 2 * T_yrs**5 * cov_g2g3
                       + 2 * T_yrs**6 * cov_g2g4
                       + 2 * T_yrs**7 * cov_g3g4)

            cum_se_dict[T_yrs] = np.sqrt(max(var_cum, 0))

        print(f"  Cum: 5yr={cum_returns[5]:.4f}, 10yr={cum_returns[10]:.4f}, "
              f"15yr={cum_returns[15]:.4f}, 20yr={cum_returns[20]:.4f}")

        results[threshold] = {
            'beta_1': beta_1,
            'beta_1_se': beta_1_se,
            'beta_2': beta_2,
            'beta_2_se': beta_2_se,
            'beta_hat': beta_hat,
            'beta_hat_se': beta_hat_se,
            'cum_returns': cum_returns,
            'cum_se': cum_se_dict,
            'gamma2': gamma2,
            'gamma3': gamma3,
            'gamma4': gamma4,
            'step1_n': int(model1.nobs),
            'step2_n': n2,
        }

    # =========================================================================
    # Format output
    # =========================================================================
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("TABLE 4B: Returns to Job Seniority Based on Various")
    output_lines.append("Remaining Job Durations in First-Step Model")
    output_lines.append("=" * 80)
    output_lines.append("")

    output_lines.append("TOP PANEL: Main Effects (standard errors in parentheses)")
    output_lines.append(f"{'':25s} {'>=0':>15s} {'>=1':>15s} {'>=3':>15s} {'>=5':>15s}")
    output_lines.append("-" * 85)

    row_b1 = f"{'Experience (beta_1)':25s}"
    row_b2 = f"{'Tenure (beta_2)':25s}"
    for t in thresholds:
        r = results[t]
        row_b1 += f" {r['beta_1']:>7.4f} ({r['beta_1_se']:.4f})"
        row_b2 += f" {r['beta_2']:>7.4f} ({r['beta_2_se']:.4f})"
    output_lines.append(row_b1)
    output_lines.append(row_b2)
    output_lines.append("")

    output_lines.append("BOTTOM PANEL: Estimated Tenure Profile (standard errors in parentheses)")
    output_lines.append(f"{'':25s} {'>=0':>15s} {'>=1':>15s} {'>=3':>15s} {'>=5':>15s}")
    output_lines.append("-" * 85)

    for T_yrs in [5, 10, 15, 20]:
        row = f"{f'{T_yrs} years':25s}"
        for t in thresholds:
            r = results[t]
            row += f" {r['cum_returns'][T_yrs]:>7.4f} ({r['cum_se'][T_yrs]:.4f})"
        output_lines.append(row)
    output_lines.append("")

    output_lines.append("SAMPLE SIZES:")
    for t in thresholds:
        r = results[t]
        output_lines.append(f"  >= {t}: Step 1 N = {r['step1_n']}, Step 2 N = {r['step2_n']}")

    output_lines.append("")
    output_lines.append("GROUND TRUTH (Paper):")
    output_lines.append(f"{'':25s} {'>=0':>15s} {'>=1':>15s} {'>=3':>15s} {'>=5':>15s}")
    output_lines.append("Paper beta_1:            " +
        "".join(f" {GROUND_TRUTH['beta_1'][t]:>7.4f} ({GROUND_TRUTH['beta_1_se'][t]:.4f})" for t in thresholds))
    output_lines.append("Paper beta_2:            " +
        "".join(f" {GROUND_TRUTH['beta_2'][t]:>7.4f} ({GROUND_TRUTH['beta_2_se'][t]:.4f})" for t in thresholds))
    for T_yrs in [5, 10, 15, 20]:
        gt_key = f'cum_{T_yrs}'
        gt_se_key = f'cum_{T_yrs}_se'
        row = f"Paper {T_yrs}yr:               "
        row += "".join(f" {GROUND_TRUTH[gt_key][t]:>7.4f} ({GROUND_TRUTH[gt_se_key][t]:.4f})" for t in thresholds)
        output_lines.append(row)

    output = "\n".join(output_lines)
    print("\n" + output)

    return output, results


def score_against_ground_truth(results=None):
    """Score results against ground truth."""
    if results is None:
        _, results = run_analysis()

    thresholds = [0, 1, 3, 5]

    print("\n" + "=" * 80)
    print("SCORING AGAINST GROUND TRUTH")
    print("=" * 80)

    scores = {}

    # 1. Step 2 coefficients (20 pts)
    print("\n--- Step 2 Coefficients [20 pts] ---")
    coef_pass = 0
    for t in thresholds:
        for name in ['beta_1', 'beta_2']:
            gt = GROUND_TRUTH[name][t]
            gen = results[t][name]
            diff = abs(gen - gt)
            match = diff <= 0.01
            if match:
                coef_pass += 1
            print(f"  >={t} {name}: gen={gen:.4f}, gt={gt:.4f}, diff={diff:.4f} {'PASS' if match else 'FAIL'}")
    scores['step2_coef'] = (coef_pass / 8) * 20

    # 2. Cumulative returns (20 pts)
    print("\n--- Cumulative Returns [20 pts] ---")
    cum_pass = 0
    for t in thresholds:
        for T_yrs in [5, 10, 15, 20]:
            gt = GROUND_TRUTH[f'cum_{T_yrs}'][t]
            gen = results[t]['cum_returns'][T_yrs]
            diff = abs(gen - gt)
            match = diff <= 0.03
            if match:
                cum_pass += 1
            print(f"  >={t} {T_yrs}yr: gen={gen:.4f}, gt={gt:.4f}, diff={diff:.4f} {'PASS' if match else 'FAIL'}")
    scores['cum_returns'] = (cum_pass / 16) * 20

    # 3. Step 1 coefficients (15 pts)
    print("\n--- Step 1 Coefficients [15 pts] ---")
    s1_pass = 0
    for t in thresholds:
        gt_b1b2 = GROUND_TRUTH['beta_1'][t] + GROUND_TRUTH['beta_2'][t]
        gen_b1b2 = results[t]['beta_hat']
        rel_err = abs(gen_b1b2 - gt_b1b2) / max(abs(gt_b1b2), 0.001)
        match = rel_err <= 0.20
        if match:
            s1_pass += 1
        print(f"  >={t}: gen={gen_b1b2:.4f}, gt={gt_b1b2:.4f}, rel_err={rel_err:.3f} {'PASS' if match else 'FAIL'}")
    scores['step1_coef'] = (s1_pass / 4) * 15

    # 4. Standard errors (10 pts)
    print("\n--- Standard Errors [10 pts] ---")
    se_pass = 0
    for t in thresholds:
        for name in ['beta_1_se', 'beta_2_se']:
            gt = GROUND_TRUTH[name][t]
            gen = results[t][name]
            rel_err = abs(gen - gt) / max(gt, 0.001)
            match = rel_err <= 0.30
            if match:
                se_pass += 1
            print(f"  >={t} {name}: gen={gen:.4f}, gt={gt:.4f}, rel_err={rel_err:.3f} {'PASS' if match else 'FAIL'}")
    scores['std_errors'] = (se_pass / 8) * 10

    # 5. Sample size (15 pts)
    print("\n--- Sample Size [15 pts] ---")
    gt_n = 10685
    gen_n = results[0]['step2_n']
    rel_err_n = abs(gen_n - gt_n) / gt_n
    n_match = rel_err_n <= 0.05
    scores['sample_size'] = 15 if n_match else max(0, 15 * max(0, 1 - rel_err_n / 0.10))
    print(f"  Step 2 N: gen={gen_n}, gt={gt_n}, rel_err={rel_err_n:.3f} {'PASS' if n_match else 'FAIL'}")

    # 6. Significance (10 pts)
    print("\n--- Significance [10 pts] ---")
    sig_pass = 0
    for t in thresholds:
        for name in ['beta_1', 'beta_2']:
            gt = GROUND_TRUTH[name][t]
            gen = results[t][name]
            gt_se = GROUND_TRUTH[f'{name}_se'][t]
            gen_se = results[t][f'{name}_se']
            gt_sig = abs(gt / gt_se) > 1.96 if gt_se > 0 else False
            gen_sig = abs(gen / gen_se) > 1.96 if gen_se > 0 else False
            match = (gt > 0) == (gen > 0) and gt_sig == gen_sig
            if match:
                sig_pass += 1
            print(f"  >={t} {name}: gen={gen:.4f}({'*' if gen_sig else ' '}), gt={gt:.4f}({'*' if gt_sig else ' '}) {'PASS' if match else 'FAIL'}")
    scores['significance'] = (sig_pass / 8) * 10

    # 7. All columns (10 pts)
    scores['columns'] = 10 if all(t in results for t in thresholds) else 0

    total = sum(scores.values())
    total_score = int(round(total))

    print(f"\n{'='*40}")
    print("SCORE BREAKDOWN:")
    for k, v in scores.items():
        print(f"  {k}: {v:.1f}")
    print(f"TOTAL SCORE: {total_score}/100")
    print(f"{'='*40}")

    return total_score


if __name__ == '__main__':
    output, results = run_analysis()
    score = score_against_ground_truth(results)
