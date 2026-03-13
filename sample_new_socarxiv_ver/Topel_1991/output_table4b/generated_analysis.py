#!/usr/bin/env python3
"""
Table 4B Replication: Returns to Job Seniority Based on Various Remaining
Job Durations in First-Step Model

Topel (1991) - "Specific Capital, Mobility, and Wages"

Two-step estimation where the first step is restricted by minimum remaining
job duration: >=0 (baseline), >=1, >=3, >=5.
The second step always uses the full sample.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import os

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'psid_panel.csv')

# CPS Real Wage Index
CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

# GNP Price Deflator (base 1982=100)
GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}

# Education category -> years mapping
EDUC_CAT_TO_YEARS = {
    0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17
}

# Ground truth from paper
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

    # =========================================================================
    # Load and prepare data
    # =========================================================================
    print("Loading PSID panel data...")
    df = pd.read_csv(data_source)
    print(f"  Raw panel: {len(df)} observations, {df['person_id'].nunique()} persons")

    # =========================================================================
    # Education recoding
    # =========================================================================
    df['education_years'] = df['education_clean'].copy()
    # 1975-1976: already in years; other years: categorical (0-8)
    cat_mask = ~df['year'].isin([1975, 1976])
    df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT_TO_YEARS)

    # Recompute experience
    df['experience'] = df['age'] - df['education_years'] - 6
    df['experience'] = df['experience'].clip(lower=0)

    # =========================================================================
    # Construct real wages (two versions)
    # =========================================================================
    # For first step (first differences): deflate by CPS wage index only
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
    df['log_real_wage_cps'] = df['log_hourly_wage'] - np.log(df['cps_index'])

    # For second step (levels): deflate by both GNP deflator and CPS wage index
    # PSID year Y reports income for year Y-1
    df['gnp_deflator'] = (df['year'] - 1).map(GNP_DEFLATOR)
    df['log_real_wage_full'] = (df['log_hourly_wage']
                                 - np.log(df['gnp_deflator'] / 100.0)
                                 - np.log(df['cps_index']))

    # =========================================================================
    # Sort and prepare within-job structure
    # =========================================================================
    df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)

    # Compute last year in each job for remaining duration
    df['last_year_in_job'] = df.groupby(['person_id', 'job_id'])['year'].transform('max')
    df['remaining_duration'] = df['last_year_in_job'] - df['year']

    # =========================================================================
    # Construct within-job first differences for Step 1
    # =========================================================================
    print("\nConstructing within-job first differences...")

    df['prev_year'] = df.groupby(['person_id', 'job_id'])['year'].shift(1)
    df['prev_log_real_wage_cps'] = df.groupby(['person_id', 'job_id'])['log_real_wage_cps'].shift(1)
    df['prev_tenure'] = df.groupby(['person_id', 'job_id'])['tenure_topel'].shift(1)
    df['prev_experience'] = df.groupby(['person_id', 'job_id'])['experience'].shift(1)

    # Keep only consecutive within-job observations
    within_job = df[
        (df['prev_year'].notna()) &
        (df['year'] - df['prev_year'] == 1)
    ].copy()

    # Compute first differences
    within_job['d_log_wage'] = within_job['log_real_wage_cps'] - within_job['prev_log_real_wage_cps']

    # Drop extreme outliers
    within_job = within_job[within_job['d_log_wage'].between(-2, 2)].copy()

    print(f"  Within-job obs (all): {len(within_job)}")

    # =========================================================================
    # Construct regressors for Step 1
    # =========================================================================
    tenure = within_job['tenure_topel']
    prev_tenure = within_job['prev_tenure']
    exp = within_job['experience']
    prev_exp = within_job['prev_experience']

    within_job['d_tenure'] = tenure - prev_tenure  # Should be 1

    # Polynomial differences
    within_job['d_tenure_sq'] = tenure**2 - prev_tenure**2
    within_job['d_tenure_cu'] = tenure**3 - prev_tenure**3
    within_job['d_tenure_qu'] = tenure**4 - prev_tenure**4

    within_job['d_exp_sq'] = exp**2 - prev_exp**2
    within_job['d_exp_cu'] = exp**3 - prev_exp**3
    within_job['d_exp_qu'] = exp**4 - prev_exp**4

    # Year dummies
    year_dummies_wj = pd.get_dummies(within_job['year'], prefix='yr', dtype=float)
    yr_cols = [c for c in year_dummies_wj.columns if c != 'yr_1971']

    # =========================================================================
    # Prepare Step 2 sample (FULL sample of wage levels)
    # =========================================================================
    print("\nPreparing Step 2 (full) sample...")

    # The full sample for step 2 includes all person-year observations with valid data
    step2_df = df.copy()

    # Apply sample restrictions
    step2_df = step2_df[step2_df['tenure_topel'] >= 1].copy()
    step2_df = step2_df[step2_df['age'].between(18, 60)].copy()
    step2_df = step2_df.dropna(subset=['log_real_wage_full', 'experience', 'education_years', 'tenure_topel'])

    # Initial experience (experience at start of current job)
    step2_df['initial_experience'] = step2_df['experience'] - step2_df['tenure_topel']
    step2_df['initial_experience'] = step2_df['initial_experience'].clip(lower=0)

    # Controls for step 2
    step2_df['married'] = step2_df['married'].fillna(0)
    step2_df['union_member'] = step2_df['union_member'].fillna(0)
    step2_df['disabled'] = step2_df['disabled'].fillna(0)
    step2_df['lives_in_smsa'] = step2_df['lives_in_smsa'].fillna(0)

    # Region dummies (fill NAs)
    for rc in ['region_ne', 'region_nc', 'region_south', 'region_west']:
        step2_df[rc] = step2_df[rc].fillna(0)

    # Year dummies for step 2
    year_dummies_s2 = pd.get_dummies(step2_df['year'], prefix='yr', dtype=float)
    yr_cols_s2 = [c for c in year_dummies_s2.columns if c != 'yr_1971']

    print(f"  Step 2 sample size: {len(step2_df)}")

    # =========================================================================
    # Two-step estimation for each remaining duration threshold
    # =========================================================================
    thresholds = [0, 1, 3, 5]
    results = {}

    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"  Remaining duration >= {threshold}")
        print(f"{'='*60}")

        # Step 1: Restrict first-step sample
        if threshold == 0:
            step1_sample = within_job.copy()
        else:
            # remaining_duration for within-job obs: use the remaining_duration from original df
            step1_sample = within_job[within_job['remaining_duration'] >= threshold].copy()

        print(f"  Step 1 sample: {len(step1_sample)} obs")

        # Step 1 regression: quartic tenure + quartic experience + year dummies
        y1 = step1_sample['d_log_wage']
        X1_vars = ['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                    'd_exp_sq', 'd_exp_cu', 'd_exp_qu']

        # Get year dummies for this subsample
        yr_dum_s1 = pd.get_dummies(step1_sample['year'], prefix='yr', dtype=float)
        yr_cols_s1 = [c for c in yr_dum_s1.columns if c != 'yr_1971']

        X1 = step1_sample[X1_vars].copy()
        for c in yr_cols_s1:
            if c in yr_dum_s1.columns:
                X1[c] = yr_dum_s1[c].values
        X1 = sm.add_constant(X1)

        valid1 = X1.notna().all(axis=1) & y1.notna()
        model1 = sm.OLS(y1[valid1], X1[valid1]).fit()

        print(f"  Step 1 N: {int(model1.nobs)}")

        # Extract first-step coefficients
        beta_hat = model1.params.get('d_tenure', 0)  # beta_1 + beta_2
        gamma2 = model1.params.get('d_tenure_sq', 0)
        gamma3 = model1.params.get('d_tenure_cu', 0)
        gamma4 = model1.params.get('d_tenure_qu', 0)
        delta2 = model1.params.get('d_exp_sq', 0)
        delta3 = model1.params.get('d_exp_cu', 0)
        delta4 = model1.params.get('d_exp_qu', 0)

        print(f"  beta_hat (b1+b2): {beta_hat:.4f}")
        print(f"  gamma2: {gamma2:.6f}, gamma3: {gamma3:.8f}, gamma4: {gamma4:.10f}")
        print(f"  delta2: {delta2:.6f}, delta3: {delta3:.8f}, delta4: {delta4:.10f}")

        # Step 2: Construct adjusted wage on FULL sample
        T = step2_df['tenure_topel'].values
        X = step2_df['experience'].values

        # Adjusted wage: subtract higher-order terms and linear tenure effect
        w_star = (step2_df['log_real_wage_full'].values
                  - beta_hat * T
                  - gamma2 * T**2
                  - gamma3 * T**3
                  - gamma4 * T**4
                  - delta2 * X**2
                  - delta3 * X**3
                  - delta4 * X**4)

        # Build Step 2 design matrix
        # Endogenous: current experience
        # Instrument: initial experience
        # Controls: education, married, union, disabled, smsa, region dummies, year dummies
        controls = ['education_years', 'married', 'union_member', 'disabled', 'lives_in_smsa',
                     'region_ne', 'region_nc', 'region_south', 'region_west']

        X2_endo = step2_df[['experience']].copy()
        X2_controls = step2_df[controls].copy()
        for c in yr_cols_s2:
            if c in year_dummies_s2.columns:
                X2_controls[c] = year_dummies_s2[c].values

        X2_controls = sm.add_constant(X2_controls)

        # Instruments: initial_experience + all controls
        Z = X2_controls.copy()
        Z['initial_experience'] = step2_df['initial_experience'].values

        # Valid observations
        valid2 = (pd.Series(w_star).notna() &
                  X2_endo.notna().all(axis=1).values &
                  X2_controls.notna().all(axis=1).values &
                  Z.notna().all(axis=1).values)

        w_star_valid = w_star[valid2]
        X2_full = pd.concat([X2_endo[valid2].reset_index(drop=True),
                              X2_controls[valid2].reset_index(drop=True)], axis=1)
        Z_full = Z[valid2].reset_index(drop=True)

        print(f"  Step 2 N: {valid2.sum()}")

        # IV regression using 2SLS
        # First stage: regress experience on initial_experience + controls
        first_stage = sm.OLS(X2_full['experience'], Z_full).fit()
        X2_full_hat = X2_full.copy()
        X2_full_hat['experience'] = first_stage.fittedvalues

        # Second stage: regress w_star on fitted experience + controls
        second_stage = sm.OLS(w_star_valid, X2_full_hat).fit()

        # Get correct IV standard errors
        # Using manual 2SLS SE correction
        # Residuals from second stage but using actual X (not fitted)
        resid_iv = w_star_valid - X2_full.values @ second_stage.params.values
        n = len(w_star_valid)
        k = X2_full.shape[1]
        sigma2_iv = np.sum(resid_iv**2) / (n - k)

        # IV variance: sigma2 * (Z'X)^{-1} (Z'Z) (X'Z)^{-1}
        # But for standard 2SLS: Var = sigma2 * (X_hat' X_hat)^{-1} X_hat' X X_hat (X_hat' X_hat)^{-1}
        # Simplified: sigma2 * (X'P_Z X)^{-1}
        X_mat = X2_full.values
        Z_mat = Z_full.values
        P_Z = Z_mat @ np.linalg.inv(Z_mat.T @ Z_mat) @ Z_mat.T
        bread = np.linalg.inv(X_mat.T @ P_Z @ X_mat)
        iv_var = sigma2_iv * bread
        iv_se = np.sqrt(np.diag(iv_var))

        beta_1 = second_stage.params['experience']
        beta_1_se = iv_se[0]  # SE for experience coefficient

        beta_2 = beta_hat - beta_1
        # SE for beta_2: sqrt(var(beta_hat) + var(beta_1))
        beta_hat_se = model1.bse.get('d_tenure', 0)
        beta_2_se = np.sqrt(beta_hat_se**2 + beta_1_se**2)

        print(f"  beta_1 (experience): {beta_1:.4f} ({beta_1_se:.4f})")
        print(f"  beta_2 (tenure): {beta_2:.4f} ({beta_2_se:.4f})")

        # Cumulative returns
        cum_returns = {}
        cum_se = {}
        for T_yrs in [5, 10, 15, 20]:
            cum_ret = (beta_2 * T_yrs
                       + gamma2 * T_yrs**2
                       + gamma3 * T_yrs**3
                       + gamma4 * T_yrs**4)
            cum_returns[T_yrs] = cum_ret

            # SE via delta method: partial derivatives w.r.t. beta_2 and gammas
            # d/d(beta_2) = T_yrs
            # Approximate SE using beta_2 SE scaled by T_yrs
            # Plus contribution from gamma terms (from step 1)
            # Simplified: SE(cum) = sqrt( T^2 * var(beta2) + ... )
            # For now, use a simplified approach
            # The variance of cum return involves var(beta_2), var(gamma2), var(gamma3), var(gamma4)
            # and covariances. We use the step1 variance-covariance matrix for gammas.
            var_beta2 = beta_2_se**2
            var_g2 = model1.cov_params().loc['d_tenure_sq', 'd_tenure_sq'] if 'd_tenure_sq' in model1.cov_params().index else 0
            var_g3 = model1.cov_params().loc['d_tenure_cu', 'd_tenure_cu'] if 'd_tenure_cu' in model1.cov_params().index else 0
            var_g4 = model1.cov_params().loc['d_tenure_qu', 'd_tenure_qu'] if 'd_tenure_qu' in model1.cov_params().index else 0

            se_cum = np.sqrt(T_yrs**2 * var_beta2
                             + T_yrs**4 * var_g2
                             + T_yrs**6 * var_g3
                             + T_yrs**8 * var_g4)
            cum_se[T_yrs] = se_cum

        print(f"  Cumulative returns: 5yr={cum_returns[5]:.4f}, 10yr={cum_returns[10]:.4f}, "
              f"15yr={cum_returns[15]:.4f}, 20yr={cum_returns[20]:.4f}")

        results[threshold] = {
            'beta_1': beta_1,
            'beta_1_se': beta_1_se,
            'beta_2': beta_2,
            'beta_2_se': beta_2_se,
            'beta_hat': beta_hat,
            'beta_hat_se': beta_hat_se,
            'cum_returns': cum_returns,
            'cum_se': cum_se,
            'step1_n': int(model1.nobs),
            'step2_n': int(valid2.sum()),
            'gamma2': gamma2,
            'gamma3': gamma3,
            'gamma4': gamma4,
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

    # Top panel
    output_lines.append("TOP PANEL: Main Effects")
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

    # Bottom panel
    output_lines.append("BOTTOM PANEL: Estimated Tenure Profile")
    output_lines.append(f"{'':25s} {'>=0':>15s} {'>=1':>15s} {'>=3':>15s} {'>=5':>15s}")
    output_lines.append("-" * 85)

    for T_yrs in [5, 10, 15, 20]:
        row = f"{f'{T_yrs} years':25s}"
        for t in thresholds:
            r = results[t]
            row += f" {r['cum_returns'][T_yrs]:>7.4f} ({r['cum_se'][T_yrs]:.4f})"
        output_lines.append(row)
    output_lines.append("")

    # Sample sizes
    output_lines.append("Sample Sizes:")
    for t in thresholds:
        r = results[t]
        output_lines.append(f"  >= {t}: Step 1 N = {r['step1_n']}, Step 2 N = {r['step2_n']}")

    # Step 1 coefficients
    output_lines.append("")
    output_lines.append("Step 1 Coefficients (from first-differenced regression):")
    for t in thresholds:
        r = results[t]
        output_lines.append(f"  >= {t}: beta_hat={r['beta_hat']:.4f} ({r['beta_hat_se']:.4f}), "
                            f"gamma2={r['gamma2']:.6f}, gamma3={r['gamma3']:.8f}, gamma4={r['gamma4']:.10f}")

    output = "\n".join(output_lines)
    print("\n" + output)

    return output, results


def score_against_ground_truth(results=None):
    """Score results against ground truth from the paper."""
    if results is None:
        _, results = run_analysis()

    thresholds = [0, 1, 3, 5]

    total_points = 0
    earned_points = 0

    print("\n" + "=" * 80)
    print("SCORING AGAINST GROUND TRUTH")
    print("=" * 80)

    # 1. Step 2 coefficients beta_1, beta_2 (20 pts)
    print("\n--- Step 2 Coefficients (beta_1, beta_2) [20 pts] ---")
    coef_points = 0
    coef_total = 0
    for t in thresholds:
        for name, gt_key in [('beta_1', 'beta_1'), ('beta_2', 'beta_2')]:
            gt = GROUND_TRUTH[gt_key][t]
            gen = results[t][gt_key]
            diff = abs(gen - gt)
            match = diff <= 0.01
            coef_total += 2.5
            if match:
                coef_points += 2.5
            print(f"  >={t} {name}: gen={gen:.4f}, gt={gt:.4f}, diff={diff:.4f} {'PASS' if match else 'FAIL'}")
    print(f"  Score: {coef_points:.1f}/{coef_total:.1f}")
    total_points += 20
    earned_points += (coef_points / coef_total) * 20

    # 2. Cumulative returns (20 pts)
    print("\n--- Cumulative Returns [20 pts] ---")
    cum_points = 0
    cum_total = 0
    for t in thresholds:
        for T_yrs in [5, 10, 15, 20]:
            gt_key = f'cum_{T_yrs}'
            gt = GROUND_TRUTH[gt_key][t]
            gen = results[t]['cum_returns'][T_yrs]
            diff = abs(gen - gt)
            match = diff <= 0.03
            cum_total += 1.25
            if match:
                cum_points += 1.25
            print(f"  >={t} {T_yrs}yr: gen={gen:.4f}, gt={gt:.4f}, diff={diff:.4f} {'PASS' if match else 'FAIL'}")
    print(f"  Score: {cum_points:.1f}/{cum_total:.1f}")
    total_points += 20
    earned_points += (cum_points / cum_total) * 20

    # 3. Step 1 coefficients (15 pts) - check beta_hat (b1+b2)
    print("\n--- Step 1 Coefficients [15 pts] ---")
    s1_points = 0
    s1_total = 0
    for t in thresholds:
        gt_b1b2 = GROUND_TRUTH['beta_1'][t] + GROUND_TRUTH['beta_2'][t]
        gen_b1b2 = results[t]['beta_hat']
        if abs(gt_b1b2) > 0.001:
            rel_err = abs(gen_b1b2 - gt_b1b2) / abs(gt_b1b2)
        else:
            rel_err = abs(gen_b1b2 - gt_b1b2)
        match = rel_err <= 0.20
        s1_total += 3.75
        if match:
            s1_points += 3.75
        print(f"  >={t}: gen b1+b2={gen_b1b2:.4f}, gt b1+b2={gt_b1b2:.4f}, rel_err={rel_err:.3f} {'PASS' if match else 'FAIL'}")
    print(f"  Score: {s1_points:.1f}/{s1_total:.1f}")
    total_points += 15
    earned_points += (s1_points / s1_total) * 15

    # 4. Standard errors (10 pts)
    print("\n--- Standard Errors [10 pts] ---")
    se_points = 0
    se_total = 0
    for t in thresholds:
        for name, gt_key in [('beta_1_se', 'beta_1_se'), ('beta_2_se', 'beta_2_se')]:
            gt = GROUND_TRUTH[gt_key][t]
            gen = results[t][name]
            if gt > 0:
                rel_err = abs(gen - gt) / gt
            else:
                rel_err = abs(gen - gt)
            match = rel_err <= 0.30
            se_total += 1.25
            if match:
                se_points += 1.25
            print(f"  >={t} {name}: gen={gen:.4f}, gt={gt:.4f}, rel_err={rel_err:.3f} {'PASS' if match else 'FAIL'}")
    print(f"  Score: {se_points:.1f}/{se_total:.1f}")
    total_points += 10
    earned_points += (se_points / se_total) * 10

    # 5. Sample size (15 pts) - check step 2 N ~ 10685
    print("\n--- Sample Size [15 pts] ---")
    # Paper says N=10,685 for the full step-2 sample
    gt_n = 10685
    gen_n = results[0]['step2_n']
    rel_err_n = abs(gen_n - gt_n) / gt_n
    n_match = rel_err_n <= 0.05
    n_points = 15 if n_match else 0
    print(f"  Step 2 N: gen={gen_n}, gt={gt_n}, rel_err={rel_err_n:.3f} {'PASS' if n_match else 'FAIL'}")
    total_points += 15
    earned_points += n_points

    # 6. Significance (10 pts) - check signs match
    print("\n--- Significance / Signs [10 pts] ---")
    sig_points = 0
    sig_total = 0
    for t in thresholds:
        for name in ['beta_1', 'beta_2']:
            gt = GROUND_TRUTH[name][t]
            gen = results[t][name]
            gt_sig = gt > 0
            gen_sig = gen > 0
            match = gt_sig == gen_sig
            sig_total += 1.25
            if match:
                sig_points += 1.25
            print(f"  >={t} {name}: gen_sign={'+'if gen>0 else'-'}, gt_sign={'+'if gt>0 else'-'} {'PASS' if match else 'FAIL'}")
    print(f"  Score: {sig_points:.1f}/{sig_total:.1f}")
    total_points += 10
    earned_points += (sig_points / sig_total) * 10

    # 7. All columns present (10 pts)
    print("\n--- All Columns Present [10 pts] ---")
    all_present = all(t in results for t in thresholds)
    col_points = 10 if all_present else 0
    print(f"  All 4 columns present: {'YES' if all_present else 'NO'}")
    total_points += 10
    earned_points += col_points

    total_score = int(round(earned_points))
    print(f"\n{'='*40}")
    print(f"TOTAL SCORE: {total_score}/100")
    print(f"{'='*40}")

    return total_score


if __name__ == '__main__':
    output, results = run_analysis()
    score = score_against_ground_truth(results)
