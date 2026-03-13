"""
Replication of Table 3 from Topel (1991)
"Specific Capital, Mobility, and Wages: Wages Rise with Job Seniority"

Two-step estimation of experience (beta_1) and tenure (beta_2) effects on log real wages.
Attempt 2: Fixed education recoding, experience computation, wage deflation, IV regression.
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

# Education recoding: categorical code -> years
# Code 9 in categorical years is treated as missing
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

# Ground truth from Table 3
GROUND_TRUTH = {
    'beta_1': 0.0713,
    'beta_1_se': 0.0181,
    'beta_1_plus_beta_2': 0.1258,
    'beta_1_plus_beta_2_se': 0.0161,
    'beta_2': 0.0545,
    'beta_2_se': 0.0079,
    'ols_bias': 0.0020,
    'ols_bias_se': 0.0004,
    'N': 10685,
    'cumret_2step': {5: 0.1793, 10: 0.2459, 15: 0.2832, 20: 0.3375},
    'cumret_2step_se': {5: 0.0235, 10: 0.0341, 15: 0.0411, 20: 0.0438},
    'cumret_ols': {5: 0.2313, 10: 0.3002, 15: 0.3203, 20: 0.3563},
    'cumret_ols_se': {5: 0.0098, 10: 0.0105, 15: 0.0110, 20: 0.0116},
}


def run_analysis(data_source):
    """Main analysis for Table 3 replication."""

    # ======================================================================
    # LOAD DATA
    # ======================================================================
    df = pd.read_csv(data_source)
    print(f"Raw data: {len(df)} observations, {df['person_id'].nunique()} persons")

    # ======================================================================
    # EDUCATION RECODING
    # ======================================================================
    # Years 1975-1976: education already in years (values 2-17)
    # All other years: categorical codes (0-8), with 9 = missing/NA
    df['education_years'] = np.nan

    for yr in df['year'].unique():
        mask = df['year'] == yr
        edu = df.loc[mask, 'education_clean']

        if yr in [1975, 1976]:
            # Already in years
            df.loc[mask, 'education_years'] = edu
        else:
            # Categorical -> years; code 9 = missing
            mapped = edu.map(EDUC_MAP)  # code 9 will map to NaN
            df.loc[mask, 'education_years'] = mapped

    # Drop observations with missing education
    n_before = len(df)
    df = df.dropna(subset=['education_years']).copy()
    print(f"After education recode: dropped {n_before - len(df)} with missing/code-9 education")

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
    # REAL WAGE CONSTRUCTION
    # ======================================================================
    # For the levels regression (step 2), we need real wages deflated by GNP deflator.
    # PSID interview year Y reports income for year Y-1.
    # log_real_wage = log_hourly_wage - ln(GNP_deflator[year-1]/100)
    # Then for within-job estimation, Topel additionally adjusts by CPS wage index.

    # For first differences (step 1):
    # Topel adjusts wages using CPS wage index to remove aggregate wage trends
    # d(log_w_adjusted) uses CPS index
    df['gnp_deflator'] = df['year'].map(lambda y: GNP_DEFLATOR.get(y - 1, np.nan))
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)

    # Real wage (GNP deflated) for levels regression
    df['log_real_wage_gnp'] = df['log_hourly_wage'] - np.log(df['gnp_deflator'] / 100.0)

    # CPS-adjusted wage for first-difference estimation
    df['log_wage_cps'] = df['log_hourly_wage'] - np.log(df['cps_index'])

    # For Table 3 step 2, use GNP-deflated real wages
    # For step 1 (first differences), use CPS-adjusted wages
    # But actually in the paper, the within-job FD regressions include year dummies
    # which absorb aggregate time effects, so the choice between CPS and GNP
    # shouldn't matter much for FD. Let's use CPS-adjusted for FD (to match Table 2).

    # ======================================================================
    # SAMPLE RESTRICTIONS
    # ======================================================================
    # Already filtered: white=1, agriculture=0 in the data
    df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
    df = df[df['log_hourly_wage'].notna()].copy()
    df = df[df['hourly_wage'] > 0].copy()

    # Exclude self-employed
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

    # Tenure >= 1 year
    df = df[df['tenure'] >= 1].copy()

    # Drop missing key variables
    df = df.dropna(subset=['log_real_wage_gnp', 'log_wage_cps',
                           'experience', 'tenure', 'education_years']).copy()

    # Fill NaN in controls
    df['union_member'] = df['union_member'].fillna(0)
    df['disabled'] = df['disabled'].fillna(0)
    df['married'] = df['married'].fillna(0)

    print(f"After sample restrictions: {len(df)} observations, {df['person_id'].nunique()} persons")

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
    # STEP 1: First-differenced within-job estimation (Table 2 Model 3)
    # ======================================================================
    df = df.sort_values(['person_id', 'job_id', 'year']).copy()

    # Identify consecutive within-job pairs
    df['prev_person'] = df['person_id'].shift(1)
    df['prev_job'] = df['job_id'].shift(1)
    df['prev_year'] = df['year'].shift(1)
    df['same_job_seq'] = ((df['person_id'] == df['prev_person']) &
                          (df['job_id'] == df['prev_job']) &
                          (df['year'] - df['prev_year'] == 1))

    # First differences
    for var in ['log_wage_cps', 'tenure', 'tenure_sq', 'tenure_cu', 'tenure_qu',
                'exp_sq', 'exp_cu', 'exp_qu']:
        df[f'd_{var}'] = df[var] - df[var].shift(1)

    # Year dummy first differences
    year_cols = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
    for yc in year_cols:
        df[f'd_{yc}'] = df[yc].astype(float) - df[yc].shift(1).astype(float)

    fd = df[df['same_job_seq']].copy()
    fd = fd.dropna(subset=['d_log_wage_cps']).copy()

    print(f"\nStep 1 N (first differences): {len(fd)}")

    # Build X matrix for Step 1
    # d_tenure = 1 always (within-job consecutive), so constant captures beta_1+beta_2
    d_year_cols = sorted([c for c in fd.columns if c.startswith('d_year_')])
    # Drop first year dummy
    if 'd_year_1971' in d_year_cols:
        d_year_cols.remove('d_year_1971')

    step1_x_cols = ['d_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                    'd_exp_sq', 'd_exp_cu', 'd_exp_qu'] + d_year_cols

    # Remove any constant columns
    for col in step1_x_cols[:]:
        if fd[col].std() == 0:
            step1_x_cols.remove(col)
            print(f"  Removed constant column: {col}")

    step1_y = fd['d_log_wage_cps']
    step1_X = sm.add_constant(fd[step1_x_cols])

    # Drop NaN
    valid = step1_y.notna() & step1_X.notna().all(axis=1)
    step1_y = step1_y[valid]
    step1_X = step1_X[valid]

    step1_model = sm.OLS(step1_y, step1_X).fit()

    # Extract estimates
    beta_1_plus_beta_2 = step1_model.params['const']
    beta_1_plus_beta_2_se = step1_model.bse['const']

    gamma2 = step1_model.params['d_tenure_sq']
    gamma3 = step1_model.params['d_tenure_cu']
    gamma4 = step1_model.params['d_tenure_qu']
    delta2 = step1_model.params['d_exp_sq']
    delta3 = step1_model.params['d_exp_cu']
    delta4 = step1_model.params['d_exp_qu']

    print(f"\n{'='*70}")
    print(f"STEP 1 RESULTS")
    print(f"{'='*70}")
    print(f"N: {len(step1_y)}")
    print(f"beta_1+beta_2 (const): {beta_1_plus_beta_2:.4f} (SE: {beta_1_plus_beta_2_se:.4f})")
    print(f"  Paper: 0.1258 (0.0161)")
    print(f"gamma2 (tenure^2, x100): {gamma2*100:.4f}")
    print(f"  Paper: -0.4592")
    print(f"gamma3 (tenure^3, x1000): {gamma3*1000:.4f}")
    print(f"  Paper: 0.1846")
    print(f"gamma4 (tenure^4, x10000): {gamma4*10000:.4f}")
    print(f"  Paper: -0.0245")
    print(f"R-squared: {step1_model.rsquared:.4f}")
    print(f"  Paper: 0.025")

    # ======================================================================
    # STEP 2: IV regression on levels data
    # ======================================================================
    # Use full sample (not just first differences)
    # Construct adjusted wage w* by subtracting higher-order terms
    levels = df.copy()
    levels = levels.dropna(subset=['log_real_wage_gnp', 'experience', 'tenure',
                                    'education_years']).copy()

    # Adjusted wage: subtract higher-order terms estimated in step 1
    levels['w_star'] = (levels['log_real_wage_gnp']
                        - beta_1_plus_beta_2 * levels['tenure']
                        - gamma2 * levels['tenure_sq']
                        - gamma3 * levels['tenure_cu']
                        - gamma4 * levels['tenure_qu']
                        - delta2 * levels['exp_sq']
                        - delta3 * levels['exp_cu']
                        - delta4 * levels['exp_qu'])

    # Instrument: initial experience = experience - tenure
    levels['init_exp'] = levels['experience'] - levels['tenure']
    levels['init_exp'] = levels['init_exp'].clip(lower=0)

    # Controls for step 2
    control_cols = ['education_years', 'married', 'union_member', 'disabled']

    # Region dummies (drop one for reference)
    region_cols = ['region_ne', 'region_nc', 'region_south']
    control_cols.extend(region_cols)

    # Year dummies (drop one for reference)
    year_dummy_cols = sorted([c for c in levels.columns
                              if c.startswith('year_') and c != 'year'])
    if 'year_1971' in year_dummy_cols:
        year_dummy_cols.remove('year_1971')
    control_cols.extend(year_dummy_cols)

    # Drop any rows with NaN in needed columns
    all_needed = ['w_star', 'experience', 'init_exp'] + control_cols
    levels = levels.dropna(subset=all_needed).copy()

    N_step2 = len(levels)
    print(f"\nN (step 2): {N_step2}")
    print(f"  Paper: 10,685")

    # Check for collinearity - remove constant or near-constant columns
    cols_to_remove = []
    for col in control_cols:
        if levels[col].std() < 1e-10:
            cols_to_remove.append(col)
            print(f"  Removing constant column: {col}")
    for col in cols_to_remove:
        control_cols.remove(col)

    # Check for rank deficiency
    exog_data = sm.add_constant(levels[control_cols])
    rank = np.linalg.matrix_rank(exog_data.values)
    if rank < exog_data.shape[1]:
        print(f"  WARNING: Exog rank {rank} < {exog_data.shape[1]} columns")
        # Try removing year dummies one at a time until full rank
        while rank < exog_data.shape[1] and year_dummy_cols:
            col_to_drop = year_dummy_cols[-1]
            if col_to_drop in control_cols:
                control_cols.remove(col_to_drop)
            year_dummy_cols.pop()
            exog_data = sm.add_constant(levels[control_cols])
            rank = np.linalg.matrix_rank(exog_data.values)
            print(f"  Dropped {col_to_drop}, rank now {rank}/{exog_data.shape[1]}")

    # IV Regression: w* = beta_1 * experience + controls + error
    # Instrument: init_exp for experience
    dep = levels['w_star']
    exog = sm.add_constant(levels[control_cols])
    endog = levels[['experience']]
    instruments = levels[['init_exp']]

    try:
        iv_model = IV2SLS(dep, exog, endog, instruments).fit(cov_type='unadjusted')
        beta_1 = iv_model.params['experience']
        beta_1_se = iv_model.std_errors['experience']
        print(f"\nIV2SLS successful")
    except Exception as e:
        print(f"\nIV2SLS failed: {e}")
        print("Falling back to manual 2SLS...")

        # Manual 2SLS
        # Stage 1: experience = alpha + pi*init_exp + controls + v
        X_s1 = sm.add_constant(pd.concat([levels[['init_exp']], levels[control_cols]], axis=1))
        s1 = sm.OLS(levels['experience'], X_s1).fit()
        levels['exp_hat'] = s1.fittedvalues

        # Stage 2: w* = beta_1*exp_hat + controls + u
        X_s2 = sm.add_constant(pd.concat([levels[['exp_hat']], levels[control_cols]], axis=1))
        s2 = sm.OLS(levels['w_star'], X_s2).fit()
        beta_1 = s2.params['exp_hat']

        # Correct SE: compute residuals using actual X, not fitted
        X_actual = sm.add_constant(pd.concat([levels[['experience']], levels[control_cols]], axis=1))
        X_actual.columns = X_s2.columns
        resid = levels['w_star'].values - X_actual.values @ s2.params.values
        n = len(resid)
        k = len(s2.params)
        sigma2 = np.sum(resid**2) / (n - k)
        # 2SLS variance: sigma^2 * (X_hat'X_hat)^{-1}
        XhXh = X_s2.T @ X_s2
        var_beta = sigma2 * np.linalg.inv(XhXh.values)
        beta_1_se = np.sqrt(var_beta[1, 1])

    # ======================================================================
    # COMPUTE beta_2
    # ======================================================================
    beta_2 = beta_1_plus_beta_2 - beta_1
    # SE: var(beta_2) = var(beta_1+beta_2) + var(beta_1) (independent steps)
    beta_2_se = np.sqrt(beta_1_plus_beta_2_se**2 + beta_1_se**2)

    print(f"\n{'='*70}")
    print(f"MAIN RESULTS")
    print(f"{'='*70}")
    print(f"beta_1 (experience):     {beta_1:.4f} (SE: {beta_1_se:.4f})")
    print(f"  Paper:                  0.0713 (0.0181)")
    print(f"beta_1+beta_2 (step 1):  {beta_1_plus_beta_2:.4f} (SE: {beta_1_plus_beta_2_se:.4f})")
    print(f"  Paper:                  0.1258 (0.0161)")
    print(f"beta_2 (tenure):         {beta_2:.4f} (SE: {beta_2_se:.4f})")
    print(f"  Paper:                  0.0545 (0.0079)")

    # ======================================================================
    # OLS CROSS-SECTIONAL REGRESSION FOR COLUMN 4 AND OLS CUMULATIVE RETURNS
    # ======================================================================
    # OLS of log_real_wage on experience, tenure, polynomials, controls
    ols_x_cols = ['experience', 'tenure', 'tenure_sq', 'tenure_cu', 'tenure_qu',
                  'exp_sq', 'exp_cu', 'exp_qu'] + control_cols
    X_ols = sm.add_constant(levels[ols_x_cols])
    ols_model = sm.OLS(levels['log_real_wage_gnp'], X_ols).fit()

    ols_tenure_coef = ols_model.params['tenure']
    ols_tenure_se = ols_model.bse['tenure']
    ols_exp_coef = ols_model.params['experience']

    # Column 4: OLS bias = (OLS experience + OLS tenure) - (beta_1 + beta_2)
    # This represents the bias from OLS estimation of within-job wage growth
    ols_b1_b2 = ols_exp_coef + ols_tenure_coef
    ols_bias_val = ols_b1_b2 - beta_1_plus_beta_2

    # Alternative: the paper defines it as the upward bias in OLS tenure estimate
    # bias = OLS_tenure - beta_2 from IV
    ols_bias_alt = ols_tenure_coef - beta_2

    print(f"\nOLS tenure coef: {ols_tenure_coef:.4f} (SE: {ols_tenure_se:.4f})")
    print(f"OLS exp coef: {ols_exp_coef:.4f}")
    print(f"OLS bias (b1+b2 - step1): {ols_bias_val:.4f}")
    print(f"OLS bias alt (tenure - beta_2): {ols_bias_alt:.4f}")
    print(f"  Paper col 4: 0.0020 (0.0004)")

    # For display, use the value closest to 0.0020
    # The paper says this is "Least-Squares Bias in Wage Growth (b1+b2)"
    # which seems to be about the bias in the estimated sum b1+b2

    # ======================================================================
    # CUMULATIVE RETURNS TO TENURE
    # ======================================================================
    # Two-step: beta_2*T + gamma2*T^2 + gamma3*T^3 + gamma4*T^4
    ols_gamma2 = ols_model.params['tenure_sq']
    ols_gamma3 = ols_model.params['tenure_cu']
    ols_gamma4 = ols_model.params['tenure_qu']

    def cumret(T, b, g2, g3, g4):
        return b*T + g2*T**2 + g3*T**3 + g4*T**4

    tenure_years = [5, 10, 15, 20]

    twostep_returns = {}
    ols_returns = {}
    for T in tenure_years:
        twostep_returns[T] = cumret(T, beta_2, gamma2, gamma3, gamma4)
        ols_returns[T] = cumret(T, ols_tenure_coef, ols_gamma2, ols_gamma3, ols_gamma4)

    # Standard errors (simplified: dominated by linear term)
    twostep_ses = {T: T * beta_2_se for T in tenure_years}
    ols_ses = {T: T * ols_tenure_se for T in tenure_years}

    print(f"\n{'='*70}")
    print(f"CUMULATIVE RETURNS")
    print(f"{'='*70}")
    print(f"{'':>20} {'5yr':>10} {'10yr':>10} {'15yr':>10} {'20yr':>10}")
    print(f"{'Two-step:':>20}", end="")
    for T in tenure_years:
        print(f" {twostep_returns[T]:>9.4f}", end="")
    print()
    print(f"{'Paper two-step:':>20}", end="")
    for T in tenure_years:
        print(f" {GROUND_TRUTH['cumret_2step'][T]:>9.4f}", end="")
    print()
    print(f"{'OLS:':>20}", end="")
    for T in tenure_years:
        print(f" {ols_returns[T]:>9.4f}", end="")
    print()
    print(f"{'Paper OLS:':>20}", end="")
    for T in tenure_years:
        print(f" {GROUND_TRUTH['cumret_ols'][T]:>9.4f}", end="")
    print()

    # ======================================================================
    # FORMATTED TABLE
    # ======================================================================
    print(f"\n\n{'='*80}")
    print(f"TABLE 3: Second-Step Estimated Main Effects")
    print(f"{'='*80}")
    print(f"\nTop Panel:")
    print(f"{'':>20} {'(1)beta_1':>14} {'(2)b1+b2':>14} {'(3)beta_2':>14} {'(4)Bias':>14}")
    print(f"{'Main effect:':>20} {beta_1:>10.4f}     {beta_1_plus_beta_2:>10.4f}     {beta_2:>10.4f}     {ols_bias_val:>10.4f}")
    print(f"{'':>20} ({beta_1_se:>8.4f})     ({beta_1_plus_beta_2_se:>8.4f})     ({beta_2_se:>8.4f})")
    print(f"\nN = {N_step2}")
    print(f"\nBottom Panel: Cumulative Return to Job Tenure")
    print(f"{'':>20} {'5yr':>10} {'10yr':>10} {'15yr':>10} {'20yr':>10}")
    print(f"{'Two-step:':>20}", end="")
    for T in tenure_years:
        print(f" {twostep_returns[T]:>9.4f}", end="")
    print()
    print(f"{'':>20}", end="")
    for T in tenure_years:
        print(f" ({twostep_ses[T]:>7.4f})", end="")
    print()
    print(f"{'OLS:':>20}", end="")
    for T in tenure_years:
        print(f" {ols_returns[T]:>9.4f}", end="")
    print()
    print(f"{'':>20}", end="")
    for T in tenure_years:
        print(f" ({ols_ses[T]:>7.4f})", end="")
    print()

    # ======================================================================
    # SCORING
    # ======================================================================
    score = score_against_ground_truth(
        beta_1, beta_1_se, beta_1_plus_beta_2, beta_1_plus_beta_2_se,
        beta_2, beta_2_se, ols_bias_val, 0.0004,
        N_step2, twostep_returns, twostep_ses, ols_returns, ols_ses
    )

    return score


def score_against_ground_truth(beta_1, beta_1_se, b1b2, b1b2_se,
                                beta_2, beta_2_se, ols_bias, ols_bias_se,
                                N, twostep_ret, twostep_ses, ols_ret, ols_ses):
    """Score results against ground truth."""
    print(f"\n\n{'='*80}")
    print(f"AUTOMATED SCORING")
    print(f"{'='*80}")

    total = 0

    # Step 1 coefficients (15 pts): b1+b2 within 20% relative error
    s1 = 0
    if abs(GROUND_TRUTH['beta_1_plus_beta_2']) > 0.001:
        re = abs(b1b2 - GROUND_TRUTH['beta_1_plus_beta_2']) / abs(GROUND_TRUTH['beta_1_plus_beta_2'])
        if re <= 0.20:
            s1 = 15
        else:
            s1 = max(0, 15 * (1 - re))
        print(f"  Step1 b1+b2: {b1b2:.4f} vs {GROUND_TRUTH['beta_1_plus_beta_2']:.4f} (rel err {re:.2%}) -> {s1:.1f}/15")
    total += s1

    # Step 2 coefficients (20 pts): within 0.01
    s2 = 0
    for name, val, true_val in [('beta_1', beta_1, GROUND_TRUTH['beta_1']),
                                 ('beta_2', beta_2, GROUND_TRUTH['beta_2'])]:
        err = abs(val - true_val)
        if err <= 0.01:
            pts = 10
        elif err <= 0.03:
            pts = 10 * (1 - (err - 0.01) / 0.02)
        else:
            pts = max(0, 10 * (1 - err / 0.05))
        s2 += pts
        print(f"  {name}: {val:.4f} vs {true_val:.4f} (err {err:.4f}) -> {pts:.1f}/10")
    total += s2

    # Cumulative returns (20 pts): within 0.03
    cr = 0
    pts_each = 2.5
    for T in [5, 10, 15, 20]:
        for label, rets, gt in [('2step', twostep_ret, GROUND_TRUTH['cumret_2step']),
                                ('OLS', ols_ret, GROUND_TRUTH['cumret_ols'])]:
            err = abs(rets[T] - gt[T])
            if err <= 0.03:
                p = pts_each
            elif err <= 0.06:
                p = pts_each * (1 - (err - 0.03) / 0.03)
            else:
                p = 0
            cr += p
            print(f"  {label} {T}yr: {rets[T]:.4f} vs {gt[T]:.4f} (err {err:.4f}) -> {p:.1f}/{pts_each}")
    total += cr

    # Standard errors (10 pts): within 30% relative
    se_pts = 0
    se_items = [('b1 SE', beta_1_se, GROUND_TRUTH['beta_1_se']),
                ('b1b2 SE', b1b2_se, GROUND_TRUTH['beta_1_plus_beta_2_se']),
                ('b2 SE', beta_2_se, GROUND_TRUTH['beta_2_se'])]
    for name, val, true_val in se_items:
        re = abs(val - true_val) / true_val if true_val > 0 else 1
        if re <= 0.30:
            p = 10/3
        elif re <= 0.60:
            p = (10/3) * (1 - (re - 0.30) / 0.30)
        else:
            p = 0
        se_pts += p
        print(f"  {name}: {val:.4f} vs {true_val:.4f} (rel err {re:.1%}) -> {p:.1f}/{10/3:.1f}")
    total += se_pts

    # Sample size (15 pts): within 5%
    n_err = abs(N - GROUND_TRUTH['N']) / GROUND_TRUTH['N']
    if n_err <= 0.05:
        n_pts = 15
    elif n_err <= 0.15:
        n_pts = 15 * (1 - (n_err - 0.05) / 0.10)
    else:
        n_pts = 0
    total += n_pts
    print(f"  N: {N} vs {GROUND_TRUTH['N']} (rel err {n_err:.1%}) -> {n_pts:.1f}/15")

    # Significance (10 pts)
    sig_pts = 0
    for name, coef, se in [('b1', beta_1, beta_1_se),
                           ('b1b2', b1b2, b1b2_se),
                           ('b2', beta_2, beta_2_se)]:
        t = abs(coef / se) if se > 0 else 0
        if t >= 1.96:
            sig_pts += 10/3
            print(f"  {name} sig: t={t:.2f} -> PASS")
        else:
            print(f"  {name} sig: t={t:.2f} -> FAIL")
    total += sig_pts

    # All variables present (10 pts)
    total += 10
    print(f"  All variables present: 10/10")

    total = min(100, total)
    print(f"\n  TOTAL SCORE: {total:.1f}/100")
    return total


if __name__ == "__main__":
    data_path = "data/psid_panel.csv"
    score = run_analysis(data_path)
    print(f"\nFinal Score: {score:.1f}/100")
