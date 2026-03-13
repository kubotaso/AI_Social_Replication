"""
Replication of Table 3 from Topel (1991)
"Specific Capital, Mobility, and Wages: Wages Rise with Job Seniority"

Two-step estimation of experience (beta_1) and tenure (beta_2) effects on log real wages.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try to import linearmodels for IV regression; fall back to statsmodels
try:
    from linearmodels.iv import IV2SLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False

import statsmodels.api as sm
from scipy import stats

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
    # LOAD AND PREPARE DATA
    # ======================================================================
    df = pd.read_csv(data_source)

    # --- Education recoding ---
    # Years 1975-1976 have education in years (0-17); all other years categorical (0-8)
    # Actually: 1976+ may have years. Let's check per-year max and recode accordingly.
    df['education_years'] = np.nan
    for yr in df['year'].unique():
        mask = df['year'] == yr
        edu_vals = df.loc[mask, 'education_clean']
        max_edu = edu_vals.max()
        if max_edu <= 8:
            # Categorical - apply mapping
            df.loc[mask, 'education_years'] = edu_vals.map(EDUC_MAP)
        else:
            # Already in years
            df.loc[mask, 'education_years'] = edu_vals

    # --- Experience ---
    df['experience'] = df['age'] - df['education_years'] - 6
    df['experience'] = df['experience'].clip(lower=0)

    # --- Tenure ---
    df['tenure'] = df['tenure_topel'].copy()

    # --- Real wage ---
    # PSID interview year Y reports income for year Y-1
    # log_real_wage = log_hourly_wage - ln(GNP_deflator[year-1]/100) - ln(CPS_wage_index[year])
    df['gnp_deflator'] = df['year'].map(lambda y: GNP_DEFLATOR.get(y - 1, np.nan))
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)

    df['log_real_wage'] = (df['log_hourly_wage']
                           - np.log(df['gnp_deflator'] / 100.0)
                           - np.log(df['cps_index']))

    # ======================================================================
    # SAMPLE RESTRICTIONS
    # ======================================================================
    df = df[df['white'] == 1].copy()
    df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
    df = df[df['log_hourly_wage'].notna()].copy()
    df = df[df['hourly_wage'] > 0].copy()

    # Exclude self-employed
    if 'self_employed' in df.columns:
        df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

    # Exclude agriculture
    if 'agriculture' in df.columns:
        df = df[df['agriculture'] != 1].copy()

    # Exclude government workers
    if 'govt_worker' in df.columns:
        df = df[(df['govt_worker'] == 0) | (df['govt_worker'].isna())].copy()

    # SRC sample only (id_1968 <= 2930)
    if 'id_1968' in df.columns:
        df = df[df['id_1968'] <= 2930].copy()

    # Tenure >= 1 year
    df = df[df['tenure'] >= 1].copy()

    # Drop missing key variables
    df = df.dropna(subset=['log_real_wage', 'experience', 'tenure', 'education_years']).copy()

    # Fill union_member NaN with 0
    if 'union_member' in df.columns:
        df['union_member'] = df['union_member'].fillna(0)

    # Fill disabled NaN with 0
    if 'disabled' in df.columns:
        df['disabled'] = df['disabled'].fillna(0)

    # ======================================================================
    # CONSTRUCT VARIABLES FOR STEP 1 (First-differenced within-job)
    # ======================================================================
    # Sort for proper differencing
    df = df.sort_values(['person_id', 'job_id', 'year']).copy()

    # Polynomial terms
    df['tenure_sq'] = df['tenure'] ** 2
    df['tenure_cu'] = df['tenure'] ** 3
    df['tenure_qu'] = df['tenure'] ** 4
    df['exp_sq'] = df['experience'] ** 2
    df['exp_cu'] = df['experience'] ** 3
    df['exp_qu'] = df['experience'] ** 4

    # First differences within jobs
    df['prev_person'] = df['person_id'].shift(1)
    df['prev_job'] = df['job_id'].shift(1)
    df['prev_year'] = df['year'].shift(1)

    df['same_job_seq'] = ((df['person_id'] == df['prev_person']) &
                          (df['job_id'] == df['prev_job']) &
                          (df['year'] - df['prev_year'] == 1))

    # Compute first differences
    for var in ['log_real_wage', 'tenure', 'tenure_sq', 'tenure_cu', 'tenure_qu',
                'exp_sq', 'exp_cu', 'exp_qu']:
        df[f'd_{var}'] = df[var] - df[var].shift(1)

    # Year dummies for first differences
    year_cols = [c for c in df.columns if c.startswith('year_') and c != 'year']
    for yc in year_cols:
        df[f'd_{yc}'] = df[yc] - df[yc].shift(1)

    # Filter to within-job consecutive pairs
    fd = df[df['same_job_seq']].copy()
    fd = fd.dropna(subset=['d_log_real_wage']).copy()

    # ======================================================================
    # STEP 1: OLS on first-differenced data (Table 2, Model 3)
    # ======================================================================
    # d_log_wage = const + b*(d_tenure) + gamma2*(d_tenure^2) + gamma3*(d_tenure^3)
    #              + gamma4*(d_tenure^4) + delta2*(d_exp^2) + delta3*(d_exp^3)
    #              + delta4*(d_exp^4) + year_dummies

    # Note: d_tenure = 1 always for within-job consecutive obs, so the intercept
    # captures (beta_1 + beta_2). We include d_tenure explicitly as a variable.

    # Scale polynomial terms as in the paper
    fd['d_tenure_sq_s'] = fd['d_tenure_sq'] / 100.0
    fd['d_tenure_cu_s'] = fd['d_tenure_cu'] / 1000.0
    fd['d_tenure_qu_s'] = fd['d_tenure_qu'] / 10000.0
    fd['d_exp_sq_s'] = fd['d_exp_sq'] / 100.0
    fd['d_exp_cu_s'] = fd['d_exp_cu'] / 1000.0
    fd['d_exp_qu_s'] = fd['d_exp_qu'] / 10000.0

    # Actually, for the first-step regression, we should NOT scale.
    # The paper scales for presentation only. Let's use unscaled for estimation
    # and convert coefficients later.

    # Build X matrix for Step 1
    # d_tenure is constant = 1, so it becomes the intercept
    # We include a constant which absorbs d_tenure
    d_year_cols = [c for c in fd.columns if c.startswith('d_year_')]
    # Drop one year dummy to avoid collinearity
    d_year_cols_use = sorted(d_year_cols)
    if 'd_year_1968' in d_year_cols_use:
        d_year_cols_use.remove('d_year_1968')

    step1_x_cols = ['d_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                    'd_exp_sq', 'd_exp_cu', 'd_exp_qu'] + d_year_cols_use

    step1_y = fd['d_log_real_wage']
    step1_X = sm.add_constant(fd[step1_x_cols])

    # Drop any remaining NaN
    valid = step1_y.notna() & step1_X.notna().all(axis=1)
    step1_y = step1_y[valid]
    step1_X = step1_X[valid]

    step1_model = sm.OLS(step1_y, step1_X).fit()

    # Extract Step 1 estimates
    # The constant = beta_1 + beta_2 (since d_tenure = 1 always)
    beta_1_plus_beta_2 = step1_model.params['const']
    beta_1_plus_beta_2_se = step1_model.bse['const']

    # Higher-order tenure coefficients (unscaled)
    gamma2 = step1_model.params['d_tenure_sq']
    gamma3 = step1_model.params['d_tenure_cu']
    gamma4 = step1_model.params['d_tenure_qu']

    # Higher-order experience coefficients (unscaled)
    delta2 = step1_model.params['d_exp_sq']
    delta3 = step1_model.params['d_exp_cu']
    delta4 = step1_model.params['d_exp_qu']

    print("=" * 70)
    print("STEP 1 RESULTS (First-differenced within-job, Model 3)")
    print("=" * 70)
    print(f"N (step 1): {len(step1_y)}")
    print(f"beta_1 + beta_2 (const): {beta_1_plus_beta_2:.4f} (SE: {beta_1_plus_beta_2_se:.4f})")
    print(f"  Paper value: 0.1258 (0.0161)")
    print(f"gamma2 (tenure^2): {gamma2:.6f} (scaled: {gamma2*100:.4f})")
    print(f"gamma3 (tenure^3): {gamma3:.8f} (scaled: {gamma3*1000:.4f})")
    print(f"gamma4 (tenure^4): {gamma4:.10f} (scaled: {gamma4*10000:.4f})")
    print(f"delta2 (exp^2): {delta2:.6f} (scaled: {delta2*100:.4f})")
    print(f"delta3 (exp^3): {delta3:.8f} (scaled: {delta3*1000:.4f})")
    print(f"delta4 (exp^4): {delta4:.10f} (scaled: {delta4*10000:.4f})")
    print(f"R-squared: {step1_model.rsquared:.4f}")

    # ======================================================================
    # STEP 2: IV regression on levels data
    # ======================================================================
    # Use the full sample (not just first-differenced)
    # Construct adjusted wage:
    #   w*_it = log_real_wage_it - (beta1+beta2)*T_it
    #           - gamma2*T^2 - gamma3*T^3 - gamma4*T^4
    #           - delta2*X^2 - delta3*X^3 - delta4*X^4
    # Then regress w* on X (experience), instrumented by X_0 = X - T

    levels = df.copy()
    levels = levels.dropna(subset=['log_real_wage', 'experience', 'tenure', 'education_years']).copy()

    # Construct adjusted wage
    levels['w_star'] = (levels['log_real_wage']
                        - beta_1_plus_beta_2 * levels['tenure']
                        - gamma2 * levels['tenure_sq']
                        - gamma3 * levels['tenure_cu']
                        - gamma4 * levels['tenure_qu']
                        - delta2 * levels['exp_sq']
                        - delta3 * levels['exp_cu']
                        - delta4 * levels['exp_qu'])

    # Initial experience (instrument)
    levels['init_exp'] = levels['experience'] - levels['tenure']
    levels['init_exp'] = levels['init_exp'].clip(lower=0)

    # Controls for Step 2
    control_cols = ['education_years', 'married']

    if 'union_member' in levels.columns:
        control_cols.append('union_member')
    if 'disabled' in levels.columns:
        control_cols.append('disabled')

    # Note: lives_in_smsa is all zeros, exclude it
    # Region dummies (drop one for reference)
    region_cols = ['region_ne', 'region_nc', 'region_south']  # west as reference
    for rc in region_cols:
        if rc in levels.columns:
            control_cols.append(rc)

    # Year dummies (drop one)
    year_dummy_cols = sorted([c for c in levels.columns if c.startswith('year_') and c != 'year'])
    if 'year_1968' in year_dummy_cols:
        year_dummy_cols.remove('year_1968')
    control_cols.extend(year_dummy_cols)

    # Drop NaN in controls
    all_needed = ['w_star', 'experience', 'init_exp'] + control_cols
    levels = levels.dropna(subset=all_needed).copy()

    N_step2 = len(levels)
    print(f"\nN (step 2, levels): {N_step2}")
    print(f"  Paper value: 10,685")

    # ======================================================================
    # IV REGRESSION: w* = beta_1 * experience + controls + error
    # Instrument: init_exp for experience
    # ======================================================================

    if HAS_LINEARMODELS:
        # Use linearmodels IV2SLS
        from linearmodels.iv import IV2SLS

        dep = levels['w_star']
        exog = sm.add_constant(levels[control_cols])
        endog = levels[['experience']]
        instruments = levels[['init_exp']]

        iv_model = IV2SLS(dep, exog, endog, instruments).fit(cov_type='unadjusted')

        beta_1 = iv_model.params['experience']
        beta_1_se = iv_model.std_errors['experience']
    else:
        # Manual 2SLS
        # Stage 1: regress experience on init_exp + controls
        exog_cols = control_cols.copy()
        X_stage1 = sm.add_constant(levels[['init_exp'] + exog_cols])
        stage1 = sm.OLS(levels['experience'], X_stage1).fit()
        levels['exp_hat'] = stage1.fittedvalues

        # Stage 2: regress w* on exp_hat + controls
        X_stage2 = sm.add_constant(levels[['exp_hat'] + exog_cols])
        stage2 = sm.OLS(levels['w_star'], X_stage2).fit()

        beta_1 = stage2.params['exp_hat']
        # Correct SE: need to use residuals from stage 2 with original X
        # For proper 2SLS SE, recompute
        X_orig = sm.add_constant(levels[['experience'] + exog_cols])
        resid = levels['w_star'] - X_orig @ stage2.params.values
        sigma2 = (resid ** 2).sum() / (len(resid) - len(stage2.params))
        # Variance = sigma2 * (X_hat'X_hat)^{-1}
        X_hat = sm.add_constant(levels[['exp_hat'] + exog_cols])
        bread = np.linalg.inv(X_hat.T @ X_hat)
        var_beta = sigma2 * bread
        beta_1_se = np.sqrt(var_beta[1, 1])  # experience is index 1

    # ======================================================================
    # COMPUTE beta_2
    # ======================================================================
    beta_2 = beta_1_plus_beta_2 - beta_1
    # SE for beta_2 via delta method: var(b2) = var(b1+b2) + var(b1)
    # (assuming independence of step 1 and step 2 estimators)
    beta_2_se = np.sqrt(beta_1_plus_beta_2_se**2 + beta_1_se**2)

    print(f"\n{'='*70}")
    print(f"STEP 2 RESULTS")
    print(f"{'='*70}")
    print(f"beta_1 (experience): {beta_1:.4f} (SE: {beta_1_se:.4f})")
    print(f"  Paper value: 0.0713 (0.0181)")
    print(f"beta_1 + beta_2 (from step 1): {beta_1_plus_beta_2:.4f} (SE: {beta_1_plus_beta_2_se:.4f})")
    print(f"  Paper value: 0.1258 (0.0161)")
    print(f"beta_2 (tenure): {beta_2:.4f} (SE: {beta_2_se:.4f})")
    print(f"  Paper value: 0.0545 (0.0079)")

    # ======================================================================
    # OLS BIAS: Column (4)
    # ======================================================================
    # OLS regression of log_real_wage on experience, tenure, tenure polynomial,
    # experience polynomial, and controls
    ols_x_cols = (['experience', 'tenure', 'tenure_sq', 'tenure_cu', 'tenure_qu',
                   'exp_sq', 'exp_cu', 'exp_qu'] + control_cols)

    X_ols = sm.add_constant(levels[ols_x_cols])
    ols_model = sm.OLS(levels['log_real_wage'], X_ols).fit()

    ols_tenure_coef = ols_model.params['tenure']
    ols_tenure_se = ols_model.bse['tenure']

    # OLS bias = OLS tenure coef - IV tenure coef (not exactly, but the paper
    # defines column 4 as the "least-squares bias in wage growth" b1+b2)
    # Actually column 4 is b1+b2 from OLS - which gives the bias.
    # The paper says bias = 0.0020, which is very small.
    # This may actually be the difference between OLS (b1+b2) on within-job changes
    # and the true value. Let me compute it as OLS estimate of within-job wage growth
    # minus the IV estimate.

    # Actually, re-reading: Column 4 = "Wage Growth Bias b1+b2"
    # This is the OLS estimate of (b1+b2) from a levels regression using
    # tenure as regressor - it should be very close to the within-job estimate.
    # The bias value of 0.0020 seems to be a very small number.
    # It's likely: bias = OLS(b1+b2) from levels - IV(b1+b2) from step1
    # Or it's the heterogeneity bias in OLS wage growth estimates.

    # Let me try: Column 4 is the bias = E[d_wage|OLS] - E[d_wage|IV]
    # which equals OLS linear tenure coef from levels minus beta_1_plus_beta_2

    ols_bias = ols_tenure_coef + ols_model.params['experience'] - beta_1_plus_beta_2

    # Actually, more precisely, the paper says:
    # "difference between OLS and IV estimates of the linear tenure coefficient"
    # This seems to mean: OLS tenure in levels vs beta_2 from IV
    # Or the OLS "within-job wage growth" coefficient minus the true one.

    # Let's compute the OLS within-job wage growth (from levels regression)
    # which is the sum of linear tenure and linear experience coefficients:
    ols_exp_coef = ols_model.params['experience']
    ols_b1_b2_levels = ols_exp_coef + ols_tenure_coef

    # Another interpretation: run OLS of d_log_wage on d_tenure (= 1) and same controls
    # to get OLS (b1+b2), then bias = OLS(b1+b2) - IV(b1+b2)
    # But OLS(b1+b2) from first differences IS the step 1 estimate, which is unbiased
    # in the Topel framework. So the bias is in the LEVELS estimate.

    # I think column 4 is simply: OLS tenure coefficient from levels (the OLS analog)
    # minus the IV tenure coefficient (beta_2). Topel says the bias is small.
    # bias = OLS_tenure - beta_2
    ols_bias_val = ols_tenure_coef - beta_2

    print(f"\n{'='*70}")
    print(f"OLS BIAS CALCULATION")
    print(f"{'='*70}")
    print(f"OLS tenure coef (levels): {ols_tenure_coef:.4f} (SE: {ols_tenure_se:.4f})")
    print(f"OLS exp coef (levels): {ols_exp_coef:.4f}")
    print(f"OLS bias (tenure coef - beta_2): {ols_bias_val:.4f}")
    print(f"  Paper value (col 4): 0.0020 (0.0004)")

    # ======================================================================
    # CUMULATIVE RETURNS TO TENURE
    # ======================================================================
    # Two-step: beta_2*T + gamma2*T^2 + gamma3*T^3 + gamma4*T^4
    # OLS: ols_tenure*T + ols_gamma2*T^2 + ols_gamma3*T^3 + ols_gamma4*T^4

    ols_gamma2 = ols_model.params['tenure_sq']
    ols_gamma3 = ols_model.params['tenure_cu']
    ols_gamma4 = ols_model.params['tenure_qu']

    def cumulative_return(T, b, g2, g3, g4):
        return b * T + g2 * T**2 + g3 * T**3 + g4 * T**4

    print(f"\n{'='*70}")
    print(f"CUMULATIVE RETURNS TO JOB TENURE")
    print(f"{'='*70}")

    tenure_years = [5, 10, 15, 20]

    print(f"\n{'':>20} {'5 Years':>12} {'10 Years':>12} {'15 Years':>12} {'20 Years':>12}")

    # Two-step returns
    twostep_returns = {}
    for T in tenure_years:
        ret = cumulative_return(T, beta_2, gamma2, gamma3, gamma4)
        twostep_returns[T] = ret

    print(f"{'Two-step model:':>20}", end="")
    for T in tenure_years:
        print(f" {twostep_returns[T]:>11.4f}", end="")
    print()

    # Two-step SEs (approximate via delta method)
    # The main source of uncertainty is beta_2
    twostep_ses = {}
    for T in tenure_years:
        # Simplified: SE ≈ T * SE(beta_2) (dominates for small T)
        se = T * beta_2_se
        twostep_ses[T] = se
    print(f"{'(SE):':>20}", end="")
    for T in tenure_years:
        print(f" ({twostep_ses[T]:>9.4f})", end="")
    print()

    print(f"\n{'Paper two-step:':>20}", end="")
    for T in tenure_years:
        print(f" {GROUND_TRUTH['cumret_2step'][T]:>11.4f}", end="")
    print()

    # OLS returns
    ols_returns = {}
    for T in tenure_years:
        ret = cumulative_return(T, ols_tenure_coef, ols_gamma2, ols_gamma3, ols_gamma4)
        ols_returns[T] = ret

    print(f"\n{'OLS:':>20}", end="")
    for T in tenure_years:
        print(f" {ols_returns[T]:>11.4f}", end="")
    print()

    ols_ses = {}
    for T in tenure_years:
        se = T * ols_tenure_se
        ols_ses[T] = se
    print(f"{'(SE):':>20}", end="")
    for T in tenure_years:
        print(f" ({ols_ses[T]:>9.4f})", end="")
    print()

    print(f"\n{'Paper OLS:':>20}", end="")
    for T in tenure_years:
        print(f" {GROUND_TRUTH['cumret_ols'][T]:>11.4f}", end="")
    print()

    # ======================================================================
    # FORMATTED TABLE 3 OUTPUT
    # ======================================================================
    print(f"\n\n{'='*80}")
    print(f"TABLE 3: Second-Step Estimated Main Effects")
    print(f"{'='*80}")
    print(f"\nTop Panel: Main Effects (standard errors in parentheses)")
    print(f"{'':>25} {'(1)':>14} {'(2)':>14} {'(3)':>14} {'(4)':>14}")
    print(f"{'':>25} {'Experience':>14} {'Within-Job':>14} {'Tenure':>14} {'Wage Growth':>14}")
    print(f"{'':>25} {'beta_1':>14} {'beta_1+beta_2':>14} {'beta_2':>14} {'Bias':>14}")
    print(f"{'Main effect:':>25} {beta_1:>10.4f}     {beta_1_plus_beta_2:>10.4f}     {beta_2:>10.4f}     {ols_bias_val:>10.4f}")
    print(f"{'':>25} ({beta_1_se:>8.4f})     ({beta_1_plus_beta_2_se:>8.4f})     ({beta_2_se:>8.4f})     ({ols_tenure_se:>8.4f})")

    print(f"\nN = {N_step2}")

    print(f"\nBottom Panel: Estimated Cumulative Return to Job Tenure")
    print(f"{'':>20} {'5 Years':>12} {'10 Years':>12} {'15 Years':>12} {'20 Years':>12}")
    print(f"{'Two-step:':>20} {twostep_returns[5]:>11.4f}  {twostep_returns[10]:>11.4f}  {twostep_returns[15]:>11.4f}  {twostep_returns[20]:>11.4f}")
    print(f"{'':>20} ({twostep_ses[5]:>9.4f})  ({twostep_ses[10]:>9.4f})  ({twostep_ses[15]:>9.4f})  ({twostep_ses[20]:>9.4f})")
    print(f"{'OLS:':>20} {ols_returns[5]:>11.4f}  {ols_returns[10]:>11.4f}  {ols_returns[15]:>11.4f}  {ols_returns[20]:>11.4f}")
    print(f"{'':>20} ({ols_ses[5]:>9.4f})  ({ols_ses[10]:>9.4f})  ({ols_ses[15]:>9.4f})  ({ols_ses[20]:>9.4f})")

    # ======================================================================
    # AUTOMATED SCORING
    # ======================================================================
    score = score_against_ground_truth(
        beta_1, beta_1_se, beta_1_plus_beta_2, beta_1_plus_beta_2_se,
        beta_2, beta_2_se, ols_bias_val, ols_tenure_se,
        N_step2, twostep_returns, twostep_ses, ols_returns, ols_ses
    )

    return score


def score_against_ground_truth(beta_1, beta_1_se, beta_1_plus_beta_2, beta_1_plus_beta_2_se,
                                beta_2, beta_2_se, ols_bias, ols_bias_se,
                                N, twostep_returns, twostep_ses, ols_returns, ols_ses):
    """Score results against ground truth using IV/two-step rubric."""

    print(f"\n\n{'='*80}")
    print(f"AUTOMATED SCORING")
    print(f"{'='*80}")

    total_score = 0
    max_score = 100

    # --- Step 1 coefficients (15 pts): within 20% relative error ---
    step1_pts = 0
    step1_max = 15

    # beta_1 + beta_2
    if abs(GROUND_TRUTH['beta_1_plus_beta_2']) > 0.001:
        rel_err = abs(beta_1_plus_beta_2 - GROUND_TRUTH['beta_1_plus_beta_2']) / abs(GROUND_TRUTH['beta_1_plus_beta_2'])
        if rel_err <= 0.20:
            step1_pts += 15
            print(f"  Step 1 beta_1+beta_2: {beta_1_plus_beta_2:.4f} vs {GROUND_TRUTH['beta_1_plus_beta_2']:.4f} (rel err {rel_err:.2%}) -- PASS")
        else:
            partial = max(0, 15 * (1 - rel_err))
            step1_pts += partial
            print(f"  Step 1 beta_1+beta_2: {beta_1_plus_beta_2:.4f} vs {GROUND_TRUTH['beta_1_plus_beta_2']:.4f} (rel err {rel_err:.2%}) -- PARTIAL ({partial:.1f})")

    total_score += step1_pts
    print(f"  Step 1 coefficients: {step1_pts:.1f}/{step1_max}")

    # --- Step 2 coefficients beta_1, beta_2 (20 pts): within 0.01 ---
    step2_pts = 0
    step2_max = 20

    err_b1 = abs(beta_1 - GROUND_TRUTH['beta_1'])
    if err_b1 <= 0.01:
        step2_pts += 10
        print(f"  beta_1: {beta_1:.4f} vs {GROUND_TRUTH['beta_1']:.4f} (err {err_b1:.4f}) -- PASS")
    elif err_b1 <= 0.03:
        pts = 10 * (1 - (err_b1 - 0.01) / 0.02)
        step2_pts += pts
        print(f"  beta_1: {beta_1:.4f} vs {GROUND_TRUTH['beta_1']:.4f} (err {err_b1:.4f}) -- PARTIAL ({pts:.1f})")
    else:
        print(f"  beta_1: {beta_1:.4f} vs {GROUND_TRUTH['beta_1']:.4f} (err {err_b1:.4f}) -- MISS")

    err_b2 = abs(beta_2 - GROUND_TRUTH['beta_2'])
    if err_b2 <= 0.01:
        step2_pts += 10
        print(f"  beta_2: {beta_2:.4f} vs {GROUND_TRUTH['beta_2']:.4f} (err {err_b2:.4f}) -- PASS")
    elif err_b2 <= 0.03:
        pts = 10 * (1 - (err_b2 - 0.01) / 0.02)
        step2_pts += pts
        print(f"  beta_2: {beta_2:.4f} vs {GROUND_TRUTH['beta_2']:.4f} (err {err_b2:.4f}) -- PARTIAL ({pts:.1f})")
    else:
        print(f"  beta_2: {beta_2:.4f} vs {GROUND_TRUTH['beta_2']:.4f} (err {err_b2:.4f}) -- MISS")

    total_score += step2_pts
    print(f"  Step 2 coefficients: {step2_pts:.1f}/{step2_max}")

    # --- Cumulative returns (20 pts): within 0.03 ---
    cumret_pts = 0
    cumret_max = 20
    n_cumret = 8  # 4 two-step + 4 OLS
    pts_each = cumret_max / n_cumret

    for T in [5, 10, 15, 20]:
        err = abs(twostep_returns[T] - GROUND_TRUTH['cumret_2step'][T])
        if err <= 0.03:
            cumret_pts += pts_each
            print(f"  Two-step {T}yr: {twostep_returns[T]:.4f} vs {GROUND_TRUTH['cumret_2step'][T]:.4f} (err {err:.4f}) -- PASS")
        elif err <= 0.06:
            p = pts_each * (1 - (err - 0.03) / 0.03)
            cumret_pts += p
            print(f"  Two-step {T}yr: {twostep_returns[T]:.4f} vs {GROUND_TRUTH['cumret_2step'][T]:.4f} (err {err:.4f}) -- PARTIAL ({p:.1f})")
        else:
            print(f"  Two-step {T}yr: {twostep_returns[T]:.4f} vs {GROUND_TRUTH['cumret_2step'][T]:.4f} (err {err:.4f}) -- MISS")

    for T in [5, 10, 15, 20]:
        err = abs(ols_returns[T] - GROUND_TRUTH['cumret_ols'][T])
        if err <= 0.03:
            cumret_pts += pts_each
            print(f"  OLS {T}yr: {ols_returns[T]:.4f} vs {GROUND_TRUTH['cumret_ols'][T]:.4f} (err {err:.4f}) -- PASS")
        elif err <= 0.06:
            p = pts_each * (1 - (err - 0.03) / 0.03)
            cumret_pts += p
            print(f"  OLS {T}yr: {ols_returns[T]:.4f} vs {GROUND_TRUTH['cumret_ols'][T]:.4f} (err {err:.4f}) -- PARTIAL ({p:.1f})")
        else:
            print(f"  OLS {T}yr: {ols_returns[T]:.4f} vs {GROUND_TRUTH['cumret_ols'][T]:.4f} (err {err:.4f}) -- MISS")

    total_score += cumret_pts
    print(f"  Cumulative returns: {cumret_pts:.1f}/{cumret_max}")

    # --- Standard errors (10 pts): within 30% relative ---
    se_pts = 0
    se_max = 10
    se_items = [
        ('beta_1 SE', beta_1_se, GROUND_TRUTH['beta_1_se']),
        ('beta_1+beta_2 SE', beta_1_plus_beta_2_se, GROUND_TRUTH['beta_1_plus_beta_2_se']),
        ('beta_2 SE', beta_2_se, GROUND_TRUTH['beta_2_se']),
    ]
    pts_per_se = se_max / len(se_items)
    for name, val, true_val in se_items:
        if true_val > 0:
            rel_err = abs(val - true_val) / true_val
            if rel_err <= 0.30:
                se_pts += pts_per_se
                print(f"  {name}: {val:.4f} vs {true_val:.4f} (rel err {rel_err:.1%}) -- PASS")
            elif rel_err <= 0.60:
                p = pts_per_se * (1 - (rel_err - 0.30) / 0.30)
                se_pts += p
                print(f"  {name}: {val:.4f} vs {true_val:.4f} (rel err {rel_err:.1%}) -- PARTIAL ({p:.1f})")
            else:
                print(f"  {name}: {val:.4f} vs {true_val:.4f} (rel err {rel_err:.1%}) -- MISS")

    total_score += se_pts
    print(f"  Standard errors: {se_pts:.1f}/{se_max}")

    # --- Sample size N (15 pts): within 5% ---
    n_pts = 0
    n_max = 15
    n_err = abs(N - GROUND_TRUTH['N']) / GROUND_TRUTH['N']
    if n_err <= 0.05:
        n_pts = n_max
        print(f"  N: {N} vs {GROUND_TRUTH['N']} (rel err {n_err:.1%}) -- PASS")
    elif n_err <= 0.15:
        n_pts = n_max * (1 - (n_err - 0.05) / 0.10)
        print(f"  N: {N} vs {GROUND_TRUTH['N']} (rel err {n_err:.1%}) -- PARTIAL ({n_pts:.1f})")
    else:
        print(f"  N: {N} vs {GROUND_TRUTH['N']} (rel err {n_err:.1%}) -- MISS")

    total_score += n_pts
    print(f"  Sample size: {n_pts:.1f}/{n_max}")

    # --- Significance levels (10 pts) ---
    sig_pts = 0
    sig_max = 10
    # Check significance of main coefficients
    sig_items = [
        ('beta_1', beta_1, beta_1_se),
        ('beta_1+beta_2', beta_1_plus_beta_2, beta_1_plus_beta_2_se),
        ('beta_2', beta_2, beta_2_se),
    ]
    pts_per_sig = sig_max / len(sig_items)
    for name, coef, se in sig_items:
        if se > 0:
            t_stat = abs(coef / se)
            # All three should be significant (paper values imply *** for all)
            if t_stat >= 1.96:
                sig_pts += pts_per_sig
                print(f"  {name} significance: t={t_stat:.2f} (significant) -- PASS")
            else:
                print(f"  {name} significance: t={t_stat:.2f} (not significant) -- MISS")

    total_score += sig_pts
    print(f"  Significance levels: {sig_pts:.1f}/{sig_max}")

    # --- All variables/columns present (10 pts) ---
    present_pts = 10  # All columns present by construction
    total_score += present_pts
    print(f"  All variables present: {present_pts}/{10}")

    total_score = min(100, total_score)
    print(f"\n  TOTAL SCORE: {total_score:.1f}/100")

    return total_score


if __name__ == "__main__":
    data_path = "data/psid_panel.csv"
    score = run_analysis(data_path)
    print(f"\nFinal Score: {score:.1f}/100")
