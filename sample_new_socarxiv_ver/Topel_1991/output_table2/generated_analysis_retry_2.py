#!/usr/bin/env python3
"""
Table 2 Replication: Models of Annual Within-Job Wage Growth
Topel (1991) - "Specific Capital, Mobility, and Wages"

First-step estimation: OLS on first-differenced within-job wage data.
Attempt 2: Improved sample restrictions and wage deflation.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import sys

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'psid_panel.csv')

# CPS Real Wage Index (from Murphy and Welch 1987, Table A1)
# These represent REAL wage trends for experience/education groups
CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

# GNP Implicit Price Deflator (1982 = 100) for converting nominal to real wages
# Source: Bureau of Economic Analysis
GNP_DEFLATOR = {
    1967: 33.36, 1968: 34.78, 1969: 36.68, 1970: 38.83, 1971: 40.49,
    1972: 41.81, 1973: 44.35, 1974: 48.61, 1975: 52.90, 1976: 55.61,
    1977: 59.02, 1978: 63.36, 1979: 68.52, 1980: 74.95, 1981: 81.94,
    1982: 86.72, 1983: 89.74
}

# Education category -> years mapping (early PSID waves use categorical codes)
# Per instruction: years 1975-1976 have education in years; all other years categorical
EDUC_CAT_TO_YEARS = {
    0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17
}


def run_analysis(data_source=DATA_FILE):
    """Run Table 2 replication."""

    # =========================================================================
    # Load and prepare data
    # =========================================================================
    print("Loading PSID panel data...")
    df = pd.read_csv(data_source)
    print(f"  Raw panel: {len(df)} observations, {df['person_id'].nunique()} persons")

    # =========================================================================
    # Recode education from categorical to years
    # =========================================================================
    # Years 1975-1976: education is already in years (0-17)
    # All other years (1968-1974, 1977-1983): education is categorical (0-8)
    df['education_years'] = df['education_clean'].copy()
    cat_mask = ~df['year'].isin([1975, 1976])
    df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT_TO_YEARS)

    # Recompute experience with correct education
    df['experience'] = df['age'] - df['education_years'] - 6
    df['experience'] = df['experience'].clip(lower=0)

    # =========================================================================
    # Wage deflation
    # =========================================================================
    # PSID wages are nominal. Need to convert to real wages.
    # Topel uses first-differenced log wages with year dummies.
    # Year dummies absorb any common time trend (inflation).
    # So for the regression, we can use nominal log wages -- the year dummies
    # will absorb the common time trend.
    #
    # However, the MEAN of d_log_wage depends on whether wages are nominal or real.
    # Paper reports mean = .026 which is real wage growth.
    # With nominal wages, mean d_log_wage ~ 0.098 (reflecting inflation).
    # But year dummies absorb this, so regression coefficients are UNAFFECTED.
    #
    # For correct mean reporting, deflate by GNP deflator.
    df['gnp_defl'] = df['year'].map(GNP_DEFLATOR)
    df['log_real_wage'] = df['log_hourly_wage'] - np.log(df['gnp_defl'] / 100.0)

    # Also try CPS index adjustment on top
    # Actually for Table 2 the key issue is just the mean -- coefficients
    # won't change. Use GNP deflator to get the right mean.

    print(f"  Mean education (years): {df['education_years'].mean():.1f}")
    print(f"  Mean experience: {df['experience'].mean():.1f}")
    print(f"  Mean log real wage: {df['log_real_wage'].mean():.3f}")

    # =========================================================================
    # SAMPLE RESTRICTIONS
    # =========================================================================
    n_start = len(df)

    # 1. White males only (already filtered in panel construction, verify)
    if 'white' in df.columns:
        df = df[df['white'] == 1].copy()
    if 'sex' in df.columns:
        df = df[df['sex'] == 1].copy()
    print(f"  After white males filter: {len(df)} (dropped {n_start - len(df)})")

    # 2. Age restriction: 18-60
    n_before = len(df)
    df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
    print(f"  After age 18-60 filter: {len(df)} (dropped {n_before - len(df)})")

    # 3. Exclude self-employed
    n_before = len(df)
    if 'self_employed' in df.columns:
        df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
    print(f"  After exclude self-employed: {len(df)} (dropped {n_before - len(df)})")

    # 4. Exclude government workers (where available)
    n_before = len(df)
    if 'govt_worker' in df.columns:
        # govt_worker = 1 means government employee; exclude them
        # Keep missing (years where variable not available) and non-govt
        df = df[(df['govt_worker'] != 1) | (df['govt_worker'].isna())].copy()
    print(f"  After exclude govt workers: {len(df)} (dropped {n_before - len(df)})")

    # 5. Exclude agriculture workers
    n_before = len(df)
    if 'agriculture' in df.columns:
        df = df[(df['agriculture'] != 1) | (df['agriculture'].isna())].copy()
    print(f"  After exclude agriculture: {len(df)} (dropped {n_before - len(df)})")

    # 6. Positive hours and wages
    n_before = len(df)
    df = df[(df['hours'] > 0) & (df['hourly_wage'] > 0)].copy()
    print(f"  After positive hours/wages: {len(df)} (dropped {n_before - len(df)})")

    # 7. Reasonable hourly wage range (drop extreme outliers)
    n_before = len(df)
    df = df[(df['hourly_wage'] >= 1.0) & (df['hourly_wage'] <= 100.0)].copy()
    print(f"  After wage range filter: {len(df)} (dropped {n_before - len(df)})")

    # 8. Minimum hours threshold (Topel likely uses full-time or near-full-time)
    n_before = len(df)
    df = df[df['hours'] >= 500].copy()
    print(f"  After hours >= 500: {len(df)} (dropped {n_before - len(df)})")

    print(f"\n  After all restrictions: {len(df)} obs, {df['person_id'].nunique()} persons")

    # =========================================================================
    # Construct within-job first differences
    # =========================================================================
    print("\nConstructing within-job first differences...")

    # Sort by person, job, year
    df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)

    # Create lagged values within the same person-job combination
    # Use groupby on person_id and job_id together for safety
    grp = df.groupby(['person_id', 'job_id'])
    df['prev_year'] = grp['year'].shift(1)
    df['prev_log_real_wage'] = grp['log_real_wage'].shift(1)
    df['prev_tenure'] = grp['tenure_topel'].shift(1)
    df['prev_experience'] = grp['experience'].shift(1)

    # Keep only consecutive within-job observations (year gap = 1)
    within_job = df[
        (df['prev_year'].notna()) &
        (df['year'] - df['prev_year'] == 1)
    ].copy()

    # Compute first differences
    within_job['d_log_wage'] = within_job['log_real_wage'] - within_job['prev_log_real_wage']

    # Drop extreme wage change outliers (likely data errors)
    within_job = within_job[within_job['d_log_wage'].between(-2, 2)].copy()

    # Additional restriction: experience >= 1 (need at least 1 year of experience)
    within_job = within_job[within_job['experience'] >= 1].copy()

    print(f"  Within-job observations: {len(within_job)}")
    print(f"  Unique persons: {within_job['person_id'].nunique()}")
    print(f"  Mean d_log_wage: {within_job['d_log_wage'].mean():.4f}")
    print(f"  Std d_log_wage: {within_job['d_log_wage'].std():.4f}")
    print(f"  Paper targets: N = 8,683, Mean = .026")
    print(f"  Mean experience: {within_job['experience'].mean():.1f}")
    print(f"  Mean tenure: {within_job['tenure_topel'].mean():.1f}")

    # =========================================================================
    # Construct regressors
    # =========================================================================

    tenure = within_job['tenure_topel'].values.astype(float)
    prev_tenure = within_job['prev_tenure'].values.astype(float)
    exp = within_job['experience'].values.astype(float)
    prev_exp = within_job['prev_experience'].values.astype(float)

    # Delta Tenure terms (tenure always increases by 1 within a job)
    within_job['d_tenure'] = tenure - prev_tenure  # Should be ~1

    # Delta Tenure^2 = tenure^2 - prev_tenure^2
    within_job['d_tenure_sq'] = (tenure**2 - prev_tenure**2)
    # Delta Tenure^3
    within_job['d_tenure_cu'] = (tenure**3 - prev_tenure**3)
    # Delta Tenure^4
    within_job['d_tenure_qu'] = (tenure**4 - prev_tenure**4)

    # Delta Experience terms (experience also increases by 1 each year)
    # Delta Experience^2
    within_job['d_exp_sq'] = (exp**2 - prev_exp**2)
    # Delta Experience^3
    within_job['d_exp_cu'] = (exp**3 - prev_exp**3)
    # Delta Experience^4
    within_job['d_exp_qu'] = (exp**4 - prev_exp**4)

    # Year dummies for the observation year
    # In the first-differenced model, year dummies capture year-specific effects
    year_dummies = pd.get_dummies(within_job['year'], prefix='yr', dtype=float)
    # Drop the first year dummy for identification
    yr_cols = sorted([c for c in year_dummies.columns])
    # Remove the first year to avoid collinearity
    yr_cols = yr_cols[1:]  # Drop earliest year

    print(f"\n  d_tenure mean: {within_job['d_tenure'].mean():.3f} (should be ~1)")
    print(f"  d_exp_sq mean: {within_job['d_exp_sq'].mean():.3f}")
    print(f"  Year dummies: {yr_cols}")

    # =========================================================================
    # Run the three models
    # =========================================================================
    results_text = []
    results_text.append("=" * 80)
    results_text.append("TABLE 2: Models of Annual Within-Job Wage Growth")
    results_text.append("PSID White Males, 1968-83")
    results_text.append(f"(Dependent Variable Is Change in Log Real Wage; Mean = {within_job['d_log_wage'].mean():.3f})")
    results_text.append("=" * 80)
    results_text.append("")

    y = within_job['d_log_wage'].values

    # Model (1): Linear tenure + quartic experience + year dummies
    X1_vars = ['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
    X1 = within_job[X1_vars].copy()
    X1 = pd.concat([X1.reset_index(drop=True), year_dummies[yr_cols].reset_index(drop=True)], axis=1)
    # Note: No constant needed -- the constant is absorbed by the mean of d_tenure
    # Actually, we DO need a constant -- the year dummies don't span the full space.
    # The constant captures the average wage growth net of year effects.
    # Wait -- in the first-differenced model, Topel includes year dummies.
    # Since d_tenure = 1 for all observations, d_tenure IS a constant.
    # So we should NOT add a separate constant -- d_tenure serves that role.
    # But Topel separately identifies the tenure coefficient...
    # Actually, think about it: d_tenure = 1 for ALL within-job obs.
    # So d_tenure IS perfectly collinear with the constant.
    # That means we can't separately identify a constant AND d_tenure.
    # The reported "d_tenure" coefficient is really the constant of the
    # first-differenced regression (= combined experience + tenure effect).
    # Include constant only, don't include d_tenure as separate regressor.
    # No wait -- Topel reports d_tenure as a variable with coefficient .1242.
    # If d_tenure = 1 always, then this IS the constant term.
    # So just use d_tenure (=1) as the constant replacement. No sm.add_constant.

    # Verify d_tenure is indeed always 1
    dt_unique = within_job['d_tenure'].unique()
    print(f"\n  d_tenure unique values: {sorted(dt_unique)[:10]}")

    # Approach: include d_tenure (which equals 1) and year dummies.
    # d_tenure acts as the constant. Its coefficient = beta_1 + beta_2.

    valid1 = np.isfinite(X1.values).all(axis=1) & np.isfinite(y)
    model1 = sm.OLS(y[valid1], X1[valid1].values, hasconst=True).fit()

    # Model (2): Quadratic tenure + quartic experience + year dummies
    X2_vars = ['d_tenure', 'd_tenure_sq', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
    X2 = within_job[X2_vars].copy()
    X2 = pd.concat([X2.reset_index(drop=True), year_dummies[yr_cols].reset_index(drop=True)], axis=1)
    valid2 = np.isfinite(X2.values).all(axis=1) & np.isfinite(y)
    model2 = sm.OLS(y[valid2], X2[valid2].values, hasconst=True).fit()

    # Model (3): Quartic tenure + quartic experience + year dummies
    X3_vars = ['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
               'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
    X3 = within_job[X3_vars].copy()
    X3 = pd.concat([X3.reset_index(drop=True), year_dummies[yr_cols].reset_index(drop=True)], axis=1)
    valid3 = np.isfinite(X3.values).all(axis=1) & np.isfinite(y)
    model3 = sm.OLS(y[valid3], X3[valid3].values, hasconst=True).fit()

    # =========================================================================
    # Assign variable names to model params for readability
    # =========================================================================
    model1_names = X1_vars + yr_cols
    model2_names = X2_vars + yr_cols
    model3_names = X3_vars + yr_cols

    # Create name -> param/se mappings
    def get_param(model, names, var):
        if var in names:
            idx = names.index(var)
            return model.params[idx], model.bse[idx]
        return None, None

    # =========================================================================
    # Format results (scaling polynomial terms by powers of 10)
    # =========================================================================

    def format_coef_val(coef, se, scale=1):
        """Format coefficient and SE, scaled."""
        if coef is not None:
            c = coef * scale
            s = se * scale
            return f"{c:>10.4f} ({s:.4f})"
        return "       ..."

    header = f"{'':30s} {'Model (1)':>20s} {'Model (2)':>20s} {'Model (3)':>20s}"
    results_text.append(header)
    results_text.append("-" * 90)

    # Delta Tenure
    c1, s1 = get_param(model1, model1_names, 'd_tenure')
    c2, s2 = get_param(model2, model2_names, 'd_tenure')
    c3, s3 = get_param(model3, model3_names, 'd_tenure')
    results_text.append(f"{'Delta Tenure':30s} {format_coef_val(c1,s1):>20s} {format_coef_val(c2,s2):>20s} {format_coef_val(c3,s3):>20s}")

    # Delta Tenure^2
    c2b, s2b = get_param(model2, model2_names, 'd_tenure_sq')
    c3b, s3b = get_param(model3, model3_names, 'd_tenure_sq')
    results_text.append(f"{'Delta Tenure^2 (x10^2)':30s} {'...':>20s} {format_coef_val(c2b,s2b,100):>20s} {format_coef_val(c3b,s3b,100):>20s}")

    # Delta Tenure^3
    c3c, s3c = get_param(model3, model3_names, 'd_tenure_cu')
    results_text.append(f"{'Delta Tenure^3 (x10^3)':30s} {'...':>20s} {'...':>20s} {format_coef_val(c3c,s3c,1000):>20s}")

    # Delta Tenure^4
    c3d, s3d = get_param(model3, model3_names, 'd_tenure_qu')
    results_text.append(f"{'Delta Tenure^4 (x10^4)':30s} {'...':>20s} {'...':>20s} {format_coef_val(c3d,s3d,10000):>20s}")

    # Delta Experience^2
    c1e, s1e = get_param(model1, model1_names, 'd_exp_sq')
    c2e, s2e = get_param(model2, model2_names, 'd_exp_sq')
    c3e, s3e = get_param(model3, model3_names, 'd_exp_sq')
    results_text.append(f"{'Delta Experience^2 (x10^2)':30s} {format_coef_val(c1e,s1e,100):>20s} {format_coef_val(c2e,s2e,100):>20s} {format_coef_val(c3e,s3e,100):>20s}")

    # Delta Experience^3
    c1f, s1f = get_param(model1, model1_names, 'd_exp_cu')
    c2f, s2f = get_param(model2, model2_names, 'd_exp_cu')
    c3f, s3f = get_param(model3, model3_names, 'd_exp_cu')
    results_text.append(f"{'Delta Experience^3 (x10^3)':30s} {format_coef_val(c1f,s1f,1000):>20s} {format_coef_val(c2f,s2f,1000):>20s} {format_coef_val(c3f,s3f,1000):>20s}")

    # Delta Experience^4
    c1g, s1g = get_param(model1, model1_names, 'd_exp_qu')
    c2g, s2g = get_param(model2, model2_names, 'd_exp_qu')
    c3g, s3g = get_param(model3, model3_names, 'd_exp_qu')
    results_text.append(f"{'Delta Experience^4 (x10^4)':30s} {format_coef_val(c1g,s1g,10000):>20s} {format_coef_val(c2g,s2g,10000):>20s} {format_coef_val(c3g,s3g,10000):>20s}")

    # R^2
    results_text.append(f"{'R^2':30s} {model1.rsquared:>20.3f} {model2.rsquared:>20.3f} {model3.rsquared:>20.3f}")

    # Standard error of regression
    se_reg1 = np.sqrt(model1.mse_resid)
    se_reg2 = np.sqrt(model2.mse_resid)
    se_reg3 = np.sqrt(model3.mse_resid)
    results_text.append(f"{'Standard error':30s} {se_reg1:>20.3f} {se_reg2:>20.3f} {se_reg3:>20.3f}")
    results_text.append(f"{'N':30s} {int(model1.nobs):>20d} {int(model2.nobs):>20d} {int(model3.nobs):>20d}")

    # =========================================================================
    # Bottom panel: Predicted within-job wage growth
    # =========================================================================
    results_text.append("")
    results_text.append("PREDICTED WITHIN-JOB WAGE GROWTH BY YEARS OF JOB TENURE")
    results_text.append("(Workers with 10 Years of Labor Market Experience)")
    results_text.append("")

    # Using Model (3) coefficients
    b_t = c3 if c3 is not None else 0
    b_t2 = get_param(model3, model3_names, 'd_tenure_sq')[0] or 0
    b_t3 = get_param(model3, model3_names, 'd_tenure_cu')[0] or 0
    b_t4 = get_param(model3, model3_names, 'd_tenure_qu')[0] or 0
    b_x2 = get_param(model3, model3_names, 'd_exp_sq')[0] or 0
    b_x3 = get_param(model3, model3_names, 'd_exp_cu')[0] or 0
    b_x4 = get_param(model3, model3_names, 'd_exp_qu')[0] or 0

    tenures = list(range(1, 11))
    predictions = []
    X0 = 10  # 10 years of experience at job start

    for T in tenures:
        # At tenure T, experience = X0 + T (since exp increases by 1 each year)
        X = X0 + T
        Xprev = X - 1

        pred = (b_t * 1 +
                b_t2 * (T**2 - (T-1)**2) +
                b_t3 * (T**3 - (T-1)**3) +
                b_t4 * (T**4 - (T-1)**4) +
                b_x2 * (X**2 - Xprev**2) +
                b_x3 * (X**3 - Xprev**3) +
                b_x4 * (X**4 - Xprev**4))
        predictions.append(pred)

    tenure_str = "Tenure (years):    " + "  ".join([f"{t:>5d}" for t in tenures])
    pred_str = "Wage growth (%):   " + "  ".join([f"{p:>5.3f}" for p in predictions])
    results_text.append(tenure_str)
    results_text.append(pred_str)

    # Paper values for comparison
    results_text.append("")
    results_text.append("Paper values:       .068  .060  .052  .046  .041  .037  .033  .030  .028  .026")

    output = "\n".join(results_text)
    print("\n" + output)

    return output


def score_against_ground_truth():
    """Score the results against ground truth from Table 2."""

    # Run analysis first to get results
    output = run_analysis()

    # Ground truth from table_summary.txt
    ground_truth = {
        'N': 8683,
        'mean_d_log_wage': 0.026,
        'model1': {
            'd_tenure': (0.1242, 0.0161),
            'd_exp_sq_x100': (-0.6051, 0.1430),
            'd_exp_cu_x1000': (0.1460, 0.0482),
            'd_exp_qu_x10000': (0.0131, 0.0054),
            'r2': 0.022,
            'se_reg': 0.218,
        },
        'model2': {
            'd_tenure': (0.1265, 0.0162),
            'd_tenure_sq_x100': (-0.0518, 0.0178),
            'd_exp_sq_x100': (-0.6144, 0.1430),
            'd_exp_cu_x1000': (0.1620, 0.0485),
            'd_exp_qu_x10000': (0.0151, 0.0055),
            'r2': 0.023,
            'se_reg': 0.218,
        },
        'model3': {
            'd_tenure': (0.1258, 0.0162),
            'd_tenure_sq_x100': (-0.4592, 0.1080),
            'd_tenure_cu_x1000': (0.1846, 0.0526),
            'd_tenure_qu_x10000': (-0.0245, 0.0079),
            'd_exp_sq_x100': (-0.4067, 0.1546),
            'd_exp_cu_x1000': (0.0989, 0.0517),
            'd_exp_qu_x10000': (0.0089, 0.0058),
            'r2': 0.025,
            'se_reg': 0.218,
        },
        'predicted_growth': [0.068, 0.060, 0.052, 0.046, 0.041, 0.037, 0.033, 0.030, 0.028, 0.026],
    }

    # Parse the output to extract generated values
    lines = output.split('\n')

    total_points = 0
    max_points = 0
    breakdown = {}

    # --- Criterion 1: Coefficient signs and magnitudes (25 pts) ---
    # We have 16 coefficient values across 3 models
    # Model 1: 4 coefficients, Model 2: 5, Model 3: 7 = 16 total
    coef_points = 0
    coef_max = 25
    coef_count = 0
    coef_total = 16

    # Extract generated values from output by parsing
    # This is a simplified scoring -- in practice we'd parse more carefully
    # For now, return the ground truth structure for reference

    print("\n\n" + "="*60)
    print("SCORING AGAINST GROUND TRUTH")
    print("="*60)
    print(f"\nGround truth N: {ground_truth['N']}")
    print(f"Ground truth mean d_log_wage: {ground_truth['mean_d_log_wage']}")
    print("\nDetailed scoring requires parsing the output.")
    print("Score will be computed after execution.")

    return ground_truth


if __name__ == '__main__':
    output = run_analysis()
    # score_against_ground_truth()  # Uncomment to score
