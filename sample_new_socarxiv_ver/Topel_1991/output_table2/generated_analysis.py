#!/usr/bin/env python3
"""
Table 2 Replication: Models of Annual Within-Job Wage Growth
Topel (1991) - "Specific Capital, Mobility, and Wages"

First-step estimation: OLS on first-differenced within-job wage data.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'psid_panel.csv')

# CPS Real Wage Index (from Murphy and Welch 1987, Table A1)
CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

# Education category -> years mapping (early PSID waves use categorical codes)
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

    # Recode education from categorical to years
    # Years 1975-1976: education is already in years (2-digit variable, 0-17)
    # All other years (1968-1974, 1977-1983): education is categorical (1-digit, 0-8)
    df['education_years'] = df['education_clean'].copy()
    cat_mask = ~df['year'].isin([1975, 1976])
    df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT_TO_YEARS)

    # Recompute experience with correct education
    df['experience'] = df['age'] - df['education_years'] - 6
    df['experience'] = df['experience'].clip(lower=0)

    # Deflate wages by CPS wage index
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
    df['log_real_wage'] = df['log_hourly_wage'] - np.log(df['cps_index'])

    print(f"  Mean education (years): {df['education_years'].mean():.1f}")
    print(f"  Mean experience: {df['experience'].mean():.1f}")
    print(f"  Mean log real wage: {df['log_real_wage'].mean():.3f}")

    # =========================================================================
    # Construct within-job first differences
    # =========================================================================
    print("\nConstructing within-job first differences...")

    # Sort by person, job, year
    df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)

    # Create lagged values within the same job
    df['prev_year'] = df.groupby('job_id')['year'].shift(1)
    df['prev_log_real_wage'] = df.groupby('job_id')['log_real_wage'].shift(1)
    df['prev_tenure'] = df.groupby('job_id')['tenure_topel'].shift(1)
    df['prev_experience'] = df.groupby('job_id')['experience'].shift(1)

    # Keep only consecutive within-job observations
    within_job = df[
        (df['prev_year'].notna()) &
        (df['year'] - df['prev_year'] == 1)
    ].copy()

    # Compute first differences
    within_job['d_log_wage'] = within_job['log_real_wage'] - within_job['prev_log_real_wage']

    # Drop extreme outliers (likely data errors)
    within_job = within_job[within_job['d_log_wage'].between(-2, 2)].copy()

    print(f"  Within-job observations: {len(within_job)}")
    print(f"  Mean d_log_wage: {within_job['d_log_wage'].mean():.4f}")
    print(f"  Paper target: N = 8,683, Mean = .026")

    # =========================================================================
    # Construct regressors
    # =========================================================================

    tenure = within_job['tenure_topel']
    prev_tenure = within_job['prev_tenure']
    exp = within_job['experience']
    prev_exp = within_job['prev_experience']

    # Delta Tenure terms (tenure always increases by 1 within a job)
    within_job['d_tenure'] = tenure - prev_tenure  # Should be 1

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

    # Year dummies (first-differenced: dummy for current year minus previous)
    # The year dummy for year t in the first-differenced model captures
    # the effect of being in year t vs t-1
    year_dummies = pd.get_dummies(within_job['year'], prefix='yr', dtype=float)
    # Drop one year for identification (1968 or the earliest year present)
    yr_cols = [c for c in year_dummies.columns if c != 'yr_1968']

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

    y = within_job['d_log_wage']

    # Model (1): Linear tenure + quartic experience
    X1_vars = ['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
    X1 = within_job[X1_vars].copy()
    X1 = pd.concat([X1, year_dummies[yr_cols]], axis=1)
    X1 = sm.add_constant(X1)

    # Drop rows with NaN
    valid1 = X1.notna().all(axis=1) & y.notna()
    model1 = sm.OLS(y[valid1], X1[valid1]).fit()

    # Model (2): Quadratic tenure + quartic experience
    X2_vars = ['d_tenure', 'd_tenure_sq', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
    X2 = within_job[X2_vars].copy()
    X2 = pd.concat([X2, year_dummies[yr_cols]], axis=1)
    X2 = sm.add_constant(X2)
    valid2 = X2.notna().all(axis=1) & y.notna()
    model2 = sm.OLS(y[valid2], X2[valid2]).fit()

    # Model (3): Quartic tenure + quartic experience
    X3_vars = ['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
               'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
    X3 = within_job[X3_vars].copy()
    X3 = pd.concat([X3, year_dummies[yr_cols]], axis=1)
    X3 = sm.add_constant(X3)
    valid3 = X3.notna().all(axis=1) & y.notna()
    model3 = sm.OLS(y[valid3], X3[valid3]).fit()

    # =========================================================================
    # Format results (scaling polynomial terms by powers of 10)
    # =========================================================================

    def format_coef(model, var, scale=1):
        """Get coefficient and SE, scaled."""
        if var in model.params.index:
            coef = model.params[var] * scale
            se = model.bse[var] * scale
            return f"{coef:>10.4f} ({se:.4f})"
        return "       ..."

    header = f"{'':30s} {'Model (1)':>20s} {'Model (2)':>20s} {'Model (3)':>20s}"
    results_text.append(header)
    results_text.append("-" * 90)

    results_text.append(f"{'Delta Tenure':30s} {format_coef(model1, 'd_tenure'):>20s} {format_coef(model2, 'd_tenure'):>20s} {format_coef(model3, 'd_tenure'):>20s}")
    results_text.append(f"{'Delta Tenure^2 (x10^2)':30s} {'...':>20s} {format_coef(model2, 'd_tenure_sq', 100):>20s} {format_coef(model3, 'd_tenure_sq', 100):>20s}")
    results_text.append(f"{'Delta Tenure^3 (x10^3)':30s} {'...':>20s} {'...':>20s} {format_coef(model3, 'd_tenure_cu', 1000):>20s}")
    results_text.append(f"{'Delta Tenure^4 (x10^4)':30s} {'...':>20s} {'...':>20s} {format_coef(model3, 'd_tenure_qu', 10000):>20s}")
    results_text.append(f"{'Delta Experience^2 (x10^2)':30s} {format_coef(model1, 'd_exp_sq', 100):>20s} {format_coef(model2, 'd_exp_sq', 100):>20s} {format_coef(model3, 'd_exp_sq', 100):>20s}")
    results_text.append(f"{'Delta Experience^3 (x10^3)':30s} {format_coef(model1, 'd_exp_cu', 1000):>20s} {format_coef(model2, 'd_exp_cu', 1000):>20s} {format_coef(model3, 'd_exp_cu', 1000):>20s}")
    results_text.append(f"{'Delta Experience^4 (x10^4)':30s} {format_coef(model1, 'd_exp_qu', 10000):>20s} {format_coef(model2, 'd_exp_qu', 10000):>20s} {format_coef(model3, 'd_exp_qu', 10000):>20s}")
    results_text.append(f"{'R^2':30s} {model1.rsquared:>20.3f} {model2.rsquared:>20.3f} {model3.rsquared:>20.3f}")
    results_text.append(f"{'Standard error':30s} {np.sqrt(model1.mse_resid):>20.3f} {np.sqrt(model2.mse_resid):>20.3f} {np.sqrt(model3.mse_resid):>20.3f}")
    results_text.append(f"{'N':30s} {int(model1.nobs):>20d} {int(model2.nobs):>20d} {int(model3.nobs):>20d}")

    # =========================================================================
    # Bottom panel: Predicted within-job wage growth
    # =========================================================================
    results_text.append("")
    results_text.append("PREDICTED WITHIN-JOB WAGE GROWTH BY YEARS OF JOB TENURE")
    results_text.append("(Workers with 10 Years of Labor Market Experience)")
    results_text.append("")

    # Using Model (3) coefficients
    b_t = model3.params.get('d_tenure', 0)
    b_t2 = model3.params.get('d_tenure_sq', 0)
    b_t3 = model3.params.get('d_tenure_cu', 0)
    b_t4 = model3.params.get('d_tenure_qu', 0)
    b_x2 = model3.params.get('d_exp_sq', 0)
    b_x3 = model3.params.get('d_exp_cu', 0)
    b_x4 = model3.params.get('d_exp_qu', 0)

    tenures = list(range(1, 11))
    predictions = []
    X0 = 10  # 10 years of experience

    for T in tenures:
        # Predicted wage growth in year T of tenure
        # d_tenure = 1
        # d_tenure_sq = T^2 - (T-1)^2 = 2T - 1
        # d_tenure_cu = T^3 - (T-1)^3 = 3T^2 - 3T + 1
        # d_tenure_qu = T^4 - (T-1)^4 = 4T^3 - 6T^2 + 4T - 1
        # Experience at tenure T with X0=10: X = X0 + T
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
    # Ground truth from table_summary.txt
    ground_truth = {
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
        'N': 8683,
        'predicted_growth': [0.068, 0.060, 0.052, 0.046, 0.041, 0.037, 0.033, 0.030, 0.028, 0.026],
    }

    print("\n\nSCORING NOT YET IMPLEMENTED — will be added after first run")
    return ground_truth


if __name__ == '__main__':
    output = run_analysis()
    score_against_ground_truth()
