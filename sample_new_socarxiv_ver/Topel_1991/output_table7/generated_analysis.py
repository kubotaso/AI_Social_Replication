#!/usr/bin/env python3
"""
Table 7 Replication: Least-Squares Models Conditioning on (Estimated) Completed Job Tenure
Topel (1991) - "Specific Capital, Mobility, and Wages"

OLS models of log wage LEVELS with completed job tenure.
5 columns: baseline, observed completed tenure (restricted/unrestricted),
imputed completed tenure (restricted/unrestricted).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

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
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5, 1972: 41.8,
    1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9, 1977: 60.6, 1978: 65.2,
    1979: 72.6, 1980: 82.4, 1981: 90.9, 1982: 100.0
}

# Education category -> years mapping
EDUC_CAT_TO_YEARS = {
    0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17
}

# Ground truth values from Table 7
GROUND_TRUTH = {
    'experience': [0.0418, 0.0379, 0.0345, 0.0397, 0.0401],
    'experience_se': [0.0013, 0.0014, 0.0015, 0.0013, 0.0014],
    'experience_sq': [-0.00079, -0.00069, -0.00072, -0.00074, -0.00073],
    'experience_sq_se': [0.00003, 0.000032, 0.000069, 0.000030, 0.000069],
    'tenure': [0.0138, -0.0015, 0.0137, 0.0060, 0.0163],
    'tenure_se': [0.0052, 0.0015, 0.0038, 0.0073, 0.0038],
    'obs_completed_tenure': [None, 0.0165, 0.0316, None, None],
    'obs_completed_tenure_se': [None, 0.0016, 0.0022, None, None],
    'x_censor': [None, -0.0025, -0.0024, None, None],
    'x_censor_se': [None, 0.0073, 0.0073, None, None],
    'imp_completed_tenure': [None, None, None, 0.0053, 0.0067],
    'imp_completed_tenure_se': [None, None, None, 0.0036, 0.0042],
    'exp_sq_interaction': [None, None, -0.00061, None, -0.00075],
    'exp_sq_interaction_se': [None, None, 0.000036, None, 0.000033],
    'tenure_interaction': [None, None, 0.0142, None, 0.0429],
    'tenure_interaction_se': [None, None, 0.0033, None, 0.0016],
    'r_squared': [0.422, 0.428, 0.432, 0.433, 0.435],
}


def run_analysis(data_source=DATA_FILE):
    """Run Table 7 replication."""

    # =========================================================================
    # Load and prepare data
    # =========================================================================
    print("Loading PSID panel data...")
    df = pd.read_csv(data_source)
    print(f"  Raw panel: {len(df)} observations, {df['person_id'].nunique()} persons")

    # Recode education from categorical to years
    # Check per-year: if max education <= 8, it's categorical
    df['education_years'] = df['education_clean'].copy()
    for yr in df['year'].unique():
        mask = df['year'] == yr
        max_ed = df.loc[mask, 'education_clean'].max()
        if max_ed <= 8:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_CAT_TO_YEARS)

    # Recompute experience with correct education
    df['experience'] = df['age'] - df['education_years'] - 6
    df['experience'] = df['experience'].clip(lower=0)

    # Deflate wages: use GNP deflator first, then CPS wage index
    # The paper says "log average hourly earnings" deflated
    # Following Topel: real_wage = nominal_wage / (GNP_deflator/100) then divided by CPS index
    df['gnp_defl'] = df['year'].map(GNP_DEFLATOR)
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)

    # log_real_wage = log(hourly_wage / (gnp_defl/100)) - log(cps_index)
    # = log(hourly_wage) - log(gnp_defl/100) - log(cps_index)
    df['log_real_wage'] = df['log_hourly_wage'] - np.log(df['gnp_defl'] / 100.0) - np.log(df['cps_index'])

    print(f"  Mean education (years): {df['education_years'].mean():.1f}")
    print(f"  Mean experience: {df['experience'].mean():.1f}")
    print(f"  Mean log real wage: {df['log_real_wage'].mean():.3f}")

    # =========================================================================
    # Fill missing control variables
    # =========================================================================
    # Union: fill missing with 0
    df['union'] = df['union_member'].fillna(0)
    # Disabled: fill missing with 0
    df['disability'] = df['disabled'].fillna(0)
    # SMSA
    df['smsa'] = df['lives_in_smsa'].fillna(0)
    # Married
    df['married_d'] = df['married'].fillna(0)

    # =========================================================================
    # Construct completed tenure variables
    # =========================================================================
    print("\nConstructing completed tenure variables...")

    # For each job, compute total observed duration in the panel
    job_stats = df.groupby('job_id').agg(
        min_year=('year', 'min'),
        max_year=('year', 'max'),
        max_tenure=('tenure_topel', 'max'),
        n_obs=('year', 'count')
    ).reset_index()

    # Observed completed tenure = total years the job lasts
    job_stats['completed_tenure_obs'] = job_stats['max_tenure']
    # Actually: max_tenure is already the 0-indexed tenure at the last observed year
    # So completed duration = max_tenure + 1 (if the job started at tenure 0)
    # But we want "completed tenure" = total years in job
    # The paper likely uses max_tenure + 1 as the completed spell length

    # Censor dummy: 1 if job is still active at end of panel (1983)
    job_stats['censor'] = (job_stats['max_year'] >= 1983).astype(float)

    print(f"  Total jobs: {len(job_stats)}")
    print(f"  Censored jobs: {job_stats['censor'].sum():.0f}")
    print(f"  Completed jobs: {(1 - job_stats['censor']).sum():.0f}")
    print(f"  Mean observed completed tenure: {job_stats['completed_tenure_obs'].mean():.2f}")

    # Merge back to person-year data
    df = df.merge(job_stats[['job_id', 'completed_tenure_obs', 'censor']], on='job_id', how='left')

    # =========================================================================
    # Imputed completed tenure
    # =========================================================================
    # Fit a model for completed tenure using observable characteristics
    # For uncensored jobs: actual completed tenure is known
    # For censored jobs: predict completed tenure

    # Strategy: Use regression on completed jobs to predict for censored ones
    # Regress completed_tenure_obs on covariates for jobs that ended

    # First, get one observation per job (use the first observation of each job)
    job_first = df.groupby('job_id').first().reset_index()

    # For uncensored jobs, the completed tenure is known
    # For censored jobs, we need to predict
    uncensored = job_first[job_first['censor'] == 0].copy()
    censored = job_first[job_first['censor'] == 1].copy()

    # Covariates for the prediction model
    pred_vars = ['experience', 'education_years', 'married_d', 'union', 'smsa', 'disability',
                 'region_ne', 'region_nc', 'region_south', 'region_west']

    # Fit prediction model on uncensored jobs
    X_pred_unc = uncensored[pred_vars].copy()
    X_pred_unc = sm.add_constant(X_pred_unc)
    y_pred_unc = uncensored['completed_tenure_obs']

    valid_unc = X_pred_unc.notna().all(axis=1) & y_pred_unc.notna()
    pred_model = sm.OLS(y_pred_unc[valid_unc], X_pred_unc[valid_unc]).fit()

    print(f"\n  Completed tenure prediction model R^2: {pred_model.rsquared:.3f}")

    # Predict for all jobs
    X_pred_all = job_first[pred_vars].copy()
    X_pred_all = sm.add_constant(X_pred_all)

    # For uncensored: use actual completed tenure
    # For censored: use predicted
    job_first['imputed_completed_tenure'] = pred_model.predict(X_pred_all)

    # For uncensored jobs, use actual observed value
    job_first.loc[job_first['censor'] == 0, 'imputed_completed_tenure'] = \
        job_first.loc[job_first['censor'] == 0, 'completed_tenure_obs']

    # Merge imputed completed tenure back to full data
    df = df.merge(job_first[['job_id', 'imputed_completed_tenure']], on='job_id', how='left')

    # =========================================================================
    # Construct model variables
    # =========================================================================
    df['experience_sq'] = df['experience'] ** 2
    df['tenure_var'] = df['tenure_topel']  # Linear tenure

    # Completed tenure interaction terms
    df['obs_ct_x_censor'] = df['completed_tenure_obs'] * df['censor']
    df['obs_ct_x_exp_sq'] = df['completed_tenure_obs'] * df['experience_sq']
    df['obs_ct_x_tenure'] = df['completed_tenure_obs'] * df['tenure_var']

    df['imp_ct_x_exp_sq'] = df['imputed_completed_tenure'] * df['experience_sq']
    df['imp_ct_x_tenure'] = df['imputed_completed_tenure'] * df['tenure_var']

    # Year dummies
    year_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971']
    # Drop one year for reference category

    # Region dummies
    region_cols = ['region_ne', 'region_nc', 'region_south']  # west is reference

    # Control variables
    control_vars = ['education_years', 'married_d', 'union', 'disability', 'smsa'] + region_cols + year_cols

    # =========================================================================
    # Sample selection
    # =========================================================================
    # Drop observations with missing key variables
    all_vars = ['log_real_wage', 'experience', 'experience_sq', 'tenure_var',
                'completed_tenure_obs', 'censor', 'imputed_completed_tenure'] + control_vars

    sample = df.dropna(subset=all_vars).copy()
    print(f"\n  Analysis sample: {len(sample)} observations")

    # =========================================================================
    # Run 5 OLS models
    # =========================================================================
    y = sample['log_real_wage']

    # Base regressors (experience quadratic + linear tenure)
    base_vars = ['experience', 'experience_sq', 'tenure_var']

    # Column (1): Baseline OLS
    X1_vars = base_vars + control_vars
    X1 = sm.add_constant(sample[X1_vars])
    model1 = sm.OLS(y, X1).fit()

    # Column (2): + observed completed tenure (restricted)
    X2_vars = base_vars + ['completed_tenure_obs', 'obs_ct_x_censor'] + control_vars
    X2 = sm.add_constant(sample[X2_vars])
    model2 = sm.OLS(y, X2).fit()

    # Column (3): + observed completed tenure + interactions (unrestricted)
    X3_vars = base_vars + ['completed_tenure_obs', 'obs_ct_x_censor',
                           'obs_ct_x_exp_sq', 'obs_ct_x_tenure'] + control_vars
    X3 = sm.add_constant(sample[X3_vars])
    model3 = sm.OLS(y, X3).fit()

    # Column (4): + imputed completed tenure (restricted)
    X4_vars = base_vars + ['imputed_completed_tenure'] + control_vars
    X4 = sm.add_constant(sample[X4_vars])
    model4 = sm.OLS(y, X4).fit()

    # Column (5): + imputed completed tenure + interactions (unrestricted)
    X5_vars = base_vars + ['imputed_completed_tenure',
                           'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars
    X5 = sm.add_constant(sample[X5_vars])
    model5 = sm.OLS(y, X5).fit()

    models = [model1, model2, model3, model4, model5]

    # =========================================================================
    # Format results
    # =========================================================================
    results_text = []
    results_text.append("=" * 100)
    results_text.append("TABLE 7: Least-Squares Models Conditioning on (Estimated) Completed Job Tenure")
    results_text.append("PSID White Males")
    results_text.append("=" * 100)
    results_text.append("")

    def fmt(model, var, width=12):
        if var in model.params.index:
            coef = model.params[var]
            se = model.bse[var]
            pval = model.pvalues[var]
            stars = ''
            if pval < 0.001:
                stars = '***'
            elif pval < 0.01:
                stars = '**'
            elif pval < 0.05:
                stars = '*'
            return f"{coef:>9.4f}{stars}", f"({se:.4f})"
        return f"{'...':>12s}", ""

    header = f"{'Variable':35s}" + "".join([f"{'('+str(i+1)+')':>14s}" for i in range(5)])
    results_text.append(header)
    results_text.append("-" * 105)

    # Key variables
    var_map = [
        ('Experience', 'experience'),
        ('Experience^2', 'experience_sq'),
        ('Tenure', 'tenure_var'),
        ('Observed completed tenure', 'completed_tenure_obs'),
        ('x censor', 'obs_ct_x_censor'),
        ('Imputed completed tenure', 'imputed_completed_tenure'),
        ('Experience^2 (interaction)', 'obs_ct_x_exp_sq'),
        ('Tenure (interaction)', 'obs_ct_x_tenure'),
    ]

    # Special handling for columns 4-5 interactions (use imputed versions)
    for label, var_name in var_map:
        coef_line = f"{label:35s}"
        se_line = f"{'':35s}"
        for i, m in enumerate(models):
            # For interactions in cols 4-5, use imputed versions
            actual_var = var_name
            if i >= 3:  # Columns 4 and 5
                if var_name == 'completed_tenure_obs':
                    actual_var = 'imputed_completed_tenure'
                elif var_name == 'obs_ct_x_censor':
                    actual_var = '__skip__'  # No censor term for imputed
                elif var_name == 'obs_ct_x_exp_sq':
                    actual_var = 'imp_ct_x_exp_sq'
                elif var_name == 'obs_ct_x_tenure':
                    actual_var = 'imp_ct_x_tenure'

            if actual_var == '__skip__':
                coef_str = f"{'...':>14s}"
                se_str = f"{'':>14s}"
            else:
                c, s = fmt(m, actual_var)
                coef_str = f"{c:>14s}"
                se_str = f"{s:>14s}" if s else f"{'':>14s}"
            coef_line += coef_str
            se_line += se_str
        results_text.append(coef_line)
        if any(fmt(m, var_name)[1] for m in models):
            results_text.append(se_line)

    results_text.append("")
    r2_line = f"{'R^2':35s}"
    n_line = f"{'N':35s}"
    for m in models:
        r2_line += f"{m.rsquared:>14.3f}"
        n_line += f"{int(m.nobs):>14d}"
    results_text.append(r2_line)
    results_text.append(n_line)

    output = "\n".join(results_text)
    print("\n" + output)

    # Also print raw coefficients for scoring
    print("\n\n=== RAW COEFFICIENTS FOR SCORING ===")
    key_vars_per_model = {
        0: ['experience', 'experience_sq', 'tenure_var'],
        1: ['experience', 'experience_sq', 'tenure_var', 'completed_tenure_obs', 'obs_ct_x_censor'],
        2: ['experience', 'experience_sq', 'tenure_var', 'completed_tenure_obs', 'obs_ct_x_censor',
            'obs_ct_x_exp_sq', 'obs_ct_x_tenure'],
        3: ['experience', 'experience_sq', 'tenure_var', 'imputed_completed_tenure'],
        4: ['experience', 'experience_sq', 'tenure_var', 'imputed_completed_tenure',
            'imp_ct_x_exp_sq', 'imp_ct_x_tenure'],
    }

    for i, m in enumerate(models):
        print(f"\nColumn ({i+1}):")
        print(f"  R^2 = {m.rsquared:.4f}")
        print(f"  N = {int(m.nobs)}")
        for var in key_vars_per_model[i]:
            if var in m.params.index:
                coef = m.params[var]
                se = m.bse[var]
                pval = m.pvalues[var]
                stars = ''
                if pval < 0.001: stars = '***'
                elif pval < 0.01: stars = '**'
                elif pval < 0.05: stars = '*'
                print(f"  {var}: coef={coef:.6f}, se={se:.6f} {stars}")

    return output, models


def score_against_ground_truth():
    """Compute score against Table 7 ground truth."""
    gt = GROUND_TRUTH

    try:
        output, models = run_analysis()
    except Exception as e:
        print(f"Runtime error: {e}")
        return 0

    total_points = 0
    max_points = 0
    details = []

    # --- Coefficient signs and magnitudes (25 pts) ---
    coef_points = 0
    coef_max = 0

    var_names = [
        ('experience', 'experience'),
        ('experience_sq', 'experience_sq'),
        ('tenure', 'tenure_var'),
    ]

    # Check main coefficients across 5 columns
    for gt_key, model_var in var_names:
        for col_idx in range(5):
            gt_val = gt[gt_key][col_idx]
            if gt_val is None:
                continue
            coef_max += 1
            if model_var in models[col_idx].params.index:
                gen_val = models[col_idx].params[model_var]
                if abs(gen_val - gt_val) <= 0.05 or (abs(gt_val) < 0.01 and abs(gen_val - gt_val) / max(abs(gt_val), 1e-6) <= 0.20):
                    coef_points += 1
                    details.append(f"  MATCH {gt_key} col({col_idx+1}): gen={gen_val:.6f} vs true={gt_val}")
                else:
                    details.append(f"  MISS {gt_key} col({col_idx+1}): gen={gen_val:.6f} vs true={gt_val}")
            else:
                details.append(f"  MISS {gt_key} col({col_idx+1}): variable not found")

    # Check completed tenure coefficients
    ct_vars = [
        ('obs_completed_tenure', 'completed_tenure_obs', [1, 2]),
        ('x_censor', 'obs_ct_x_censor', [1, 2]),
        ('imp_completed_tenure', 'imputed_completed_tenure', [3, 4]),
        ('exp_sq_interaction', ['obs_ct_x_exp_sq', 'imp_ct_x_exp_sq'], [2, 4]),
        ('tenure_interaction', ['obs_ct_x_tenure', 'imp_ct_x_tenure'], [2, 4]),
    ]

    for gt_key, model_vars, cols in ct_vars:
        for i, col_idx in enumerate(cols):
            gt_val = gt[gt_key][col_idx]
            if gt_val is None:
                continue
            coef_max += 1
            if isinstance(model_vars, list):
                mv = model_vars[i] if i < len(model_vars) else model_vars[0]
            else:
                mv = model_vars
            if mv in models[col_idx].params.index:
                gen_val = models[col_idx].params[mv]
                tol = 0.05
                if abs(gt_val) < 0.01:
                    match = abs(gen_val - gt_val) / max(abs(gt_val), 1e-6) <= 0.20
                else:
                    match = abs(gen_val - gt_val) <= tol
                if match:
                    coef_points += 1
                    details.append(f"  MATCH {gt_key} col({col_idx+1}): gen={gen_val:.6f} vs true={gt_val}")
                else:
                    details.append(f"  MISS {gt_key} col({col_idx+1}): gen={gen_val:.6f} vs true={gt_val}")
            else:
                details.append(f"  MISS {gt_key} col({col_idx+1}): variable {mv} not found")

    coef_score = 25 * (coef_points / max(coef_max, 1))
    details.append(f"\nCoefficients: {coef_points}/{coef_max} matched => {coef_score:.1f}/25 pts")

    # --- Standard errors (15 pts) ---
    se_points = 0
    se_max = 0

    se_vars = [
        ('experience_se', 'experience', list(range(5))),
        ('experience_sq_se', 'experience_sq', list(range(5))),
        ('tenure_se', 'tenure_var', list(range(5))),
    ]

    for gt_key, model_var, cols in se_vars:
        for col_idx in cols:
            gt_val = gt[gt_key][col_idx]
            if gt_val is None:
                continue
            se_max += 1
            if model_var in models[col_idx].params.index:
                gen_val = models[col_idx].bse[model_var]
                if abs(gen_val - gt_val) <= 0.02:
                    se_points += 1
                    details.append(f"  SE MATCH {gt_key} col({col_idx+1}): gen={gen_val:.6f} vs true={gt_val}")
                else:
                    details.append(f"  SE MISS {gt_key} col({col_idx+1}): gen={gen_val:.6f} vs true={gt_val}")

    se_score = 15 * (se_points / max(se_max, 1))
    details.append(f"\nStandard errors: {se_points}/{se_max} matched => {se_score:.1f}/15 pts")

    # --- Sample size (15 pts) ---
    # Paper says ~13,128
    target_n = 13128
    gen_n = int(models[0].nobs)
    n_ratio = abs(gen_n - target_n) / target_n
    if n_ratio <= 0.05:
        n_score = 15
    elif n_ratio <= 0.10:
        n_score = 10
    elif n_ratio <= 0.20:
        n_score = 5
    else:
        n_score = 0
    details.append(f"\nSample size: gen={gen_n} vs target~{target_n}, ratio_err={n_ratio:.3f} => {n_score}/15 pts")

    # --- Significance levels (25 pts) ---
    sig_points = 0
    sig_max = 0

    def get_stars(pval):
        if pval < 0.001: return '***'
        elif pval < 0.01: return '**'
        elif pval < 0.05: return '*'
        return ''

    def true_stars(coef, se):
        if se == 0: return ''
        t = abs(coef / se)
        if t > 3.291: return '***'
        elif t > 2.576: return '**'
        elif t > 1.96: return '*'
        return ''

    # Check significance for all reported coefficients
    sig_checks = [
        ('experience', 'experience', list(range(5))),
        ('experience_sq', 'experience_sq', list(range(5))),
        ('tenure', 'tenure_var', list(range(5))),
    ]

    for gt_key, model_var, cols in sig_checks:
        for col_idx in cols:
            gt_coef = gt[gt_key][col_idx]
            gt_se_val = gt[gt_key.replace('experience_sq', 'experience_sq_se').replace('experience', 'experience_se').replace('tenure', 'tenure_se')
                          if '_se' not in gt_key else gt_key][col_idx]
            # Get correct SE key
            se_key = gt_key + '_se'
            if se_key not in gt:
                continue
            gt_se_val = gt[se_key][col_idx]
            if gt_coef is None or gt_se_val is None:
                continue
            sig_max += 1
            true_s = true_stars(gt_coef, gt_se_val)
            if model_var in models[col_idx].params.index:
                gen_s = get_stars(models[col_idx].pvalues[model_var])
                if gen_s == true_s:
                    sig_points += 1

    sig_score = 25 * (sig_points / max(sig_max, 1))
    details.append(f"\nSignificance: {sig_points}/{sig_max} matched => {sig_score:.1f}/25 pts")

    # --- All variables present (10 pts) ---
    var_present = 0
    var_total = 8  # experience, exp^2, tenure, obs_ct, censor, imp_ct, exp_sq_int, tenure_int

    check_presence = [
        ('experience', 0), ('experience_sq', 0), ('tenure_var', 0),
        ('completed_tenure_obs', 1), ('obs_ct_x_censor', 1),
        ('imputed_completed_tenure', 3),
        ('obs_ct_x_exp_sq', 2), ('obs_ct_x_tenure', 2),
    ]
    for var, col in check_presence:
        if var in models[col].params.index:
            var_present += 1

    var_score = 10 * (var_present / var_total)
    details.append(f"\nVariables present: {var_present}/{var_total} => {var_score:.1f}/10 pts")

    # --- R-squared (10 pts) ---
    r2_points = 0
    for i in range(5):
        gen_r2 = models[i].rsquared
        true_r2 = gt['r_squared'][i]
        if abs(gen_r2 - true_r2) <= 0.02:
            r2_points += 1
            details.append(f"  R2 MATCH col({i+1}): gen={gen_r2:.4f} vs true={true_r2}")
        else:
            details.append(f"  R2 MISS col({i+1}): gen={gen_r2:.4f} vs true={true_r2}")

    r2_score = 10 * (r2_points / 5)
    details.append(f"\nR-squared: {r2_points}/5 matched => {r2_score:.1f}/10 pts")

    # Total
    total_score = coef_score + se_score + n_score + sig_score + var_score + r2_score

    print("\n\n" + "=" * 60)
    print("SCORING BREAKDOWN")
    print("=" * 60)
    for d in details:
        print(d)
    print(f"\n{'='*60}")
    print(f"TOTAL SCORE: {total_score:.1f} / 100")
    print(f"{'='*60}")

    return total_score


if __name__ == '__main__':
    score = score_against_ground_truth()
