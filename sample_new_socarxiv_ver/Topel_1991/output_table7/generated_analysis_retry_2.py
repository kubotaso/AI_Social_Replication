#!/usr/bin/env python3
"""
Table 7 Replication: Least-Squares Models Conditioning on (Estimated) Completed Job Tenure
Topel (1991) - "Specific Capital, Mobility, and Wages"

Attempt 2: Fixed education recoding, CPS-only deflation, improved completed tenure,
           fixed scoring function, improved imputed tenure via hazard model.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
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

# Education category -> years mapping (early PSID waves use categorical codes 0-8)
# Code 9 appears occasionally, mapped to 17 (advanced/professional)
EDUC_CAT_TO_YEARS = {
    0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17, 9: 17
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

    # =========================================================================
    # Education recoding
    # =========================================================================
    # Years 1975-1976: education is already in years (max > 8)
    # All other years: education is categorical (0-8 or 0-9)
    df['education_years'] = df['education_clean'].copy()
    for yr in df['year'].unique():
        mask = df['year'] == yr
        max_ed = df.loc[mask, 'education_clean'].max()
        if max_ed <= 9:  # Categorical
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_CAT_TO_YEARS)

    # Recompute experience with correct education
    df['experience'] = df['age'] - df['education_years'] - 6
    df['experience'] = df['experience'].clip(lower=1)

    # =========================================================================
    # Wage deflation - CPS wage index only
    # =========================================================================
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
    df['log_real_wage'] = df['log_hourly_wage'] - np.log(df['cps_index'])

    print(f"  Mean education (years): {df['education_years'].mean():.1f}")
    print(f"  Mean experience: {df['experience'].mean():.1f}")
    print(f"  Mean log real wage: {df['log_real_wage'].mean():.3f}")

    # =========================================================================
    # Control variables
    # =========================================================================
    df['union'] = df['union_member'].fillna(0)
    df['disability'] = df['disabled'].fillna(0)
    df['smsa'] = df['lives_in_smsa'].fillna(0)
    df['married_d'] = df['married'].fillna(0)

    # =========================================================================
    # Construct completed tenure variables
    # =========================================================================
    print("\nConstructing completed tenure variables...")

    # For each job, compute total observed duration in the panel
    # tenure_topel starts at 1 for first year of job
    job_stats = df.groupby('job_id').agg(
        min_year=('year', 'min'),
        max_year=('year', 'max'),
        max_tenure=('tenure_topel', 'max'),
        min_tenure=('tenure_topel', 'min'),
        n_obs=('year', 'count')
    ).reset_index()

    # Observed completed tenure = max tenure value for the job
    # (since tenure starts at 1, max_tenure = total years in job)
    job_stats['completed_tenure_obs'] = job_stats['max_tenure'].astype(float)

    # Censor dummy: 1 if job is still active at end of panel (1983)
    job_stats['censor'] = (job_stats['max_year'] >= 1983).astype(float)

    print(f"  Total jobs: {len(job_stats)}")
    print(f"  Censored jobs: {job_stats['censor'].sum():.0f}")
    print(f"  Completed jobs: {(1 - job_stats['censor']).sum():.0f}")
    print(f"  Mean observed completed tenure: {job_stats['completed_tenure_obs'].mean():.2f}")

    # Merge back to person-year data
    df = df.merge(job_stats[['job_id', 'completed_tenure_obs', 'censor']], on='job_id', how='left')

    # =========================================================================
    # Imputed completed tenure via discrete-time hazard model
    # =========================================================================
    print("\nFitting hazard model for imputed completed tenure...")

    # Create a person-year-level indicator for job ending
    # A job ends in year t if the person is observed in year t but not in
    # the same job in year t+1
    df_sorted = df.sort_values(['person_id', 'job_id', 'year']).copy()

    # For each observation, check if this is the last year of the job
    df_sorted['next_year_in_job'] = df_sorted.groupby('job_id')['year'].shift(-1)
    df_sorted['job_ends'] = 0
    # Job ends if there's no next year in job, AND the max_year for this job < 1983
    # (if max_year == 1983, it's censored, not an actual ending)
    df_sorted['job_max_year'] = df_sorted.groupby('job_id')['year'].transform('max')
    df_sorted.loc[
        (df_sorted['next_year_in_job'].isna()) & (df_sorted['job_max_year'] < 1983),
        'job_ends'
    ] = 1

    # Fit logistic regression for job ending
    hazard_vars = ['experience', 'education_years', 'married_d', 'union',
                   'smsa', 'disability', 'tenure_topel',
                   'region_ne', 'region_nc', 'region_south', 'region_west']

    # Add experience^2 and tenure^2 to hazard model
    df_sorted['exp_sq_h'] = df_sorted['experience'] ** 2
    df_sorted['ten_sq_h'] = df_sorted['tenure_topel'] ** 2
    hazard_vars_ext = hazard_vars + ['exp_sq_h', 'ten_sq_h']

    # Only use non-censored observations AND non-final-year observations for hazard
    # Actually, include all observations: job_ends=1 for final obs of completed jobs,
    # job_ends=0 for all others (including censored job observations)
    hazard_sample = df_sorted.dropna(subset=hazard_vars_ext + ['job_ends']).copy()

    X_haz = sm.add_constant(hazard_sample[hazard_vars_ext])
    y_haz = hazard_sample['job_ends']

    try:
        hazard_model = sm.Logit(y_haz, X_haz).fit(disp=0, maxiter=100)
        print(f"  Hazard model pseudo-R^2: {hazard_model.prsquared:.3f}")

        # Predict survival probabilities and expected remaining tenure
        # For each observation, predict the hazard of job ending
        X_pred = sm.add_constant(df_sorted[hazard_vars_ext].fillna(0))
        df_sorted['hazard_prob'] = hazard_model.predict(X_pred)

        # Imputed completed tenure = current tenure + expected remaining tenure
        # Expected remaining tenure approximated by 1/hazard_rate
        # Or more precisely: sum of survival probabilities from current year forward
        # Simple approach: E[remaining] = (1-h)/h for geometric hazard
        df_sorted['expected_remaining'] = (1 - df_sorted['hazard_prob']) / df_sorted['hazard_prob'].clip(lower=0.01)
        df_sorted['expected_remaining'] = df_sorted['expected_remaining'].clip(upper=30)

        df_sorted['imputed_completed_tenure'] = df_sorted['tenure_topel'] + df_sorted['expected_remaining']

        # For uncensored jobs, use actual completed tenure
        uncensored_mask = df_sorted['censor'] == 0
        df_sorted.loc[uncensored_mask, 'imputed_completed_tenure'] = \
            df_sorted.loc[uncensored_mask, 'completed_tenure_obs']

        print(f"  Mean imputed completed tenure: {df_sorted['imputed_completed_tenure'].mean():.2f}")

    except Exception as e:
        print(f"  Hazard model failed: {e}")
        print("  Falling back to OLS prediction")
        # Fallback: predict completed tenure from observables
        job_first = df_sorted.groupby('job_id').first().reset_index()
        uncensored = job_first[job_first['censor'] == 0].copy()

        pred_vars = ['experience', 'education_years', 'married_d', 'union',
                     'smsa', 'disability', 'region_ne', 'region_nc', 'region_south']
        X_unc = sm.add_constant(uncensored[pred_vars].fillna(0))
        y_unc = uncensored['completed_tenure_obs']
        pred_model = sm.OLS(y_unc, X_unc).fit()

        X_all = sm.add_constant(df_sorted[pred_vars].fillna(0))
        df_sorted['imputed_completed_tenure'] = pred_model.predict(X_all)
        df_sorted.loc[df_sorted['censor'] == 0, 'imputed_completed_tenure'] = \
            df_sorted.loc[df_sorted['censor'] == 0, 'completed_tenure_obs']

    # Copy back to main df
    df = df_sorted.copy()

    # =========================================================================
    # Construct model variables
    # =========================================================================
    df['experience_sq'] = df['experience'] ** 2
    df['tenure_var'] = df['tenure_topel'].astype(float)

    # Completed tenure interaction terms
    # For observed completed tenure (cols 2-3):
    df['obs_ct_x_censor'] = df['completed_tenure_obs'] * df['censor']
    df['obs_ct_x_exp_sq'] = df['completed_tenure_obs'] * df['experience_sq']
    df['obs_ct_x_tenure'] = df['completed_tenure_obs'] * df['tenure_var']

    # For imputed completed tenure (cols 4-5):
    df['imp_ct_x_exp_sq'] = df['imputed_completed_tenure'] * df['experience_sq']
    df['imp_ct_x_tenure'] = df['imputed_completed_tenure'] * df['tenure_var']

    # Year dummies (drop 1971 as reference)
    year_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971']

    # Region dummies (drop west as reference)
    region_cols = ['region_ne', 'region_nc', 'region_south']

    # Control variables
    control_vars = ['education_years', 'married_d', 'union', 'disability', 'smsa'] + region_cols + year_cols

    # =========================================================================
    # Sample selection
    # =========================================================================
    all_vars = ['log_real_wage', 'experience', 'experience_sq', 'tenure_var',
                'completed_tenure_obs', 'censor', 'imputed_completed_tenure'] + control_vars

    sample = df.dropna(subset=all_vars).copy()
    print(f"\n  Analysis sample: {len(sample)} observations")

    # =========================================================================
    # Run 5 OLS models
    # =========================================================================
    y = sample['log_real_wage']

    # Base regressors
    base_vars = ['experience', 'experience_sq', 'tenure_var']

    # Column (1): Baseline OLS
    X1_vars = base_vars + control_vars
    X1 = sm.add_constant(sample[X1_vars])
    model1 = sm.OLS(y, X1).fit()

    # Column (2): + observed completed tenure (restricted)
    # In the restricted model (eq 17), completed tenure enters as level shift
    # with censor interaction
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
    # Format and print results
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 7: Least-Squares Models Conditioning on (Estimated) Completed Job Tenure")
    print("PSID White Males")
    print("=" * 100)

    def fmt(model, var):
        if var in model.params.index:
            coef = model.params[var]
            se = model.bse[var]
            pval = model.pvalues[var]
            stars = ''
            if pval < 0.001: stars = '***'
            elif pval < 0.01: stars = '**'
            elif pval < 0.05: stars = '*'
            return f"{coef:>10.5f}{stars}", f"({se:.5f})"
        return f"{'...':>13s}", ""

    print(f"\n{'Variable':40s}" + "".join([f"{'('+str(i+1)+')':>16s}" for i in range(5)]))
    print("-" * 120)

    # Define which variable to use for each model column
    rows = [
        ('Experience', ['experience'] * 5),
        ('Experience^2', ['experience_sq'] * 5),
        ('Tenure', ['tenure_var'] * 5),
        ('Observed completed tenure', [None, 'completed_tenure_obs', 'completed_tenure_obs', None, None]),
        ('x censor', [None, 'obs_ct_x_censor', 'obs_ct_x_censor', None, None]),
        ('Imputed completed tenure', [None, None, None, 'imputed_completed_tenure', 'imputed_completed_tenure']),
        ('Experience^2 (interaction)', [None, None, 'obs_ct_x_exp_sq', None, 'imp_ct_x_exp_sq']),
        ('Tenure (interaction)', [None, None, 'obs_ct_x_tenure', None, 'imp_ct_x_tenure']),
    ]

    for label, var_list in rows:
        coef_line = f"{label:40s}"
        se_line = f"{'':40s}"
        has_se = False
        for i in range(5):
            var = var_list[i]
            if var is None:
                coef_line += f"{'...':>16s}"
                se_line += f"{'':>16s}"
            else:
                c, s = fmt(models[i], var)
                coef_line += f"{c:>16s}"
                if s:
                    se_line += f"{s:>16s}"
                    has_se = True
                else:
                    se_line += f"{'':>16s}"
        print(coef_line)
        if has_se:
            print(se_line)

    print("")
    r2_line = f"{'R^2':40s}"
    n_line = f"{'N':40s}"
    for m in models:
        r2_line += f"{m.rsquared:>16.3f}"
        n_line += f"{int(m.nobs):>16d}"
    print(r2_line)
    print(n_line)

    # Print raw coefficients for scoring
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

    return models


def score_against_ground_truth():
    """Compute score against Table 7 ground truth."""
    gt = GROUND_TRUTH

    try:
        models = run_analysis()
    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()
        return 0

    total_points = 0
    details = []

    # Map ground truth keys to model variable names per column
    coef_map = {
        'experience': {i: 'experience' for i in range(5)},
        'experience_sq': {i: 'experience_sq' for i in range(5)},
        'tenure': {i: 'tenure_var' for i in range(5)},
        'obs_completed_tenure': {1: 'completed_tenure_obs', 2: 'completed_tenure_obs'},
        'x_censor': {1: 'obs_ct_x_censor', 2: 'obs_ct_x_censor'},
        'imp_completed_tenure': {3: 'imputed_completed_tenure', 4: 'imputed_completed_tenure'},
        'exp_sq_interaction': {2: 'obs_ct_x_exp_sq', 4: 'imp_ct_x_exp_sq'},
        'tenure_interaction': {2: 'obs_ct_x_tenure', 4: 'imp_ct_x_tenure'},
    }

    se_map = {
        'experience': 'experience_se',
        'experience_sq': 'experience_sq_se',
        'tenure': 'tenure_se',
        'obs_completed_tenure': 'obs_completed_tenure_se',
        'x_censor': 'x_censor_se',
        'imp_completed_tenure': 'imp_completed_tenure_se',
        'exp_sq_interaction': 'exp_sq_interaction_se',
        'tenure_interaction': 'tenure_interaction_se',
    }

    # --- Coefficient signs and magnitudes (25 pts) ---
    coef_points = 0
    coef_max = 0
    for gt_key, col_vars in coef_map.items():
        for col_idx, model_var in col_vars.items():
            gt_val = gt[gt_key][col_idx]
            if gt_val is None:
                continue
            coef_max += 1
            if model_var in models[col_idx].params.index:
                gen_val = models[col_idx].params[model_var]
                # Tolerance: within 0.05 absolute, or 20% relative for small values
                if abs(gt_val) < 0.01:
                    match = abs(gen_val - gt_val) / max(abs(gt_val), 1e-8) <= 0.20
                else:
                    match = abs(gen_val - gt_val) <= 0.05
                if match:
                    coef_points += 1
                    details.append(f"  COEF MATCH {gt_key} col({col_idx+1}): gen={gen_val:.6f} vs true={gt_val}")
                else:
                    details.append(f"  COEF MISS {gt_key} col({col_idx+1}): gen={gen_val:.6f} vs true={gt_val} (diff={abs(gen_val-gt_val):.6f})")
            else:
                details.append(f"  COEF MISS {gt_key} col({col_idx+1}): variable {model_var} not found")

    coef_score = 25 * (coef_points / max(coef_max, 1))
    details.append(f"\nCoefficients: {coef_points}/{coef_max} matched => {coef_score:.1f}/25 pts")

    # --- Standard errors (15 pts) ---
    se_points = 0
    se_max = 0
    for gt_key, col_vars in coef_map.items():
        se_key = se_map.get(gt_key)
        if se_key is None or se_key not in gt:
            continue
        for col_idx, model_var in col_vars.items():
            gt_se = gt[se_key][col_idx]
            if gt_se is None:
                continue
            se_max += 1
            if model_var in models[col_idx].params.index:
                gen_se = models[col_idx].bse[model_var]
                if abs(gen_se - gt_se) <= 0.02:
                    se_points += 1
                    details.append(f"  SE MATCH {gt_key} col({col_idx+1}): gen={gen_se:.6f} vs true={gt_se}")
                else:
                    details.append(f"  SE MISS {gt_key} col({col_idx+1}): gen={gen_se:.6f} vs true={gt_se}")

    se_score = 15 * (se_points / max(se_max, 1))
    details.append(f"\nStandard errors: {se_points}/{se_max} matched => {se_score:.1f}/15 pts")

    # --- Sample size (15 pts) ---
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
    details.append(f"\nSample size: gen={gen_n} vs target~{target_n}, err={n_ratio:.3f} => {n_score}/15 pts")

    # --- Significance levels (25 pts) ---
    sig_points = 0
    sig_max = 0

    def get_stars(pval):
        if pval < 0.001: return '***'
        elif pval < 0.01: return '**'
        elif pval < 0.05: return '*'
        return ''

    def true_stars(coef, se):
        if se == 0 or se is None: return ''
        t = abs(coef / se)
        if t > 3.291: return '***'
        elif t > 2.576: return '**'
        elif t > 1.96: return '*'
        return ''

    for gt_key, col_vars in coef_map.items():
        se_key = se_map.get(gt_key)
        if se_key is None or se_key not in gt:
            continue
        for col_idx, model_var in col_vars.items():
            gt_coef = gt[gt_key][col_idx]
            gt_se = gt[se_key][col_idx]
            if gt_coef is None or gt_se is None:
                continue
            sig_max += 1
            ts = true_stars(gt_coef, gt_se)
            if model_var in models[col_idx].params.index:
                gs = get_stars(models[col_idx].pvalues[model_var])
                if gs == ts:
                    sig_points += 1
                    details.append(f"  SIG MATCH {gt_key} col({col_idx+1}): gen={gs} true={ts}")
                else:
                    details.append(f"  SIG MISS {gt_key} col({col_idx+1}): gen={gs} true={ts}")

    sig_score = 25 * (sig_points / max(sig_max, 1))
    details.append(f"\nSignificance: {sig_points}/{sig_max} matched => {sig_score:.1f}/25 pts")

    # --- All variables present (10 pts) ---
    var_checks = [
        ('experience', 0), ('experience_sq', 0), ('tenure_var', 0),
        ('completed_tenure_obs', 1), ('obs_ct_x_censor', 1),
        ('imputed_completed_tenure', 3),
        ('obs_ct_x_exp_sq', 2), ('obs_ct_x_tenure', 2),
    ]
    var_present = sum(1 for var, col in var_checks if var in models[col].params.index)
    var_score = 10 * (var_present / len(var_checks))
    details.append(f"\nVariables present: {var_present}/{len(var_checks)} => {var_score:.1f}/10 pts")

    # --- R-squared (10 pts) ---
    r2_points = 0
    for i in range(5):
        gen_r2 = models[i].rsquared
        true_r2 = gt['r_squared'][i]
        if abs(gen_r2 - true_r2) <= 0.02:
            r2_points += 1
            details.append(f"  R2 MATCH col({i+1}): gen={gen_r2:.4f} vs true={true_r2}")
        else:
            details.append(f"  R2 MISS col({i+1}): gen={gen_r2:.4f} vs true={true_r2} (diff={abs(gen_r2-true_r2):.4f})")
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
