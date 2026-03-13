#!/usr/bin/env python3
"""
Table 7 Replication - Attempt 3
Topel (1991) - "Specific Capital, Mobility, and Wages"

Changes from attempt 2:
- Use GNP deflator for levels wage (+ CPS index)
- Use lagged GNP deflator (year-1) as Table 3 does
- Improved hazard model for imputed completed tenure
- Focus on matching coefficient magnitudes
- Accept that interaction sign may differ due to data limitations
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

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
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5, 1972: 41.8,
    1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9, 1977: 60.6, 1978: 65.2,
    1979: 72.6, 1980: 82.4, 1981: 90.9, 1982: 100.0
}

EDUC_CAT_TO_YEARS = {
    0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17, 9: 17
}

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

    print("Loading PSID panel data...")
    df = pd.read_csv(data_source)
    print(f"  Raw panel: {len(df)} observations, {df['person_id'].nunique()} persons")

    # Education recoding
    df['education_years'] = df['education_clean'].copy()
    for yr in df['year'].unique():
        mask = df['year'] == yr
        max_ed = df.loc[mask, 'education_clean'].max()
        if max_ed <= 9:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_CAT_TO_YEARS)

    # Experience
    df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=1)
    df['experience_sq'] = df['experience'] ** 2

    # Wage deflation: GNP deflator (lagged by 1 year, since PSID income is from prior year)
    # + CPS wage index
    df['gnp_defl'] = df['year'].map(lambda y: GNP_DEFLATOR.get(y - 1, GNP_DEFLATOR.get(y, 100.0)))
    df['cps_idx'] = df['year'].map(CPS_WAGE_INDEX)
    df['log_real_wage'] = df['log_hourly_wage'] - np.log(df['gnp_defl'] / 100.0) - np.log(df['cps_idx'])

    print(f"  Mean education (years): {df['education_years'].mean():.1f}")
    print(f"  Mean experience: {df['experience'].mean():.1f}")
    print(f"  Mean log real wage: {df['log_real_wage'].mean():.3f}")

    # Controls
    df['union'] = df['union_member'].fillna(0)
    df['disability'] = df['disabled'].fillna(0)
    df['smsa'] = df['lives_in_smsa'].fillna(0)
    df['married_d'] = df['married'].fillna(0)

    # Tenure (linear)
    df['tenure_var'] = df['tenure_topel'].astype(float)

    # Completed tenure
    print("\nConstructing completed tenure...")
    df['ct_obs'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
    df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)

    # Interaction: ct * censor
    df['ct_x_censor'] = df['ct_obs'] * df['censor']

    # Interactions for unrestricted model
    df['ct_x_exp_sq'] = df['ct_obs'] * df['experience_sq']
    df['ct_x_tenure'] = df['ct_obs'] * df['tenure_var']

    print(f"  Mean completed tenure: {df['ct_obs'].mean():.2f}")
    print(f"  Censored fraction: {df['censor'].mean():.3f}")

    # Imputed completed tenure via discrete-time hazard model
    print("\nFitting hazard model for imputed completed tenure...")
    df_sorted = df.sort_values(['person_id', 'job_id', 'year']).copy()

    # Job ending indicator
    df_sorted['next_year_in_job'] = df_sorted.groupby(['person_id', 'job_id'])['year'].shift(-1)
    df_sorted['job_max_year'] = df_sorted.groupby('job_id')['year'].transform('max')
    df_sorted['job_ends'] = 0
    df_sorted.loc[
        (df_sorted['next_year_in_job'].isna()) & (df_sorted['job_max_year'] < 1983),
        'job_ends'
    ] = 1

    # Hazard model covariates (simpler set to avoid singularity)
    haz_vars = ['experience', 'education_years', 'married_d', 'tenure_var']

    # Fill NaN before fitting
    for v in haz_vars:
        df_sorted[v] = df_sorted[v].fillna(0)

    haz_sample = df_sorted.dropna(subset=['job_ends']).copy()
    X_haz = sm.add_constant(haz_sample[haz_vars])
    y_haz = haz_sample['job_ends']

    try:
        haz_model = sm.Logit(y_haz, X_haz).fit(disp=0, maxiter=100, method='bfgs')
        print(f"  Hazard model pseudo-R2: {haz_model.prsquared:.3f}")

        # Predict hazard for all observations
        X_all_haz = sm.add_constant(df_sorted[haz_vars])
        df_sorted['hazard_prob'] = haz_model.predict(X_all_haz)
        df_sorted['hazard_prob'] = df_sorted['hazard_prob'].clip(lower=0.01, upper=0.99)

        # Expected remaining tenure
        df_sorted['exp_remaining'] = (1 - df_sorted['hazard_prob']) / df_sorted['hazard_prob']
        df_sorted['exp_remaining'] = df_sorted['exp_remaining'].clip(upper=30)

        # Imputed completed tenure = current tenure + expected remaining
        df_sorted['imp_ct'] = df_sorted['tenure_var'] + df_sorted['exp_remaining']

        # For uncensored jobs, use actual observed completed tenure
        df_sorted.loc[df_sorted['censor'] == 0, 'imp_ct'] = df_sorted.loc[df_sorted['censor'] == 0, 'ct_obs']

        print(f"  Mean imputed completed tenure: {df_sorted['imp_ct'].mean():.2f}")
        df = df_sorted.copy()

    except Exception as e:
        print(f"  Hazard model failed: {e}")
        # Fallback: predict from OLS on uncensored jobs (use first obs per job)
        job_first = df.groupby('job_id').first().reset_index()
        uncensored = job_first[job_first['censor'] == 0].copy()
        pred_vars = ['experience', 'education_years', 'married_d', 'tenure_var']
        for v in pred_vars:
            uncensored[v] = uncensored[v].fillna(0)
            df[v] = df[v].fillna(0)
        X_unc = sm.add_constant(uncensored[pred_vars])
        y_unc = uncensored['ct_obs']
        pred_model = sm.OLS(y_unc, X_unc).fit()
        print(f"  Prediction model R2: {pred_model.rsquared:.3f}")

        X_all = sm.add_constant(df[pred_vars])
        df['imp_ct'] = pred_model.predict(X_all)
        df.loc[df['censor'] == 0, 'imp_ct'] = df.loc[df['censor'] == 0, 'ct_obs']
        print(f"  Mean imputed completed tenure: {df['imp_ct'].mean():.2f}")

    # Imputed CT interactions
    df['imp_ct_x_exp_sq'] = df['imp_ct'] * df['experience_sq']
    df['imp_ct_x_tenure'] = df['imp_ct'] * df['tenure_var']

    # Year dummies (drop reference)
    yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]

    # Region dummies
    region_cols = ['region_ne', 'region_nc', 'region_south']

    # Controls
    control_vars = ['education_years', 'married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

    # Sample
    all_vars = ['log_real_wage', 'experience', 'experience_sq', 'tenure_var',
                'ct_obs', 'censor', 'imp_ct'] + control_vars
    sample = df.dropna(subset=all_vars).copy()
    print(f"\n  Analysis sample: {len(sample)} observations")

    # Base variables
    base_vars = ['experience', 'experience_sq', 'tenure_var']
    y = sample['log_real_wage']

    # Column (1): Baseline
    X1 = sm.add_constant(sample[base_vars + control_vars])
    model1 = sm.OLS(y, X1).fit()

    # Column (2): Restricted observed CT
    X2 = sm.add_constant(sample[base_vars + ['ct_obs', 'ct_x_censor'] + control_vars])
    model2 = sm.OLS(y, X2).fit()

    # Column (3): Unrestricted observed CT
    X3 = sm.add_constant(sample[base_vars + ['ct_obs', 'ct_x_censor',
                                              'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])
    model3 = sm.OLS(y, X3).fit()

    # Column (4): Restricted imputed CT
    X4 = sm.add_constant(sample[base_vars + ['imp_ct'] + control_vars])
    model4 = sm.OLS(y, X4).fit()

    # Column (5): Unrestricted imputed CT
    X5 = sm.add_constant(sample[base_vars + ['imp_ct',
                                              'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])
    model5 = sm.OLS(y, X5).fit()

    models = [model1, model2, model3, model4, model5]

    # Print results
    print("\n" + "=" * 100)
    print("TABLE 7: Least-Squares Models Conditioning on (Estimated) Completed Job Tenure")
    print("PSID White Males")
    print("=" * 100)

    def fmt(model, var):
        if var in model.params.index:
            coef = model.params[var]
            se = model.bse[var]
            pval = model.pvalues[var]
            stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            return f"{coef:>10.5f}{stars}", f"({se:.5f})"
        return f"{'...':>13s}", ""

    print(f"\n{'Variable':40s}" + "".join([f"{'('+str(i+1)+')':>16s}" for i in range(5)]))
    print("-" * 120)

    rows = [
        ('Experience', ['experience'] * 5),
        ('Experience^2', ['experience_sq'] * 5),
        ('Tenure', ['tenure_var'] * 5),
        ('Observed completed tenure', [None, 'ct_obs', 'ct_obs', None, None]),
        ('x censor', [None, 'ct_x_censor', 'ct_x_censor', None, None]),
        ('Imputed completed tenure', [None, None, None, 'imp_ct', 'imp_ct']),
        ('Experience^2 (interaction)', [None, None, 'ct_x_exp_sq', None, 'imp_ct_x_exp_sq']),
        ('Tenure (interaction)', [None, None, 'ct_x_tenure', None, 'imp_ct_x_tenure']),
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
                c, s_str = fmt(models[i], var)
                coef_line += f"{c:>16s}"
                if s_str:
                    se_line += f"{s_str:>16s}"
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

    # Raw coefficients
    print("\n\n=== RAW COEFFICIENTS ===")
    key_vars = {
        0: ['experience', 'experience_sq', 'tenure_var'],
        1: ['experience', 'experience_sq', 'tenure_var', 'ct_obs', 'ct_x_censor'],
        2: ['experience', 'experience_sq', 'tenure_var', 'ct_obs', 'ct_x_censor',
            'ct_x_exp_sq', 'ct_x_tenure'],
        3: ['experience', 'experience_sq', 'tenure_var', 'imp_ct'],
        4: ['experience', 'experience_sq', 'tenure_var', 'imp_ct',
            'imp_ct_x_exp_sq', 'imp_ct_x_tenure'],
    }

    for i, m in enumerate(models):
        print(f"\nColumn ({i+1}):")
        print(f"  R^2 = {m.rsquared:.4f}")
        print(f"  N = {int(m.nobs)}")
        for var in key_vars[i]:
            if var in m.params.index:
                coef = m.params[var]
                se = m.bse[var]
                pval = m.pvalues[var]
                stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                print(f"  {var}: coef={coef:.6f}, se={se:.6f} {stars}")

    return models


def score_against_ground_truth():
    """Compute score."""
    gt = GROUND_TRUTH
    try:
        models = run_analysis()
    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()
        return 0

    details = []

    coef_map = {
        'experience': {i: 'experience' for i in range(5)},
        'experience_sq': {i: 'experience_sq' for i in range(5)},
        'tenure': {i: 'tenure_var' for i in range(5)},
        'obs_completed_tenure': {1: 'ct_obs', 2: 'ct_obs'},
        'x_censor': {1: 'ct_x_censor', 2: 'ct_x_censor'},
        'imp_completed_tenure': {3: 'imp_ct', 4: 'imp_ct'},
        'exp_sq_interaction': {2: 'ct_x_exp_sq', 4: 'imp_ct_x_exp_sq'},
        'tenure_interaction': {2: 'ct_x_tenure', 4: 'imp_ct_x_tenure'},
    }

    se_map = {k: k + '_se' for k in coef_map}

    # Coefficients (25 pts)
    coef_pts, coef_max = 0, 0
    for gt_key, col_vars in coef_map.items():
        for col_idx, model_var in col_vars.items():
            gt_val = gt[gt_key][col_idx]
            if gt_val is None:
                continue
            coef_max += 1
            if model_var in models[col_idx].params.index:
                gen_val = models[col_idx].params[model_var]
                if abs(gt_val) < 0.01:
                    match = abs(gen_val - gt_val) / max(abs(gt_val), 1e-8) <= 0.20
                else:
                    match = abs(gen_val - gt_val) <= 0.05
                if match:
                    coef_pts += 1
                    details.append(f"  COEF MATCH {gt_key} col({col_idx+1}): {gen_val:.6f} vs {gt_val}")
                else:
                    details.append(f"  COEF MISS {gt_key} col({col_idx+1}): {gen_val:.6f} vs {gt_val}")
            else:
                details.append(f"  COEF MISS {gt_key} col({col_idx+1}): var not found")
    coef_score = 25 * coef_pts / max(coef_max, 1)
    details.append(f"\nCoefficients: {coef_pts}/{coef_max} => {coef_score:.1f}/25")

    # SEs (15 pts)
    se_pts, se_max = 0, 0
    for gt_key, col_vars in coef_map.items():
        se_key = se_map[gt_key]
        if se_key not in gt:
            continue
        for col_idx, model_var in col_vars.items():
            gt_se = gt[se_key][col_idx]
            if gt_se is None:
                continue
            se_max += 1
            if model_var in models[col_idx].params.index:
                gen_se = models[col_idx].bse[model_var]
                if abs(gen_se - gt_se) <= 0.02:
                    se_pts += 1
                    details.append(f"  SE MATCH {gt_key} col({col_idx+1})")
                else:
                    details.append(f"  SE MISS {gt_key} col({col_idx+1}): {gen_se:.6f} vs {gt_se}")
    se_score = 15 * se_pts / max(se_max, 1)
    details.append(f"\nSEs: {se_pts}/{se_max} => {se_score:.1f}/15")

    # N (15 pts)
    gen_n = int(models[0].nobs)
    n_ratio = abs(gen_n - 13128) / 13128
    n_score = 15 if n_ratio <= 0.05 else 10 if n_ratio <= 0.10 else 5 if n_ratio <= 0.20 else 0
    details.append(f"\nN: {gen_n} vs ~13128, err={n_ratio:.3f} => {n_score}/15")

    # Significance (25 pts)
    def stars(coef, se):
        if se == 0: return ''
        t = abs(coef / se)
        return '***' if t > 3.291 else '**' if t > 2.576 else '*' if t > 1.96 else ''

    sig_pts, sig_max = 0, 0
    for gt_key, col_vars in coef_map.items():
        se_key = se_map[gt_key]
        if se_key not in gt:
            continue
        for col_idx, model_var in col_vars.items():
            gt_c, gt_s = gt[gt_key][col_idx], gt[se_key][col_idx]
            if gt_c is None or gt_s is None:
                continue
            sig_max += 1
            ts = stars(gt_c, gt_s)
            if model_var in models[col_idx].params.index:
                pval = models[col_idx].pvalues[model_var]
                gs = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                if gs == ts:
                    sig_pts += 1
    sig_score = 25 * sig_pts / max(sig_max, 1)
    details.append(f"\nSignificance: {sig_pts}/{sig_max} => {sig_score:.1f}/25")

    # Variables (10 pts)
    var_checks = [('experience', 0), ('experience_sq', 0), ('tenure_var', 0),
                  ('ct_obs', 1), ('ct_x_censor', 1), ('imp_ct', 3),
                  ('ct_x_exp_sq', 2), ('ct_x_tenure', 2)]
    var_pts = sum(1 for v, c in var_checks if v in models[c].params.index)
    var_score = 10 * var_pts / len(var_checks)
    details.append(f"\nVariables: {var_pts}/{len(var_checks)} => {var_score:.1f}/10")

    # R-squared (10 pts)
    r2_pts = 0
    for i in range(5):
        if abs(models[i].rsquared - gt['r_squared'][i]) <= 0.02:
            r2_pts += 1
            details.append(f"  R2 MATCH col({i+1}): {models[i].rsquared:.4f} vs {gt['r_squared'][i]}")
        else:
            details.append(f"  R2 MISS col({i+1}): {models[i].rsquared:.4f} vs {gt['r_squared'][i]}")
    r2_score = 10 * r2_pts / 5
    details.append(f"\nR2: {r2_pts}/5 => {r2_score:.1f}/10")

    total = coef_score + se_score + n_score + sig_score + var_score + r2_score
    print("\n" + "=" * 60)
    print("SCORING BREAKDOWN")
    print("=" * 60)
    for d in details:
        print(d)
    print(f"\nTOTAL: {total:.1f}/100")

    return total


if __name__ == '__main__':
    score = score_against_ground_truth()
