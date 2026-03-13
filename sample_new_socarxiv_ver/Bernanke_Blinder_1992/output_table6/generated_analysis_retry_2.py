"""
Replication of Table 6 from Bernanke and Blinder (1992)
"Estimated Slope of Supply Function of Nonborrowed Reserves"

Attempt 2:
- Scale coefficients by 0.01 (paper reports response to 1% NBR innovation)
- Try LIML estimator for robustness to weak instruments
- Use linearmodels IV2SLS with proper specification
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from linearmodels.iv import IV2SLS, IVLIML
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def run_iv_regression(y, x_endog, z_instruments, method='2sls'):
    """
    Run IV regression: y = a + b*x_endog, instrumented by z_instruments.
    Returns (coefficient, standard_error) for the slope.
    """
    y_s = pd.Series(y, name='y') if not isinstance(y, pd.Series) else y
    x_df = pd.DataFrame(x_endog, columns=['x']) if not isinstance(x_endog, pd.DataFrame) else x_endog
    z_df = pd.DataFrame(z_instruments) if not isinstance(z_instruments, pd.DataFrame) else z_instruments

    # Ensure aligned indices
    idx = y_s.index
    exog = pd.DataFrame(np.ones(len(idx)), index=idx, columns=['const'])

    if method == 'liml':
        result = IVLIML(dependent=y_s, exog=exog, endog=x_df, instruments=z_df).fit()
    else:
        result = IV2SLS(dependent=y_s, exog=exog, endog=x_df, instruments=z_df).fit()

    # Get slope coefficient (last one, after constant)
    coeff = result.params.iloc[-1]
    se = result.std_errors.iloc[-1]
    return coeff, se


def run_analysis(data_source):
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')
    df.index.freq = 'MS'

    # Define instrument sets
    instrument_sets = {
        'Set A': ['log_industrial_production', 'log_capacity_utilization', 'log_employment'],
        'Set B': ['unemp_male_2554', 'log_housing_starts', 'log_personal_income_real'],
        'Set C': ['log_retail_sales_real', 'log_consumption_real'],  # durable goods missing
    }

    policy_vars = {
        'FUNDS': 'funds_rate',
        'FFBOND': 'ffbond'
    }

    nbr_var = 'log_nonborrowed_reserves_real'

    # Sample period
    sample_start = '1959-08'
    sample_end = '1979-09'

    results = {}

    for set_name, macro_vars in instrument_sets.items():
        results[set_name] = {}
        for policy_name, policy_col in policy_vars.items():
            # Construct VAR variables
            var_cols = macro_vars + [nbr_var, policy_col]

            # Subset to sample period
            var_data = df.loc[sample_start:sample_end, var_cols].copy()
            var_data = var_data.dropna()

            # Estimate VAR with 6 lags
            model = VAR(var_data)
            var_result = model.fit(maxlags=6, ic=None, trend='c')

            # Extract residuals
            resid = var_result.resid

            # Run IV regression: both 2SLS and LIML
            y = resid[policy_col]
            x = resid[[nbr_var]]
            z = resid[macro_vars]

            # 2SLS
            beta_2sls, se_2sls = run_iv_regression(y, x, z, method='2sls')

            # LIML
            beta_liml, se_liml = run_iv_regression(y, x, z, method='liml')

            # OLS for comparison
            ols = sm.OLS(y.values, sm.add_constant(resid[nbr_var].values)).fit()
            beta_ols = ols.params[1]
            se_ols = ols.bse[1]

            # First stage diagnostics
            fs = sm.OLS(resid[nbr_var].values, sm.add_constant(resid[macro_vars].values)).fit()

            # Paper reports: response to 1% innovation in NBR
            # Since NBR is in log, 1% = 0.01 in log units
            # So reported coefficient = raw_beta * 0.01
            scale = 0.01

            results[set_name][policy_name] = {
                'coeff_2sls': beta_2sls * scale,
                'se_2sls': se_2sls * scale,
                'coeff_liml': beta_liml * scale,
                'se_liml': se_liml * scale,
                'coeff_ols': beta_ols * scale,
                'se_ols': se_ols * scale,
                'raw_beta_2sls': beta_2sls,
                'nobs': len(y),
                'fs_f': fs.fvalue,
                'fs_r2': fs.rsquared,
            }

    # Format output
    output_lines = []
    output_lines.append("Table 6: Estimated Slope of Supply Function of Nonborrowed Reserves")
    output_lines.append("=" * 70)
    output_lines.append(f"Sample: {sample_start} to {sample_end}, 6 lags in VAR")
    output_lines.append("Coefficients scaled: response of policy rate (pp) to 1% NBR innovation")
    output_lines.append("")

    # Main results table (using 2SLS, which is what the paper uses)
    output_lines.append("=== 2SLS Results (paper's method) ===")
    output_lines.append(f"{'':20s} {'FUNDS':>12s} {'FFBOND':>12s}")
    output_lines.append("-" * 44)

    for set_name in ['Set A', 'Set B', 'Set C']:
        funds = results[set_name]['FUNDS']
        ffbond = results[set_name]['FFBOND']
        output_lines.append(f"{set_name:20s} {funds['coeff_2sls']:12.4f} {ffbond['coeff_2sls']:12.4f}")
        output_lines.append(f"{'':20s} ({funds['se_2sls']:10.4f}) ({ffbond['se_2sls']:10.4f})")
        output_lines.append("")

    output_lines.append("")
    output_lines.append("=== LIML Results (robust to weak instruments) ===")
    output_lines.append(f"{'':20s} {'FUNDS':>12s} {'FFBOND':>12s}")
    output_lines.append("-" * 44)

    for set_name in ['Set A', 'Set B', 'Set C']:
        funds = results[set_name]['FUNDS']
        ffbond = results[set_name]['FFBOND']
        output_lines.append(f"{set_name:20s} {funds['coeff_liml']:12.4f} {ffbond['coeff_liml']:12.4f}")
        output_lines.append(f"{'':20s} ({funds['se_liml']:10.4f}) ({ffbond['se_liml']:10.4f})")
        output_lines.append("")

    output_lines.append("")
    output_lines.append("=== OLS Results (for comparison) ===")
    output_lines.append(f"{'':20s} {'FUNDS':>12s} {'FFBOND':>12s}")
    output_lines.append("-" * 44)

    for set_name in ['Set A', 'Set B', 'Set C']:
        funds = results[set_name]['FUNDS']
        ffbond = results[set_name]['FFBOND']
        output_lines.append(f"{set_name:20s} {funds['coeff_ols']:12.4f} {ffbond['coeff_ols']:12.4f}")
        output_lines.append(f"{'':20s} ({funds['se_ols']:10.4f}) ({ffbond['se_ols']:10.4f})")
        output_lines.append("")

    output_lines.append("")
    output_lines.append("=== First Stage Diagnostics ===")
    for set_name in ['Set A', 'Set B', 'Set C']:
        funds = results[set_name]['FUNDS']
        output_lines.append(f"{set_name}: F={funds['fs_f']:.3f}, R2={funds['fs_r2']:.4f}, N={funds['nobs']}")

    result_text = '\n'.join(output_lines)
    print(result_text)

    # Score using 2SLS results (paper's method)
    score_results = {}
    for sn in ['Set A', 'Set B', 'Set C']:
        score_results[sn] = {}
        for pn in ['FUNDS', 'FFBOND']:
            score_results[sn][pn] = {
                'coeff': results[sn][pn]['coeff_2sls'],
                'se': results[sn][pn]['se_2sls'],
            }

    score, breakdown = score_against_ground_truth(score_results)
    print(f"\n\nAUTOMATED SCORE (2SLS): {score}/100")
    print(breakdown)

    # Also score LIML
    score_results_liml = {}
    for sn in ['Set A', 'Set B', 'Set C']:
        score_results_liml[sn] = {}
        for pn in ['FUNDS', 'FFBOND']:
            score_results_liml[sn][pn] = {
                'coeff': results[sn][pn]['coeff_liml'],
                'se': results[sn][pn]['se_liml'],
            }

    score_liml, breakdown_liml = score_against_ground_truth(score_results_liml)
    print(f"\n\nAUTOMATED SCORE (LIML): {score_liml}/100")
    print(breakdown_liml)

    # Also score OLS
    score_results_ols = {}
    for sn in ['Set A', 'Set B', 'Set C']:
        score_results_ols[sn] = {}
        for pn in ['FUNDS', 'FFBOND']:
            score_results_ols[sn][pn] = {
                'coeff': results[sn][pn]['coeff_ols'],
                'se': results[sn][pn]['se_ols'],
            }

    score_ols, breakdown_ols = score_against_ground_truth(score_results_ols)
    print(f"\n\nAUTOMATED SCORE (OLS): {score_ols}/100")
    print(breakdown_ols)

    return result_text


def score_against_ground_truth(results):
    """Score results against ground truth from the paper."""
    ground_truth = {
        'Set A': {'FUNDS': {'coeff': -0.021, 'se': 0.023},
                  'FFBOND': {'coeff': -0.011, 'se': 0.015}},
        'Set B': {'FUNDS': {'coeff': -0.0068, 'se': 0.0104},
                  'FFBOND': {'coeff': -0.0072, 'se': 0.0092}},
        'Set C': {'FUNDS': {'coeff': -0.014, 'se': 0.016},
                  'FFBOND': {'coeff': -0.014, 'se': 0.016}},
    }

    breakdown_lines = []

    # Criterion 1: All rows present (15 pts)
    all_present = True
    for sn in ['Set A', 'Set B', 'Set C']:
        for pn in ['FUNDS', 'FFBOND']:
            if sn not in results or pn not in results[sn]:
                all_present = False
    presence_score = 15 if all_present else 0
    breakdown_lines.append(f"Presence (all sets/policies): {presence_score}/15")

    # Criterion 2: Signs correct (25 pts)
    sign_count = 0
    total_signs = 6
    for sn in ['Set A', 'Set B', 'Set C']:
        for pn in ['FUNDS', 'FFBOND']:
            gt = ground_truth[sn][pn]['coeff']
            gen = results[sn][pn]['coeff']
            if (gt < 0 and gen < 0) or (gt > 0 and gen > 0) or (gt == 0 and abs(gen) < 0.001):
                sign_count += 1
                breakdown_lines.append(f"  {sn} {pn}: SIGN OK (gen={gen:.6f}, gt={gt:.4f})")
            else:
                breakdown_lines.append(f"  {sn} {pn}: SIGN WRONG (gen={gen:.6f}, gt={gt:.4f})")
    sign_score = int(25 * sign_count / total_signs)
    breakdown_lines.append(f"Signs correct: {sign_count}/{total_signs} -> {sign_score}/25")

    # Criterion 3: Data values accuracy (60 pts)
    coeff_matches = 0
    total_coeffs = 6
    for sn in ['Set A', 'Set B', 'Set C']:
        for pn in ['FUNDS', 'FFBOND']:
            gt = ground_truth[sn][pn]['coeff']
            gen = results[sn][pn]['coeff']
            abs_err = abs(gen - gt)
            rel_err = abs_err / max(abs(gt), 1e-10) if gt != 0 else abs_err
            match = abs_err <= 0.005 or rel_err <= 0.20
            if match:
                coeff_matches += 1
            breakdown_lines.append(f"  {sn} {pn}: gen={gen:.6f}, gt={gt:.4f}, abs_err={abs_err:.6f}, rel_err={rel_err:.2%}, match={match}")

    accuracy_score = int(60 * coeff_matches / total_coeffs)
    breakdown_lines.append(f"Accuracy: {coeff_matches}/{total_coeffs} -> {accuracy_score}/60")

    total = presence_score + sign_score + accuracy_score
    breakdown = '\n'.join(breakdown_lines)
    return total, breakdown


if __name__ == "__main__":
    result = run_analysis("bb1992_data.csv")
