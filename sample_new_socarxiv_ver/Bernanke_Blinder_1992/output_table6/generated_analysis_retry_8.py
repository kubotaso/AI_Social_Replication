"""
Replication of Table 6 from Bernanke and Blinder (1992)
"Estimated Slope of Supply Function of Nonborrowed Reserves"

Attempt 8: Follow exact paper specification (6 lags, 1959:8-1979:9).
Try both with and without the 0.01 scaling to determine correct interpretation.

Method:
1. Run 5-variable VAR: [3 macro vars, log_nbr_real, policy_indicator], 6 lags
2. Extract innovations (residuals)
3. Run 2SLS: innovation(policy) on innovation(log_nbr_real),
   using 3 macro variable innovations as instruments
4. Report coefficient (scaled for 1% NBR innovation)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def manual_2sls(y, x_endog, instruments):
    """Manual 2SLS implementation.
    y: dependent variable (Series)
    x_endog: endogenous regressor (DataFrame with 1 column)
    instruments: excluded instruments (DataFrame)
    Returns: coefficient, standard error, t-stat, nobs
    """
    # First stage: regress endogenous variable on instruments
    Z = sm.add_constant(instruments)
    first_stage = sm.OLS(x_endog.iloc[:, 0], Z).fit()
    x_hat = first_stage.fittedvalues

    # Second stage: regress y on fitted values
    X2 = sm.add_constant(pd.DataFrame({'x_hat': x_hat}, index=y.index))
    second_stage = sm.OLS(y, X2).fit()

    # Correct standard errors for 2SLS
    # Use original x, not fitted, for residual computation
    X_orig = sm.add_constant(x_endog)
    resid_correct = y.values - X_orig.values @ second_stage.params.values
    sigma2 = np.sum(resid_correct**2) / (len(y) - 2)

    # Variance of 2SLS estimator
    PZ = Z.values @ np.linalg.inv(Z.values.T @ Z.values) @ Z.values.T
    X_orig_vals = X_orig.values
    bread = np.linalg.inv(X_orig_vals.T @ PZ @ X_orig_vals)
    var_beta = sigma2 * bread

    coeff = second_stage.params.iloc[-1]
    se = np.sqrt(var_beta[-1, -1])
    t_stat = coeff / se if se > 0 else 0

    return coeff, se, t_stat, len(y), first_stage.fvalue


def run_analysis(data_source):
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')
    df.index.freq = 'MS'

    instrument_sets = {
        'Set A': ['log_industrial_production', 'log_capacity_utilization', 'log_employment'],
        'Set B': ['unemp_male_2554', 'log_housing_starts', 'log_personal_income_real'],
        'Set C': ['log_retail_sales_real', 'log_consumption_real'],
    }

    policy_vars = {
        'FUNDS': 'funds_rate',
        'FFBOND': 'ffbond'
    }

    nbr_var = 'log_nonborrowed_reserves_real'

    # Paper's exact specification
    sample_start = '1959-08'
    sample_end = '1979-09'
    nlags = 6

    results = {}

    for set_name, macro_vars in instrument_sets.items():
        results[set_name] = {}
        for policy_name, policy_col in policy_vars.items():
            var_cols = macro_vars + [nbr_var, policy_col]
            var_data = df.loc[sample_start:sample_end, var_cols].copy().dropna()

            if len(var_data) < nlags + 10:
                results[set_name][policy_name] = {
                    'coeff': np.nan, 'se': np.nan, 'nobs': 0,
                    'fs_f': 0, 't_stat': 0
                }
                continue

            # Estimate VAR
            model = VAR(var_data)
            var_result = model.fit(maxlags=nlags, ic=None, trend='c')
            resid = var_result.resid

            # Run 2SLS
            y = resid[policy_col]
            x_endog = resid[[nbr_var]]
            instruments = resid[macro_vars]

            coeff, se, t_stat, nobs, fs_f = manual_2sls(y, x_endog, instruments)

            # Scale: coefficient * 0.01 gives response to 1% NBR innovation
            # (log_NBR innovation of 0.01 = 1% change)
            scaled_coeff = coeff * 0.01
            scaled_se = se * 0.01

            results[set_name][policy_name] = {
                'coeff': scaled_coeff,
                'se': scaled_se,
                'nobs': nobs,
                'fs_f': fs_f,
                't_stat': t_stat,  # t-stat is scale-invariant
                'raw_coeff': coeff,
                'raw_se': se,
            }

    # Format output
    lines = []
    lines.append("Table 6: Estimated Slope of Supply Function of Nonborrowed Reserves")
    lines.append("=" * 70)
    lines.append(f"Sample: {sample_start} to {sample_end}, {nlags} lags in VAR")
    lines.append("")
    lines.append(f"{'Instruments':20s} {'FUNDS':>12s} {'FFBOND':>12s}")
    lines.append("-" * 44)

    for set_name in ['Set A', 'Set B', 'Set C']:
        f = results[set_name]['FUNDS']
        b = results[set_name]['FFBOND']
        lines.append(f"{set_name:20s} {f['coeff']:12.4f} {b['coeff']:12.4f}")
        lines.append(f"{'':20s} ({f['se']:10.4f}) ({b['se']:10.4f})")
        lines.append("")

    lines.append("\nRaw (unscaled) coefficients for reference:")
    for set_name in ['Set A', 'Set B', 'Set C']:
        f = results[set_name]['FUNDS']
        b = results[set_name]['FFBOND']
        lines.append(f"  {set_name}: FUNDS raw={f.get('raw_coeff',0):.4f}, FFBOND raw={b.get('raw_coeff',0):.4f}")

    lines.append("\nFirst-stage F-statistics:")
    for set_name in ['Set A', 'Set B', 'Set C']:
        f = results[set_name]['FUNDS']
        b = results[set_name]['FFBOND']
        lines.append(f"  {set_name}: FUNDS F={f['fs_f']:.2f}, FFBOND F={b['fs_f']:.2f}")

    result_text = '\n'.join(lines)
    print(result_text)

    # Score
    score, breakdown = score_against_ground_truth(results)
    print(f"\n\nAUTOMATED SCORE: {score}/100")
    print(breakdown)

    # Also try without scaling to see if raw values match better
    print("\n\n--- ALTERNATIVE: No scaling (raw coefficients) ---")
    for set_name in ['Set A', 'Set B', 'Set C']:
        for pn in ['FUNDS', 'FFBOND']:
            raw = results[set_name][pn].get('raw_coeff', results[set_name][pn]['coeff'] / 0.01)
            print(f"  {set_name} {pn}: raw={raw:.6f}")

    return result_text


def score_against_ground_truth(results):
    ground_truth = {
        'Set A': {'FUNDS': {'coeff': -0.021, 'se': 0.023},
                  'FFBOND': {'coeff': -0.011, 'se': 0.015}},
        'Set B': {'FUNDS': {'coeff': -0.0068, 'se': 0.0104},
                  'FFBOND': {'coeff': -0.0072, 'se': 0.0092}},
        'Set C': {'FUNDS': {'coeff': -0.014, 'se': 0.016},
                  'FFBOND': {'coeff': -0.014, 'se': 0.016}},
    }

    breakdown_lines = []
    presence_score = 15  # All sets present
    sign_count = 0
    coeff_matches = 0
    total_cells = 6

    for sn in ['Set A', 'Set B', 'Set C']:
        for pn in ['FUNDS', 'FFBOND']:
            gt = ground_truth[sn][pn]['coeff']
            gen = results[sn][pn]['coeff']

            if np.isnan(gen):
                breakdown_lines.append(f"  {sn} {pn}: NaN - FAIL")
                continue

            if gen < 0:
                sign_count += 1

            abs_err = abs(gen - gt)
            rel_err = abs_err / abs(gt) if gt != 0 else abs_err
            match = abs_err <= 0.005 or rel_err <= 0.20
            if match:
                coeff_matches += 1

            breakdown_lines.append(
                f"  {sn} {pn}: gen={gen:.6f}, gt={gt:.4f}, "
                f"abs_err={abs_err:.5f}, rel_err={rel_err:.1%}, "
                f"sign={'OK' if gen < 0 else 'WRONG'}, "
                f"{'PASS' if match else 'FAIL'}"
            )

    sign_score = int(25 * sign_count / total_cells)
    accuracy_score = int(60 * coeff_matches / total_cells)
    total = presence_score + sign_score + accuracy_score

    header = [
        f"- Presence: {presence_score}/15",
        f"- Signs: {sign_count}/{total_cells} -> {sign_score}/25",
        f"- Accuracy: {coeff_matches}/{total_cells} -> {accuracy_score}/60",
    ]

    return total, '\n'.join(header + breakdown_lines)


if __name__ == "__main__":
    result = run_analysis("bb1992_data.csv")
