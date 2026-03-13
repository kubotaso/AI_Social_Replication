"""
Replication of Table 6 from Bernanke and Blinder (1992)
"Estimated Slope of Supply Function of Nonborrowed Reserves"

Method:
1. For each instrument set, run a 5-variable VAR with 6 lags
2. Extract VAR residuals (innovations)
3. Run IV (2SLS) regression of innovation(FUNDS/FFBOND) on innovation(NBR),
   using the 3 macro instrument innovations as IVs
4. Report beta coefficient and standard error
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def manual_2sls(y, X_endog, Z):
    """
    Manual 2SLS estimation.
    y: dependent variable (T,)
    X_endog: endogenous regressor (T,) - will add constant
    Z: instruments (T, k) - will add constant

    Returns: coefficient on X_endog, standard error
    """
    T = len(y)

    # Add constant to endogenous regressor
    X = sm.add_constant(X_endog)

    # Add constant to instruments
    Z_with_const = sm.add_constant(Z)

    # First stage: regress X_endog on instruments
    first_stage = sm.OLS(X_endog, Z_with_const).fit()
    X_hat_endog = first_stage.fittedvalues

    # Create X_hat with constant
    X_hat = sm.add_constant(X_hat_endog)

    # Second stage: regress y on X_hat
    second_stage = sm.OLS(y, X_hat).fit()
    beta = second_stage.params

    # Correct standard errors: use original X, not X_hat
    # Residuals from second stage using original X
    resid = y - X @ beta
    sigma2 = np.sum(resid**2) / (T - X.shape[1])

    # Variance: sigma2 * (X_hat'X_hat)^{-1} * X_hat'X * (X_hat'X_hat)^{-1}
    # But standard 2SLS formula is: sigma2 * (X'Pz X)^{-1}
    # where Pz = Z(Z'Z)^{-1}Z'
    Pz = Z_with_const @ np.linalg.inv(Z_with_const.T @ Z_with_const) @ Z_with_const.T
    XPzX_inv = np.linalg.inv(X.T @ Pz @ X)
    var_beta = sigma2 * XPzX_inv
    se = np.sqrt(np.diag(var_beta))

    # Return coefficient and SE for the slope (index 1, after constant)
    return beta[1], se[1]


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
            # Construct VAR variables: [macro1, macro2, macro3, NBR, policy]
            var_cols = macro_vars + [nbr_var, policy_col]

            # Subset to sample period
            var_data = df.loc[sample_start:sample_end, var_cols].copy()
            var_data = var_data.dropna()

            # Estimate VAR with 6 lags
            model = VAR(var_data)
            var_result = model.fit(maxlags=6, ic=None, trend='c')

            # Extract residuals
            resid = var_result.resid

            # Innovations for instruments (macro variables)
            Z = resid[macro_vars].values

            # Innovation for NBR (endogenous regressor)
            X_endog = resid[nbr_var].values

            # Innovation for policy variable (dependent variable)
            y = resid[policy_col].values

            # Run 2SLS
            beta, se = manual_2sls(y, X_endog, Z)

            results[set_name][policy_name] = {
                'coeff': beta,
                'se': se,
                'nobs': len(y),
                'var_nobs': var_result.nobs
            }

    # Format output
    output_lines = []
    output_lines.append("Table 6: Estimated Slope of Supply Function of Nonborrowed Reserves")
    output_lines.append("=" * 70)
    output_lines.append(f"Sample: {sample_start} to {sample_end}, 6 lags in VAR")
    output_lines.append("")
    output_lines.append(f"{'':20s} {'FUNDS':>12s} {'FFBOND':>12s}")
    output_lines.append("-" * 44)

    for set_name in ['Set A', 'Set B', 'Set C']:
        funds = results[set_name]['FUNDS']
        ffbond = results[set_name]['FFBOND']
        output_lines.append(f"{set_name:20s} {funds['coeff']:12.4f} {ffbond['coeff']:12.4f}")
        output_lines.append(f"{'':20s} ({funds['se']:10.4f}) ({ffbond['se']:10.4f})")
        output_lines.append(f"{'':20s} [N={funds['nobs']:3d}]       [N={ffbond['nobs']:3d}]")
        output_lines.append("")

    result_text = '\n'.join(output_lines)
    print(result_text)

    # Also print detailed results for scoring
    print("\n\nDetailed results for scoring:")
    print("=" * 50)
    for set_name in ['Set A', 'Set B', 'Set C']:
        for policy_name in ['FUNDS', 'FFBOND']:
            r = results[set_name][policy_name]
            print(f"{set_name} {policy_name}: coeff={r['coeff']:.6f}, SE={r['se']:.6f}, N={r['nobs']}")

    # Score against ground truth
    score, breakdown = score_against_ground_truth(results)
    print(f"\n\nAUTOMATED SCORE: {score}/100")
    print(breakdown)

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
    sign_score = int(25 * sign_count / total_signs)
    breakdown_lines.append(f"Signs correct: {sign_count}/{total_signs} -> {sign_score}/25")

    # Criterion 3: Data values accuracy (60 pts)
    # Each coefficient within 20% relative error or 0.005 absolute
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
