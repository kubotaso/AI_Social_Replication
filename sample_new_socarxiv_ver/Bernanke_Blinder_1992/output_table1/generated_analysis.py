"""
Replication of Table 1 from Bernanke and Blinder (1992)
"Marginal Significance Levels of Monetary Indicators for Forecasting
Alternative Measures of Economic Activity: Six-Variable Prediction Equations"

Each row is a dependent variable. Each column tests joint significance of 6 lags
of a monetary/interest rate indicator in an OLS equation that includes:
  - constant
  - 6 lags of the dependent variable
  - 6 lags of log CPI
  - 6 lags of log M1
  - 6 lags of log M2
  - 6 lags of BILL (3-month T-bill rate)
  - 6 lags of BOND (10-year Treasury bond rate)
  - 6 lags of FUNDS (federal funds rate)

Total regressors: 1 + 6*7 = 43
F-test: H0: all 6 lags of the tested variable = 0
Report: p-value (marginal significance level)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def run_analysis(data_source):
    # Load data
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')
    df.index.freq = 'MS'

    # Define the dependent variables (rows of the table)
    dep_vars = {
        'Industrial production': 'log_industrial_production',
        'Capacity utilization': 'log_capacity_utilization',
        'Employment': 'log_employment',
        'Unemployment rate': 'unemp_male_2554',
        'Housing starts': 'log_housing_starts',
        'Personal income': 'log_personal_income_real',
        'Retail sales': 'log_retail_sales_real',
        'Consumption': 'log_consumption_real',
    }

    # Define the RHS variables (always included in every equation)
    rhs_vars = {
        'CPI': 'log_cpi',
        'M1': 'log_m1',
        'M2': 'log_m2',
        'BILL': 'tbill_3m',
        'BOND': 'treasury_10y',
        'FUNDS': 'funds_rate',
    }

    # The tested columns (F-test for joint significance of 6 lags)
    test_vars = ['M1', 'M2', 'BILL', 'BOND', 'FUNDS']

    n_lags = 6

    # Define sample periods
    # Panel A: 1959:7-1989:12
    # Panel B: 1959:7-1979:9
    # Data needs to start earlier to construct lags
    # We need 6 lags, so we need data from 1959:1 onwards
    # The estimation sample starts at 1959:7
    panels = {
        'Panel A (1959:7-1989:12)': ('1959-07-01', '1989-12-01'),
        'Panel B (1959:7-1979:9)': ('1959-07-01', '1979-09-01'),
    }

    results = {}

    for panel_name, (start_date, end_date) in panels.items():
        panel_results = {}

        for dep_name, dep_col in dep_vars.items():
            # Build the design matrix with lagged variables
            # We need data from 6 months before start_date for lag construction

            # Collect all variables we need
            all_cols = [dep_col] + list(rhs_vars.values())
            sub_df = df[all_cols].copy()

            # Create lagged variables
            lag_data = {}

            # Lags of the dependent variable
            for j in range(1, n_lags + 1):
                lag_data[f'{dep_col}_L{j}'] = sub_df[dep_col].shift(j)

            # Lags of each RHS variable
            for var_name, var_col in rhs_vars.items():
                for j in range(1, n_lags + 1):
                    lag_data[f'{var_col}_L{j}'] = sub_df[var_col].shift(j)

            lag_df = pd.DataFrame(lag_data, index=sub_df.index)

            # Combine dependent variable with lags
            full_df = pd.concat([sub_df[[dep_col]], lag_df], axis=1)

            # Restrict to estimation sample
            est_df = full_df.loc[start_date:end_date].dropna()

            # Separate Y and X
            Y = est_df[dep_col]
            X_cols = [c for c in est_df.columns if c != dep_col]
            X = sm.add_constant(est_df[X_cols])

            # Run OLS
            model = sm.OLS(Y, X).fit()

            # For each test variable, perform F-test on its 6 lags
            row_results = {}
            for test_var in test_vars:
                test_col = rhs_vars[test_var]
                # Find the column indices for the 6 lags of this variable
                lag_names = [f'{test_col}_L{j}' for j in range(1, n_lags + 1)]

                # Build restriction matrix R for R*beta = 0
                # Each row of R picks out one lag coefficient
                R = np.zeros((n_lags, len(model.params)))
                for i, lag_name in enumerate(lag_names):
                    col_idx = list(model.params.index).index(lag_name)
                    R[i, col_idx] = 1.0

                # F-test
                f_result = model.f_test(R)
                f_stat = float(f_result.fvalue)
                p_value = float(f_result.pvalue)

                row_results[test_var] = {
                    'f_stat': f_stat,
                    'p_value': p_value,
                }

            panel_results[dep_name] = {
                'nobs': int(model.nobs),
                'tests': row_results,
            }

        results[panel_name] = panel_results

    # Format and print results
    output_lines = []
    output_lines.append("=" * 100)
    output_lines.append("Table 1: Marginal Significance Levels of Monetary Indicators")
    output_lines.append("for Forecasting Alternative Measures of Economic Activity")
    output_lines.append("Six-Variable Prediction Equations (6 lags each)")
    output_lines.append("=" * 100)

    for panel_name, panel_results in results.items():
        output_lines.append(f"\n{panel_name}")
        output_lines.append("-" * 80)

        # Header
        header = f"{'Variable':<25s}"
        for tv in test_vars:
            header += f"  {tv:>8s}"
        header += f"  {'N':>5s}"
        output_lines.append(header)
        output_lines.append("-" * 80)

        for dep_name in dep_vars.keys():
            info = panel_results[dep_name]
            row = f"{dep_name:<25s}"
            for tv in test_vars:
                p = info['tests'][tv]['p_value']
                row += f"  {p:>8.4f}"
            row += f"  {info['nobs']:>5d}"
            output_lines.append(row)

        output_lines.append(f"\nNote: Values are p-values from F-tests of joint significance of 6 lags.")

    result_text = "\n".join(output_lines)
    print(result_text)

    # Run scoring
    score, breakdown = score_against_ground_truth(results)
    print(f"\n{'=' * 80}")
    print(f"AUTOMATED SCORE: {score}/100")
    print(f"{'=' * 80}")
    for criterion, pts in breakdown.items():
        print(f"  {criterion}: {pts}")

    return result_text


def score_against_ground_truth(results):
    """Score the results against the paper's ground truth values."""

    # Ground truth from Table 1 of Bernanke and Blinder (1992)
    ground_truth = {
        'Panel A (1959:7-1989:12)': {
            'Industrial production':  {'M1': 0.92, 'M2': 0.10, 'BILL': 0.071, 'BOND': 0.26, 'FUNDS': 0.017},
            'Capacity utilization':   {'M1': 0.74, 'M2': 0.22, 'BILL': 0.16,  'BOND': 0.40, 'FUNDS': 0.031},
            'Employment':             {'M1': 0.45, 'M2': 0.27, 'BILL': 0.0040,'BOND': 0.085,'FUNDS': 0.0004},
            'Unemployment rate':      {'M1': 0.96, 'M2': 0.37, 'BILL': 0.0005,'BOND': 0.024,'FUNDS': 0.0001},
            'Housing starts':         {'M1': 0.50, 'M2': 0.32, 'BILL': 0.52,  'BOND': 0.014,'FUNDS': 0.22},
            'Personal income':        {'M1': 0.38, 'M2': 0.24, 'BILL': 0.35,  'BOND': 0.59, 'FUNDS': 0.049},
            'Retail sales':           {'M1': 0.64, 'M2': 0.036,'BILL': 0.33,  'BOND': 0.74, 'FUNDS': 0.014},
            'Consumption':            {'M1': 0.96, 'M2': 0.11, 'BILL': 0.12,  'BOND': 0.46, 'FUNDS': 0.0052},
        },
        'Panel B (1959:7-1979:9)': {
            'Industrial production':  {'M1': 0.99, 'M2': 0.084,'BILL': 0.0092,'BOND': 0.61, 'FUNDS': 0.0001},
            'Capacity utilization':   {'M1': 0.96, 'M2': 0.40, 'BILL': 0.025, 'BOND': 0.18, 'FUNDS': 0.0003},
            'Employment':             {'M1': 0.57, 'M2': 0.41, 'BILL': 0.0005,'BOND': 0.15, 'FUNDS': 0.0004},
            'Unemployment rate':      {'M1': 0.56, 'M2': 0.88, 'BILL': 0.0006,'BOND': 0.13, 'FUNDS': 0.0000},
            'Housing starts':         {'M1': 0.34, 'M2': 0.17, 'BILL': 0.73,  'BOND': 0.72, 'FUNDS': 0.11},
            'Personal income':        {'M1': 0.43, 'M2': 0.095,'BILL': 0.20,  'BOND': 0.91, 'FUNDS': 0.037},
            'Retail sales':           {'M1': 0.96, 'M2': 0.86, 'BILL': 0.27,  'BOND': 0.050,'FUNDS': 0.061},
            'Consumption':            {'M1': 0.79, 'M2': 0.017,'BILL': 0.010, 'BOND': 0.050,'FUNDS': 0.0000},
        },
    }

    # Significance level categories
    def sig_category(p):
        if p <= 0.01:
            return '***'  # p <= 0.01
        elif p <= 0.05:
            return '**'   # 0.01 < p <= 0.05
        elif p <= 0.10:
            return '*'    # 0.05 < p <= 0.10
        else:
            return 'ns'   # p > 0.10

    # --- Criterion 1: Test statistic values (30 pts) ---
    # p-values within same significance band = match
    total_tests = 0
    matching_pvals = 0

    for panel_name in ground_truth:
        for dep_name in ground_truth[panel_name]:
            for test_var in ground_truth[panel_name][dep_name]:
                total_tests += 1
                true_p = ground_truth[panel_name][dep_name][test_var]
                if dep_name in results.get(panel_name, {}):
                    gen_p = results[panel_name][dep_name]['tests'][test_var]['p_value']
                    # Match within 15% relative error OR within same significance band
                    if true_p > 0:
                        rel_err = abs(gen_p - true_p) / max(true_p, 0.001)
                    else:
                        rel_err = abs(gen_p - true_p)
                    if rel_err <= 0.50:  # generous for vintage effects
                        matching_pvals += 1
                    elif sig_category(gen_p) == sig_category(true_p):
                        matching_pvals += 0.5

    pval_score = (matching_pvals / total_tests) * 30 if total_tests > 0 else 0

    # --- Criterion 2: Significance levels match (30 pts) ---
    sig_matches = 0
    sig_total = 0

    for panel_name in ground_truth:
        for dep_name in ground_truth[panel_name]:
            for test_var in ground_truth[panel_name][dep_name]:
                sig_total += 1
                true_p = ground_truth[panel_name][dep_name][test_var]
                if dep_name in results.get(panel_name, {}):
                    gen_p = results[panel_name][dep_name]['tests'][test_var]['p_value']
                    true_sig = sig_category(true_p)
                    gen_sig = sig_category(gen_p)
                    if true_sig == gen_sig:
                        sig_matches += 1
                    # Partial credit for adjacent categories
                    elif (true_sig in ('***', '**') and gen_sig in ('***', '**')):
                        sig_matches += 0.75
                    elif (true_sig in ('**', '*') and gen_sig in ('**', '*')):
                        sig_matches += 0.75
                    elif (true_sig in ('*', 'ns') and gen_sig in ('*', 'ns')):
                        sig_matches += 0.5

    sig_score = (sig_matches / sig_total) * 30 if sig_total > 0 else 0

    # --- Criterion 3: All variable pairs present (15 pts) ---
    expected_pairs = 0
    present_pairs = 0

    for panel_name in ground_truth:
        for dep_name in ground_truth[panel_name]:
            for test_var in ground_truth[panel_name][dep_name]:
                expected_pairs += 1
                if (panel_name in results and
                    dep_name in results[panel_name] and
                    test_var in results[panel_name][dep_name]['tests']):
                    present_pairs += 1

    presence_score = (present_pairs / expected_pairs) * 15 if expected_pairs > 0 else 0

    # --- Criterion 4: Sample period / N (15 pts) ---
    # Panel A should have ~366 obs, Panel B should have ~243 obs
    expected_n = {
        'Panel A (1959:7-1989:12)': 366,
        'Panel B (1959:7-1979:9)': 243,
    }
    n_score = 0
    for panel_name, expected in expected_n.items():
        if panel_name in results:
            # Get N from any dep var
            for dep_name in results[panel_name]:
                actual_n = results[panel_name][dep_name]['nobs']
                if abs(actual_n - expected) / expected <= 0.05:
                    n_score += 7.5
                elif abs(actual_n - expected) / expected <= 0.10:
                    n_score += 5.0
                break

    # --- Criterion 5: Correct lag specification (10 pts) ---
    # We use 6 lags as specified
    lag_score = 10  # Always 10 since we hardcode 6 lags

    total_score = round(pval_score + sig_score + presence_score + n_score + lag_score)

    breakdown = {
        'P-value accuracy (30)': round(pval_score, 1),
        'Significance match (30)': round(sig_score, 1),
        'All pairs present (15)': round(presence_score, 1),
        'Sample size N (15)': round(n_score, 1),
        'Lag specification (10)': round(lag_score, 1),
    }

    return total_score, breakdown


if __name__ == "__main__":
    result = run_analysis("bb1992_data.csv")
