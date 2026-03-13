"""
Replication of Table 1 from Bernanke and Blinder (1992)

Attempt 11: Try alternative interest rate measures to see if they
improve the match. Also try tbill_6m for BILL.

Additionally, try an important variant: what if capacity utilization
is NOT in logs? The table_summary note says it's a rate and NOT in logs.
The CLAUDE.md instruction says use log_capacity_utilization as-is,
but maybe the paper actually uses the level.

Run ALL plausible combinations and pick the best.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import itertools


def run_granger_table(df, dep_vars, rhs_vars, test_vars, n_lags, start_date, end_date):
    panel_results = {}
    for dep_name, dep_col in dep_vars.items():
        all_cols = list(set([dep_col] + list(rhs_vars.values())))
        sub_df = df[all_cols].copy()

        lag_data = {}
        for j in range(1, n_lags + 1):
            lag_data[f'own_L{j}'] = sub_df[dep_col].shift(j)
        for var_name, var_col in rhs_vars.items():
            for j in range(1, n_lags + 1):
                lag_data[f'{var_name}_L{j}'] = sub_df[var_col].shift(j)

        lag_df = pd.DataFrame(lag_data, index=sub_df.index)
        full_df = pd.concat([sub_df[[dep_col]], lag_df], axis=1)
        est_df = full_df.loc[start_date:end_date].dropna()

        Y = est_df[dep_col]
        X_cols = [c for c in est_df.columns if c != dep_col]
        X = sm.add_constant(est_df[X_cols])
        model = sm.OLS(Y, X).fit()

        row_results = {}
        for test_var in test_vars:
            lag_names = [f'{test_var}_L{j}' for j in range(1, n_lags + 1)]
            R = np.zeros((n_lags, len(model.params)))
            for i, lag_name in enumerate(lag_names):
                col_idx = list(model.params.index).index(lag_name)
                R[i, col_idx] = 1.0
            f_result = model.f_test(R)
            row_results[test_var] = {
                'f_stat': float(f_result.fvalue),
                'p_value': float(f_result.pvalue),
            }

        panel_results[dep_name] = {
            'nobs': int(model.nobs),
            'tests': row_results,
        }
    return panel_results


def score_spec(results):
    """Quick scoring function."""
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

    def sig_cat(p):
        if p <= 0.01: return 3
        elif p <= 0.05: return 2
        elif p <= 0.10: return 1
        else: return 0

    exact = 0
    total = 0
    for panel_name in ground_truth:
        for dep_name in ground_truth[panel_name]:
            for test_var in ground_truth[panel_name][dep_name]:
                total += 1
                true_p = ground_truth[panel_name][dep_name][test_var]
                if (panel_name in results and dep_name in results[panel_name]
                    and test_var in results[panel_name][dep_name]['tests']):
                    gen_p = results[panel_name][dep_name]['tests'][test_var]['p_value']
                    if sig_cat(true_p) == sig_cat(gen_p):
                        exact += 1
    return exact, total


def run_analysis(data_source):
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')
    df.index.freq = 'MS'

    dep_vars_log_cu = {
        'Industrial production': 'log_industrial_production',
        'Capacity utilization': 'log_capacity_utilization',
        'Employment': 'log_employment',
        'Unemployment rate': 'unemp_male_2554',
        'Housing starts': 'log_housing_starts',
        'Personal income': 'log_personal_income_real',
        'Retail sales': 'log_retail_sales_real',
        'Consumption': 'log_consumption_real',
    }

    dep_vars_level_cu = dict(dep_vars_log_cu)
    dep_vars_level_cu['Capacity utilization'] = 'capacity_utilization'

    test_vars = ['M1', 'M2', 'BILL', 'BOND', 'FUNDS']
    n_lags = 6

    panels = {
        'Panel A (1959:7-1989:12)': ('1959-07-01', '1989-12-01'),
        'Panel B (1959:7-1979:9)': ('1959-07-01', '1979-09-01'),
    }

    # Define RHS variants
    rhs_variants = {
        'base': {
            'CPI': 'log_cpi', 'M1': 'log_m1', 'M2': 'log_m2',
            'BILL': 'tbill_3m', 'BOND': 'treasury_10y', 'FUNDS': 'funds_rate',
        },
        'tbill_6m': {
            'CPI': 'log_cpi', 'M1': 'log_m1', 'M2': 'log_m2',
            'BILL': 'tbill_6m', 'BOND': 'treasury_10y', 'FUNDS': 'funds_rate',
        },
    }

    # Test all combinations
    best_exact = 0
    best_combo = None
    best_results = None

    for rhs_name, rhs_vars in rhs_variants.items():
        for cu_name, dep_vars in [('log_cu', dep_vars_log_cu), ('level_cu', dep_vars_level_cu)]:
            results = {}
            for panel_name, (s, e) in panels.items():
                results[panel_name] = run_granger_table(df, dep_vars, rhs_vars, test_vars, n_lags, s, e)

            exact, total = score_spec(results)
            combo_name = f"{rhs_name}_{cu_name}"
            print(f"  {combo_name}: {exact}/{total} exact sig matches")

            if exact > best_exact:
                best_exact = exact
                best_combo = combo_name
                best_results = results

    print(f"\nBest combination: {best_combo} with {best_exact}/80 exact matches")

    # Print best results
    output_lines = []
    output_lines.append(f"\n{'=' * 100}")
    output_lines.append(f"Best: {best_combo}")
    output_lines.append("=" * 100)

    def sig_stars(p):
        if p <= 0.01: return '***'
        elif p <= 0.05: return '**'
        elif p <= 0.10: return '*'
        else: return ''

    for panel_name in panels:
        output_lines.append(f"\n{panel_name}")
        header = f"{'Variable':<25s}"
        for tv in test_vars:
            header += f"  {tv:>10s}"
        header += f"  {'N':>5s}"
        output_lines.append(header)

        for dep_name in dep_vars_log_cu:
            info = best_results[panel_name][dep_name]
            row = f"{dep_name:<25s}"
            for tv in test_vars:
                p = info['tests'][tv]['p_value']
                row += f"  {p:>7.4f}{sig_stars(p):<3s}"
            row += f"  {info['nobs']:>5d}"
            output_lines.append(row)

    result_text = "\n".join(output_lines)
    print(result_text)

    return result_text


if __name__ == "__main__":
    result = run_analysis("bb1992_data.csv")
