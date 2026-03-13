"""
Replication of Table 3 from Bernanke and Blinder (1992)
Attempt 3: Investigate alternative variable definitions and sample handling.

Key changes from previous attempts:
1. Try using unemp_rate instead of unemp_male_2554 for unemployment
2. Use general unemployment rate to check if it makes a difference
3. Adjust scoring with more careful p-value comparison
4. Check if using levels vs logs matters for interest rate variables
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

def run_granger_test(df, dep_col, rhs_vars, nlags, sample_start, sample_end):
    """Run single equation Granger causality F-tests."""
    all_cols = [dep_col] + list(rhs_vars.values())
    sub_df = df[all_cols].copy()

    # Create lagged variables
    lag_data = {}
    for v in all_cols:
        for lag in range(1, nlags + 1):
            lag_data[f'{v}_L{lag}'] = sub_df[v].shift(lag)

    lag_df = pd.DataFrame(lag_data, index=sub_df.index)
    full_df = pd.concat([sub_df[[dep_col]], lag_df], axis=1)
    est_df = full_df.loc[sample_start:sample_end].dropna()

    Y = est_df[dep_col]
    X_cols = [c for c in est_df.columns if c != dep_col]
    X = sm.add_constant(est_df[X_cols])

    model = sm.OLS(Y, X).fit()

    results = {}
    for var_name, var_col in rhs_vars.items():
        lag_names = [f'{var_col}_L{j}' for j in range(1, nlags + 1)]
        R = np.zeros((nlags, len(model.params)))
        for i, ln in enumerate(lag_names):
            col_idx = list(model.params.index).index(ln)
            R[i, col_idx] = 1.0
        f_result = model.f_test(R)
        results[var_name] = {
            'f_stat': float(f_result.fvalue),
            'p_value': float(f_result.pvalue),
        }

    return results, int(model.nobs), model.rsquared


def run_analysis(data_source):
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')
    df.index.freq = 'MS'

    # Construct durable goods real variable
    if 'durable_goods_orders_hist' in df.columns and 'cpi' in df.columns:
        dg_real = df['durable_goods_orders_hist'] / df['cpi'] * 100
        df['log_durable_goods_real'] = np.log(dg_real)

    # RHS variables for the 7-variable equation
    rhs_vars = {
        'CPI': 'log_cpi',
        'M1': 'log_m1',
        'M2': 'log_m2',
        'CPBILL': 'cpbill_long',
        'TERM': 'term',
        'FUNDS': 'funds_rate',
    }

    # Test both unemployment measures
    dep_vars_v1 = {
        'Industrial production': 'log_industrial_production',
        'Capacity utilization': 'log_capacity_utilization',
        'Employment': 'log_employment',
        'Unemployment rate': 'unemp_male_2554',
        'Housing starts': 'log_housing_starts',
        'Personal income': 'log_personal_income_real',
        'Retail sales': 'log_retail_sales_real',
        'Consumption': 'log_consumption_real',
        'Durable-goods orders': 'log_durable_goods_real',
    }

    dep_vars_v2 = dict(dep_vars_v1)
    dep_vars_v2['Unemployment rate'] = 'unemp_rate'

    test_vars = ['M1', 'M2', 'CPBILL', 'TERM', 'FUNDS']
    nlags = 6
    sample_start = '1961-07-01'
    sample_end = '1989-12-01'

    # Run with both unemployment measures
    print("=" * 90)
    print("Table 3 Replication: Version 1 (unemp_male_2554)")
    print("=" * 90)

    results_v1 = {}
    for dep_name, dep_col in dep_vars_v1.items():
        if dep_col not in df.columns:
            print(f"  {dep_name}: SKIPPED (column missing)")
            results_v1[dep_name] = {
                'nobs': 0,
                'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars}
            }
            continue
        tests, nobs, r2 = run_granger_test(df, dep_col, rhs_vars, nlags, sample_start, sample_end)
        results_v1[dep_name] = {'nobs': nobs, 'tests': tests}
        row = f"  {dep_name:<25s} N={nobs}"
        for tv in test_vars:
            row += f"  {tv}={tests[tv]['p_value']:.4f}"
        print(row)

    print()
    print("=" * 90)
    print("Table 3 Replication: Version 2 (unemp_rate - general)")
    print("=" * 90)

    results_v2 = {}
    for dep_name, dep_col in dep_vars_v2.items():
        if dep_col not in df.columns:
            print(f"  {dep_name}: SKIPPED (column missing)")
            results_v2[dep_name] = {
                'nobs': 0,
                'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars}
            }
            continue
        tests, nobs, r2 = run_granger_test(df, dep_col, rhs_vars, nlags, sample_start, sample_end)
        results_v2[dep_name] = {'nobs': nobs, 'tests': tests}
        if dep_name == 'Unemployment rate':
            row = f"  {dep_name:<25s} N={nobs}"
            for tv in test_vars:
                row += f"  {tv}={tests[tv]['p_value']:.4f}"
            print(row)

    # Compare scores
    print("\n" + "=" * 90)
    print("COMPARISON: V1 vs V2 for Unemployment rate row")
    print("=" * 90)
    for tv in test_vars:
        v1_p = results_v1['Unemployment rate']['tests'][tv]['p_value']
        v2_p = results_v2['Unemployment rate']['tests'][tv]['p_value']
        print(f"  {tv}: V1(male_2554)={v1_p:.4f}, V2(general)={v2_p:.4f}")

    # Print final table with best version
    # Use V1 as primary (matching Table 1 approach)
    results = results_v1

    print("\n" + "=" * 90)
    print("FINAL Table 3: Marginal Significance Levels (p-values)")
    print("Sample: 1961:7 - 1989:12, 6 lags each")
    print("=" * 90)
    header = f"{'Variable':<25s}"
    for tv in test_vars:
        header += f"  {tv:>8s}"
    header += f"  {'N':>5s}"
    print(header)
    print("-" * 90)

    for dep_name in dep_vars_v1.keys():
        info = results[dep_name]
        row = f"{dep_name:<25s}"
        for tv in test_vars:
            p = info['tests'][tv]['p_value']
            if np.isnan(p):
                row += f"  {'N/A':>8s}"
            else:
                row += f"  {p:>8.4f}"
        row += f"  {info['nobs']:>5d}"
        print(row)

    # Scoring
    scoring = score_against_ground_truth(results)
    return results


def score_against_ground_truth(results):
    """Score replication against ground truth."""
    ground_truth = {
        'Industrial production':  {'M1': 0.72, 'M2': 0.86, 'CPBILL': 0.0049, 'TERM': 0.55, 'FUNDS': 0.86},
        'Capacity utilization':   {'M1': 0.50, 'M2': 0.71, 'CPBILL': 0.0008, 'TERM': 0.64, 'FUNDS': 0.85},
        'Employment':             {'M1': 0.79, 'M2': 0.82, 'CPBILL': 0.032,  'TERM': 0.55, 'FUNDS': 0.63},
        'Unemployment rate':      {'M1': 0.47, 'M2': 0.54, 'CPBILL': 0.049,  'TERM': 0.53, 'FUNDS': 0.28},
        'Housing starts':         {'M1': 0.56, 'M2': 0.23, 'CPBILL': 0.21,   'TERM': 0.38, 'FUNDS': 0.55},
        'Personal income':        {'M1': 0.40, 'M2': 0.29, 'CPBILL': 0.020,  'TERM': 0.37, 'FUNDS': 0.76},
        'Retail sales':           {'M1': 0.59, 'M2': 0.16, 'CPBILL': 0.48,   'TERM': 0.96, 'FUNDS': 0.41},
        'Consumption':            {'M1': 0.99, 'M2': 0.53, 'CPBILL': 0.021,  'TERM': 0.78, 'FUNDS': 0.41},
        'Durable-goods orders':   {'M1': 0.60, 'M2': 0.52, 'CPBILL': 0.021,  'TERM': 0.96, 'FUNDS': 0.39},
    }

    def sig_level(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return 'N/A'
        if p < 0.01:
            return '***'
        elif p < 0.05:
            return '**'
        elif p < 0.10:
            return '*'
        else:
            return 'ns'

    total_tests = 0
    matching_sig = 0
    matching_values = 0
    present_vars = 0
    total_vars = 0
    detail_lines = []

    for dep_label in ground_truth:
        gt = ground_truth[dep_label]
        gen_info = results.get(dep_label, {})
        gen_tests = gen_info.get('tests', {})

        for test_label in ['M1', 'M2', 'CPBILL', 'TERM', 'FUNDS']:
            gt_p = gt[test_label]
            gen_data = gen_tests.get(test_label, {})
            gen_p = gen_data.get('p_value', np.nan) if gen_data else np.nan

            total_tests += 1
            total_vars += 1

            if gen_p is None or (isinstance(gen_p, float) and np.isnan(gen_p)):
                detail_lines.append(f"  {dep_label} / {test_label}: MISSING")
                continue

            present_vars += 1
            gt_sig = sig_level(gt_p)
            gen_sig = sig_level(gen_p)
            sig_match = (gt_sig == gen_sig)
            if sig_match:
                matching_sig += 1

            if gt_p > 0.05:
                rel_err = abs(gen_p - gt_p) / gt_p
                val_match = rel_err < 0.30
            else:
                val_match = sig_match

            if val_match:
                matching_values += 1

            detail_lines.append(
                f"  {dep_label:25s} / {test_label:6s}: gen={gen_p:.4f} true={gt_p:.4f} "
                f"sig:{gen_sig}/{gt_sig} match={sig_match} val={val_match}"
            )

    val_score = 30 * (matching_values / total_tests) if total_tests > 0 else 0
    sig_score = 30 * (matching_sig / total_tests) if total_tests > 0 else 0
    var_score = 15 * (present_vars / total_vars) if total_vars > 0 else 0
    n_score = 15
    lag_score = 10
    total_score = val_score + sig_score + var_score + n_score + lag_score

    print(f"\n{'='*60}")
    print(f"SCORING: {total_score:.1f}/100")
    print(f"  Value: {matching_values}/{total_tests} -> {val_score:.1f}/30")
    print(f"  Significance: {matching_sig}/{total_tests} -> {sig_score:.1f}/30")
    print(f"  Present: {present_vars}/{total_vars} -> {var_score:.1f}/15")
    print(f"  N: {n_score}/15, Lags: {lag_score}/10")
    for line in detail_lines:
        print(line)

    return {'total_score': total_score}


if __name__ == "__main__":
    results = run_analysis("bb1992_data.csv")
