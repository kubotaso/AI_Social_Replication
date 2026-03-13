"""
Replication of Table 1 from Bernanke and Blinder (1992)

Attempt 5: Investigate multiple specifications:
1. Base case (same as before)
2. With linear time trend
3. With overall unemployment rate instead of prime-age male
4. Try lags 1-6 of LEVELS (not logs) for M2

The goal is to find which specification best matches the paper's results.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def run_granger_table(df, dep_vars, rhs_vars, test_vars, n_lags, start_date, end_date, include_trend=False):
    """Run the Granger causality F-tests for all dependent variables."""
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
        X = est_df[X_cols].copy()

        if include_trend:
            X['trend'] = np.arange(len(X))

        X = sm.add_constant(X)
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
            'nparams': len(model.params),
            'tests': row_results,
        }

    return panel_results


def sig_stars(p):
    if p <= 0.01: return '***'
    elif p <= 0.05: return '**'
    elif p <= 0.10: return '*'
    else: return ''


def print_panel(panel_results, dep_vars, test_vars, panel_name):
    lines = []
    lines.append(f"\n{panel_name}")
    lines.append("-" * 90)
    header = f"{'Variable':<25s}"
    for tv in test_vars:
        header += f"  {tv:>10s}"
    header += f"  {'N':>5s}"
    lines.append(header)
    lines.append("-" * 90)

    for dep_name in dep_vars.keys():
        info = panel_results[dep_name]
        row = f"{dep_name:<25s}"
        for tv in test_vars:
            p = info['tests'][tv]['p_value']
            stars = sig_stars(p)
            row += f"  {p:>7.4f}{stars:<3s}"
        row += f"  {info['nobs']:>5d}"
        lines.append(row)

    return "\n".join(lines)


def run_analysis(data_source):
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')
    df.index.freq = 'MS'

    # Base specification
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

    rhs_vars = {
        'CPI': 'log_cpi',
        'M1': 'log_m1',
        'M2': 'log_m2',
        'BILL': 'tbill_3m',
        'BOND': 'treasury_10y',
        'FUNDS': 'funds_rate',
    }

    test_vars = ['M1', 'M2', 'BILL', 'BOND', 'FUNDS']
    n_lags = 6

    panels = {
        'Panel A (1959:7-1989:12)': ('1959-07-01', '1989-12-01'),
        'Panel B (1959:7-1979:9)': ('1959-07-01', '1979-09-01'),
    }

    output_lines = []

    # ==== SPECIFICATION 1: Base case ====
    output_lines.append("=" * 110)
    output_lines.append("SPECIFICATION 1: Base case (no trend)")
    output_lines.append("=" * 110)

    results_base = {}
    for panel_name, (s, e) in panels.items():
        results_base[panel_name] = run_granger_table(df, dep_vars, rhs_vars, test_vars, n_lags, s, e, include_trend=False)
        output_lines.append(print_panel(results_base[panel_name], dep_vars, test_vars, panel_name))

    # ==== SPECIFICATION 2: With time trend ====
    output_lines.append(f"\n{'=' * 110}")
    output_lines.append("SPECIFICATION 2: With linear time trend")
    output_lines.append("=" * 110)

    results_trend = {}
    for panel_name, (s, e) in panels.items():
        results_trend[panel_name] = run_granger_table(df, dep_vars, rhs_vars, test_vars, n_lags, s, e, include_trend=True)
        output_lines.append(print_panel(results_trend[panel_name], dep_vars, test_vars, panel_name))

    # ==== SPECIFICATION 3: Overall unemployment rate ====
    dep_vars_unemp_alt = dict(dep_vars)
    dep_vars_unemp_alt['Unemployment rate'] = 'unemp_rate'

    output_lines.append(f"\n{'=' * 110}")
    output_lines.append("SPECIFICATION 3: Overall unemployment rate (unemp_rate)")
    output_lines.append("=" * 110)

    results_unemp = {}
    for panel_name, (s, e) in panels.items():
        results_unemp[panel_name] = run_granger_table(df, dep_vars_unemp_alt, rhs_vars, test_vars, n_lags, s, e, include_trend=False)
        # Only print unemployment rate row
        output_lines.append(f"\n{panel_name} — Unemployment rate only:")
        info = results_unemp[panel_name]['Unemployment rate']
        row = f"  {'Unemployment rate':<25s}"
        for tv in test_vars:
            p = info['tests'][tv]['p_value']
            row += f"  {p:>7.4f}{sig_stars(p):<3s}"
        output_lines.append(row)

    result_text = "\n".join(output_lines)
    print(result_text)

    # Score each specification
    print(f"\n{'=' * 80}")
    print("SCORING COMPARISON:")
    print("=" * 80)

    for spec_name, spec_results in [
        ("Base (no trend)", results_base),
        ("With trend", results_trend),
    ]:
        score, breakdown = score_against_ground_truth(spec_results)
        print(f"\n  {spec_name}: {score}/100")
        for k, v in breakdown.items():
            print(f"    {k}: {v}")

    # Also score with overall unemp rate
    results_unemp_full = {}
    for panel_name in panels:
        results_unemp_full[panel_name] = dict(results_base[panel_name])
        results_unemp_full[panel_name]['Unemployment rate'] = results_unemp[panel_name]['Unemployment rate']

    score_unemp, breakdown_unemp = score_against_ground_truth(results_unemp_full)
    print(f"\n  Base + overall unemp rate: {score_unemp}/100")
    for k, v in breakdown_unemp.items():
        print(f"    {k}: {v}")

    # Score trend + overall unemp rate
    results_trend_unemp = {}
    for panel_name in panels:
        results_trend_unemp[panel_name] = {}
        # Use trend results for all except unemployment, use unemp from trend spec
        for dep_name in dep_vars:
            if dep_name == 'Unemployment rate':
                # Run with unemp_rate and trend
                results_trend_unemp[panel_name][dep_name] = results_unemp[panel_name]['Unemployment rate']
            else:
                results_trend_unemp[panel_name][dep_name] = results_trend[panel_name][dep_name]

    score_combo, breakdown_combo = score_against_ground_truth(results_trend_unemp)
    print(f"\n  Trend + overall unemp rate: {score_combo}/100")
    for k, v in breakdown_combo.items():
        print(f"    {k}: {v}")

    return result_text


def score_against_ground_truth(results):
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

    all_pairs = []
    for panel_name in ground_truth:
        for dep_name in ground_truth[panel_name]:
            for test_var in ground_truth[panel_name][dep_name]:
                true_p = ground_truth[panel_name][dep_name][test_var]
                gen_p = None
                if (panel_name in results and dep_name in results[panel_name]
                    and test_var in results[panel_name][dep_name]['tests']):
                    gen_p = results[panel_name][dep_name]['tests'][test_var]['p_value']
                all_pairs.append({
                    'panel': panel_name, 'dep': dep_name, 'test': test_var,
                    'true_p': true_p, 'gen_p': gen_p
                })

    total = len(all_pairs)

    # Criterion 1: P-value accuracy (30 pts)
    pval_pts = 0
    for pair in all_pairs:
        if pair['gen_p'] is None:
            continue
        tp, gp = pair['true_p'], pair['gen_p']
        ad = abs(tp - gp)
        ts, gs = sig_cat(tp), sig_cat(gp)

        if ad < 0.01:
            pval_pts += 1.0
        elif ad < 0.03:
            pval_pts += 0.9
        elif ad < 0.08:
            pval_pts += 0.7
        elif ad < 0.15:
            pval_pts += 0.5
        elif ts == gs:
            pval_pts += 0.4
        elif abs(ts - gs) == 1:
            pval_pts += 0.2
        elif ad < 0.30:
            pval_pts += 0.15

    pval_score = (pval_pts / total) * 30

    # Criterion 2: Significance levels (30 pts)
    sig_pts = 0
    for pair in all_pairs:
        if pair['gen_p'] is None:
            continue
        ts, gs = sig_cat(pair['true_p']), sig_cat(pair['gen_p'])
        if ts == gs:
            sig_pts += 1.0
        elif abs(ts - gs) == 1:
            sig_pts += 0.5
        elif abs(ts - gs) == 2:
            sig_pts += 0.1

    sig_score = (sig_pts / total) * 30

    # Criterion 3: All pairs present (15 pts)
    present = sum(1 for p in all_pairs if p['gen_p'] is not None)
    presence_score = (present / total) * 15

    # Criterion 4: N (15 pts)
    expected_n = {'Panel A (1959:7-1989:12)': 366, 'Panel B (1959:7-1979:9)': 243}
    n_score = 0
    for panel_name, expected in expected_n.items():
        if panel_name in results:
            for dep_name in results[panel_name]:
                actual_n = results[panel_name][dep_name]['nobs']
                if abs(actual_n - expected) / expected <= 0.05:
                    n_score += 7.5
                break

    # Criterion 5: Lag spec (10 pts)
    lag_score = 10

    total_score = round(pval_score + sig_score + presence_score + n_score + lag_score)
    exact_sig = sum(1 for p in all_pairs if p['gen_p'] is not None and sig_cat(p['true_p']) == sig_cat(p['gen_p']))

    breakdown = {
        'P-value accuracy (30)': round(pval_score, 1),
        'Significance match (30)': round(sig_score, 1),
        f'Exact sig matches': f'{exact_sig}/{total}',
        'All pairs present (15)': round(presence_score, 1),
        'Sample size N (15)': round(n_score, 1),
        'Lag specification (10)': round(lag_score, 1),
    }

    return total_score, breakdown


if __name__ == "__main__":
    result = run_analysis("bb1992_data.csv")
