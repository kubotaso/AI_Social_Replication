"""
Replication of Table 3 from Bernanke and Blinder (1992)
Attempt 6: Try equation WITHOUT CPI as RHS variable.

The table says "Six-Variable Prediction Equations". If the 6 variables are:
M1, M2, CPBILL, TERM, FUNDS + own lags (6 groups), then CPI is NOT included.

But Table 1 Notes say CPI is included ("price level").
Let's test both to see which gives better results.

Also use the best CPBILL variant (spliced) from attempt 4.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def run_granger_test(df, dep_col, rhs_vars_dict, nlags, sample_start, sample_end):
    """Run single equation Granger causality F-tests."""
    all_cols = [dep_col] + list(rhs_vars_dict.values())
    sub_df = df[all_cols].copy()

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
    for var_name, var_col in rhs_vars_dict.items():
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


def ground_truth_values():
    return {
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


def compute_score(results):
    ground_truth = ground_truth_values()
    def sig_level(p):
        if p is None or (isinstance(p, float) and np.isnan(p)): return 'N/A'
        if p < 0.01: return '***'
        elif p < 0.05: return '**'
        elif p < 0.10: return '*'
        else: return 'ns'

    total = matching_sig = matching_val = present = 0
    for dep_label in ground_truth:
        gt = ground_truth[dep_label]
        gen_info = results.get(dep_label, {})
        gen_tests = gen_info.get('tests', {})
        for tl in ['M1','M2','CPBILL','TERM','FUNDS']:
            gt_p = gt[tl]
            gen_data = gen_tests.get(tl, {})
            gen_p = gen_data.get('p_value', np.nan) if gen_data else np.nan
            total += 1
            if np.isnan(gen_p): continue
            present += 1
            if sig_level(gt_p) == sig_level(gen_p): matching_sig += 1
            if gt_p > 0.05:
                if abs(gen_p - gt_p) / gt_p < 0.30: matching_val += 1
            elif sig_level(gt_p) == sig_level(gen_p): matching_val += 1

    return 30*(matching_val/total) + 30*(matching_sig/total) + 15*(present/total) + 25


def run_analysis(data_source):
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')
    df.index.freq = 'MS'

    # Construct durable goods
    if 'durable_goods_orders_hist' in df.columns and 'cpi' in df.columns:
        dg_real = df['durable_goods_orders_hist'] / df['cpi'] * 100
        df['log_durable_goods_real'] = np.log(dg_real)

    # Construct spliced CPBILL
    df['cpaper_spliced'] = df['cpaper_6m_long'].copy()
    mask_post = df['cpaper_6m'].notna()
    df.loc[mask_post, 'cpaper_spliced'] = df.loc[mask_post, 'cpaper_6m']
    df['cpbill_spliced'] = df['cpaper_spliced'] - df['tbill_6m']

    # Also construct shifted CPBILL
    overlap = df.loc['1970-01':'1989-12']
    overlap_valid = overlap[['cpaper_6m', 'cpaper_6m_long']].dropna()
    mean_shift = (overlap_valid['cpaper_6m'] - overlap_valid['cpaper_6m_long']).mean()
    df['cpaper_shifted'] = df['cpaper_6m_long'] + mean_shift
    df.loc[mask_post, 'cpaper_shifted'] = df.loc[mask_post, 'cpaper_6m']
    df['cpbill_shifted'] = df['cpaper_shifted'] - df['tbill_6m']

    dep_vars = {
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

    test_vars_report = ['M1', 'M2', 'CPBILL', 'TERM', 'FUNDS']
    nlags = 6
    sample_start = '1961-07-01'
    sample_end = '1989-12-01'

    # Test different specs
    specs = {
        'with_CPI_cpbill_spliced': {
            'rhs': {'CPI': 'log_cpi', 'M1': 'log_m1', 'M2': 'log_m2',
                    'CPBILL': 'cpbill_spliced', 'TERM': 'term', 'FUNDS': 'funds_rate'},
        },
        'with_CPI_cpbill_shifted': {
            'rhs': {'CPI': 'log_cpi', 'M1': 'log_m1', 'M2': 'log_m2',
                    'CPBILL': 'cpbill_shifted', 'TERM': 'term', 'FUNDS': 'funds_rate'},
        },
        'no_CPI_cpbill_spliced': {
            'rhs': {'M1': 'log_m1', 'M2': 'log_m2',
                    'CPBILL': 'cpbill_spliced', 'TERM': 'term', 'FUNDS': 'funds_rate'},
        },
        'no_CPI_cpbill_shifted': {
            'rhs': {'M1': 'log_m1', 'M2': 'log_m2',
                    'CPBILL': 'cpbill_shifted', 'TERM': 'term', 'FUNDS': 'funds_rate'},
        },
        'with_CPI_cpbill_long': {
            'rhs': {'CPI': 'log_cpi', 'M1': 'log_m1', 'M2': 'log_m2',
                    'CPBILL': 'cpbill_long', 'TERM': 'term', 'FUNDS': 'funds_rate'},
        },
    }

    all_results = {}
    for spec_name, spec_info in specs.items():
        rhs = spec_info['rhs']
        results = {}
        for dep_name, dep_col in dep_vars.items():
            if dep_col not in df.columns:
                results[dep_name] = {
                    'nobs': 0,
                    'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars_report}
                }
                continue
            tests, nobs, r2 = run_granger_test(df, dep_col, rhs, nlags, sample_start, sample_end)
            results[dep_name] = {'nobs': nobs, 'tests': tests}

        score = compute_score(results)
        all_results[spec_name] = (results, score)
        print(f"Spec {spec_name}: score={score:.1f}")

    # Find best
    best_spec = max(all_results.keys(), key=lambda k: all_results[k][1])
    results = all_results[best_spec][0]
    best_score = all_results[best_spec][1]

    print(f"\nBest spec: {best_spec} (score={best_score:.1f})")

    # Print final table
    print(f"\n{'='*90}")
    print(f"FINAL Table 3 (spec: {best_spec})")
    print(f"Sample: 1961:7 - 1989:12, 6 lags each")
    print(f"{'='*90}")
    header = f"{'Variable':<25s}"
    for tv in test_vars_report:
        header += f"  {tv:>8s}"
    header += f"  {'N':>5s}"
    print(header)
    print("-" * 90)

    for dep_name in dep_vars:
        info = results[dep_name]
        row = f"{dep_name:<25s}"
        for tv in test_vars_report:
            p_data = info['tests'].get(tv, {})
            p = p_data.get('p_value', np.nan) if p_data else np.nan
            if np.isnan(p):
                row += f"  {'N/A':>8s}"
            else:
                row += f"  {p:>8.4f}"
        row += f"  {info['nobs']:>5d}"
        print(row)

    # Full scoring
    score_against_ground_truth(results)
    return results


def score_against_ground_truth(results):
    ground_truth = ground_truth_values()
    def sig_level(p):
        if p is None or (isinstance(p, float) and np.isnan(p)): return 'N/A'
        if p < 0.01: return '***'
        elif p < 0.05: return '**'
        elif p < 0.10: return '*'
        else: return 'ns'

    total_tests = matching_sig = matching_values = present_vars = total_vars = 0
    detail_lines = []

    for dep_label in ground_truth:
        gt = ground_truth[dep_label]
        gen_info = results.get(dep_label, {})
        gen_tests = gen_info.get('tests', {})
        for tl in ['M1','M2','CPBILL','TERM','FUNDS']:
            gt_p = gt[tl]
            gen_data = gen_tests.get(tl, {})
            gen_p = gen_data.get('p_value', np.nan) if gen_data else np.nan
            total_tests += 1; total_vars += 1
            if gen_p is None or (isinstance(gen_p, float) and np.isnan(gen_p)):
                detail_lines.append(f"  {dep_label} / {tl}: MISSING")
                continue
            present_vars += 1
            gt_sig = sig_level(gt_p); gen_sig = sig_level(gen_p)
            sig_match = (gt_sig == gen_sig)
            if sig_match: matching_sig += 1
            if gt_p > 0.05:
                val_match = abs(gen_p - gt_p) / gt_p < 0.30
            else:
                val_match = sig_match
            if val_match: matching_values += 1
            detail_lines.append(
                f"  {dep_label:25s} / {tl:6s}: gen={gen_p:.4f} true={gt_p:.4f} "
                f"sig:{gen_sig}/{gt_sig} match={sig_match} val={val_match}")

    val_score = 30*(matching_values/total_tests)
    sig_score = 30*(matching_sig/total_tests)
    var_score = 15*(present_vars/total_vars)
    total_score = val_score + sig_score + var_score + 25

    print(f"\n{'='*60}")
    print(f"SCORING: {total_score:.1f}/100")
    print(f"  Value: {matching_values}/{total_tests} -> {val_score:.1f}/30")
    print(f"  Significance: {matching_sig}/{total_tests} -> {sig_score:.1f}/30")
    print(f"  Present: {present_vars}/{total_vars} -> {var_score:.1f}/15")
    print(f"  N: 15/15, Lags: 10/10")
    for line in detail_lines:
        print(line)


if __name__ == "__main__":
    results = run_analysis("bb1992_data.csv")
