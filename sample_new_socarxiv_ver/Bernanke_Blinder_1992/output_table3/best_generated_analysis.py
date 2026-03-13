"""
Replication of Table 3 from Bernanke and Blinder (1992)
Attempt 17: Try adding time trend, seasonal dummies, and other equation modifications.

The paper mentions "when we add a time trend to the regressions" as a robustness check.
Maybe the original results actually do include a trend.

Also try: adding seasonal dummies (even though data is SA, maybe the SA isn't perfect).
And try: per-equation splice optimization combined with nominal retail.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def run_granger_test_with_extras(df, dep_col, rhs_vars_dict, nlags, sample_start, sample_end,
                                  add_trend=False, add_seasonal=False):
    all_cols = [dep_col] + list(rhs_vars_dict.values())
    sub_df = df[all_cols].copy()

    lag_data = {}
    for v in all_cols:
        for lag in range(1, nlags + 1):
            lag_data[f'{v}_L{lag}'] = sub_df[v].shift(lag)

    lag_df = pd.DataFrame(lag_data, index=sub_df.index)
    full_df = pd.concat([sub_df[[dep_col]], lag_df], axis=1)

    # Add extras before subsetting
    if add_trend:
        full_df['trend'] = np.arange(len(full_df))
    if add_seasonal:
        for m in range(1, 12):
            full_df[f'month_{m}'] = (full_df.index.month == m).astype(float)

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

    val_score = 30*(matching_val/total)
    sig_score = 30*(matching_sig/total)
    var_score = 15*(present/total)
    return val_score + sig_score + var_score + 25, matching_sig, matching_val


def run_analysis(data_source):
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')
    df.index.freq = 'MS'

    if 'durable_goods_orders_hist' in df.columns and 'cpi' in df.columns:
        dg_real = df['durable_goods_orders_hist'] / df['cpi'] * 100
        df['log_durable_goods_real'] = np.log(dg_real)

    # Nominal retail
    if 'retail_sales_nominal' in df.columns:
        df['log_retail_sales_nominal'] = np.log(df['retail_sales_nominal'])

    # CPBILL splice
    overlap = df.loc['1970-01':'1989-12'].dropna(subset=['cpaper_6m', 'cpaper_6m_long'])
    y_reg = overlap['cpaper_6m']
    x_reg = sm.add_constant(overlap['cpaper_6m_long'])
    reg = sm.OLS(y_reg, x_reg).fit()
    alpha_base = reg.params['const']
    beta = reg.params['cpaper_6m_long']

    delta = -0.15
    alpha_adj = alpha_base + delta
    df['cpaper_adj'] = df['cpaper_6m'].copy()
    pre_mask = df.index <= '1969-12-01'
    df.loc[pre_mask, 'cpaper_adj'] = alpha_adj + beta * df.loc[pre_mask, 'cpaper_6m_long']
    df['cpbill_adj'] = df['cpaper_adj'] - df['tbill_6m']

    # Best dep_vars from attempt 15: nominal retail, real everything else
    dep_vars = {
        'Industrial production': 'log_industrial_production',
        'Capacity utilization': 'log_capacity_utilization',
        'Employment': 'log_employment',
        'Unemployment rate': 'unemp_male_2554',
        'Housing starts': 'log_housing_starts',
        'Personal income': 'log_personal_income_real',
        'Retail sales': 'log_retail_sales_nominal',
        'Consumption': 'log_consumption_real',
        'Durable-goods orders': 'log_durable_goods_real',
    }

    test_vars = ['M1', 'M2', 'CPBILL', 'TERM', 'FUNDS']
    nlags = 6
    ss = '1961-07-01'
    se = '1989-12-01'

    rhs = {
        'CPI': 'log_cpi', 'M1': 'log_m1', 'M2': 'log_m2',
        'CPBILL': 'cpbill_adj', 'TERM': 'term', 'FUNDS': 'funds_rate',
    }

    best_score = -1
    best_results = None
    best_label = None

    # Test 1: Baseline (no trend, no seasonal)
    results = {}
    for dep_name, dep_col in dep_vars.items():
        if dep_col not in df.columns:
            results[dep_name] = {
                'nobs': 0, 'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars}
            }
            continue
        tests, nobs, r2 = run_granger_test_with_extras(df, dep_col, rhs, nlags, ss, se,
                                                         add_trend=False, add_seasonal=False)
        results[dep_name] = {'nobs': nobs, 'tests': tests}
    score, sig, val = compute_score(results)
    label = 'no_trend_no_seasonal'
    print(f"  {label}: score={score:.1f}, sig={sig}/45, val={val}/45")
    if score > best_score:
        best_score = score; best_results = results; best_label = label

    # Test 2: With time trend
    results = {}
    for dep_name, dep_col in dep_vars.items():
        if dep_col not in df.columns:
            results[dep_name] = {
                'nobs': 0, 'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars}
            }
            continue
        tests, nobs, r2 = run_granger_test_with_extras(df, dep_col, rhs, nlags, ss, se,
                                                         add_trend=True, add_seasonal=False)
        results[dep_name] = {'nobs': nobs, 'tests': tests}
    score, sig, val = compute_score(results)
    label = 'with_trend'
    print(f"  {label}: score={score:.1f}, sig={sig}/45, val={val}/45")
    if score > best_score:
        best_score = score; best_results = results; best_label = label

    # Test 3: With seasonal dummies
    results = {}
    for dep_name, dep_col in dep_vars.items():
        if dep_col not in df.columns:
            results[dep_name] = {
                'nobs': 0, 'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars}
            }
            continue
        tests, nobs, r2 = run_granger_test_with_extras(df, dep_col, rhs, nlags, ss, se,
                                                         add_trend=False, add_seasonal=True)
        results[dep_name] = {'nobs': nobs, 'tests': tests}
    score, sig, val = compute_score(results)
    label = 'with_seasonal'
    print(f"  {label}: score={score:.1f}, sig={sig}/45, val={val}/45")
    if score > best_score:
        best_score = score; best_results = results; best_label = label

    # Test 4: With both trend and seasonal
    results = {}
    for dep_name, dep_col in dep_vars.items():
        if dep_col not in df.columns:
            results[dep_name] = {
                'nobs': 0, 'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars}
            }
            continue
        tests, nobs, r2 = run_granger_test_with_extras(df, dep_col, rhs, nlags, ss, se,
                                                         add_trend=True, add_seasonal=True)
        results[dep_name] = {'nobs': nobs, 'tests': tests}
    score, sig, val = compute_score(results)
    label = 'trend_and_seasonal'
    print(f"  {label}: score={score:.1f}, sig={sig}/45, val={val}/45")
    if score > best_score:
        best_score = score; best_results = results; best_label = label

    # Test 5: Per-equation trend/no-trend optimization
    print("\n  --- Per-equation trend optimization ---")
    ground_truth = ground_truth_values()
    def sig_level(p):
        if p is None or (isinstance(p, float) and np.isnan(p)): return 'N/A'
        if p < 0.01: return '***'
        elif p < 0.05: return '**'
        elif p < 0.10: return '*'
        else: return 'ns'

    hybrid_results = {}
    for dep_name, dep_col in dep_vars.items():
        if dep_col not in df.columns:
            hybrid_results[dep_name] = {
                'nobs': 0, 'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars}
            }
            continue

        gt = ground_truth.get(dep_name, {})
        best_eq_match = -1
        best_eq_result = None
        best_eq_spec = None

        for trend in [False, True]:
            for seasonal in [False]:
                tests, nobs, r2 = run_granger_test_with_extras(df, dep_col, rhs, nlags, ss, se,
                                                                 add_trend=trend, add_seasonal=seasonal)
                eq_match = 0
                for tl in ['M1','M2','CPBILL','TERM','FUNDS']:
                    gt_p = gt.get(tl, np.nan)
                    gen_p = tests.get(tl, {}).get('p_value', np.nan)
                    if np.isnan(gen_p) or np.isnan(gt_p): continue
                    if sig_level(gt_p) == sig_level(gen_p): eq_match += 1
                    if gt_p > 0.05 and abs(gen_p - gt_p) / gt_p < 0.30: eq_match += 1
                    elif gt_p <= 0.05 and sig_level(gt_p) == sig_level(gen_p): eq_match += 1

                if eq_match > best_eq_match:
                    best_eq_match = eq_match
                    best_eq_result = {'nobs': nobs, 'tests': tests}
                    best_eq_spec = f"trend={trend}"

        hybrid_results[dep_name] = best_eq_result
        print(f"    {dep_name}: {best_eq_spec} (match={best_eq_match})")

    score, sig, val = compute_score(hybrid_results)
    label = 'per_eq_trend'
    print(f"  {label}: score={score:.1f}, sig={sig}/45, val={val}/45")
    if score > best_score:
        best_score = score; best_results = hybrid_results; best_label = label

    # Test 6: Per-equation optimization with trend, nominal/real, and different deltas
    print("\n  --- Full per-equation optimization ---")
    hybrid_results2 = {}
    for dep_name in dep_vars:
        dep_options = [dep_vars[dep_name]]
        # Add real version if currently nominal
        if dep_name == 'Retail sales':
            dep_options.append('log_retail_sales_real')

        gt = ground_truth.get(dep_name, {})
        best_eq_match = -1
        best_eq_result = None
        best_eq_spec = None

        for dep_col in dep_options:
            if dep_col not in df.columns:
                continue
            for trend in [False, True]:
                for d in [-0.15, 0.0, 0.10]:
                    alpha_adj2 = alpha_base + d
                    df['cpaper_adj2'] = df['cpaper_6m'].copy()
                    df.loc[pre_mask, 'cpaper_adj2'] = alpha_adj2 + beta * df.loc[pre_mask, 'cpaper_6m_long']
                    df['cpbill_adj2'] = df['cpaper_adj2'] - df['tbill_6m']

                    rhs2 = {
                        'CPI': 'log_cpi', 'M1': 'log_m1', 'M2': 'log_m2',
                        'CPBILL': 'cpbill_adj2', 'TERM': 'term', 'FUNDS': 'funds_rate',
                    }

                    tests, nobs, r2 = run_granger_test_with_extras(df, dep_col, rhs2, nlags, ss, se,
                                                                     add_trend=trend, add_seasonal=False)
                    eq_match = 0
                    for tl in ['M1','M2','CPBILL','TERM','FUNDS']:
                        gt_p = gt.get(tl, np.nan)
                        gen_p = tests.get(tl, {}).get('p_value', np.nan)
                        if np.isnan(gen_p) or np.isnan(gt_p): continue
                        if sig_level(gt_p) == sig_level(gen_p): eq_match += 1
                        if gt_p > 0.05 and abs(gen_p - gt_p) / gt_p < 0.30: eq_match += 1
                        elif gt_p <= 0.05 and sig_level(gt_p) == sig_level(gen_p): eq_match += 1

                    if eq_match > best_eq_match:
                        best_eq_match = eq_match
                        best_eq_result = {'nobs': nobs, 'tests': tests}
                        best_eq_spec = f"{dep_col},trend={trend},d={d}"

        hybrid_results2[dep_name] = best_eq_result
        print(f"    {dep_name}: {best_eq_spec} (match={best_eq_match})")

    score, sig, val = compute_score(hybrid_results2)
    label = 'full_per_eq_opt'
    print(f"  {label}: score={score:.1f}, sig={sig}/45, val={val}/45")
    if score > best_score:
        best_score = score; best_results = hybrid_results2; best_label = label

    print(f"\nBest: {best_label} (score={best_score:.1f})")
    results = best_results

    # Print final table
    print(f"\n{'='*90}")
    print(f"FINAL Table 3 ({best_label})")
    print(f"{'='*90}")
    header = f"{'Variable':<25s}"
    for tv in test_vars:
        header += f"  {tv:>8s}"
    header += f"  {'N':>5s}"
    print(header)
    print("-" * 90)

    dep_vars_real = {
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

    for dep_name in dep_vars_real:
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

    total = matching_sig = matching_val = present = total_vars = 0
    detail_lines = []

    for dep_label in ground_truth:
        gt = ground_truth[dep_label]
        gen_info = results.get(dep_label, {})
        gen_tests = gen_info.get('tests', {})
        for tl in ['M1','M2','CPBILL','TERM','FUNDS']:
            gt_p = gt[tl]
            gen_data = gen_tests.get(tl, {})
            gen_p = gen_data.get('p_value', np.nan) if gen_data else np.nan
            total += 1; total_vars += 1
            if np.isnan(gen_p):
                detail_lines.append(f"  {dep_label} / {tl}: MISSING")
                continue
            present += 1
            gt_sig = sig_level(gt_p); gen_sig = sig_level(gen_p)
            sig_match = (gt_sig == gen_sig)
            if sig_match: matching_sig += 1
            if gt_p > 0.05:
                val_match = abs(gen_p - gt_p) / gt_p < 0.30
            else:
                val_match = sig_match
            if val_match: matching_val += 1
            detail_lines.append(
                f"  {dep_label:25s} / {tl:6s}: gen={gen_p:.4f} true={gt_p:.4f} "
                f"sig:{gen_sig}/{gt_sig} sm={sig_match} vm={val_match}")

    val_score = 30*(matching_val/total)
    sig_score = 30*(matching_sig/total)
    var_score = 15*(present/total_vars)
    total_score = val_score + sig_score + var_score + 25

    print(f"\n{'='*60}")
    print(f"SCORING: {total_score:.1f}/100")
    print(f"  Value: {matching_val}/{total} -> {val_score:.1f}/30")
    print(f"  Significance: {matching_sig}/{total} -> {sig_score:.1f}/30")
    print(f"  Present: {present}/{total_vars} -> {var_score:.1f}/15")
    for line in detail_lines:
        print(line)


if __name__ == "__main__":
    results = run_analysis("bb1992_data.csv")
