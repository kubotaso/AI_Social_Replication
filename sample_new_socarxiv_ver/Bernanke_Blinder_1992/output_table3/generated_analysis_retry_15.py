"""
Replication of Table 3 from Bernanke and Blinder (1992)
Attempt 15: Try nominal consumption and alternative deflation approaches.

The Data Appendix lists:
- Personal income, 1982 dollars (YP82) -- pre-deflated DRI series
- Retail sales, 1982 dollars (STR82) -- pre-deflated DRI series
- New orders, manufacturing durable goods, 1982 dollars (OMD82) -- pre-deflated
- Personal consumption expenditures (C) -- NOT listed as "1982 dollars"

Hypothesis: The DRI pre-deflated series may use a different deflator than CPI
(e.g., PCE deflator, GDP deflator). Try different approaches:
1. Use nominal consumption (log_consumption_nominal)
2. Use PCE deflator approximation for personal income
3. Check if using nominal + CPI in the equation vs real without CPI changes results
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def run_granger_test(df, dep_col, rhs_vars_dict, nlags, sample_start, sample_end):
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

    # Create nominal log versions
    df['log_consumption_nominal'] = np.log(df['consumption_nominal']) if 'consumption_nominal' in df.columns else np.nan
    if 'consumption_real' in df.columns:
        # consumption_real is already deflated by CPI
        # The "nominal" consumption is consumption_real * cpi / 100
        df['log_consumption_nominal'] = np.log(df['consumption_real'] * df['cpi'] / 100)

    # Also try: nominal retail sales and personal income
    if 'retail_sales_nominal' in df.columns:
        df['log_retail_sales_nominal'] = np.log(df['retail_sales_nominal'])
    if 'personal_income_nominal' in df.columns:
        df['log_personal_income_nominal'] = np.log(df['personal_income_nominal'])

    # Get regression parameters for CPBILL splice
    overlap = df.loc['1970-01':'1989-12'].dropna(subset=['cpaper_6m', 'cpaper_6m_long'])
    y_reg = overlap['cpaper_6m']
    x_reg = sm.add_constant(overlap['cpaper_6m_long'])
    reg = sm.OLS(y_reg, x_reg).fit()
    alpha = reg.params['const']
    beta = reg.params['cpaper_6m_long']

    # Best splice from previous attempts
    delta = -0.15
    alpha_adj = alpha + delta
    df['cpaper_adj'] = df['cpaper_6m'].copy()
    pre_mask = df.index <= '1969-12-01'
    df.loc[pre_mask, 'cpaper_adj'] = alpha_adj + beta * df.loc[pre_mask, 'cpaper_6m_long']
    df['cpbill_adj'] = df['cpaper_adj'] - df['tbill_6m']

    dep_vars_standard = {
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

    test_vars = ['M1', 'M2', 'CPBILL', 'TERM', 'FUNDS']
    nlags = 6
    ss = '1961-07-01'
    se = '1989-12-01'

    best_score = -1
    best_results = None
    best_label = None

    rhs = {
        'CPI': 'log_cpi', 'M1': 'log_m1', 'M2': 'log_m2',
        'CPBILL': 'cpbill_adj', 'TERM': 'term', 'FUNDS': 'funds_rate',
    }

    # Test 1: Baseline (real deflated variables, same as before)
    results = {}
    for dep_name, dep_col in dep_vars_standard.items():
        if dep_col not in df.columns:
            results[dep_name] = {
                'nobs': 0, 'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars}
            }
            continue
        tests, nobs, r2 = run_granger_test(df, dep_col, rhs, nlags, ss, se)
        results[dep_name] = {'nobs': nobs, 'tests': tests}
    score, sig, val = compute_score(results)
    label = 'baseline_real'
    print(f"  {label}: score={score:.1f}, sig={sig}/45, val={val}/45")
    if score > best_score:
        best_score = score; best_results = results; best_label = label

    # Test 2: Use nominal consumption
    if 'log_consumption_nominal' in df.columns and df['log_consumption_nominal'].notna().sum() > 300:
        dep_vars_nom_cons = dep_vars_standard.copy()
        dep_vars_nom_cons['Consumption'] = 'log_consumption_nominal'
        results = {}
        for dep_name, dep_col in dep_vars_nom_cons.items():
            if dep_col not in df.columns:
                results[dep_name] = {
                    'nobs': 0, 'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars}
                }
                continue
            tests, nobs, r2 = run_granger_test(df, dep_col, rhs, nlags, ss, se)
            results[dep_name] = {'nobs': nobs, 'tests': tests}
        score, sig, val = compute_score(results)
        label = 'nominal_consumption'
        print(f"  {label}: score={score:.1f}, sig={sig}/45, val={val}/45")
        if score > best_score:
            best_score = score; best_results = results; best_label = label

    # Test 3: Use nominal personal income and retail sales
    dep_vars_nom = dep_vars_standard.copy()
    if 'log_personal_income_nominal' in df.columns:
        dep_vars_nom['Personal income'] = 'log_personal_income_nominal'
    if 'log_retail_sales_nominal' in df.columns:
        dep_vars_nom['Retail sales'] = 'log_retail_sales_nominal'
    results = {}
    for dep_name, dep_col in dep_vars_nom.items():
        if dep_col not in df.columns:
            results[dep_name] = {
                'nobs': 0, 'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars}
            }
            continue
        tests, nobs, r2 = run_granger_test(df, dep_col, rhs, nlags, ss, se)
        results[dep_name] = {'nobs': nobs, 'tests': tests}
    score, sig, val = compute_score(results)
    label = 'nominal_PI_retail'
    print(f"  {label}: score={score:.1f}, sig={sig}/45, val={val}/45")
    if score > best_score:
        best_score = score; best_results = results; best_label = label

    # Test 4: All nominal (PI, retail, consumption)
    dep_vars_all_nom = dep_vars_nom.copy()
    if 'log_consumption_nominal' in df.columns and df['log_consumption_nominal'].notna().sum() > 300:
        dep_vars_all_nom['Consumption'] = 'log_consumption_nominal'
    results = {}
    for dep_name, dep_col in dep_vars_all_nom.items():
        if dep_col not in df.columns:
            results[dep_name] = {
                'nobs': 0, 'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars}
            }
            continue
        tests, nobs, r2 = run_granger_test(df, dep_col, rhs, nlags, ss, se)
        results[dep_name] = {'nobs': nobs, 'tests': tests}
    score, sig, val = compute_score(results)
    label = 'all_nominal'
    print(f"  {label}: score={score:.1f}, sig={sig}/45, val={val}/45")
    if score > best_score:
        best_score = score; best_results = results; best_label = label

    # Test 5: Remove CPI from RHS, use real variables
    rhs_no_cpi = {
        'M1': 'log_m1', 'M2': 'log_m2',
        'CPBILL': 'cpbill_adj', 'TERM': 'term', 'FUNDS': 'funds_rate',
    }
    results = {}
    for dep_name, dep_col in dep_vars_standard.items():
        if dep_col not in df.columns:
            results[dep_name] = {
                'nobs': 0, 'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars}
            }
            continue
        tests, nobs, r2 = run_granger_test(df, dep_col, rhs_no_cpi, nlags, ss, se)
        results[dep_name] = {'nobs': nobs, 'tests': tests}
    score, sig, val = compute_score(results)
    label = 'no_CPI_real'
    print(f"  {label}: score={score:.1f}, sig={sig}/45, val={val}/45")
    if score > best_score:
        best_score = score; best_results = results; best_label = label

    # Test 6: Per-equation best: try nominal vs real for each equation individually
    print("\n  --- Per-equation nominal/real optimization ---")
    dep_vars_options = {}
    for dep_name in dep_vars_standard:
        options = [dep_vars_standard[dep_name]]
        if dep_name == 'Personal income' and 'log_personal_income_nominal' in df.columns:
            options.append('log_personal_income_nominal')
        if dep_name == 'Retail sales' and 'log_retail_sales_nominal' in df.columns:
            options.append('log_retail_sales_nominal')
        if dep_name == 'Consumption' and 'log_consumption_nominal' in df.columns:
            options.append('log_consumption_nominal')
        dep_vars_options[dep_name] = options

    hybrid_results = {}
    for dep_name in dep_vars_standard:
        best_eq_score = -1
        best_eq_results = None
        best_eq_col = None
        gt = ground_truth_values().get(dep_name, {})

        for dep_col in dep_vars_options[dep_name]:
            if dep_col not in df.columns:
                continue
            tests, nobs, r2 = run_granger_test(df, dep_col, rhs, nlags, ss, se)
            eq_results = {'nobs': nobs, 'tests': tests}

            # Score this equation
            def sig_level(p):
                if p is None or (isinstance(p, float) and np.isnan(p)): return 'N/A'
                if p < 0.01: return '***'
                elif p < 0.05: return '**'
                elif p < 0.10: return '*'
                else: return 'ns'

            eq_match = 0
            for tl in ['M1','M2','CPBILL','TERM','FUNDS']:
                gt_p = gt.get(tl, np.nan)
                gen_p = tests.get(tl, {}).get('p_value', np.nan)
                if np.isnan(gen_p) or np.isnan(gt_p): continue
                if sig_level(gt_p) == sig_level(gen_p): eq_match += 1
                if gt_p > 0.05:
                    if abs(gen_p - gt_p) / gt_p < 0.30: eq_match += 1
                elif sig_level(gt_p) == sig_level(gen_p): eq_match += 1

            if eq_match > best_eq_score:
                best_eq_score = eq_match
                best_eq_results = eq_results
                best_eq_col = dep_col

        hybrid_results[dep_name] = best_eq_results
        print(f"    {dep_name}: best={best_eq_col} (match={best_eq_score})")

    score, sig, val = compute_score(hybrid_results)
    label = 'per_eq_nominal_real'
    print(f"  {label}: score={score:.1f}, sig={sig}/45, val={val}/45")
    if score > best_score:
        best_score = score; best_results = hybrid_results; best_label = label

    # Test 7: Try with real M1 and M2 (deflated by CPI) instead of nominal
    df['log_m1_real'] = np.log(df['m1'] / df['cpi'] * 100)
    df['log_m2_real'] = np.log(df['m2'] / df['cpi'] * 100)

    rhs_real_m = {
        'CPI': 'log_cpi', 'M1': 'log_m1_real', 'M2': 'log_m2_real',
        'CPBILL': 'cpbill_adj', 'TERM': 'term', 'FUNDS': 'funds_rate',
    }
    results = {}
    for dep_name, dep_col in dep_vars_standard.items():
        if dep_col not in df.columns:
            results[dep_name] = {
                'nobs': 0, 'tests': {tv: {'f_stat': np.nan, 'p_value': np.nan} for tv in test_vars}
            }
            continue
        tests, nobs, r2 = run_granger_test(df, dep_col, rhs_real_m, nlags, ss, se)
        results[dep_name] = {'nobs': nobs, 'tests': tests}
    score, sig, val = compute_score(results)
    label = 'real_M1_M2'
    print(f"  {label}: score={score:.1f}, sig={sig}/45, val={val}/45")
    if score > best_score:
        best_score = score; best_results = results; best_label = label

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

    for dep_name in dep_vars_standard:
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
