"""Test different specifications to see which reduces mismatches most."""
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

# Ground truth
gt = {
    'A': {
        'Industrial production':  {'M1': 0.92, 'M2': 0.10, 'BILL': 0.071, 'BOND': 0.26, 'FUNDS': 0.017},
        'Capacity utilization':   {'M1': 0.74, 'M2': 0.22, 'BILL': 0.16,  'BOND': 0.40, 'FUNDS': 0.031},
        'Employment':             {'M1': 0.45, 'M2': 0.27, 'BILL': 0.0040,'BOND': 0.085,'FUNDS': 0.0004},
        'Unemployment rate':      {'M1': 0.96, 'M2': 0.37, 'BILL': 0.0005,'BOND': 0.024,'FUNDS': 0.0001},
        'Housing starts':         {'M1': 0.50, 'M2': 0.32, 'BILL': 0.52,  'BOND': 0.014,'FUNDS': 0.22},
        'Personal income':        {'M1': 0.38, 'M2': 0.24, 'BILL': 0.35,  'BOND': 0.59, 'FUNDS': 0.049},
        'Retail sales':           {'M1': 0.64, 'M2': 0.036,'BILL': 0.33,  'BOND': 0.74, 'FUNDS': 0.014},
        'Consumption':            {'M1': 0.96, 'M2': 0.11, 'BILL': 0.12,  'BOND': 0.46, 'FUNDS': 0.0052},
    },
    'B': {
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

def s(p):
    if p <= 0.01: return 3
    if p <= 0.05: return 2
    if p <= 0.10: return 1
    return 0

def run_spec(dep_vars, rhs_vars, test_vars, panels, label):
    """Run a specification and count mismatches."""
    n_lags = 6
    mismatches = 0
    total = 0

    for panel_key, (panel_name, s_date, e_date) in panels.items():
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
            est_df = full_df.loc[s_date:e_date].dropna()

            Y = est_df[dep_col]
            X_cols = [c for c in est_df.columns if c != dep_col]
            X = sm.add_constant(est_df[X_cols])
            model = sm.OLS(Y, X).fit()

            for test_var in test_vars:
                lag_names = [f'{test_var}_L{j}' for j in range(1, n_lags + 1)]
                R = np.zeros((n_lags, len(model.params)))
                for i, lag_name in enumerate(lag_names):
                    col_idx = list(model.params.index).index(lag_name)
                    R[i, col_idx] = 1.0
                f_result = model.f_test(R)
                gen_p = float(f_result.pvalue)

                if dep_name in gt[panel_key] and test_var in gt[panel_key][dep_name]:
                    true_p = gt[panel_key][dep_name][test_var]
                    total += 1
                    if s(true_p) != s(gen_p):
                        mismatches += 1

    print(f"{label}: {total - mismatches}/{total} exact matches ({mismatches} mismatches)")
    return total - mismatches

# Baseline specification
dep_vars_base = {
    'Industrial production': 'log_industrial_production',
    'Capacity utilization': 'log_capacity_utilization',
    'Employment': 'log_employment',
    'Unemployment rate': 'unemp_male_2554',
    'Housing starts': 'log_housing_starts',
    'Personal income': 'log_personal_income_real',
    'Retail sales': 'log_retail_sales_real',
    'Consumption': 'log_consumption_real',
}

rhs_vars_base = {
    'CPI': 'log_cpi', 'M1': 'log_m1', 'M2': 'log_m2',
    'BILL': 'tbill_3m', 'BOND': 'treasury_10y', 'FUNDS': 'funds_rate',
}

test_vars = ['M1', 'M2', 'BILL', 'BOND', 'FUNDS']
panels = {
    'A': ('Panel A', '1959-07-01', '1989-12-01'),
    'B': ('Panel B', '1959-07-01', '1979-09-01'),
}

run_spec(dep_vars_base, rhs_vars_base, test_vars, panels, "Baseline (current best)")

# Test 1: Use capacity_utilization in levels instead of logs
dep_vars_test1 = dep_vars_base.copy()
dep_vars_test1['Capacity utilization'] = 'capacity_utilization'
run_spec(dep_vars_test1, rhs_vars_base, test_vars, panels, "Test 1: CU in levels")

# Test 2: Use overall unemployment rate
dep_vars_test2 = dep_vars_base.copy()
dep_vars_test2['Unemployment rate'] = 'unemp_rate'
run_spec(dep_vars_test2, rhs_vars_base, test_vars, panels, "Test 2: Overall unemp rate")

# Test 3: Use 6-month T-bill instead of 3-month
rhs_vars_test3 = rhs_vars_base.copy()
rhs_vars_test3['BILL'] = 'tbill_6m'
run_spec(dep_vars_base, rhs_vars_test3, test_vars, panels, "Test 3: 6-month T-bill")

# Test 4: Panel B end at 1979:12 instead of 1979:9
panels_test4 = {
    'A': ('Panel A', '1959-07-01', '1989-12-01'),
    'B': ('Panel B', '1959-07-01', '1979-12-01'),
}
run_spec(dep_vars_base, rhs_vars_base, test_vars, panels_test4, "Test 4: Panel B end 1979:12")

# Test 5: Use 1-year Treasury instead of 10-year for BOND
rhs_vars_test5 = rhs_vars_base.copy()
rhs_vars_test5['BOND'] = 'treasury_1y'
run_spec(dep_vars_base, rhs_vars_test5, test_vars, panels, "Test 5: 1-year Treasury for BOND")

# Test 6: Combine CU levels + overall unemp
dep_vars_test6 = dep_vars_base.copy()
dep_vars_test6['Capacity utilization'] = 'capacity_utilization'
dep_vars_test6['Unemployment rate'] = 'unemp_rate'
run_spec(dep_vars_test6, rhs_vars_base, test_vars, panels, "Test 6: CU levels + overall unemp")

# Test 7: Use tbill_6m for BILL + treasury_1y for BOND
rhs_vars_test7 = rhs_vars_base.copy()
rhs_vars_test7['BILL'] = 'tbill_6m'
rhs_vars_test7['BOND'] = 'treasury_1y'
run_spec(dep_vars_base, rhs_vars_test7, test_vars, panels, "Test 7: 6m bill + 1y bond")

# Test 8: Use discount_rate instead of funds_rate for FUNDS
rhs_vars_test8 = rhs_vars_base.copy()
rhs_vars_test8['FUNDS'] = 'discount_rate'
run_spec(dep_vars_base, rhs_vars_test8, test_vars, panels, "Test 8: Discount rate for FUNDS")
