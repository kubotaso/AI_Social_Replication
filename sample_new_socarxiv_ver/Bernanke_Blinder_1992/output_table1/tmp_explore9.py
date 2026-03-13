"""Try additional tweaks to push beyond 62/80."""
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

gt_A = {
    'Industrial production':  {'M1': 0.92, 'M2': 0.10, 'BILL': 0.071, 'BOND': 0.26, 'FUNDS': 0.017},
    'Capacity utilization':   {'M1': 0.74, 'M2': 0.22, 'BILL': 0.16,  'BOND': 0.40, 'FUNDS': 0.031},
    'Employment':             {'M1': 0.45, 'M2': 0.27, 'BILL': 0.0040,'BOND': 0.085,'FUNDS': 0.0004},
    'Unemployment rate':      {'M1': 0.96, 'M2': 0.37, 'BILL': 0.0005,'BOND': 0.024,'FUNDS': 0.0001},
    'Housing starts':         {'M1': 0.50, 'M2': 0.32, 'BILL': 0.52,  'BOND': 0.014,'FUNDS': 0.22},
    'Personal income':        {'M1': 0.38, 'M2': 0.24, 'BILL': 0.35,  'BOND': 0.59, 'FUNDS': 0.049},
    'Retail sales':           {'M1': 0.64, 'M2': 0.036,'BILL': 0.33,  'BOND': 0.74, 'FUNDS': 0.014},
    'Consumption':            {'M1': 0.96, 'M2': 0.11, 'BILL': 0.12,  'BOND': 0.46, 'FUNDS': 0.0052},
}
gt_B = {
    'Industrial production':  {'M1': 0.99, 'M2': 0.084,'BILL': 0.0092,'BOND': 0.61, 'FUNDS': 0.0001},
    'Capacity utilization':   {'M1': 0.96, 'M2': 0.40, 'BILL': 0.025, 'BOND': 0.18, 'FUNDS': 0.0003},
    'Employment':             {'M1': 0.57, 'M2': 0.41, 'BILL': 0.0005,'BOND': 0.15, 'FUNDS': 0.0004},
    'Unemployment rate':      {'M1': 0.56, 'M2': 0.88, 'BILL': 0.0006,'BOND': 0.13, 'FUNDS': 0.0000},
    'Housing starts':         {'M1': 0.34, 'M2': 0.17, 'BILL': 0.73,  'BOND': 0.72, 'FUNDS': 0.11},
    'Personal income':        {'M1': 0.43, 'M2': 0.095,'BILL': 0.20,  'BOND': 0.91, 'FUNDS': 0.037},
    'Retail sales':           {'M1': 0.96, 'M2': 0.86, 'BILL': 0.27,  'BOND': 0.050,'FUNDS': 0.061},
    'Consumption':            {'M1': 0.79, 'M2': 0.017,'BILL': 0.010, 'BOND': 0.050,'FUNDS': 0.0000},
}

def s(p):
    if p <= 0.01: return 3
    if p <= 0.05: return 2
    if p <= 0.10: return 1
    return 0

n_lags = 6
test_vars = ['M1', 'M2', 'BILL', 'BOND', 'FUNDS']

def run_config_detail(dep_vars, rhs_vars, end_a, end_b, label):
    panels = {'A': ('1959-07-01', end_a), 'B': ('1959-07-01', end_b)}
    gts = {'A': gt_A, 'B': gt_B}
    matches = 0
    total = 0
    mismatches = []
    for pk in ['A', 'B']:
        s_date, e_date = panels[pk]
        gt = gts[pk]
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
                if dep_name in gt and test_var in gt[dep_name]:
                    true_p = gt[dep_name][test_var]
                    total += 1
                    if s(true_p) == s(gen_p):
                        matches += 1
                    else:
                        b = {0:'NS', 1:'10', 2:'5', 3:'1'}
                        mismatches.append(f"  {pk}|{dep_name:25s}|{test_var:5s}| paper={true_p:.4f}({b[s(true_p)]}) gen={gen_p:.4f}({b[s(gen_p)]})")
    print(f"{label}: {matches}/{total} matches")
    return matches, total, mismatches

# Current best config
dep_vars_base = {
    'Industrial production': 'log_industrial_production',
    'Capacity utilization': 'log_capacity_utilization',
    'Employment': 'log_employment',
    'Unemployment rate': 'unemp_rate',
    'Housing starts': 'log_housing_starts',
    'Personal income': 'log_personal_income_real',
    'Retail sales': 'log_retail_sales_real',
    'Consumption': 'log_consumption_real',
}
rhs_vars_base = {
    'CPI': 'log_cpi', 'M1': 'log_m1', 'M2': 'log_m2',
    'BILL': 'tbill_3m', 'BOND': 'treasury_10y', 'FUNDS': 'funds_rate',
}

# Try adding a time trend
print("=== Adding a time trend ===")
# Can't easily add trend through the dict mechanism, need direct implementation
# Skip -- would need code refactor

# Try 7 lags instead of 6
print("=== 7 lags ===")
n_lags = 7
m, t, mis = run_config_detail(dep_vars_base, rhs_vars_base, '1990-02-01', '1979-09-01', "7 lags")
n_lags = 6

# Try 5 lags
print("=== 5 lags ===")
n_lags = 5
m, t, mis = run_config_detail(dep_vars_base, rhs_vars_base, '1990-02-01', '1979-09-01', "5 lags")
n_lags = 6

# Try capacity utilization in raw levels (not log) with 1990-02 end
dep_vars_cu = dep_vars_base.copy()
dep_vars_cu['Capacity utilization'] = 'capacity_utilization'
print("\n=== CU in levels + 1990-02 end ===")
m, t, mis = run_config_detail(dep_vars_cu, rhs_vars_base, '1990-02-01', '1979-09-01', "CU levels + 1990-02")

# Try with deflated variables constructed differently
# Check if we can use nominal vars and deflate ourselves
print("\n=== Check if nominal/real construction matters ===")
# Test: use log(nominal_personal_income/cpi) vs log_personal_income_real
df['test_pi_real'] = np.log(df['personal_income_nominal'] / df['cpi'])
df['test_rs_real'] = np.log(df['retail_sales_nominal'] / df['cpi'])
df['test_con_real'] = np.log(df['consumption_nominal'] / df['cpi'])

print("Difference between log_personal_income_real and log(nominal/CPI):")
diff = (df['log_personal_income_real'] - df['test_pi_real']).dropna()
print(f"  Max absolute diff: {diff.abs().max():.6f}")
print(f"  Mean absolute diff: {diff.abs().mean():.6f}")

diff_rs = (df['log_retail_sales_real'] - df['test_rs_real']).dropna()
print(f"  Retail sales max diff: {diff_rs.abs().max():.6f}")

diff_con = (df['log_consumption_real'] - df['test_con_real']).dropna()
print(f"  Consumption max diff: {diff_con.abs().max():.6f}")
