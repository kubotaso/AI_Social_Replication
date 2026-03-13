"""Test more specifications with the overall unemployment rate."""
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

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

def run_spec(dep_vars, rhs_vars, test_vars, panels, label, detail=False):
    n_lags = 6
    mismatches = 0
    total = 0
    details = []

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
                        if detail:
                            b = {0:'NS', 1:'10%', 2:'5%', 3:'1%'}
                            details.append(f"  {panel_key}|{dep_name:25s}|{test_var:5s}| paper={true_p:.4f}({b[s(true_p)]}) gen={gen_p:.4f}({b[s(gen_p)]})")

    print(f"{label}: {total - mismatches}/{total} exact matches ({mismatches} mismatches)")
    if detail:
        for d in details:
            print(d)
    return total - mismatches

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

# Test with overall unemp rate -- detail the mismatches
dep_vars_unemp = dep_vars_base.copy()
dep_vars_unemp['Unemployment rate'] = 'unemp_rate'
print("\n=== Overall unemployment rate (detail) ===")
run_spec(dep_vars_unemp, rhs_vars_base, test_vars, panels, "Overall unemp", detail=True)

# Test: first differences of all log variables (growth rates)
print("\n=== First differences of log variables ===")
# Need to create first-differenced variables
df['dlog_ip'] = df['log_industrial_production'].diff()
df['dlog_cu'] = df['log_capacity_utilization'].diff()
df['dlog_emp'] = df['log_employment'].diff()
df['d_unemp'] = df['unemp_male_2554'].diff()
df['dlog_hs'] = df['log_housing_starts'].diff()
df['dlog_pi'] = df['log_personal_income_real'].diff()
df['dlog_rs'] = df['log_retail_sales_real'].diff()
df['dlog_con'] = df['log_consumption_real'].diff()
df['dlog_cpi'] = df['log_cpi'].diff()
df['dlog_m1'] = df['log_m1'].diff()
df['dlog_m2'] = df['log_m2'].diff()
df['d_bill'] = df['tbill_3m'].diff()
df['d_bond'] = df['treasury_10y'].diff()
df['d_funds'] = df['funds_rate'].diff()

dep_vars_fd = {
    'Industrial production': 'dlog_ip',
    'Capacity utilization': 'dlog_cu',
    'Employment': 'dlog_emp',
    'Unemployment rate': 'd_unemp',
    'Housing starts': 'dlog_hs',
    'Personal income': 'dlog_pi',
    'Retail sales': 'dlog_rs',
    'Consumption': 'dlog_con',
}

rhs_vars_fd = {
    'CPI': 'dlog_cpi', 'M1': 'dlog_m1', 'M2': 'dlog_m2',
    'BILL': 'd_bill', 'BOND': 'd_bond', 'FUNDS': 'd_funds',
}

run_spec(dep_vars_fd, rhs_vars_fd, test_vars, panels, "First differences", detail=True)

# Test: Use commercial paper rate (cpaper_6m) for BILL
if 'cpaper_6m' in df.columns or 'cpaper_6m_long' in df.columns:
    cp_col = 'cpaper_6m' if 'cpaper_6m' in df.columns else 'cpaper_6m_long'
    rhs_vars_cp = rhs_vars_base.copy()
    rhs_vars_cp['BILL'] = cp_col
    print(f"\n=== Commercial paper rate for BILL ({cp_col}) ===")
    run_spec(dep_vars_base, rhs_vars_cp, test_vars, panels, f"CP rate ({cp_col})", detail=False)

# Check if cpaper_6m exists
for col in df.columns:
    if 'paper' in col.lower() or 'cp' in col.lower():
        print(f"  Found: {col}")
