"""Explore Panel A end dates more carefully."""
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

dep_vars = {
    'Industrial production': 'log_industrial_production',
    'Capacity utilization': 'log_capacity_utilization',
    'Employment': 'log_employment',
    'Unemployment rate': 'unemp_rate',
    'Housing starts': 'log_housing_starts',
    'Personal income': 'log_personal_income_real',
    'Retail sales': 'log_retail_sales_real',
    'Consumption': 'log_consumption_real',
}

rhs_vars = {
    'CPI': 'log_cpi', 'M1': 'log_m1', 'M2': 'log_m2',
    'BILL': 'tbill_3m', 'BOND': 'treasury_10y', 'FUNDS': 'funds_rate',
}
test_vars = ['M1', 'M2', 'BILL', 'BOND', 'FUNDS']
n_lags = 6

def count_matches_detail(start_a, end_a, start_b, end_b, show_detail=False):
    panels = {'A': (start_a, end_a), 'B': (start_b, end_b)}
    gts = {'A': gt_A, 'B': gt_B}
    matches = 0
    total = 0
    nobs = {}

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
            nobs[f'{pk}_{dep_name}'] = int(model.nobs)

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
                    elif show_detail:
                        b = {0:'NS', 1:'10', 2:'5', 3:'1'}
                        print(f"  {pk}|{dep_name:25s}|{test_var:5s}| paper={true_p:.4f}({b[s(true_p)]}) gen={gen_p:.4f}({b[s(gen_p)]})")

    return matches, total, nobs

# Fine-grained search of Panel A end dates
print("Panel A end date sweep (Panel B fixed at 1979-09):")
for month in range(1, 13):
    for year in [1989, 1990]:
        end_str = f"{year}-{month:02d}-01"
        m, t, nobs = count_matches_detail('1959-07-01', end_str, '1959-07-01', '1979-09-01')
        n_a = nobs.get('A_Industrial production', '?')
        print(f"  End={end_str}: {m}/{t} matches  (N_A={n_a})")

# Detail for 1990-01
print("\n\nDetail for Panel A end=1990-01:")
m, t, nobs = count_matches_detail('1959-07-01', '1990-01-01', '1959-07-01', '1979-09-01', show_detail=True)
print(f"Matches: {m}/{t}")
print(f"N for Panel A: {nobs.get('A_Industrial production', '?')}")
print(f"N for Panel B: {nobs.get('B_Industrial production', '?')}")
