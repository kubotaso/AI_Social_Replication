"""Quick validation: replicate a few cells of Table 1 Panel A to confirm data works."""

import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

# Sample period for Panel A
sample = df.loc['1959-07-01':'1989-12-01'].copy()

# RHS variables (always included)
rhs_vars = {
    'CPI': 'log_cpi',
    'M1': 'log_m1',
    'M2': 'log_m2',
    'BILL': 'tbill_3m',
    'BOND': 'treasury_10y',
    'FUNDS': 'funds_rate',
}

# Test variables (columns of Table 1)
test_vars = ['M1', 'M2', 'BILL', 'BOND', 'FUNDS']

# Forecasted variables (rows of Table 1)
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

NLAGS = 6

def run_granger_test(dep_col, sample_df):
    """Run the 6-variable prediction equation and return F-test p-values."""
    # Build lagged variables
    data = pd.DataFrame(index=sample_df.index)

    # Dependent variable and its lags
    data['y'] = sample_df[dep_col]
    for j in range(1, NLAGS + 1):
        data[f'y_L{j}'] = sample_df[dep_col].shift(j)

    # RHS variable lags
    for name, col in rhs_vars.items():
        for j in range(1, NLAGS + 1):
            data[f'{name}_L{j}'] = sample_df[col].shift(j)

    # Drop NaN rows (from lagging)
    data = data.dropna()

    # Dependent and independent variables
    y = data['y']
    X_cols = [c for c in data.columns if c != 'y']
    X = sm.add_constant(data[X_cols])

    # Fit OLS
    model = sm.OLS(y, X).fit()

    # F-test for each test variable
    results = {}
    for test_name in test_vars:
        # Find the columns corresponding to this variable's lags
        lag_cols = [f'{test_name}_L{j}' for j in range(1, NLAGS + 1)]
        # Build restriction matrix
        restriction = np.zeros((NLAGS, len(X.columns)))
        for i, col_name in enumerate(lag_cols):
            col_idx = list(X.columns).index(col_name)
            restriction[i, col_idx] = 1.0

        f_test = model.f_test(restriction)
        p_value = float(f_test.pvalue)
        results[test_name] = p_value

    return results, len(y)

# Run for all dependent variables
print("Table 1, Panel A (1959:7-1989:12)")
print("=" * 70)
print(f"{'Variable':<25} {'M1':>8} {'M2':>8} {'BILL':>8} {'BOND':>8} {'FUNDS':>8}")
print("-" * 70)

for dep_name, dep_col in dep_vars.items():
    try:
        results, nobs = run_granger_test(dep_col, sample)
        row = f"{dep_name:<25}"
        for tv in test_vars:
            p = results[tv]
            if p < 0.001:
                row += f" {p:>8.4f}"
            else:
                row += f" {p:>8.2f}"
        print(row)
    except Exception as e:
        print(f"{dep_name:<25} ERROR: {e}")

print(f"\nN observations: {nobs}")

# Compare key values with paper
print("\n\nVALIDATION: Comparing key values with paper")
print("=" * 50)
paper_vals = {
    ('Industrial production', 'FUNDS'): 0.017,
    ('Unemployment rate', 'FUNDS'): 0.0001,
    ('Employment', 'BILL'): 0.004,
    ('Retail sales', 'M2'): 0.036,
}

results_all = {}
for dep_name, dep_col in dep_vars.items():
    results_all[dep_name], _ = run_granger_test(dep_col, sample)

for (dep, col), paper_p in paper_vals.items():
    our_p = results_all[dep][col]
    match = "MATCH" if (paper_p < 0.05) == (our_p < 0.05) else "MISMATCH"
    print(f"  {dep} / {col}: paper={paper_p:.4f}, ours={our_p:.4f} - significance {match}")
