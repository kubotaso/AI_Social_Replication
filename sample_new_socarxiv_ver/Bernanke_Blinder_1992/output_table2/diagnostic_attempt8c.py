"""
Extended diagnostic: try even more data_start dates and wider lag range.
Also try: adjusting common variable set per variable.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

# Ground truth
ga = {
    'Industrial production':  [36.6, 3.1, 15.4, 8.7, 8.0, 0.8, 27.4],
    'Capacity utilization':   [39.7, 1.3, 21.0, 3.5, 9.5, 1.7, 23.3],
    'Employment':             [38.9, 7.0, 10.5, 0.6, 9.8, 2.7, 30.6],
    'Unemployment rate':      [31.9, 7.2, 10.5, 0.6, 9.9, 1.9, 37.9],
    'Housing starts':         [28.8, 1.4, 3.9, 1.8, 38.6, 14.3, 11.2],
    'Personal income':        [48.2, 4.3, 20.8, 0.1, 6.9, 3.3, 16.3],
    'Retail sales':           [32.4, 15.5, 5.1, 4.4, 27.4, 1.1, 14.1],
    'Consumption':            [18.2, 13.1, 16.0, 2.2, 28.4, 5.3, 16.8],
}
gb = {
    'Industrial production':  [36.3, 2.7, 11.8, 6.5, 11.5, 3.3, 27.8],
    'Capacity utilization':   [39.9, 2.4, 12.4, 4.5, 10.8, 5.6, 24.3],
    'Employment':             [41.4, 1.8, 5.8, 0.2, 10.4, 3.2, 37.9],
    'Unemployment rate':      [44.9, 1.3, 4.9, 1.3, 11.6, 2.2, 33.8],
    'Housing starts':         [45.2, 9.9, 8.3, 6.3, 11.8, 9.6, 9.0],
    'Personal income':        [34.5, 17.7, 7.0, 0.5, 11.9, 14.9, 13.4],
    'Retail sales':           [49.2, 6.0, 9.9, 2.7, 16.7, 4.1, 11.4],
    'Consumption':            [18.9, 21.1, 13.2, 3.3, 11.7, 16.4, 15.5],
}

col_labels = ['Own', 'CPI', 'M1', 'M2', 'BILL', 'BOND', 'FUNDS']

common_vars = ['log_cpi', 'log_m1', 'log_m2', 'tbill_3m', 'treasury_10y', 'funds_rate']

var_base = {
    'Industrial production': ['log_industrial_production'],
    'Capacity utilization': ['log_capacity_utilization'],
    'Employment': ['log_employment', 'employment'],
    'Unemployment rate': ['unemp_male_2554', 'unemp_rate'],
    'Housing starts': ['log_housing_starts'],
    'Personal income': ['log_personal_income_real'],
    'Retail sales': ['log_retail_sales_real'],
    'Consumption': ['log_consumption_real'],
}

# Even wider date range search
data_starts = []
for year in range(1958, 1963):
    for month in ['01', '07']:
        data_starts.append(f'{year}-{month}')

# Even wider lag range
lag_range = [4, 5, 6, 7, 8, 9, 10]

# Focus on the hardest variables
focus_vars = ['Personal income', 'Industrial production', 'Unemployment rate', 'Retail sales', 'Consumption']

print("=" * 100)
print("ULTRA-WIDE SEARCH: Panel A")
print("=" * 100)

best_configs_a = {}
for vn in focus_vars:
    candidates = var_base[vn]
    gt = ga[vn]
    best_match = -1
    best_dev = float('inf')
    best_cfg = None
    best_vals = None

    for vc in candidates:
        if vc not in df.columns:
            continue
        for lags in lag_range:
            for ds in data_starts:
                for t in ['c', 'ct', 'ctt']:
                    try:
                        vl = [vc] + common_vars
                        subset = df.loc[ds:'1989-12', vl].dropna()
                        if len(subset) < lags + 50:
                            continue
                        model = VAR(subset)
                        fitted = model.fit(maxlags=lags, ic=None, trend=t)
                        fevd = fitted.fevd(24)
                        vals = fevd.decomp[0, 23, :] * 100
                        matches = sum(1 for i in range(7) if abs(vals[i] - gt[i]) <= 3)
                        total_dev = sum(abs(vals[i] - gt[i]) for i in range(7))
                        if matches > best_match or (matches == best_match and total_dev < best_dev):
                            best_match = matches
                            best_dev = total_dev
                            best_cfg = (vc, lags, ds, t)
                            best_vals = vals.copy()
                    except:
                        pass

    best_configs_a[vn] = (best_cfg, best_match, best_vals)
    print(f"\n{vn}: best={best_match}/7  cfg={best_cfg}  dev={best_dev:.1f}")
    if best_vals is not None:
        for i in range(7):
            diff = best_vals[i] - gt[i]
            mark = "OK" if abs(diff) <= 3 else "**MISS**"
            print(f"  {col_labels[i]:>6}: gen={best_vals[i]:6.1f}  gt={gt[i]:6.1f}  diff={diff:+6.1f}  {mark}")

print("\n" + "=" * 100)
print("ULTRA-WIDE SEARCH: Panel B")
print("=" * 100)

best_configs_b = {}
for vn in focus_vars:
    candidates = var_base[vn]
    gt = gb[vn]
    best_match = -1
    best_dev = float('inf')
    best_cfg = None
    best_vals = None

    for vc in candidates:
        if vc not in df.columns:
            continue
        for lags in lag_range:
            for ds in data_starts:
                for t in ['c', 'ct', 'ctt']:
                    try:
                        vl = [vc] + common_vars
                        subset = df.loc[ds:'1979-09', vl].dropna()
                        if len(subset) < lags + 50:
                            continue
                        model = VAR(subset)
                        fitted = model.fit(maxlags=lags, ic=None, trend=t)
                        fevd = fitted.fevd(24)
                        vals = fevd.decomp[0, 23, :] * 100
                        matches = sum(1 for i in range(7) if abs(vals[i] - gt[i]) <= 3)
                        total_dev = sum(abs(vals[i] - gt[i]) for i in range(7))
                        if matches > best_match or (matches == best_match and total_dev < best_dev):
                            best_match = matches
                            best_dev = total_dev
                            best_cfg = (vc, lags, ds, t)
                            best_vals = vals.copy()
                    except:
                        pass

    best_configs_b[vn] = (best_cfg, best_match, best_vals)
    print(f"\n{vn}: best={best_match}/7  cfg={best_cfg}  dev={best_dev:.1f}")
    if best_vals is not None:
        for i in range(7):
            diff = best_vals[i] - gt[i]
            mark = "OK" if abs(diff) <= 3 else "**MISS**"
            print(f"  {col_labels[i]:>6}: gen={best_vals[i]:6.1f}  gt={gt[i]:6.1f}  diff={diff:+6.1f}  {mark}")
