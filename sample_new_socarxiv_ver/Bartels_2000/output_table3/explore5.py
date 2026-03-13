import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit

df = pd.read_csv('anes_cumulative.csv', usecols=['VCF0004','VCF0301','VCF0707','VCF0902'], low_memory=False)

# For 1982, try different VCF0902 code mappings
print("=== 1982 VCF0902 code investigation ===")
sub82 = df[(df['VCF0004']==1982) & (df['VCF0707'].isin([1,2]))].copy()
print(f"N = {len(sub82)}")
print("VCF0902 codes:", sorted(sub82['VCF0902'].dropna().unique()))

# Try variations for 1982:
# Variation 1: Code 40 as Dem incumbent (-1) instead of open seat
# Variation 2: Code 49 as open seat (already the case)
# Variation 3: Exclude codes 13, 19, 29 (treat them as non-standard)

dem_inc = [12, 13, 14, 19]
rep_inc = [21, 23, 24, 29]

def run_probit_for_year(df, year, dem_codes, rep_codes, include_null=True):
    sub = df[(df['VCF0004']==year) & (df['VCF0707'].isin([1,2]))].copy()
    if not include_null:
        sub = sub[sub['VCF0902'].notna()]
    sub['vote_rep'] = (sub['VCF0707'] == 2).astype(int)
    sub['strong'] = 0
    sub['weak'] = 0
    sub['leaner'] = 0
    sub.loc[sub['VCF0301'] == 7, 'strong'] = 1
    sub.loc[sub['VCF0301'] == 1, 'strong'] = -1
    sub.loc[sub['VCF0301'] == 6, 'weak'] = 1
    sub.loc[sub['VCF0301'] == 2, 'weak'] = -1
    sub.loc[sub['VCF0301'] == 5, 'leaner'] = 1
    sub.loc[sub['VCF0301'] == 3, 'leaner'] = -1
    sub['incumbency'] = 0
    sub.loc[sub['VCF0902'].isin(dem_codes), 'incumbency'] = -1
    sub.loc[sub['VCF0902'].isin(rep_codes), 'incumbency'] = 1
    y = sub['vote_rep']
    X = sm.add_constant(sub[['strong', 'weak', 'leaner', 'incumbency']])
    model = Probit(y, X)
    result = model.fit(disp=0)
    return len(sub), result

# 1982 ground truth: LL=-265.7, R2=0.45
# Current result: LL=-267.5, diff=1.8

# Test variations for 1982
variations_82 = [
    ("Standard (12,13,14,19 vs 21,23,24,29)", [12,13,14,19], [21,23,24,29]),
    ("Only 12,14 vs 21,24", [12,14], [21,24]),
    ("12,14,19 vs 21,24,29", [12,14,19], [21,24,29]),
    ("Add 40 as Dem inc", [12,13,14,19,40], [21,23,24,29]),
    ("Code 13,19 -> open seat", [12,14], [21,24]),
]

for name, dem, rep in variations_82:
    n, result = run_probit_for_year(df, 1982, dem, rep, include_null=True)
    print(f"\n  {name}:")
    print(f"    N={n}, LL={result.llf:.1f}, R2={result.prsquared:.4f}")
    print(f"    Strong={result.params['strong']:.3f}, Weak={result.params['weak']:.3f}, Leaner={result.params['leaner']:.3f}")
    print(f"    Inc={result.params['incumbency']:.3f}, Const={result.params['const']:.3f}")

# Now check 1976
print("\n\n=== 1976 VCF0902 code investigation ===")
sub76 = df[(df['VCF0004']==1976) & (df['VCF0707'].isin([1,2]))].copy()
print(f"N total = {len(sub76)}")
print("VCF0902 codes:", sorted(sub76['VCF0902'].dropna().unique()))

# 1976 ground truth: LL=-482.0
# Current (include null): LL=-479.7, diff=2.3
# Without null: LL=-478.9, N=1076

# Try some VCF0301 filter variations for 1976
for name, dem, rep in [("Standard", [12,13,14,19], [21,23,24,29])]:
    for inc_null in [True, False]:
        n, result = run_probit_for_year(df, 1976, dem, rep, include_null=inc_null)
        print(f"\n  {name}, include_null={inc_null}:")
        print(f"    N={n}, LL={result.llf:.1f}, R2={result.prsquared:.4f}")

# Check 1994 more carefully
print("\n\n=== 1994 investigation ===")
sub94 = df[(df['VCF0004']==1994) & (df['VCF0707'].isin([1,2]))].copy()
print(f"N = {len(sub94)}")
print("VCF0902 codes:", sorted(sub94['VCF0902'].dropna().unique()))

# 1994 ground truth: strong=1.471, weak=0.706
# Current: strong=1.466 (diff=0.005), weak=0.730 (diff=0.024)
# weak is within 0.05 so it passes. But 0.024 is notable.

# Check if code 23 in 1994 matters
for name, dem, rep in [
    ("Standard (incl 23)", [12,13,14,19], [21,23,24,29]),
    ("Without 23", [12,13,14,19], [21,24,29]),
]:
    n, result = run_probit_for_year(df, 1994, dem, rep, include_null=True)
    print(f"\n  {name}:")
    print(f"    N={n}, LL={result.llf:.1f}")
    print(f"    Strong={result.params['strong']:.3f}, Weak={result.params['weak']:.3f}")
    print(f"    Inc={result.params['incumbency']:.3f}")
