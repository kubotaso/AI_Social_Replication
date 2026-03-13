"""Test different OLS control sets for constrained OLS approach"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

df = pd.read_csv('data/psid_panel.csv')
df['pnum'] = df['person_id'] % 1000
df = df[df['pnum'].isin([1, 170, 3])].copy()
df['education_years'] = np.nan
for yr in df['year'].unique():
    mask = df['year'] == yr
    if yr in [1975, 1976]:
        df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean']
    else:
        df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_MAP)
df = df.dropna(subset=['education_years']).copy()
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df['tenure'] = df['tenure_topel'].copy()
df['gnp_def'] = df['year'].map(lambda y: GNP_DEFLATOR.get(y - 1, np.nan))
df['log_real_gnp'] = df['log_hourly_wage'] - np.log(df['gnp_def'] / 100.0)
df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
df = df[df['hourly_wage'] > 0].copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
df = df[df['tenure'] >= 1].copy()
df = df.dropna(subset=['log_real_gnp', 'experience', 'tenure']).copy()
df['union_member'] = df['union_member'].fillna(0)
df['disabled'] = df['disabled'].fillna(0)
df['married'] = df['married'].fillna(0)
for k in [2, 3, 4]:
    df[f'tenure_{k}'] = df['tenure'] ** k
    df[f'exp_{k}'] = df['experience'] ** k
for yr in range(1970, 1984):
    col = f'year_{yr}'
    if col not in df.columns:
        df[col] = (df['year'] == yr).astype(int)

# Occupation dummies
for occ in range(10):
    col = f'occ_{occ}'
    if col in df.columns:
        df[col] = df[col].fillna(0)

ctrl_base = ['education_years', 'married', 'union_member', 'disabled',
             'region_ne', 'region_nc', 'region_south']
yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
yr_dums = yr_dums[1:]

g2, g3, g4 = -0.004592, 0.0001846, -0.00000245
wage_var = 'log_real_gnp'

# Construct adjusted wage (subtracting step 1 higher-order tenure terms)
df['wage_adj'] = df[wage_var] - g2 * df['tenure_2'] - g3 * df['tenure_3'] - g4 * df['tenure_4']

targets = {5: 0.2313, 10: 0.3002, 15: 0.3203, 20: 0.3563}

def test_ols(label, ctrl_list, extra_vars=None):
    xvars = ['experience', 'tenure', 'exp_2', 'exp_3', 'exp_4'] + ctrl_list + yr_dums
    if extra_vars:
        xvars = xvars + extra_vars
    dfc = df.dropna(subset=xvars + ['wage_adj']).copy()
    dfc = dfc.replace([np.inf, -np.inf], np.nan).dropna(subset=xvars + ['wage_adj'])
    X = sm.add_constant(dfc[xvars])
    ols = sm.OLS(dfc['wage_adj'], X).fit()
    a1 = ols.params['tenure']
    a1_se = ols.bse['tenure']
    results = {}
    for T in [5, 10, 15, 20]:
        cr = a1 * T + g2 * T**2 + g3 * T**3 + g4 * T**4
        results[T] = cr
    cr15_err = abs(results[15] - targets[15])
    status = "PASS" if cr15_err <= 0.03 else "FAIL"
    print(f"{label:40s}: a1={a1:.6f}(SE {a1_se:.6f}), 5yr={results[5]:.4f}, 10yr={results[10]:.4f}, 15yr={results[15]:.4f}({status}), 20yr={results[20]:.4f}, N={len(dfc)}")
    return a1

# Baseline
test_ols("Baseline", ctrl_base)

# With occupation dummies
occ_dums = [f'occ_{i}' for i in range(1, 10)]
test_ols("+ Occupation", ctrl_base, occ_dums)

# Without year dummies
print()
xvars = ['experience', 'tenure', 'exp_2', 'exp_3', 'exp_4'] + ctrl_base
dfc = df.dropna(subset=xvars + ['wage_adj']).copy()
X = sm.add_constant(dfc[xvars])
ols = sm.OLS(dfc['wage_adj'], X).fit()
a1 = ols.params['tenure']
for T in [5, 10, 15, 20]:
    cr = a1 * T + g2 * T**2 + g3 * T**3 + g4 * T**4
    status = "PASS" if abs(cr - targets[T]) <= 0.03 else "FAIL"
    if T == 15 or T == 20:
        print(f"No yr dummies: {T}yr={cr:.4f} (err={abs(cr-targets[T]):.4f}) {status}")

# Quadratic experience
test_ols("Quadratic exp", ctrl_base)

# With region_west
test_ols("+ region_west", ctrl_base + ['region_west'])

# Drop disabled
test_ols("No disabled", [c for c in ctrl_base if c != 'disabled'])

# With education squared
df['education_sq'] = df['education_years'] ** 2
test_ols("+ education^2", ctrl_base, ['education_sq'])

# With experience*tenure interaction
df['exp_ten'] = df['experience'] * df['tenure']
test_ols("+ exp*tenure", ctrl_base, ['exp_ten'])

# Using the standard (not constrained) quartic OLS
print("\nStandard OLS quartic (NOT constrained):")
xvars = ['experience', 'tenure', 'tenure_2', 'tenure_3', 'tenure_4',
         'exp_2', 'exp_3', 'exp_4'] + ctrl_base + yr_dums
dfc = df.dropna(subset=xvars + [wage_var]).copy().replace([np.inf, -np.inf], np.nan).dropna(subset=xvars + [wage_var])
X = sm.add_constant(dfc[xvars])
ols = sm.OLS(dfc[wage_var], X).fit()
for T in [5, 10, 15, 20]:
    cr = (ols.params['tenure']*T + ols.params['tenure_2']*T**2
          + ols.params['tenure_3']*T**3 + ols.params['tenure_4']*T**4)
    print(f"  {T}yr: {cr:.4f} (target: {targets[T]:.4f})")

# With occupation dummies in standard quartic
xvars_occ = xvars + occ_dums
dfc_occ = df.dropna(subset=xvars_occ + [wage_var]).copy().replace([np.inf, -np.inf], np.nan).dropna(subset=xvars_occ + [wage_var])
X_occ = sm.add_constant(dfc_occ[xvars_occ])
ols_occ = sm.OLS(dfc_occ[wage_var], X_occ).fit()
print("\nStandard OLS quartic with occ dummies:")
for T in [5, 10, 15, 20]:
    cr = (ols_occ.params['tenure']*T + ols_occ.params['tenure_2']*T**2
          + ols_occ.params['tenure_3']*T**3 + ols_occ.params['tenure_4']*T**4)
    print(f"  {T}yr: {cr:.4f} (target: {targets[T]:.4f})")
