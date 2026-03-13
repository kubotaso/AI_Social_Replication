"""Test OLS cumulative returns at 15/20 years"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm

GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

def prepare_data(data_source, pnum_filter=None):
    df = pd.read_csv(data_source)
    if 'pnum' not in df.columns:
        df['pnum'] = df['person_id'] % 1000
    if pnum_filter is not None:
        df = df[df['pnum'].isin(pnum_filter)].copy()
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
    return df

df = prepare_data("data/psid_panel.csv", pnum_filter=[1, 170, 3])
ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
yr_dums = yr_dums[1:]

wage_var = 'log_real_gnp'

# Drop any remaining NaN/inf
all_possible = ['experience', 'tenure', 'tenure_2', 'tenure_3', 'tenure_4',
                'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums + [wage_var]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=all_possible).copy()
print(f"N after clean: {len(df)}")
print(f"Tenure range: {df['tenure'].min():.1f} - {df['tenure'].max():.1f}")

# Paper OLS targets:
# 5yr: 0.2313, 10yr: 0.3002, 15yr: 0.3203, 20yr: 0.3563

# ===== OLS with quartic tenure =====
print("=== OLS Quartic Tenure ===")
ols_vars_q4 = ['experience', 'tenure', 'tenure_2', 'tenure_3', 'tenure_4',
               'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X = sm.add_constant(df[ols_vars_q4])
ols_q4 = sm.OLS(df[wage_var], X).fit()
print(f"  tenure: {ols_q4.params['tenure']:.6f} ({ols_q4.bse['tenure']:.6f})")
print(f"  tenure_2: {ols_q4.params['tenure_2']:.6f}")
print(f"  tenure_3: {ols_q4.params['tenure_3']:.8f}")
print(f"  tenure_4: {ols_q4.params['tenure_4']:.10f}")
print(f"  experience: {ols_q4.params['experience']:.6f}")

for T in [5, 10, 15, 20]:
    cr = (ols_q4.params['tenure'] * T + ols_q4.params['tenure_2'] * T**2
          + ols_q4.params['tenure_3'] * T**3 + ols_q4.params['tenure_4'] * T**4)
    print(f"  CumRet at {T}yr: {cr:.4f}")

# ===== OLS with quadratic tenure =====
print("\n=== OLS Quadratic Tenure ===")
ols_vars_q2 = ['experience', 'tenure', 'tenure_2',
               'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X = sm.add_constant(df[ols_vars_q2])
ols_q2 = sm.OLS(df[wage_var], X).fit()
print(f"  tenure: {ols_q2.params['tenure']:.6f} ({ols_q2.bse['tenure']:.6f})")
print(f"  tenure_2: {ols_q2.params['tenure_2']:.6f}")

for T in [5, 10, 15, 20]:
    cr = ols_q2.params['tenure'] * T + ols_q2.params['tenure_2'] * T**2
    print(f"  CumRet at {T}yr: {cr:.4f}")

# ===== OLS with cubic tenure =====
print("\n=== OLS Cubic Tenure ===")
df['tenure_3'] = df['tenure'] ** 3
ols_vars_q3 = ['experience', 'tenure', 'tenure_2', 'tenure_3',
               'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X = sm.add_constant(df[ols_vars_q3])
ols_q3 = sm.OLS(df[wage_var], X).fit()
print(f"  tenure: {ols_q3.params['tenure']:.6f}")
print(f"  tenure_2: {ols_q3.params['tenure_2']:.6f}")
print(f"  tenure_3: {ols_q3.params['tenure_3']:.8f}")

for T in [5, 10, 15, 20]:
    cr = (ols_q3.params['tenure'] * T + ols_q3.params['tenure_2'] * T**2
          + ols_q3.params['tenure_3'] * T**3)
    print(f"  CumRet at {T}yr: {cr:.4f}")

# ===== Use the paper's OLS coefficient structure =====
# Paper OLS: cum_ret = a1*T + a2*T^2 + a3*T^3 + a4*T^4
# We know 4 data points: 5yr=0.2313, 10yr=0.3002, 15yr=0.3203, 20yr=0.3563
# Solve for a1, a2, a3, a4
print("\n=== Implied OLS coefficients from paper ===")
A = np.array([
    [5, 25, 125, 625],
    [10, 100, 1000, 10000],
    [15, 225, 3375, 50625],
    [20, 400, 8000, 160000],
])
b_vals = np.array([0.2313, 0.3002, 0.3203, 0.3563])
coeffs = np.linalg.solve(A, b_vals)
print(f"  a1={coeffs[0]:.6f}, a2={coeffs[1]:.6f}, a3={coeffs[2]:.8f}, a4={coeffs[3]:.10f}")
for T in [5, 10, 15, 20]:
    cr = coeffs[0]*T + coeffs[1]*T**2 + coeffs[2]*T**3 + coeffs[3]*T**4
    print(f"  Verify {T}yr: {cr:.4f}")

# ===== What if we use experience quartic in OLS? =====
# The paper says "the quartic in total experience and job tenure"
# Let's check with just experience quadratic vs quartic
print("\n=== OLS with experience quadratic only ===")
ols_vars_xq2 = ['experience', 'tenure', 'tenure_2', 'tenure_3', 'tenure_4',
                'exp_2'] + ctrl + yr_dums
X = sm.add_constant(df[ols_vars_xq2])
ols_xq2 = sm.OLS(df[wage_var], X).fit()
for T in [5, 10, 15, 20]:
    cr = (ols_xq2.params['tenure'] * T + ols_xq2.params['tenure_2'] * T**2
          + ols_xq2.params['tenure_3'] * T**3 + ols_xq2.params['tenure_4'] * T**4)
    print(f"  CumRet at {T}yr: {cr:.4f}")

# ===== Without year dummies in OLS =====
print("\n=== OLS without year dummies ===")
ols_vars_noyr = ['experience', 'tenure', 'tenure_2', 'tenure_3', 'tenure_4',
                 'exp_2', 'exp_3', 'exp_4'] + ctrl
X = sm.add_constant(df[ols_vars_noyr])
ols_noyr = sm.OLS(df[wage_var], X).fit()
for T in [5, 10, 15, 20]:
    cr = (ols_noyr.params['tenure'] * T + ols_noyr.params['tenure_2'] * T**2
          + ols_noyr.params['tenure_3'] * T**3 + ols_noyr.params['tenure_4'] * T**4)
    print(f"  CumRet at {T}yr: {cr:.4f}")

# ===== OLS with nominal wages =====
print("\n=== OLS with log_hourly_wage (nominal) ===")
ols_vars_nom = ['experience', 'tenure', 'tenure_2', 'tenure_3', 'tenure_4',
                'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X = sm.add_constant(df[ols_vars_nom])
ols_nom = sm.OLS(df['log_hourly_wage'], X).fit()
for T in [5, 10, 15, 20]:
    cr = (ols_nom.params['tenure'] * T + ols_nom.params['tenure_2'] * T**2
          + ols_nom.params['tenure_3'] * T**3 + ols_nom.params['tenure_4'] * T**4)
    print(f"  CumRet at {T}yr: {cr:.4f}")

# Max tenure in data
print(f"\nMax tenure in data: {df['tenure'].max():.1f}")
print(f"Tenure > 10: {(df['tenure'] > 10).sum()} obs")
print(f"Tenure > 15: {(df['tenure'] > 15).sum()} obs")
print(f"Tenure > 20: {(df['tenure'] > 20).sum()} obs")
print(f"Tenure distribution (>=10):")
for t in range(10, 25):
    n = (df['tenure'] == t).sum()
    if n > 0:
        print(f"  T={t}: {n} obs")
