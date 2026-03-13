"""
Test OLS cumulative returns using HYBRID approach:
OLS linear tenure coefficient + step 1 higher-order terms.

The insight: OLS cross-section only biases the LINEAR tenure coefficient.
The higher-order terms come from within-job estimation (step 1) which
is consistently estimated.
"""
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

ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
yr_dums = yr_dums[1:]

wage_var = 'log_real_gnp'
all_vars = ['experience', 'tenure', 'tenure_2', 'tenure_3', 'tenure_4',
            'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums + [wage_var]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=all_vars).copy()

# Step 1 higher-order terms (from Table 2 Model 3)
g2, g3, g4 = -0.004592, 0.0001846, -0.00000245

# Our beta_2
beta_2 = 0.0543

# OLS with quartic - get the linear tenure coefficient
ols_q4_vars = ['experience', 'tenure', 'tenure_2', 'tenure_3', 'tenure_4',
               'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X = sm.add_constant(df[ols_q4_vars])
ols_q4 = sm.OLS(df[wage_var], X).fit()
ols_linear_q4 = ols_q4.params['tenure']
ols_linear_se_q4 = ols_q4.bse['tenure']
print(f"OLS quartic: linear tenure coef = {ols_linear_q4:.6f} (SE {ols_linear_se_q4:.6f})")
print(f"OLS bias (quartic) = OLS_ten - beta_2 = {ols_linear_q4 - beta_2:.4f}")

# OLS with quadratic - get the linear tenure coefficient
ols_q2_vars = ['experience', 'tenure', 'tenure_2',
               'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X = sm.add_constant(df[ols_q2_vars])
ols_q2 = sm.OLS(df[wage_var], X).fit()
ols_linear_q2 = ols_q2.params['tenure']
ols_linear_se_q2 = ols_q2.bse['tenure']
print(f"OLS quadratic: linear tenure coef = {ols_linear_q2:.6f} (SE {ols_linear_se_q2:.6f})")
print(f"OLS bias (quadratic) = {ols_linear_q2 - beta_2:.4f}")

# OLS with just linear tenure
ols_lin_vars = ['experience', 'tenure',
                'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X = sm.add_constant(df[ols_lin_vars])
ols_lin = sm.OLS(df[wage_var], X).fit()
ols_linear_lin = ols_lin.params['tenure']
ols_linear_se_lin = ols_lin.bse['tenure']
print(f"OLS linear only: tenure coef = {ols_linear_lin:.6f} (SE {ols_linear_se_lin:.6f})")
print(f"OLS bias (linear) = {ols_linear_lin - beta_2:.4f}")

# Paper targets
targets = {5: 0.2313, 10: 0.3002, 15: 0.3203, 20: 0.3563}

# HYBRID approach: OLS linear tenure + step 1 higher-order terms
print(f"\n=== HYBRID: OLS linear tenure + step1 higher-order ===")

# Try different OLS linear values
for label, ols_ten_val, ols_ten_se in [
    ("OLS quartic linear", ols_linear_q4, ols_linear_se_q4),
    ("OLS quadratic linear", ols_linear_q2, ols_linear_se_q2),
    ("OLS linear only", ols_linear_lin, ols_linear_se_lin),
]:
    print(f"\n  {label}: OLS_ten = {ols_ten_val:.6f}")
    print(f"  bias = {ols_ten_val - beta_2:.4f} (paper bias: 0.0020)")
    for T in [5, 10, 15, 20]:
        cr = ols_ten_val * T + g2*T**2 + g3*T**3 + g4*T**4
        se = T * ols_ten_se
        err = abs(cr - targets[T])
        status = "PASS" if err <= 0.03 else "FAIL"
        print(f"    {T}yr: {cr:.4f} (paper: {targets[T]:.4f}, err: {err:.4f}) {status}")

# What OLS linear tenure coefficient would give the best fit to paper OLS cumrets?
print(f"\n=== Optimal OLS linear tenure coefficient ===")
# Minimize sum of squared errors in cumulative returns
from scipy.optimize import minimize_scalar
def obj(ols_ten):
    return sum((ols_ten*T + g2*T**2 + g3*T**3 + g4*T**4 - targets[T])**2
               for T in [5, 10, 15, 20])
res = minimize_scalar(obj, bounds=(0.04, 0.12), method='bounded')
ols_ten_opt = res.x
print(f"  Optimal OLS linear tenure = {ols_ten_opt:.6f}")
print(f"  Implied OLS bias = {ols_ten_opt - beta_2:.4f}")
for T in [5, 10, 15, 20]:
    cr = ols_ten_opt * T + g2*T**2 + g3*T**3 + g4*T**4
    print(f"    {T}yr: {cr:.4f} (paper: {targets[T]:.4f}, err: {abs(cr-targets[T]):.4f})")

# What about using the paper's OLS bias directly?
print(f"\n=== Using paper's OLS bias = 0.0020 ===")
ols_ten_from_bias = beta_2 + 0.0020  # = 0.0563
print(f"  OLS linear tenure = {ols_ten_from_bias:.6f}")
for T in [5, 10, 15, 20]:
    cr = ols_ten_from_bias * T + g2*T**2 + g3*T**3 + g4*T**4
    print(f"    {T}yr: {cr:.4f} (paper: {targets[T]:.4f}, err: {abs(cr-targets[T]):.4f})")
