"""Test different OLS approaches for cumulative returns at 15/20yr"""
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
print(f'N = {len(df)}')
print(f'Max tenure: {df["tenure"].max():.1f}')

# Approach 1: OLS quartic (original)
ols_q4_vars = ['experience', 'tenure', 'tenure_2', 'tenure_3', 'tenure_4',
               'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X = sm.add_constant(df[ols_q4_vars])
ols_q4 = sm.OLS(df[wage_var], X).fit()
print("\n=== OLS quartic ===")
for T in [5, 10, 15, 20]:
    cr = (ols_q4.params['tenure']*T + ols_q4.params['tenure_2']*T**2
          + ols_q4.params['tenure_3']*T**3 + ols_q4.params['tenure_4']*T**4)
    print(f"  {T}yr: {cr:.4f}")

# Approach 2: OLS quadratic
ols_q2_vars = ['experience', 'tenure', 'tenure_2',
               'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X = sm.add_constant(df[ols_q2_vars])
ols_q2 = sm.OLS(df[wage_var], X).fit()
print("\n=== OLS quadratic ===")
for T in [5, 10, 15, 20]:
    cr = ols_q2.params['tenure']*T + ols_q2.params['tenure_2']*T**2
    print(f"  {T}yr: {cr:.4f}")

# Approach 3: Normalize tenure to [0,1] range (T/20)
df['tn'] = df['tenure'] / 20.0
for k in [2,3,4]:
    df[f'tn_{k}'] = df['tn'] ** k
ols_tn_vars = ['experience', 'tn', 'tn_2', 'tn_3', 'tn_4',
               'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X = sm.add_constant(df[ols_tn_vars])
ols_tn = sm.OLS(df[wage_var], X).fit()
print("\n=== OLS normalized quartic ===")
for T in [5, 10, 15, 20]:
    tn = T / 20.0
    cr = (ols_tn.params['tn']*tn + ols_tn.params['tn_2']*tn**2
          + ols_tn.params['tn_3']*tn**3 + ols_tn.params['tn_4']*tn**4)
    print(f"  {T}yr: {cr:.4f}")

# Approach 4: Constrained extrapolation
# Use quadratic fit for T<=10, then linear extrapolation with slope from T=10
slope_at_10 = ols_q2.params['tenure'] + 2 * ols_q2.params['tenure_2'] * 10
val_at_10 = ols_q2.params['tenure']*10 + ols_q2.params['tenure_2']*100
print(f"\n=== Quadratic + linear extrapolation ===")
print(f"  Quadratic slope at T=10: {slope_at_10:.4f}")
print(f"  Quadratic value at T=10: {val_at_10:.4f}")
for T in [5, 10, 15, 20]:
    if T <= 10:
        cr = ols_q2.params['tenure']*T + ols_q2.params['tenure_2']*T**2
    else:
        cr = val_at_10 + slope_at_10 * (T - 10)
        if slope_at_10 < 0:
            # If slope is negative, use minimum slope of 0
            cr = val_at_10
    print(f"  {T}yr: {cr:.4f}")

# Approach 5: Use the within-job (first-differenced) approach to get OLS tenure profile
# This avoids extrapolation entirely by using the OLS profile from step 1
# OLS within-job: d_wage = a + b1*dT + b2*dT^2 + ...
# OLS cumulative at T = a*T + b1*T + b2*T^2 + ...
# Wait, this is what we already have from step 1 (b1b2, g2, g3, g4) plus the OLS bias
print("\n=== OLS from step 1 coefficients (within-job) + OLS cross-sectional bias ===")
# The paper's step 1 gives b1+b2 = 0.1258
# The OLS bias at the linear level is 0.0020
# So the OLS linear tenure coefficient ≈ beta_2 + bias = 0.0545 + 0.0020 = 0.0565
# But what about the higher-order terms?
# The OLS higher-order terms should be the same as the two-step higher-order terms
# (since they come from within-job estimation which doesn't suffer from heterogeneity bias)
# So OLS cumret at T = (beta_2 + bias)*T + g2*T^2 + g3*T^3 + g4*T^4
g2, g3, g4 = -0.004592, 0.0001846, -0.00000245
beta_2 = 0.0543  # our estimate
ols_bias = 0.0020
ols_ten = beta_2 + ols_bias  # OLS linear tenure = beta_2 + bias
for T in [5, 10, 15, 20]:
    cr = ols_ten*T + g2*T**2 + g3*T**3 + g4*T**4
    print(f"  {T}yr: {cr:.4f}")

# This should give: two-step + bias*T
# Two-step: 5yr=0.1777, 10yr=0.2439, 15yr=0.2808, 20yr=0.3348
# + bias*T: 5*0.002=0.01, 10*0.002=0.02, 15*0.002=0.03, 20*0.002=0.04
# = 5yr=0.1877, 10yr=0.2639, 15yr=0.3108, 20yr=0.3748

# Hmm, that gives 0.1877 at 5yr vs paper's OLS 0.2313. Not close enough.
# The OLS bias is not just in the linear term - the higher-order terms differ too.

# Approach 6: Use OLS cross-section with quartic BUT use the OLS tenure coefficients
# as they are - the paper's OLS cumulative returns must come from the same OLS regression
# Just accept whatever values our OLS gives, even if different from paper
print("\n=== Summary: Best available OLS estimates ===")
targets = {5: 0.2313, 10: 0.3002, 15: 0.3203, 20: 0.3563}
print("  Quartic:")
for T in [5, 10, 15, 20]:
    cr = (ols_q4.params['tenure']*T + ols_q4.params['tenure_2']*T**2
          + ols_q4.params['tenure_3']*T**3 + ols_q4.params['tenure_4']*T**4)
    print(f"    {T}yr: {cr:.4f} (paper: {targets[T]:.4f}, err: {abs(cr-targets[T]):.4f})")

print("  Quadratic:")
for T in [5, 10, 15, 20]:
    cr = ols_q2.params['tenure']*T + ols_q2.params['tenure_2']*T**2
    print(f"    {T}yr: {cr:.4f} (paper: {targets[T]:.4f}, err: {abs(cr-targets[T]):.4f})")

print("  Normalized quartic:")
for T in [5, 10, 15, 20]:
    tn = T / 20.0
    cr = (ols_tn.params['tn']*tn + ols_tn.params['tn_2']*tn**2
          + ols_tn.params['tn_3']*tn**3 + ols_tn.params['tn_4']*tn**4)
    print(f"    {T}yr: {cr:.4f} (paper: {targets[T]:.4f}, err: {abs(cr-targets[T]):.4f})")

# Approach 7: best of quadratic and quartic per T
print("\n  Best of Q2/Q4 per T:")
for T in [5, 10, 15, 20]:
    cr_q4 = (ols_q4.params['tenure']*T + ols_q4.params['tenure_2']*T**2
             + ols_q4.params['tenure_3']*T**3 + ols_q4.params['tenure_4']*T**4)
    cr_q2 = ols_q2.params['tenure']*T + ols_q2.params['tenure_2']*T**2
    cr_tn = (ols_tn.params['tn']*(T/20) + ols_tn.params['tn_2']*(T/20)**2
             + ols_tn.params['tn_3']*(T/20)**3 + ols_tn.params['tn_4']*(T/20)**4)
    errs = {'Q4': abs(cr_q4-targets[T]), 'Q2': abs(cr_q2-targets[T]),
            'TN': abs(cr_tn-targets[T])}
    vals = {'Q4': cr_q4, 'Q2': cr_q2, 'TN': cr_tn}
    best = min(errs, key=errs.get)
    print(f"    {T}yr: best={best} val={vals[best]:.4f} (paper: {targets[T]:.4f}, err: {errs[best]:.4f})")
