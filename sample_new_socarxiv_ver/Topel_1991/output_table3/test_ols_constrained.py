"""OLS with constrained quartic: fix higher-order terms from step 1"""
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

targets = {5: 0.2313, 10: 0.3002, 15: 0.3203, 20: 0.3563}

# Step 1 higher-order terms
g2 = -0.004592
g3 = 0.0001846
g4 = -0.00000245

# Approach: Subtract the step 1 higher-order terms from wages,
# then run OLS with just linear tenure
# y_adj = log_wage - g2*T^2 - g3*T^3 - g4*T^4
# OLS: y_adj = ... + a1*T + ...
# Then cumret(T) = a1*T + g2*T^2 + g3*T^3 + g4*T^4

df['wage_adj_t'] = df[wage_var] - g2 * df['tenure_2'] - g3 * df['tenure_3'] - g4 * df['tenure_4']

# OLS with just linear tenure on the adjusted wage
ols_vars = ['experience', 'tenure', 'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X = sm.add_constant(df[ols_vars])
ols = sm.OLS(df['wage_adj_t'], X).fit()
a1 = ols.params['tenure']
a1_se = ols.bse['tenure']
print(f"Constrained OLS: linear tenure = {a1:.6f} (SE {a1_se:.6f})")
print(f"OLS bias = {a1 - 0.0543:.4f}")

for T in [5, 10, 15, 20]:
    cr = a1 * T + g2 * T**2 + g3 * T**3 + g4 * T**4
    err = abs(cr - targets[T])
    status = "PASS" if err <= 0.03 else "FAIL"
    print(f"  {T}yr: {cr:.4f} (paper: {targets[T]:.4f}, err: {err:.4f}) {status}")

# Alternative: use both quadratic with step 1 cubic/quartic
# y_adj2 = y - g3*T^3 - g4*T^4
# OLS: y_adj2 = ... + a1*T + a2*T^2 + ...
df['wage_adj_t2'] = df[wage_var] - g3 * df['tenure_3'] - g4 * df['tenure_4']
ols_vars2 = ['experience', 'tenure', 'tenure_2', 'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X2 = sm.add_constant(df[ols_vars2])
ols2 = sm.OLS(df['wage_adj_t2'], X2).fit()
a1_2 = ols2.params['tenure']
a2_2 = ols2.params['tenure_2']
print(f"\nPartially constrained (linear+quad free, cubic+quartic from step 1):")
print(f"  tenure: {a1_2:.6f}, tenure_2: {a2_2:.6f}")
for T in [5, 10, 15, 20]:
    cr = a1_2 * T + a2_2 * T**2 + g3 * T**3 + g4 * T**4
    err = abs(cr - targets[T])
    status = "PASS" if err <= 0.03 else "FAIL"
    print(f"  {T}yr: {cr:.4f} (paper: {targets[T]:.4f}, err: {err:.4f}) {status}")

# What about: subtract ALL step 1 polynomial terms and use just linear OLS?
df['wage_adj_all'] = df[wage_var] - g2 * df['tenure_2'] - g3 * df['tenure_3'] - g4 * df['tenure_4']
ols_vars3 = ['experience', 'tenure', 'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X3 = sm.add_constant(df[ols_vars3])
ols3 = sm.OLS(df['wage_adj_all'], X3).fit()
print(f"\nFully constrained (only linear free, all higher from step 1):")
print(f"  tenure: {ols3.params['tenure']:.6f}")
for T in [5, 10, 15, 20]:
    cr = ols3.params['tenure'] * T + g2 * T**2 + g3 * T**3 + g4 * T**4
    err = abs(cr - targets[T])
    status = "PASS" if err <= 0.03 else "FAIL"
    print(f"  {T}yr: {cr:.4f} (paper: {targets[T]:.4f}, err: {err:.4f}) {status}")

# Try: the "bias" approach
# The paper says column (4) is the "OLS bias in wage growth"
# bias = b1+b2_OLS - b1+b2_IV. But wait, b1+b2 is from STEP 1 (within job),
# not from cross-section OLS. Column (4) says ".0020 (.0004)"
# This is very small. It means the OLS cross-section tenure coefficient
# is only 0.002 higher than the IV estimate.
# So: OLS_ten_linear ≈ beta_2 + 0.0020 = 0.0565
# But our OLS gives 0.066 (quadratic) or 0.101 (quartic).
# The paper's OLS must use a quartic specification where the linear coefficient
# is much lower (0.0565).

# Actually re-reading: column (4) is "Wage Growth Bias b1+b2"
# This seems to be the bias in the WITHIN-JOB wage growth estimate, not
# the cross-sectional OLS. Let me re-read...
# The paper says: "Column (4) reports the estimated bias parameter that arises
# in the least-squares (within-job) estimate of tenure effects on within-job
# wage growth."
# So column (4) is about the WITHIN-JOB OLS, not the cross-section OLS!
# The bias is in the step 1 estimate (b1+b2), not in the cross-section.

# In that case, the OLS cumulative returns in the bottom panel may use a
# DIFFERENT OLS specification - a standard cross-sectional wage regression.
# Let me check what OLS specification gives the paper's values.

# From the earlier test: the paper's implied quartic coefficients are:
# a1=0.072518, a2=-0.006416, a3=0.00024927, a4=-0.0000032600
# Our quartic gives:
# a1=0.101278, a2=-0.009434, a3=0.0001482, a4=0.00001182

# The paper has a4 NEGATIVE (concave at high T), ours is POSITIVE (convex)
# This is the key difference causing extrapolation to blow up.

# What if we constrain a4 = g4 (from step 1)?
df['wage_adj_g4'] = df[wage_var] - g4 * df['tenure_4']
ols_vars4 = ['experience', 'tenure', 'tenure_2', 'tenure_3',
             'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X4 = sm.add_constant(df[ols_vars4])
ols4 = sm.OLS(df['wage_adj_g4'], X4).fit()
print(f"\nConstrained a4=g4={g4:.10f}:")
print(f"  tenure: {ols4.params['tenure']:.6f}")
print(f"  tenure_2: {ols4.params['tenure_2']:.6f}")
print(f"  tenure_3: {ols4.params['tenure_3']:.8f}")
for T in [5, 10, 15, 20]:
    cr = (ols4.params['tenure'] * T + ols4.params['tenure_2'] * T**2
          + ols4.params['tenure_3'] * T**3 + g4 * T**4)
    err = abs(cr - targets[T])
    status = "PASS" if err <= 0.03 else "FAIL"
    print(f"  {T}yr: {cr:.4f} (paper: {targets[T]:.4f}, err: {err:.4f}) {status}")

# What about constraining a3=g3 and a4=g4?
df['wage_adj_g34'] = df[wage_var] - g3 * df['tenure_3'] - g4 * df['tenure_4']
ols_vars5 = ['experience', 'tenure', 'tenure_2',
             'exp_2', 'exp_3', 'exp_4'] + ctrl + yr_dums
X5 = sm.add_constant(df[ols_vars5])
ols5 = sm.OLS(df['wage_adj_g34'], X5).fit()
print(f"\nConstrained a3=g3, a4=g4:")
print(f"  tenure: {ols5.params['tenure']:.6f}")
print(f"  tenure_2: {ols5.params['tenure_2']:.6f}")
for T in [5, 10, 15, 20]:
    cr = (ols5.params['tenure'] * T + ols5.params['tenure_2'] * T**2
          + g3 * T**3 + g4 * T**4)
    err = abs(cr - targets[T])
    status = "PASS" if err <= 0.03 else "FAIL"
    print(f"  {T}yr: {cr:.4f} (paper: {targets[T]:.4f}, err: {err:.4f}) {status}")
