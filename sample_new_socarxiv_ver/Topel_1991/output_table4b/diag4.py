"""
Diagnostic: Use X_0 (initial experience) directly as regressor (not IV).
Based on equation (10): y - x'Gamma_hat = X_0 * beta_1 + F*gamma + e
"""
import pandas as pd, numpy as np, statsmodels.api as sm

df = pd.read_csv('data/psid_panel.csv')
df = df[~df['region'].isin([5, 6])]

EDUC = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17,9:17}
df['ey'] = df['education_clean'].copy()
m = ~df['year'].isin([1975, 1976])
df.loc[m, 'ey'] = df.loc[m, 'education_clean'].map(EDUC)
df = df.dropna(subset=['ey'])
df['exp'] = (df['age'] - df['ey'] - 6).clip(lower=0)
df['ie'] = (df['exp'] - df['tenure_topel']).clip(lower=0)

CPS = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,
       1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,
       1982:1.103,1983:1.089}
GNP = {1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,1973:44.4,1974:48.9,
       1975:53.6,1976:56.9,1977:60.6,1978:65.2,1979:72.6,1980:82.4,1981:90.9,1982:100.0}

df['cps'] = df['year'].map(CPS)
df['gnp'] = (df['year']-1).map(GNP)
df['lrw'] = df['log_hourly_wage'] - np.log(df['gnp']/100.0) - np.log(df['cps'])

for c in ['married','union_member','disabled','region_ne','region_nc','region_south']:
    df[c] = df[c].fillna(0)

# Paper step 1 coefficients (Table 2, Model 3)
beta_hat = 0.1258  # b1+b2
gamma2 = -0.004592  # T^2
gamma3 = 0.0001846  # T^3
gamma4 = -0.00000245  # T^4
delta2 = -0.004067  # X^2
delta3 = 0.0000989  # X^3
delta4 = 0.00000089  # X^4

T = df['tenure_topel'].values.astype(float)
X = df['exp'].values.astype(float)

# Construct adjusted wage per eq (10):
# w_star = log_wage - beta_hat*T - gamma(T^2,T^3,T^4) - delta(X^2,X^3,X^4)
# Then regress w_star on X_0 + controls

w_star = (df['lrw'].values
          - beta_hat * T
          - gamma2 * T**2 - gamma3 * T**3 - gamma4 * T**4
          - delta2 * X**2 - delta3 * X**3 - delta4 * X**4)

df['w_star'] = w_star

# Step 2: OLS of w_star on X_0 (initial experience) + controls
controls = ['ey', 'married', 'union_member', 'disabled', 'region_ne', 'region_nc', 'region_south']
yr = pd.get_dummies(df['year'], prefix='yr', dtype=float)
yr_cols = sorted(yr.columns.tolist())[1:]

df2 = df.reset_index(drop=True)
X_reg = df2[['ie'] + controls].copy()
for c in yr_cols:
    X_reg[c] = yr[c].values
X_reg = sm.add_constant(X_reg)

valid = X_reg.notna().all(axis=1) & df2['w_star'].notna()
model = sm.OLS(df2.loc[valid, 'w_star'], X_reg[valid]).fit()

beta_1 = model.params['ie']
beta_1_se = model.bse['ie']
beta_2 = beta_hat - beta_1
print(f"OLS on X_0: beta_1 = {beta_1:.4f} ({beta_1_se:.4f})")
print(f"beta_2 = {beta_hat} - {beta_1:.4f} = {beta_2:.4f}")
print(f"Expected: beta_1 = 0.0713 (0.0181), beta_2 = 0.0545 (0.0079)")
print(f"N = {int(model.nobs)} (expected: 10,685)")
print()

# Also try with CPS-only deflation
df['lrw_cps'] = df['log_hourly_wage'] - np.log(df['cps'])
w_star_cps = (df['lrw_cps'].values
              - beta_hat * T
              - gamma2 * T**2 - gamma3 * T**3 - gamma4 * T**4
              - delta2 * X**2 - delta3 * X**3 - delta4 * X**4)
df['w_star_cps'] = w_star_cps
model_cps = sm.OLS(df2.loc[valid, 'w_star_cps'], X_reg[valid]).fit()
print(f"CPS-only: beta_1 = {model_cps.params['ie']:.4f} ({model_cps.bse['ie']:.4f})")

# Also try nominal wages
w_star_nom = (df['log_hourly_wage'].values
              - beta_hat * T
              - gamma2 * T**2 - gamma3 * T**3 - gamma4 * T**4
              - delta2 * X**2 - delta3 * X**3 - delta4 * X**4)
df['w_star_nom'] = w_star_nom
model_nom = sm.OLS(df2.loc[valid, 'w_star_nom'], X_reg[valid]).fit()
print(f"Nominal: beta_1 = {model_nom.params['ie']:.4f} ({model_nom.bse['ie']:.4f})")
