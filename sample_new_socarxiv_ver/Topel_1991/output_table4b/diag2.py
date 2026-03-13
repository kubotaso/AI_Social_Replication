"""
Diagnostic: Use paper's step-1 coefficients to see if step 2 gives right beta_1.
"""
import pandas as pd, numpy as np, statsmodels.api as sm
from linearmodels.iv import IV2SLS

df = pd.read_csv('data/psid_panel.csv')
df = df[~df['region'].isin([5, 6])]

# Education recoding
EDUC = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17,9:17}
df['ey'] = df['education_clean'].copy()
m = ~df['year'].isin([1975, 1976])
df.loc[m, 'ey'] = df.loc[m, 'education_clean'].map(EDUC)
df = df.dropna(subset=['ey'])

df['exp'] = (df['age'] - df['ey'] - 6).clip(lower=0)

CPS = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,
       1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,
       1982:1.103,1983:1.089}
GNP = {1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,1973:44.4,1974:48.9,
       1975:53.6,1976:56.9,1977:60.6,1978:65.2,1979:72.6,1980:82.4,1981:90.9,1982:100.0}

df['cps'] = df['year'].map(CPS)
df['gnp'] = (df['year']-1).map(GNP)
df['lrw'] = df['log_hourly_wage'] - np.log(df['gnp']/100.0) - np.log(df['cps'])

df['ie'] = (df['exp'] - df['tenure_topel']).clip(lower=0)

for c in ['married','union_member','disabled','lives_in_smsa','region_ne','region_nc','region_south','region_west']:
    df[c] = df[c].fillna(0)

# Use PAPER'S step 1 coefficients (Table 2, Model 3):
# beta_hat (b1+b2) = .1258
# gamma2 (d_tenure_sq) = -.4592/100 = -.004592
# gamma3 (d_tenure_cu) = .1846/1000 = .0001846
# gamma4 (d_tenure_qu) = -.0245/10000 = -.00000245
# delta2 (d_exp_sq) = -.4067/100 = -.004067
# delta3 (d_exp_cu) = .0989/1000 = .0000989
# delta4 (d_exp_qu) = .0089/10000 = .00000089

beta_hat = 0.1258
gamma2 = -0.004592
gamma3 = 0.0001846
gamma4 = -0.00000245
delta2 = -0.004067
delta3 = 0.0000989
delta4 = 0.00000089

T = df['tenure_topel'].values.astype(float)
X = df['exp'].values.astype(float)

df['w_star'] = (df['lrw'].values
                - beta_hat * T
                - gamma2 * T**2
                - gamma3 * T**3
                - gamma4 * T**4
                - delta2 * X**2
                - delta3 * X**3
                - delta4 * X**4)

print("w_star stats:", df['w_star'].describe().to_string())
print()

# Step 2 IV regression
controls = ['ey', 'married', 'union_member', 'disabled', 'region_ne', 'region_nc', 'region_south']
yr = pd.get_dummies(df['year'], prefix='yr', dtype=float)
yr_cols = sorted(yr.columns.tolist())[1:]

exog = df[controls].copy()
for c in yr_cols:
    exog[c] = yr[c].values
exog = sm.add_constant(exog)

df2 = df.reset_index(drop=True)
exog = exog.reset_index(drop=True)

valid = df2[['w_star','exp','ie']].notna().all(axis=1) & exog.notna().all(axis=1)

print(f"Step 2 N: {valid.sum()}")

iv = IV2SLS(
    dependent=df2.loc[valid, 'w_star'],
    exog=exog[valid],
    endog=df2.loc[valid, ['exp']],
    instruments=df2.loc[valid, ['ie']]
).fit()

print(f"\nbeta_1 (experience): {iv.params['exp']:.4f} ({iv.std_errors['exp']:.4f})")
print(f"Expected: 0.0713 (0.0181)")
print()
beta_1 = iv.params['exp']
beta_2 = beta_hat - beta_1
print(f"beta_2 = {beta_hat} - {beta_1:.4f} = {beta_2:.4f}")
print(f"Expected beta_2: 0.0545")
print()

# Also try OLS for comparison
ols_X = df2.loc[valid, ['exp']].copy()
for c in exog.columns:
    ols_X[c] = exog.loc[valid, c].values
ols_model = sm.OLS(df2.loc[valid, 'w_star'], ols_X).fit()
print(f"OLS beta_1: {ols_model.params['exp']:.4f} ({ols_model.bse['exp']:.4f})")
print()

# Try without year dummies to see if that matters
exog2 = df[controls].copy().reset_index(drop=True)
exog2 = sm.add_constant(exog2)

iv2 = IV2SLS(
    dependent=df2.loc[valid, 'w_star'],
    exog=exog2[valid],
    endog=df2.loc[valid, ['exp']],
    instruments=df2.loc[valid, ['ie']]
).fit()
print(f"IV without year dummies: beta_1={iv2.params['exp']:.4f} ({iv2.std_errors['exp']:.4f})")
