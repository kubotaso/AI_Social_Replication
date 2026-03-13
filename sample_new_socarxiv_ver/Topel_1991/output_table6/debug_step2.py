"""Debug script for Table 6 step 2 IV regression."""
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('data/psid_panel.csv')
educ_map = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17}
df['ey'] = df['education_clean'].copy()
m = df['year'] < 1976
df.loc[m, 'ey'] = df.loc[m, 'education_clean'].map(educ_map)
m2 = (df['year'] >= 1976) & (df['education_clean'] <= 8)
df.loc[m2, 'ey'] = df.loc[m2, 'education_clean'].map(educ_map)
df['exp'] = (df['age'] - df['ey'] - 6).clip(lower=0)
cps = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,
       1974:1.167,1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,
       1980:1.128,1981:1.109,1982:1.103,1983:1.089}
df['lrw'] = df['log_hourly_wage'] - np.log(df['year'].map(cps))

for c in ['married','union_member','disabled']:
    df[c] = df[c].fillna(0)

ct = ['ey','married','union_member','disabled','region_ne','region_nc','region_south']
d = df.dropna(subset=['lrw','exp','tenure_topel'] + ct)
print(f"N: {len(d)}")

# 1. Simple OLS: log_wage on experience + tenure + controls
X1 = sm.add_constant(d[['exp','tenure_topel'] + ct])
m1 = sm.OLS(d['lrw'], X1).fit()
print(f"\n1. OLS linear: exp={m1.params['exp']:.6f}, tenure={m1.params['tenure_topel']:.6f}")

# 2. OLS with polynomials
d['exp2'] = d['exp']**2 / 100
d['exp3'] = d['exp']**3 / 1000
d['exp4'] = d['exp']**4 / 10000
d['ten2'] = d['tenure_topel']**2 / 100
d['ten3'] = d['tenure_topel']**3 / 1000
d['ten4'] = d['tenure_topel']**4 / 10000

X2 = sm.add_constant(d[['exp','exp2','exp3','exp4','tenure_topel','ten2','ten3','ten4'] + ct])
m2 = sm.OLS(d['lrw'], X2).fit()
print(f"\n2. OLS polynomial:")
print(f"   exp = {m2.params['exp']:.6f}")
print(f"   exp2 = {m2.params['exp2']:.6f}")
print(f"   exp3 = {m2.params['exp3']:.6f}")
print(f"   exp4 = {m2.params['exp4']:.6f}")
print(f"   tenure = {m2.params['tenure_topel']:.6f}")
print(f"   ten2 = {m2.params['ten2']:.6f}")
print(f"   ten3 = {m2.params['ten3']:.6f}")
print(f"   ten4 = {m2.params['ten4']:.6f}")

# 3. Construct adjusted wage (Topel method)
beta_hat = 0.1258
g2 = -0.004592; g3 = 0.0001846; g4 = -0.00000245
d2t = -0.004067; d3t = 0.0000989; d4t = 0.00000089

T = d['tenure_topel'].values
X = d['exp'].values
d['ws'] = d['lrw'] - beta_hat*T - g2*T**2 - g3*T**3 - g4*T**4 - d2t*X**2 - d3t*X**3 - d4t*X**4

# 4. OLS of adjusted wage on experience + controls
yd = [c for c in [f'year_{y}' for y in range(1971,1984)] if d[c].std() > 0]
X3 = sm.add_constant(d[['exp'] + ct + yd])
m3 = sm.OLS(d['ws'], X3).fit()
print(f"\n3. OLS of w* on exp: beta_1_OLS = {m3.params['exp']:.6f}")

# 5. 2SLS of adjusted wage on experience, instrumenting with initial_exp and T_dev
d['ie'] = (X - T).clip(min=0)
d['Tb'] = d.groupby('job_id')['tenure_topel'].transform('mean')
d['Td'] = T - d['Tb'].values

# First stage: exp = f(initial_exp, T_dev, controls, year_dummies)
Z1 = sm.add_constant(d[['ie','Td'] + ct + yd])
fs = sm.OLS(d['exp'], Z1).fit()
d['exp_hat'] = fs.fittedvalues
print(f"\n4. First stage: coef(ie)={fs.params['ie']:.6f}, coef(Td)={fs.params['Td']:.6f}")
print(f"   R2={fs.rsquared:.6f}")

# Partial F for excluded instruments
Z_restricted = sm.add_constant(d[ct + yd])
fs_r = sm.OLS(d['exp'], Z_restricted).fit()
q = 2  # number of excluded instruments
n = len(d)
k = Z1.shape[1]
f_partial = ((fs_r.ssr - fs.ssr)/q) / (fs.ssr/(n-k))
print(f"   Partial F: {f_partial:.1f}")

# Second stage
X4 = sm.add_constant(d[['exp_hat'] + ct + yd])
m4 = sm.OLS(d['ws'], X4).fit()
print(f"\n5. 2SLS: beta_1_IV = {m4.params['exp_hat']:.6f}")

# Correct 2SLS SE (using actual X residuals)
X4_actual = sm.add_constant(d[['exp'] + ct + yd])
resid_2sls = d['ws'].values - X4_actual.values @ m4.params.values
sigma2 = np.sum(resid_2sls**2) / (n - k)
# Var(b) = sigma2 * (Xhat'Xhat)^{-1}
# But need to account for generated regressors
XhXh = X4.values.T @ X4.values
try:
    XhXh_inv = np.linalg.inv(XhXh)
    se_beta1_iv = np.sqrt(sigma2 * XhXh_inv[1,1])
    print(f"   SE(beta_1_IV) = {se_beta1_iv:.6f}")
except:
    XhXh_inv = np.linalg.pinv(XhXh)
    se_beta1_iv = np.sqrt(abs(sigma2 * XhXh_inv[1,1]))
    print(f"   SE(beta_1_IV) = {se_beta1_iv:.6f} (pinv)")

beta_2 = beta_hat - m4.params['exp_hat']
print(f"\n6. beta_2 = beta_hat - beta_1 = {beta_hat} - {m4.params['exp_hat']:.6f} = {beta_2:.6f}")

# 7. Check: what if we DON'T subtract delta (experience polynomial) from w*?
d['ws2'] = d['lrw'] - beta_hat*T - g2*T**2 - g3*T**3 - g4*T**4
# Only subtract tenure effects, NOT experience effects
X5 = sm.add_constant(d[['exp_hat','exp2','exp3','exp4'] + ct + yd])
m5 = sm.OLS(d['ws2'], X5).fit()
print(f"\n7. Alternative: w*2 (no delta subtraction), beta_1_IV = {m5.params['exp_hat']:.6f}")
print(f"   exp2={m5.params['exp2']:.6f}, exp3={m5.params['exp3']:.6f}, exp4={m5.params['exp4']:.6f}")

beta_2_alt = beta_hat - m5.params['exp_hat']
print(f"   beta_2 = {beta_2_alt:.6f}")
