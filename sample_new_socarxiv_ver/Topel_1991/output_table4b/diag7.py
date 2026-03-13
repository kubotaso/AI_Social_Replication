import pandas as pd, numpy as np
df = pd.read_csv('data/psid_panel.csv')
df = df[~df['region'].isin([5, 6])]
EDUC = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17,9:17}
df['ey'] = df['education_clean'].copy()
m = ~df['year'].isin([1975, 1976])
df.loc[m, 'ey'] = df.loc[m, 'education_clean'].map(EDUC)
df = df.dropna(subset=['ey'])
df['exp'] = (df['age'] - df['ey'] - 6).clip(lower=0)
df['ie'] = (df['exp'] - df['tenure_topel']).clip(lower=0)

GNP = {1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,1973:44.4,1974:48.9,
       1975:53.6,1976:56.9,1977:60.6,1978:65.2,1979:72.6,1980:82.4,1981:90.9,1982:100.0}
CPS = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,
       1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,
       1982:1.103,1983:1.089}
df['cps'] = df['year'].map(CPS)
df['gnp'] = (df['year']-1).map(GNP)
df['lrw'] = df['log_hourly_wage'] - np.log(df['gnp']/100.0) - np.log(df['cps'])

print("Correlations with log_hourly_wage:")
print(f"  ie: {df[['ie','log_hourly_wage']].corr().iloc[0,1]:.4f}")
print(f"  exp: {df[['exp','log_hourly_wage']].corr().iloc[0,1]:.4f}")
print(f"  age: {df[['age','log_hourly_wage']].corr().iloc[0,1]:.4f}")
print(f"  tenure: {df[['tenure_topel','log_hourly_wage']].corr().iloc[0,1]:.4f}")
print(f"  ey: {df[['ey','log_hourly_wage']].corr().iloc[0,1]:.4f}")
print()

# After adjusting for higher-order polynomials, what's the raw w* look like?
beta_hat = 0.1258
gamma2 = -0.004592; gamma3 = 0.0001846; gamma4 = -0.00000245
delta2 = -0.004067; delta3 = 0.0000989; delta4 = 0.00000089

T = df['tenure_topel'].values.astype(float)
X = df['exp'].values.astype(float)
w_star = df['lrw'].values - beta_hat*T - gamma2*T**2 - gamma3*T**3 - gamma4*T**4 - delta2*X**2 - delta3*X**3 - delta4*X**4
df['w_star'] = w_star

print("Correlations with w_star (adjusted wage):")
print(f"  ie: {df[['ie','w_star']].corr().iloc[0,1]:.4f}")
print(f"  exp: {df[['exp','w_star']].corr().iloc[0,1]:.4f}")
print(f"  ey: {df[['ey','w_star']].corr().iloc[0,1]:.4f}")
print()

# Check if the issue is the high-order experience polynomial
# At high experience values, the polynomial terms dominate
print("For workers with exp > 30:")
hi = df[df['exp'] > 30]
print(f"  N: {len(hi)}")
print(f"  mean exp poly (delta2*X^2+...): {(delta2*hi['exp']**2 + delta3*hi['exp']**3 + delta4*hi['exp']**4).mean():.4f}")
lo = df[df['exp'] <= 10]
print(f"For workers with exp <= 10:")
print(f"  N: {len(lo)}")
print(f"  mean exp poly: {(delta2*lo['exp']**2 + delta3*lo['exp']**3 + delta4*lo['exp']**4).mean():.4f}")
print()

# Maybe the issue is the polynomial blows up for high experience
# Paper coefficients are: delta2=-0.004067, delta3=0.0000989, delta4=0.00000089
# At X=40: delta2*1600 + delta3*64000 + delta4*2560000 = -6.51 + 6.33 + 2.28 = 2.10
# At X=50: delta2*2500 + delta3*125000 + delta4*6250000 = -10.17 + 12.36 + 5.56 = 7.75
print("Polynomial values at different experience levels:")
for x in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    poly_val = delta2*x**2 + delta3*x**3 + delta4*x**4
    print(f"  X={x}: delta2*X^2+delta3*X^3+delta4*X^4 = {poly_val:.4f}")
