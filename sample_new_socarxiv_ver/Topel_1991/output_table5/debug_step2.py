"""Debug script to test different step 2 approaches."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/psid_panel.csv')

# Prep
em = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17}
df['ed'] = df['education_clean'].map(em).fillna(12)
df['exp'] = (df['age'] - df['ed'] - 6).clip(lower=0)
df['ten'] = df['tenure_topel']

gnp = {1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,
       1973:44.4,1974:48.9,1975:53.6,1976:56.9,1977:60.6,1978:65.2,
       1979:72.6,1980:82.4,1981:90.9,1982:100.0}
cps = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,
       1974:1.167,1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,
       1980:1.128,1981:1.109,1982:1.103,1983:1.089}

df['lrw'] = df['log_hourly_wage'] - np.log((df['year']-1).map(gnp)/100.0) - np.log(df['year'].map(cps))

# Occupation mapping
occ = df['occ_1digit'].copy()
m3 = occ > 9
t = occ[m3]
mp = pd.Series(0, index=t.index, dtype=int)
mp[(t>=1)&(t<=195)]=1; mp[(t>=201)&(t<=245)]=2; mp[(t>=260)&(t<=395)]=4
mp[(t>=401)&(t<=580)]=5; mp[(t>=601)&(t<=695)]=6; mp[(t>=701)&(t<=785)]=7
mp[(t>=801)&(t<=824)]=9; mp[(t>=900)&(t<=965)]=8
occ[m3] = mp
df['occ'] = occ

# Filter
df = df[df['ten'] >= 1].dropna(subset=['log_hourly_wage', 'lrw']).copy()
df = df[~df['occ'].isin([0, 3, 9])].copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

df['x0'] = (df['exp'] - df['ten']).clip(lower=0)
for c in ['married','disabled','lives_in_smsa','union_member',
          'region_ne','region_nc','region_south']:
    df[c] = df[c].fillna(0)

# Paper's step 1 values (from Table 3 full sample)
b12 = 0.1258
g2 = -0.004592; g3 = 0.0001846; g4 = -0.00000245
d2 = -0.006051; d3 = 0.0002067; d4 = -0.00000238

# w* construction
df['ws'] = (df['lrw']
            - b12 * df['ten']
            - g2 * df['ten']**2 - g3 * df['ten']**3 - g4 * df['ten']**4
            - d2 * df['exp']**2 - d3 * df['exp']**3 - d4 * df['exp']**4)

# PS subsample
ps = df[df['occ'].isin([1, 2, 4, 8])].copy()
yr = [f'year_{y}' for y in range(1969, 1984)]
ct = ['ed', 'married', 'disabled', 'lives_in_smsa', 'union_member',
      'region_ne', 'region_nc', 'region_south']

print(f"PS N = {len(ps)}")

# Approach 1: OLS on X_0
X1 = sm.add_constant(ps[['x0'] + ct + yr].astype(float))
y = ps['ws'].astype(float)
m1 = sm.OLS(y, X1).fit()
print(f"OLS on X_0: beta_1 = {m1.params['x0']:.4f} (SE = {m1.bse['x0']:.4f})")

# Approach 2: OLS on X (current experience)
X2 = sm.add_constant(ps[['exp'] + ct + yr].astype(float))
m2 = sm.OLS(y, X2).fit()
print(f"OLS on X:   beta_1 = {m2.params['exp']:.4f} (SE = {m2.bse['exp']:.4f})")

# 2SLS skipped (singular matrix with all controls + year dummies)
print("2SLS: skipped (singular matrix)")

# Check diagnostics
endog = ps['exp'].values.astype(np.float64)
instrument = ps['x0'].values.astype(np.float64)
print(f"\nDiagnostics:")
print(f"  Mean X = {endog.mean():.2f}, Mean X_0 = {instrument.mean():.2f}")
print(f"  Mean T = {ps['ten'].mean():.2f}")
print(f"  Correlation(X, X_0) = {np.corrcoef(endog, instrument)[0,1]:.4f}")

# Also try: OLS on X_0 WITHOUT year dummies
X3 = sm.add_constant(ps[['x0'] + ct].astype(float))
m3 = sm.OLS(y, X3).fit()
print(f"\nOLS on X_0 (no year dummies): beta_1 = {m3.params['x0']:.4f} (SE = {m3.bse['x0']:.4f})")

# Also try: OLS on X_0 with NO controls
X4 = sm.add_constant(ps[['x0']].astype(float))
m4 = sm.OLS(y, X4).fit()
print(f"OLS on X_0 (no controls): beta_1 = {m4.params['x0']:.4f} (SE = {m4.bse['x0']:.4f})")

# What if we DON'T subtract beta_hat*T? (the w* only has higher-order terms removed)
# Then w* ~ alpha + (beta_1+beta_2)*T + beta_1*(X-T) + epsilon
# = alpha + beta_1*X + beta_2*T + epsilon
# Need to include T in regression and use X_0 as instrument for X
df['ws2'] = (df['lrw']
             - g2 * df['ten']**2 - g3 * df['ten']**3 - g4 * df['ten']**4
             - d2 * df['exp']**2 - d3 * df['exp']**3 - d4 * df['exp']**4)

ps2 = df[df['occ'].isin([1, 2, 4, 8])].copy()
y2 = ps2['ws2'].astype(float)

# OLS on X_0 + tenure (this should give beta_1 and beta_2 separately)
X5 = sm.add_constant(ps2[['x0', 'ten'] + ct + yr].astype(float))
m5 = sm.OLS(y2, X5).fit()
print(f"\nw* (no linear T removed), OLS on X_0 + T:")
print(f"  beta_1 (X_0) = {m5.params['x0']:.4f} (SE = {m5.bse['x0']:.4f})")
print(f"  b1+b2  (T)   = {m5.params['ten']:.4f} (SE = {m5.bse['ten']:.4f})")
print(f"  beta_2 = b1+b2 - beta_1 = {m5.params['ten'] - m5.params['x0']:.4f}")

# Try without controls to see raw relationship
X6 = sm.add_constant(ps2[['x0', 'ten']].astype(float))
m6 = sm.OLS(y2, X6).fit()
print(f"\nw* (no controls):")
print(f"  beta_1 (X_0) = {m6.params['x0']:.4f}")
print(f"  b1+b2  (T)   = {m6.params['ten']:.4f}")

print(f"\nPaper targets: beta_1=0.0707, beta_2=0.0601, b1+b2=0.1309")
