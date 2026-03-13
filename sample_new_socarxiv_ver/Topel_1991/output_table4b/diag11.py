"""
Critical test: Shift tenure down by 1 (paper uses 0-based tenure).
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

CPS = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,
       1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,
       1982:1.103,1983:1.089}
df['cps'] = df['year'].map(CPS)
df['lrw'] = df['log_hourly_wage'] - np.log(df['cps'])

for c in ['married','union_member','disabled','region_ne','region_nc','region_south']:
    df[c] = df[c].fillna(0)

# SHIFT TENURE DOWN BY 1 (paper uses 0-based)
df['tenure'] = df['tenure_topel'] - 1  # Now starts at 0
df['ie'] = (df['exp'] - df['tenure']).clip(lower=0)

df = df.sort_values(['person_id','job_id','year']).reset_index(drop=True)

# First differences
df['prev_yr'] = df.groupby(['person_id','job_id'])['year'].shift(1)
df['prev_lrw'] = df.groupby(['person_id','job_id'])['lrw'].shift(1)
df['prev_ten'] = df.groupby(['person_id','job_id'])['tenure'].shift(1)
df['prev_exp'] = df.groupby(['person_id','job_id'])['exp'].shift(1)

wj = df[(df['prev_yr'].notna()) & (df['year'] - df['prev_yr'] == 1)].copy()
wj['dlw'] = wj['lrw'] - wj['prev_lrw']
wj = wj[wj['dlw'].between(-2, 2)].copy()

t = wj['tenure']; pt = wj['prev_ten']
x = wj['exp']; px = wj['prev_exp']
wj['d_ten'] = t - pt
wj['d_ten_sq'] = t**2 - pt**2
wj['d_ten_cu'] = t**3 - pt**3
wj['d_ten_qu'] = t**4 - pt**4
wj['d_exp_sq'] = x**2 - px**2
wj['d_exp_cu'] = x**3 - px**3
wj['d_exp_qu'] = x**4 - px**4

# Step 1
y1 = wj['dlw']
X1_vars = ['d_ten', 'd_ten_sq', 'd_ten_cu', 'd_ten_qu', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
X1 = wj[X1_vars].copy()
yr1 = pd.get_dummies(wj['year'], prefix='yr', dtype=float)
yr_cols1 = sorted(yr1.columns.tolist())[1:]
for c in yr_cols1:
    X1[c] = yr1[c].values
X1 = sm.add_constant(X1)
valid1 = X1.notna().all(axis=1) & y1.notna()
m1 = sm.OLS(y1[valid1], X1[valid1]).fit()

bhat = m1.params['d_ten']
bhat_se = m1.bse['d_ten']
g2 = m1.params['d_ten_sq']
g3 = m1.params['d_ten_cu']
g4 = m1.params['d_ten_qu']
d2 = m1.params['d_exp_sq']
d3 = m1.params['d_exp_cu']
d4 = m1.params['d_exp_qu']

print(f"Step 1 N: {int(m1.nobs)}")
print(f"beta_hat (b1+b2): {bhat:.4f} ({bhat_se:.4f})")
print(f"Paper b1+b2: 0.1258 (0.0162)")
print()
print(f"gamma2 (raw): {g2:.6f}")
print(f"gamma2 scaled (x100): {g2*100:.4f}")
print(f"Paper gamma2 scaled: -0.4592")
print()
print(f"gamma3 (raw): {g3:.8f}")
print(f"gamma3 scaled (x1000): {g3*1000:.4f}")
print(f"Paper gamma3 scaled: 0.1846")
print()
print(f"gamma4 (raw): {g4:.10f}")
print(f"gamma4 scaled (x10000): {g4*10000:.4f}")
print(f"Paper gamma4 scaled: -0.0245")
print()

# Step 2 with shifted tenure
T = df['tenure'].values.astype(float)
X = df['exp'].values.astype(float)

w_star = df['lrw'].values - bhat*T - g2*T**2 - g3*T**3 - g4*T**4 - d2*X**2 - d3*X**3 - d4*X**4

controls = ['ey', 'married', 'union_member', 'disabled', 'region_ne', 'region_nc', 'region_south']
yr2 = pd.get_dummies(df['year'], prefix='yr', dtype=float)
yr_cols2 = sorted(yr2.columns.tolist())[1:]

df2 = df.reset_index(drop=True)
X_reg = df2[['ie'] + controls].copy()
for c in yr_cols2:
    X_reg[c] = yr2[c].values
X_reg = sm.add_constant(X_reg)

ws = pd.Series(w_star).reset_index(drop=True)
valid = X_reg.notna().all(axis=1) & ws.notna()
m2 = sm.OLS(ws[valid], X_reg[valid]).fit()

b1 = m2.params['ie']
b1_se = m2.bse['ie']
b1_se_corr = np.sqrt(b1_se**2 + bhat_se**2)
b2 = bhat - b1
print(f"beta_1 = {b1:.4f} ({b1_se_corr:.4f})")
print(f"beta_2 = {b2:.4f}")
print(f"Paper: beta_1 = 0.0713, beta_2 = 0.0545")
print()

# Cumulative returns
for Ty in [5, 10, 15, 20]:
    cum = b2*Ty + g2*Ty**2 + g3*Ty**3 + g4*Ty**4
    print(f"  {Ty}yr: {cum:.4f}")

print()
print("Paper cumulative returns:")
for Ty, pv in [(5, 0.1793), (10, 0.2459), (15, 0.2832), (20, 0.3375)]:
    print(f"  {Ty}yr: {pv}")
