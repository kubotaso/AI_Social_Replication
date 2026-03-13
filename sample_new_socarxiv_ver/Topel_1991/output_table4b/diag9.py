"""Try different ie definitions and check which gives closer beta_1."""
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

df = df.sort_values(['person_id','job_id','year']).reset_index(drop=True)

# Step 1: our coefficients
df['prev_yr'] = df.groupby(['person_id','job_id'])['year'].shift(1)
df['prev_lrw'] = df.groupby(['person_id','job_id'])['lrw'].shift(1)
df['prev_ten'] = df.groupby(['person_id','job_id'])['tenure_topel'].shift(1)
df['prev_exp'] = df.groupby(['person_id','job_id'])['exp'].shift(1)

wj = df[(df['prev_yr'].notna()) & (df['year'] - df['prev_yr'] == 1)].copy()
wj['dlw'] = wj['lrw'] - wj['prev_lrw']
wj = wj[wj['dlw'].between(-2, 2)].copy()

t = wj['tenure_topel']; pt = wj['prev_ten']
x = wj['exp']; px = wj['prev_exp']
wj['d_ten'] = t - pt
wj['d_ten_sq'] = t**2 - pt**2
wj['d_ten_cu'] = t**3 - pt**3
wj['d_ten_qu'] = t**4 - pt**4
wj['d_exp_sq'] = x**2 - px**2
wj['d_exp_cu'] = x**3 - px**3
wj['d_exp_qu'] = x**4 - px**4

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
g2 = m1.params['d_ten_sq']
g3 = m1.params['d_ten_cu']
g4 = m1.params['d_ten_qu']
d2 = m1.params['d_exp_sq']
d3 = m1.params['d_exp_cu']
d4 = m1.params['d_exp_qu']

print(f"beta_hat = {bhat:.4f}")

# Try different ie definitions
T = df['tenure_topel'].values.astype(float)
X = df['exp'].values.astype(float)

w_star = df['lrw'].values - bhat*T - g2*T**2 - g3*T**3 - g4*T**4 - d2*X**2 - d3*X**3 - d4*X**4

controls = ['ey', 'married', 'union_member', 'disabled', 'region_ne', 'region_nc', 'region_south']
yr2 = pd.get_dummies(df['year'], prefix='yr', dtype=float)
yr_cols2 = sorted(yr2.columns.tolist())[1:]

for ie_name, ie_def in [
    ('ie = exp - tenure', df['exp'] - df['tenure_topel']),
    ('ie = exp - tenure + 1', df['exp'] - df['tenure_topel'] + 1),
    ('ie = exp - tenure (clipped)', (df['exp'] - df['tenure_topel']).clip(lower=0)),
    ('ie = age - edu - 6 - tenure', (df['age'] - df['ey'] - 6 - df['tenure_topel']).clip(lower=0)),
    ('ie = age at job start', (df['age'] - df['tenure_topel']).clip(lower=18)),
]:
    df2 = df.copy()
    df2['ie'] = ie_def
    df2 = df2.reset_index(drop=True)

    X_reg = df2[['ie'] + controls].copy()
    for c in yr_cols2:
        X_reg[c] = yr2[c].values
    X_reg = sm.add_constant(X_reg)

    ws = pd.Series(w_star).reset_index(drop=True)
    valid = X_reg.notna().all(axis=1) & ws.notna()
    m2 = sm.OLS(ws[valid], X_reg[valid]).fit()

    b1 = m2.params['ie']
    b1_se = m2.bse['ie']
    b2 = bhat - b1
    print(f"  {ie_name}: beta_1={b1:.4f} ({b1_se:.4f}), beta_2={b2:.4f}")

# What if we use age directly instead of experience?
print("\nUsing age as proxy for experience:")
df2 = df.copy().reset_index(drop=True)
df2['ie_age'] = df2['age'] - df2['tenure_topel']
X_reg2 = df2[['ie_age'] + controls].copy()
for c in yr_cols2:
    X_reg2[c] = yr2[c].values
X_reg2 = sm.add_constant(X_reg2)
ws = pd.Series(w_star).reset_index(drop=True)
valid = X_reg2.notna().all(axis=1) & ws.notna()
m3 = sm.OLS(ws[valid], X_reg2[valid]).fit()
print(f"  age - tenure: beta_1={m3.params['ie_age']:.4f} ({m3.bse['ie_age']:.4f})")
