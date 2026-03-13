"""
Use our OWN step 1 coefficients (not paper's) and try step 2 with X_0 as regressor.
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
df['lrw_cps'] = df['log_hourly_wage'] - np.log(df['cps'])
df['lrw_full'] = df['log_hourly_wage'] - np.log(df['gnp']/100.0) - np.log(df['cps'])

for c in ['married','union_member','disabled','region_ne','region_nc','region_south']:
    df[c] = df[c].fillna(0)

df = df.sort_values(['person_id','job_id','year']).reset_index(drop=True)

# Step 1: Within-job first differences
df['prev_yr'] = df.groupby(['person_id','job_id'])['year'].shift(1)
df['prev_lrw'] = df.groupby(['person_id','job_id'])['lrw_cps'].shift(1)
df['prev_ten'] = df.groupby(['person_id','job_id'])['tenure_topel'].shift(1)
df['prev_exp'] = df.groupby(['person_id','job_id'])['exp'].shift(1)

wj = df[(df['prev_yr'].notna()) & (df['year'] - df['prev_yr'] == 1)].copy()
wj['dlw'] = wj['lrw_cps'] - wj['prev_lrw']
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

beta_hat = m1.params['d_ten']
g2 = m1.params['d_ten_sq']
g3 = m1.params['d_ten_cu']
g4 = m1.params['d_ten_qu']
d2 = m1.params['d_exp_sq']
d3 = m1.params['d_exp_cu']
d4 = m1.params['d_exp_qu']

print(f"Step 1 N: {int(m1.nobs)}")
print(f"beta_hat: {beta_hat:.4f}, g2: {g2:.6f}, g3: {g3:.8f}, g4: {g4:.10f}")
print(f"d2: {d2:.6f}, d3: {d3:.8f}, d4: {d4:.10f}")
print()

# Step 2: construct w* with OUR step 1 coefficients
T = df['tenure_topel'].values.astype(float)
X = df['exp'].values.astype(float)

# Try different wage definitions and see which gives positive beta_1
for wage_name, wage_col in [('CPS', 'lrw_cps'), ('GNP+CPS', 'lrw_full'), ('Nominal', 'log_hourly_wage')]:
    w_star = (df[wage_col].values - beta_hat*T - g2*T**2 - g3*T**3 - g4*T**4
              - d2*X**2 - d3*X**3 - d4*X**4)

    controls = ['ey', 'married', 'union_member', 'disabled', 'region_ne', 'region_nc', 'region_south']
    yr2 = pd.get_dummies(df['year'], prefix='yr', dtype=float)
    yr_cols2 = sorted(yr2.columns.tolist())[1:]

    X2 = df[['ie'] + controls].copy().reset_index(drop=True)
    for c in yr_cols2:
        X2[c] = yr2[c].values
    X2 = sm.add_constant(X2)

    y2 = pd.Series(w_star).reset_index(drop=True)
    valid2 = X2.notna().all(axis=1) & y2.notna()
    m2 = sm.OLS(y2[valid2], X2[valid2]).fit()

    b1 = m2.params['ie']
    b1_se = m2.bse['ie']
    b2 = beta_hat - b1
    print(f"{wage_name}: beta_1={b1:.4f} ({b1_se:.4f}), beta_2={b2:.4f}, N={int(m2.nobs)}")

# Also try: what if we DON'T subtract exp polynomial?
print()
print("Without exp polynomial:")
for wage_name, wage_col in [('GNP+CPS', 'lrw_full')]:
    w_star = (df[wage_col].values - beta_hat*T - g2*T**2 - g3*T**3 - g4*T**4)
    y2 = pd.Series(w_star).reset_index(drop=True)
    controls = ['ey', 'married', 'union_member', 'disabled', 'region_ne', 'region_nc', 'region_south']
    yr2 = pd.get_dummies(df['year'], prefix='yr', dtype=float)
    yr_cols2 = sorted(yr2.columns.tolist())[1:]
    X2 = df[['ie'] + controls].copy().reset_index(drop=True)
    for c in yr_cols2:
        X2[c] = yr2[c].values
    X2 = sm.add_constant(X2)
    valid2 = X2.notna().all(axis=1) & y2.notna()
    m2 = sm.OLS(y2[valid2], X2[valid2]).fit()
    b1 = m2.params['ie']
    b1_se = m2.bse['ie']
    b2 = beta_hat - b1
    print(f"  {wage_name}: beta_1={b1:.4f} ({b1_se:.4f}), beta_2={b2:.4f}")

# Check: what's the exp polynomial doing?
print()
print("Exp polynomial values (our coefficients):")
for xv in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    poly = d2*xv**2 + d3*xv**3 + d4*xv**4
    print(f"  X={xv}: {poly:.4f}")
