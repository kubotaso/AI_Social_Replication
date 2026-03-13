"""Check d_exp distribution and try using paper's coefficients with our data."""
import pandas as pd, numpy as np, statsmodels.api as sm
from linearmodels.iv import IV2SLS

df = pd.read_csv('data/psid_panel.csv')
df['pnum'] = df['person_id'] % 1000
df = df[df['pnum'].isin([1, 170])].copy()
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
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
CPS = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,
       1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,
       1982:1.103,1983:1.089}
df['cps'] = df['year'].map(CPS)
df['lrw'] = df['log_hourly_wage'] - np.log(df['cps'])

GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}

df = df[(df['age'] >= 18) & (df['age'] <= 60)]
df = df[df['hourly_wage'] > 0]
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())]
df = df[df['tenure'] >= 1]
df = df.dropna(subset=['lrw', 'experience', 'tenure'])
df = df.sort_values(['person_id','job_id','year']).reset_index(drop=True)

# Check d_exp distribution
df['prev_yr'] = df.groupby(['person_id','job_id'])['year'].shift(1)
df['prev_exp'] = df.groupby(['person_id','job_id'])['experience'].shift(1)
fd = df[(df['prev_yr'].notna()) & (df['year'] - df['prev_yr'] == 1)].copy()
fd['d_exp'] = fd['experience'] - fd['prev_exp']
print(f'd_exp unique values: {sorted(fd["d_exp"].unique())}')
print(f'd_exp value counts:')
print(fd['d_exp'].value_counts())
print()

# The reason the delta terms are near zero:
# When d_exp is always 1, d_exp_sq = x^2 - (x-1)^2 = 2x-1 (linear in x)
# d_exp_cu = x^3 - (x-1)^3 = 3x^2 - 3x + 1 (quadratic in x)
# These are NOT zero - they capture the polynomial structure.
# Let me check if the issue is that year dummies absorb the experience effects.

# Try WITHOUT year dummies to see if deltas change
df2 = df.copy()
df2['prev_lrw'] = df2.groupby(['person_id','job_id'])['lrw'].shift(1)
df2['prev_ten'] = df2.groupby(['person_id','job_id'])['tenure'].shift(1)
fd2 = df2[(df2['prev_yr'].notna()) & (df2['year'] - df2['prev_yr'] == 1)].copy()
fd2['dlw'] = fd2['lrw'] - fd2['prev_lrw']
fd2 = fd2[fd2['dlw'].between(-2, 2)].copy()
t, pt = fd2['tenure'], fd2['prev_ten']
x, px = fd2['experience'], fd2['prev_exp']
fd2['d_ten_sq'] = t**2 - pt**2
fd2['d_ten_cu'] = t**3 - pt**3
fd2['d_ten_qu'] = t**4 - pt**4
fd2['d_exp_sq'] = x**2 - px**2
fd2['d_exp_cu'] = x**3 - px**3
fd2['d_exp_qu'] = x**4 - px**4

X_vars = ['d_ten_sq', 'd_ten_cu', 'd_ten_qu', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
X = sm.add_constant(fd2[X_vars])
y = fd2['dlw']
valid = X.notna().all(axis=1) & y.notna()
m = sm.OLS(y[valid], X[valid]).fit()
print(f'Without year dummies:')
print(f'  b1+b2 = {m.params["const"]:.4f} ({m.bse["const"]:.4f})')
print(f'  g2*100 = {m.params["d_ten_sq"]*100:.4f}')
print(f'  g3*1000 = {m.params["d_ten_cu"]*1000:.4f}')
print(f'  g4*10000 = {m.params["d_ten_qu"]*10000:.4f}')
print(f'  d2*100 = {m.params["d_exp_sq"]*100:.4f}')
print(f'  d3*1000 = {m.params["d_exp_cu"]*1000:.4f}')
print(f'  d4*10000 = {m.params["d_exp_qu"]*10000:.4f}')
print()

# Now WITH year dummies
yr_dum = pd.get_dummies(fd2['year'], prefix='yr', dtype=float)
yr_cols = sorted(yr_dum.columns.tolist())[1:]
X2 = fd2[X_vars].copy()
for c in yr_cols:
    X2[c] = yr_dum[c].values
X2 = sm.add_constant(X2)
m2 = sm.OLS(y[valid], X2[valid]).fit()
print(f'With year dummies:')
print(f'  b1+b2 = {m2.params["const"]:.4f} ({m2.bse["const"]:.4f})')
print(f'  g2*100 = {m2.params["d_ten_sq"]*100:.4f}')
print(f'  g3*1000 = {m2.params["d_ten_cu"]*1000:.4f}')
print(f'  g4*10000 = {m2.params["d_ten_qu"]*10000:.4f}')
print(f'  d2*100 = {m2.params["d_exp_sq"]*100:.4f}')
print(f'  d3*1000 = {m2.params["d_exp_cu"]*1000:.4f}')
print(f'  d4*10000 = {m2.params["d_exp_qu"]*10000:.4f}')
print()

# Now try using paper's STEP 1 coefficients for step 2
# Use CPS-adjusted wage for levels (same as step 1)
df['init_exp'] = (df['experience'] - df['tenure']).clip(lower=0)
# Paper step 1 values
bhat = 0.1258
g2 = -0.004592
g3 = 0.0001846
g4 = -0.00000245
d2 = -0.006051
d3 = 0.0002067
d4 = -0.00000238

T = df['tenure'].values
X_exp = df['experience'].values
df['w_star_cps'] = (df['lrw'] - bhat*T - g2*T**2 - g3*T**3 - g4*T**4
                    - d2*X_exp**2 - d3*X_exp**3 - d4*X_exp**4)

ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_cols2 = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_use = [c for c in yr_cols2 if df[c].std() > 1e-10][1:]

# Step 2 on CPS-adjusted wages
lev = df.dropna(subset=['w_star_cps', 'experience', 'init_exp'] + ctrl).copy()

exog = sm.add_constant(lev[ctrl + yr_use])
r = np.linalg.matrix_rank(exog.values)
current_yr = yr_use.copy()
while r < exog.shape[1] and current_yr:
    current_yr.pop()
    exog = sm.add_constant(lev[ctrl + current_yr])
    r = np.linalg.matrix_rank(exog.values)

dep = lev['w_star_cps']
exog_final = sm.add_constant(lev[ctrl + current_yr])
endog = lev[['experience']]
instruments = lev[['init_exp']]

iv = IV2SLS(dep, exog_final, endog, instruments).fit(cov_type='unadjusted')
print(f'Step 2 IV with paper step 1, CPS wages:')
print(f'  N = {len(lev)}')
print(f'  beta_1 = {iv.params["experience"]:.4f} ({iv.std_errors["experience"]:.4f})')
print(f'  beta_2 = {bhat - iv.params["experience"]:.4f}')
print(f'  Paper: beta_1=0.0713 (0.0181), beta_2=0.0545')
print()

# Also try OLS on X_0 directly (not IV)
X_ols = sm.add_constant(lev[['init_exp'] + ctrl + current_yr])
m_ols = sm.OLS(dep, X_ols).fit()
print(f'Step 2 OLS on X_0 with paper step 1, CPS wages:')
print(f'  N = {len(lev)}')
print(f'  beta_1 = {m_ols.params["init_exp"]:.4f} ({m_ols.bse["init_exp"]:.4f})')
print(f'  beta_2 = {bhat - m_ols.params["init_exp"]:.4f}')
