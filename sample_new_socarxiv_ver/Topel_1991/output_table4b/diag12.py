"""Compare step 1 specifications: d_ten as regressor vs constant."""
import pandas as pd, numpy as np, statsmodels.api as sm

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
df = df[(df['age'] >= 18) & (df['age'] <= 60)]
df = df[df['hourly_wage'] > 0]
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())]
df = df[df['tenure'] >= 1]
df = df.dropna(subset=['lrw', 'experience', 'tenure'])
df = df.sort_values(['person_id','job_id','year']).reset_index(drop=True)
df['prev_yr'] = df.groupby(['person_id','job_id'])['year'].shift(1)
df['prev_lrw'] = df.groupby(['person_id','job_id'])['lrw'].shift(1)
df['prev_ten'] = df.groupby(['person_id','job_id'])['tenure'].shift(1)
df['prev_exp'] = df.groupby(['person_id','job_id'])['experience'].shift(1)
fd = df[(df['prev_yr'].notna()) & (df['year'] - df['prev_yr'] == 1)].copy()
fd['dlw'] = fd['lrw'] - fd['prev_lrw']
fd = fd[fd['dlw'].between(-2, 2)].copy()
t, pt = fd['tenure'], fd['prev_ten']
x, px = fd['experience'], fd['prev_exp']
fd['d_ten'] = t - pt
fd['d_ten_sq'] = t**2 - pt**2
fd['d_ten_cu'] = t**3 - pt**3
fd['d_ten_qu'] = t**4 - pt**4
fd['d_exp_sq'] = x**2 - px**2
fd['d_exp_cu'] = x**3 - px**3
fd['d_exp_qu'] = x**4 - px**4
yr_dum = pd.get_dummies(fd['year'], prefix='yr', dtype=float)
yr_cols = sorted(yr_dum.columns.tolist())[1:]
y = fd['dlw']

# Version A: with d_ten, NO constant
X_vars = ['d_ten', 'd_ten_sq', 'd_ten_cu', 'd_ten_qu', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
X = fd[X_vars].copy()
for c in yr_cols:
    X[c] = yr_dum[c].values
valid = X.notna().all(axis=1) & y.notna()
m = sm.OLS(y[valid], X[valid]).fit()
print(f'Version A (d_ten, no constant):')
print(f'  N = {int(m.nobs)}')
print(f'  b1+b2 = {m.params["d_ten"]:.4f} ({m.bse["d_ten"]:.4f})')
print(f'  g2*100 = {m.params["d_ten_sq"]*100:.4f}')
print(f'  g3*1000 = {m.params["d_ten_cu"]*1000:.4f}')
print(f'  g4*10000 = {m.params["d_ten_qu"]*10000:.4f}')
print(f'  d2*100 = {m.params["d_exp_sq"]*100:.4f}')
print(f'  d3*1000 = {m.params["d_exp_cu"]*1000:.4f}')
print(f'  d4*10000 = {m.params["d_exp_qu"]*10000:.4f}')
print()

# Version B: with constant, NO d_ten
X_vars2 = ['d_ten_sq', 'd_ten_cu', 'd_ten_qu', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
X2 = fd[X_vars2].copy()
for c in yr_cols:
    X2[c] = yr_dum[c].values
X2 = sm.add_constant(X2)
m2 = sm.OLS(y[valid], X2[valid]).fit()
print(f'Version B (constant, no d_ten):')
print(f'  N = {int(m2.nobs)}')
print(f'  const (=b1+b2) = {m2.params["const"]:.4f} ({m2.bse["const"]:.4f})')
print(f'  g2*100 = {m2.params["d_ten_sq"]*100:.4f}')
print(f'  g3*1000 = {m2.params["d_ten_cu"]*1000:.4f}')
print(f'  g4*10000 = {m2.params["d_ten_qu"]*10000:.4f}')
print(f'  d2*100 = {m2.params["d_exp_sq"]*100:.4f}')
print(f'  d3*1000 = {m2.params["d_exp_cu"]*1000:.4f}')
print(f'  d4*10000 = {m2.params["d_exp_qu"]*10000:.4f}')
print()

# Check: are d_ten values always 1?
print(f'd_ten unique values: {sorted(fd.loc[valid, "d_ten"].unique())}')
print(f'd_ten mean: {fd.loc[valid, "d_ten"].mean():.4f}')
print()

# Version C: use paper's step 1 coefficients, compute step 2
from linearmodels.iv import IV2SLS
GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}
df['gnp_def'] = df['year'].map(lambda y: GNP_DEFLATOR.get(y - 1, np.nan))
df['log_real_wage'] = df['log_hourly_wage'] - np.log(df['gnp_def'] / 100.0)
df['init_exp'] = (df['experience'] - df['tenure']).clip(lower=0)

# Use our Version A step 1 coefficients
bhat = m.params['d_ten']
g2 = m.params['d_ten_sq']
g3 = m.params['d_ten_cu']
g4 = m.params['d_ten_qu']
d2 = m.params['d_exp_sq']
d3 = m.params['d_exp_cu']
d4 = m.params['d_exp_qu']

T = df['tenure'].values
X_exp = df['experience'].values
df['w_star'] = (df['log_real_wage'] - bhat*T - g2*T**2 - g3*T**3 - g4*T**4
                - d2*X_exp**2 - d3*X_exp**3 - d4*X_exp**4)

ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_cols2 = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_use = [c for c in yr_cols2 if df[c].std() > 1e-10][1:]

lev = df.dropna(subset=['w_star', 'experience', 'init_exp', 'log_real_wage'] + ctrl).copy()

# Check rank
exog = sm.add_constant(lev[ctrl + yr_use])
r = np.linalg.matrix_rank(exog.values)
current_yr = yr_use.copy()
while r < exog.shape[1] and current_yr:
    current_yr.pop()
    exog = sm.add_constant(lev[ctrl + current_yr])
    r = np.linalg.matrix_rank(exog.values)

dep = lev['w_star']
exog_final = sm.add_constant(lev[ctrl + current_yr])
endog = lev[['experience']]
instruments = lev[['init_exp']]

iv = IV2SLS(dep, exog_final, endog, instruments).fit(cov_type='unadjusted')
beta_1 = iv.params['experience']
beta_1_se = iv.std_errors['experience']
beta_1_se_mt = np.sqrt(beta_1_se**2 + m.bse['d_ten']**2)
beta_2 = bhat - beta_1

print(f'Version C: Step 2 IV with own step 1 (Version A):')
print(f'  N = {len(lev)}')
print(f'  beta_1 = {beta_1:.4f} (naive SE={beta_1_se:.4f}, MT SE={beta_1_se_mt:.4f})')
print(f'  beta_2 = {beta_2:.4f}')
print(f'  Paper: beta_1=0.0713 (0.0181), beta_2=0.0545')
print()

# Cumulative returns
for Ty in [5, 10, 15, 20]:
    cum = beta_2*Ty + g2*Ty**2 + g3*Ty**3 + g4*Ty**4
    print(f'  {Ty}yr: {cum:.4f} (paper: ', end='')
    paper_cum = {5: 0.1793, 10: 0.2459, 15: 0.2832, 20: 0.3375}
    print(f'{paper_cum[Ty]})')

print(f'\nPaper: b1+b2=0.1258 (0.0162), g2=-0.4592, g3=0.1846, g4=-0.0245')
print(f'       d2=-0.6051, d3=0.2067, d4=-0.0238')
