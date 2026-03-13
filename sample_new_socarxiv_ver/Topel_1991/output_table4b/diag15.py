"""Fix NaN filling, rank issues, and try full pipeline."""
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

# Fill NaN BEFORE sample restrictions
for c in ['married', 'union_member', 'disabled', 'region_ne', 'region_nc', 'region_south', 'region_west']:
    df[c] = df[c].fillna(0)

df = df[(df['age'] >= 18) & (df['age'] <= 60)]
df = df[df['hourly_wage'] > 0]
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())]
df = df[df['tenure'] >= 1]
df = df.dropna(subset=['lrw', 'experience', 'tenure'])
df = df.sort_values(['person_id','job_id','year']).reset_index(drop=True)
df['init_exp'] = (df['experience'] - df['tenure']).clip(lower=0)

# Remaining job duration
last_yr = df.groupby(['person_id','job_id'])['year'].transform('max')
df['remaining_dur'] = last_yr - df['year']

print(f'Full sample: {len(df)} obs, {df["person_id"].nunique()} persons')

# Step 1 with own coefficients
df['prev_yr'] = df.groupby(['person_id','job_id'])['year'].shift(1)
df['prev_lrw'] = df.groupby(['person_id','job_id'])['lrw'].shift(1)
df['prev_ten'] = df.groupby(['person_id','job_id'])['tenure'].shift(1)
df['prev_exp'] = df.groupby(['person_id','job_id'])['experience'].shift(1)

fd = df[(df['prev_yr'].notna()) & (df['year'] - df['prev_yr'] == 1)].copy()
fd['dlw'] = fd['lrw'] - fd['prev_lrw']
fd = fd[fd['dlw'].between(-2, 2)].copy()

t, pt = fd['tenure'], fd['prev_ten']
x, px = fd['experience'], fd['prev_exp']
fd['d_ten_sq'] = t**2 - pt**2
fd['d_ten_cu'] = t**3 - pt**3
fd['d_ten_qu'] = t**4 - pt**4
fd['d_exp_sq'] = x**2 - px**2
fd['d_exp_cu'] = x**3 - px**3
fd['d_exp_qu'] = x**4 - px**4

yr_dum = pd.get_dummies(fd['year'], prefix='yr', dtype=float)
yr_cols_fd = sorted(yr_dum.columns.tolist())[1:]

X_vars = ['d_ten_sq', 'd_ten_cu', 'd_ten_qu', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
X = fd[X_vars].copy()
for c in yr_cols_fd:
    X[c] = yr_dum[c].values
X = sm.add_constant(X)
y = fd['dlw']
valid = X.notna().all(axis=1) & y.notna()
m = sm.OLS(y[valid], X[valid]).fit()

bhat = m.params['const']
bhat_se = m.bse['const']
g2 = m.params['d_ten_sq']
g3 = m.params['d_ten_cu']
g4 = m.params['d_ten_qu']
d2 = m.params['d_exp_sq']
d3 = m.params['d_exp_cu']
d4 = m.params['d_exp_qu']

print(f'\nStep 1 (own, >=0): N={int(m.nobs)}, b1+b2={bhat:.4f} ({bhat_se:.4f})')
print(f'  g2*100={g2*100:.4f}, g3*1000={g3*1000:.4f}, g4*10000={g4*10000:.4f}')
print(f'  d2*100={d2*100:.4f}, d3*1000={d3*1000:.4f}, d4*10000={d4*10000:.4f}')
print(f'  Paper: b1+b2=0.1258, g2=-0.4592, g3=0.1846, g4=-0.0245')
print(f'         d2=-0.6051, d3=0.2067, d4=-0.0238')

# Step 2: check controls for rank
ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']

yr_cols_lev = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_use = [c for c in yr_cols_lev if df[c].std() > 1e-10][1:]

T = df['tenure'].values
X_exp = df['experience'].values
df['w_star'] = (df['lrw'] - bhat*T - g2*T**2 - g3*T**3 - g4*T**4
                - d2*X_exp**2 - d3*X_exp**3 - d4*X_exp**4)

lev = df.dropna(subset=['w_star', 'experience', 'init_exp'] + ctrl + yr_use).copy()
print(f'\nStep 2 sample: {len(lev)} obs')

# Simpler rank check - just drop one year dummy
exog = sm.add_constant(lev[ctrl + yr_use])
r = np.linalg.matrix_rank(exog.values)
print(f'Rank: {r}, ncols: {exog.shape[1]}')

# The issue is that region dummies + year dummies may be collinear.
# Let me check: do all 4 region dummies sum to 1 for most obs?
region_sum = lev['region_ne'] + lev['region_nc'] + lev['region_south'] + lev['region_west']
print(f'Region sum: min={region_sum.min()}, max={region_sum.max()}, mean={region_sum.mean():.3f}')
print(f'Region sum == 1: {(region_sum == 1).sum()}, == 0: {(region_sum == 0).sum()}')

# Drop region_west (4th dummy) to avoid collinearity with constant
ctrl_adj = ['education_years', 'married', 'union_member', 'disabled',
            'region_ne', 'region_nc', 'region_south']
exog2 = sm.add_constant(lev[ctrl_adj + yr_use])
r2 = np.linalg.matrix_rank(exog2.values)
print(f'After dropping region_west: Rank: {r2}, ncols: {exog2.shape[1]}')

# Try IV with 3 region dummies
dep = lev['w_star']
exog_final = sm.add_constant(lev[ctrl_adj + yr_use])
endog = lev[['experience']]
instruments = lev[['init_exp']]

try:
    iv = IV2SLS(dep, exog_final, endog, instruments).fit(cov_type='unadjusted')
    beta_1 = iv.params['experience']
    beta_1_se = iv.std_errors['experience']
    beta_1_se_mt = np.sqrt(beta_1_se**2 + bhat_se**2)
    beta_2 = bhat - beta_1
    print(f'\nStep 2 IV (own step 1):')
    print(f'  N = {len(lev)}')
    print(f'  beta_1 = {beta_1:.4f} (naive={beta_1_se:.4f}, MT={beta_1_se_mt:.4f})')
    print(f'  beta_2 = {beta_2:.4f}')
    print(f'  Paper: beta_1=0.0713, beta_2=0.0545')

    for Ty in [5, 10, 15, 20]:
        cum = beta_2*Ty + g2*Ty**2 + g3*Ty**3 + g4*Ty**4
        paper = {5: 0.1793, 10: 0.2459, 15: 0.2832, 20: 0.3375}
        print(f'  {Ty}yr: {cum:.4f} (paper: {paper[Ty]})')
except Exception as e:
    print(f'IV failed: {e}')

# Also try with paper's step 1 coefficients
bhat_p = 0.1258
g2_p, g3_p, g4_p = -0.004592, 0.0001846, -0.00000245
d2_p, d3_p, d4_p = -0.006051, 0.0002067, -0.00000238

df['w_star_p'] = (df['lrw'] - bhat_p*T - g2_p*T**2 - g3_p*T**3 - g4_p*T**4
                  - d2_p*X_exp**2 - d3_p*X_exp**3 - d4_p*X_exp**4)

lev2 = df.dropna(subset=['w_star_p', 'experience', 'init_exp'] + ctrl_adj + yr_use).copy()
dep2 = lev2['w_star_p']
exog2 = sm.add_constant(lev2[ctrl_adj + yr_use])
endog2 = lev2[['experience']]
instruments2 = lev2[['init_exp']]

try:
    iv2 = IV2SLS(dep2, exog2, endog2, instruments2).fit(cov_type='unadjusted')
    print(f'\nStep 2 IV (paper step 1):')
    print(f'  N = {len(lev2)}')
    print(f'  beta_1 = {iv2.params["experience"]:.4f} ({iv2.std_errors["experience"]:.4f})')
    print(f'  beta_2 = {bhat_p - iv2.params["experience"]:.4f}')
    print(f'  Paper: beta_1=0.0713, beta_2=0.0545')
except Exception as e:
    print(f'IV with paper step 1 failed: {e}')
