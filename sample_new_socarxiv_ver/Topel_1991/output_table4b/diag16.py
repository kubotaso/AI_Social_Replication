"""
Try 0-based tenure with head-of-household and full pipeline including IV step 2.
Also check if the paper uses a linear+quadratic experience term in step 1
(not just quartic higher-order terms).
"""
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

CPS = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,
       1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,
       1982:1.103,1983:1.089}
df['cps'] = df['year'].map(CPS)
df['lrw'] = df['log_hourly_wage'] - np.log(df['cps'])
for c in ['married', 'union_member', 'disabled', 'region_ne', 'region_nc', 'region_south', 'region_west']:
    df[c] = df[c].fillna(0)
df = df[(df['age'] >= 18) & (df['age'] <= 60)]
df = df[df['hourly_wage'] > 0]
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())]
df = df[df['tenure_topel'] >= 1]
df = df.dropna(subset=['lrw', 'experience', 'tenure_topel'])

# Test both tenure definitions
for tenure_shift, label in [(0, '1-based'), (1, '0-based')]:
    df_t = df.copy()
    df_t['tenure'] = df_t['tenure_topel'] - tenure_shift
    df_t = df_t.sort_values(['person_id','job_id','year']).reset_index(drop=True)
    df_t['init_exp'] = (df_t['experience'] - df_t['tenure']).clip(lower=0)

    # Remaining job duration
    last_yr = df_t.groupby(['person_id','job_id'])['year'].transform('max')
    df_t['remaining_dur'] = last_yr - df_t['year']

    # Step 1
    df_t['prev_yr'] = df_t.groupby(['person_id','job_id'])['year'].shift(1)
    df_t['prev_lrw'] = df_t.groupby(['person_id','job_id'])['lrw'].shift(1)
    df_t['prev_ten'] = df_t.groupby(['person_id','job_id'])['tenure'].shift(1)
    df_t['prev_exp'] = df_t.groupby(['person_id','job_id'])['experience'].shift(1)

    fd = df_t[(df_t['prev_yr'].notna()) & (df_t['year'] - df_t['prev_yr'] == 1)].copy()
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

    X_vars = ['d_ten', 'd_ten_sq', 'd_ten_cu', 'd_ten_qu', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
    X = fd[X_vars].copy()
    for c in yr_cols:
        X[c] = yr_dum[c].values
    # NO constant (d_ten=1 absorbs it)
    y = fd['dlw']
    valid = X.notna().all(axis=1) & y.notna()
    m = sm.OLS(y[valid], X[valid]).fit()

    bhat = m.params['d_ten']
    bhat_se = m.bse['d_ten']
    g2 = m.params['d_ten_sq']
    g3 = m.params['d_ten_cu']
    g4 = m.params['d_ten_qu']
    d2 = m.params['d_exp_sq']
    d3 = m.params['d_exp_cu']
    d4 = m.params['d_exp_qu']

    # Step 2
    T = df_t['tenure'].values.astype(float)
    X_exp = df_t['experience'].values.astype(float)
    df_t['w_star'] = (df_t['lrw'] - bhat*T - g2*T**2 - g3*T**3 - g4*T**4
                      - d2*X_exp**2 - d3*X_exp**3 - d4*X_exp**4)

    ctrl = ['education_years', 'married', 'union_member', 'disabled',
            'region_ne', 'region_nc', 'region_south']
    yr_cols2 = sorted([c for c in df_t.columns if c.startswith('year_') and c != 'year'])
    yr_use = [c for c in yr_cols2 if df_t[c].std() > 1e-10][1:]

    lev = df_t.dropna(subset=['w_star', 'experience', 'init_exp'] + ctrl + yr_use).copy()

    dep = lev['w_star']
    exog = sm.add_constant(lev[ctrl + yr_use])
    endog = lev[['experience']]
    instruments = lev[['init_exp']]

    iv = IV2SLS(dep, exog, endog, instruments).fit(cov_type='unadjusted')
    beta_1 = iv.params['experience']
    beta_1_se = iv.std_errors['experience']
    beta_1_se_mt = np.sqrt(beta_1_se**2 + bhat_se**2)
    beta_2 = bhat - beta_1

    print(f'\n{"="*60}')
    print(f'TENURE: {label} (shift={tenure_shift})')
    print(f'{"="*60}')
    print(f'Step 1: N={int(m.nobs)}, b1+b2={bhat:.4f} ({bhat_se:.4f})')
    print(f'  g2*100={g2*100:.4f}, g3*1000={g3*1000:.4f}, g4*10000={g4*10000:.4f}')
    print(f'  d2*100={d2*100:.4f}, d3*1000={d3*1000:.4f}, d4*10000={d4*10000:.4f}')
    print(f'Step 2: N={len(lev)}')
    print(f'  beta_1={beta_1:.4f} (naive={beta_1_se:.4f}, MT={beta_1_se_mt:.4f})')
    print(f'  beta_2={beta_2:.4f}')
    for Ty in [5, 10, 15, 20]:
        cum = beta_2*Ty + g2*Ty**2 + g3*Ty**3 + g4*Ty**4
        print(f'  {Ty}yr: {cum:.4f}')

print(f'\nPaper: b1+b2=0.1258, beta_1=0.0713, beta_2=0.0545')
print(f'       g2=-0.4592, g3=0.1846, g4=-0.0245')
print(f'       d2=-0.6051, d3=0.2067, d4=-0.0238')
print(f'       5yr=0.1793, 10yr=0.2459, 15yr=0.2832, 20yr=0.3375')
