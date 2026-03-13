"""
Test: without d_exp==1 filter, and check if restricting tenure range helps gamma terms.
Also test using both d_exp and d_tenure as regressors when d_exp != 1.
"""
import pandas as pd, numpy as np, statsmodels.api as sm

df = pd.read_csv('data/psid_panel.csv')
df['pnum'] = df['person_id'] % 1000
df = df[df['pnum'].isin([1, 170])].copy()
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
df['education_years'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map({**EDUC_MAP, 9: np.nan})
def get_fixed_educ(group):
    good_years = group[group['year'].isin([1975, 1976])]['education_years'].dropna()
    if len(good_years) > 0:
        return good_years.iloc[0]
    mapped = group['education_years'].dropna()
    if len(mapped) > 0:
        modes = mapped.mode()
        return modes.iloc[0] if len(modes) > 0 else mapped.median()
    return np.nan
person_educ = df.groupby('person_id').apply(get_fixed_educ)
df['education_fixed'] = df['person_id'].map(person_educ)
df = df[df['education_fixed'].notna()].copy()
df['experience'] = (df['age'] - df['education_fixed'] - 6).clip(lower=0)
df['tenure'] = df['tenure_topel'] - 1  # 0-based
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
df = df.dropna(subset=['lrw', 'experience', 'tenure'])
df = df.sort_values(['person_id','job_id','year']).reset_index(drop=True)

last_yr = df.groupby(['person_id','job_id'])['year'].transform('max')
df['remaining_dur'] = last_yr - df['year']

grp = df.groupby(['person_id','job_id'])
df['prev_yr'] = grp['year'].shift(1)
df['prev_lrw'] = grp['lrw'].shift(1)
df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
df['prev_ten'] = grp['tenure'].shift(1)
df['prev_exp'] = grp['experience'].shift(1)

fd = df[(df['prev_yr'].notna()) & (df['year'] - df['prev_yr'] == 1)].copy()
fd['dlw_nom'] = fd['log_hourly_wage'] - fd['prev_log_wage']
fd['dlw_cps'] = fd['lrw'] - fd['prev_lrw']
fd['d_exp'] = fd['experience'] - fd['prev_exp']

# Tenure distribution
print(f"Tenure range: {fd['tenure'].min()} to {fd['tenure'].max()}")
print(f"Tenure distribution (0-based):")
print(fd['tenure'].describe())
print()

# Test different configurations
configs = [
    ("A: All obs, CPS, d_exp filter", fd[fd['d_exp']==1].copy(), 'dlw_cps'),
    ("B: All obs, CPS, no d_exp filter", fd.copy(), 'dlw_cps'),
    ("C: All obs, Nominal, d_exp filter", fd[fd['d_exp']==1].copy(), 'dlw_nom'),
    ("D: All obs, Nominal, no d_exp filter", fd.copy(), 'dlw_nom'),
    ("E: Tenure<=15, CPS, no d_exp filter", fd[fd['tenure']<=15].copy(), 'dlw_cps'),
    ("F: Tenure<=20, CPS, no d_exp filter", fd[fd['tenure']<=20].copy(), 'dlw_cps'),
]

for label, data, wage_col in configs:
    data = data[data[wage_col].between(-2, 2)].copy()

    t, pt = data['tenure'], data['prev_ten']
    x, px = data['experience'], data['prev_exp']
    data['d_ten_sq'] = t**2 - pt**2
    data['d_ten_cu'] = t**3 - pt**3
    data['d_ten_qu'] = t**4 - pt**4
    data['d_exp_sq'] = x**2 - px**2
    data['d_exp_cu'] = x**3 - px**3
    data['d_exp_qu'] = x**4 - px**4

    yr_dum = pd.get_dummies(data['year'], prefix='yr', dtype=float)
    yr_cols = sorted(yr_dum.columns.tolist())[1:]

    X_vars = ['d_ten_sq', 'd_ten_cu', 'd_ten_qu', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
    X = data[X_vars].copy()
    for c in yr_cols:
        X[c] = yr_dum[c].values
    X = sm.add_constant(X)
    y = data[wage_col]
    valid = X.notna().all(axis=1) & y.notna()
    m = sm.OLS(y[valid], X[valid]).fit()

    b = m.params['const']
    g2 = m.params['d_ten_sq']
    g3 = m.params['d_ten_cu']
    g4 = m.params['d_ten_qu']
    d2 = m.params['d_exp_sq']

    # Cumulative returns at 5, 10, 15, 20 (using beta_2 = b - 0.07)
    beta_2_approx = b * 0.5  # rough split
    cum5 = beta_2_approx*5 + g2*25 + g3*125 + g4*625
    cum10 = beta_2_approx*10 + g2*100 + g3*1000 + g4*10000
    cum15 = beta_2_approx*15 + g2*225 + g3*3375 + g4*50625
    cum20 = beta_2_approx*20 + g2*400 + g3*8000 + g4*160000

    print(f'{label}: N={int(m.nobs)}, b1+b2={b:.4f} ({m.bse["const"]:.4f})')
    print(f'  g2*100={g2*100:.4f}, g3*1000={g3*1000:.4f}, g4*10000={g4*10000:.4f}')
    print(f'  d2*100={d2*100:.4f}')
    print(f'  Approx cum (b2=b/2): 5yr={cum5:.4f}, 10yr={cum10:.4f}, 15yr={cum15:.4f}, 20yr={cum20:.4f}')
    print()

print(f'Paper: b1+b2=0.1258, g2=-0.4592, g3=0.1846, g4=-0.0245')
print(f'       d2=-0.6051 (or -0.4067 in model 3)')
print(f'       5yr=0.1793, 10yr=0.2459, 15yr=0.2832, 20yr=0.3375')
