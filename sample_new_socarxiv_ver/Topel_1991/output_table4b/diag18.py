"""
Test: 1-based tenure + fixed education + d_exp=1 filter + head of household.
Also try without head-of-household restriction to see impact on gamma.
And try with the full (non-fixed) education to get more N.
"""
import pandas as pd, numpy as np, statsmodels.api as sm

df0 = pd.read_csv('data/psid_panel.csv')
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

configs = [
    ("A: HoH, fixEd, 1-based, dexp=1", True, True, True, True),
    ("B: HoH, fixEd, 1-based, no dexp", True, True, True, False),
    ("C: NoHoH, fixEd, 1-based, dexp=1", False, True, True, True),
    ("D: NoHoH, fixEd, 1-based, no dexp", False, True, True, False),
    ("E: NoHoH, yearEd, 1-based, no dexp", False, False, True, False),
    ("F: HoH, yearEd, 1-based, no dexp", True, False, True, False),
]

for label, use_hoh, fix_edu, one_based, dexp_filter in configs:
    df = df0.copy()

    if use_hoh:
        df['pnum'] = df['person_id'] % 1000
        df = df[df['pnum'].isin([1, 170])].copy()

    if fix_edu:
        df['education_years'] = df['education_clean'].copy()
        cat_mask = ~df['year'].isin([1975, 1976])
        df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map({**EDUC_MAP, 9: np.nan})
        def get_fixed_educ(group):
            good_years = group[group['year'].isin([1975, 1976])]['education_years'].dropna()
            if len(good_years) > 0: return good_years.iloc[0]
            mapped = group['education_years'].dropna()
            if len(mapped) > 0:
                modes = mapped.mode()
                return modes.iloc[0] if len(modes) > 0 else mapped.median()
            return np.nan
        person_educ = df.groupby('person_id').apply(get_fixed_educ)
        df['education_fixed'] = df['person_id'].map(person_educ)
        df = df[df['education_fixed'].notna()].copy()
        df['experience'] = (df['age'] - df['education_fixed'] - 6).clip(lower=0)
    else:
        df['education_years'] = df['education_clean'].copy()
        cat_mask = ~df['year'].isin([1975, 1976])
        df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_MAP)
        df = df.dropna(subset=['education_years']).copy()
        df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)

    if one_based:
        df['tenure'] = df['tenure_topel']
    else:
        df['tenure'] = df['tenure_topel'] - 1

    CPS = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,
           1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,
           1982:1.103,1983:1.089}
    df['cps'] = df['year'].map(CPS)
    df['lrw'] = df['log_hourly_wage'] - np.log(df['cps'])
    for c in ['married','union_member','disabled','region_ne','region_nc','region_south','region_west']:
        df[c] = df[c].fillna(0)
    df = df[(df['age'] >= 18) & (df['age'] <= 60)]
    df = df[df['hourly_wage'] > 0]
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())]
    df = df[df['tenure_topel'] >= 1]
    df = df.dropna(subset=['lrw', 'experience', 'tenure'])
    df = df.sort_values(['person_id','job_id','year']).reset_index(drop=True)

    grp = df.groupby(['person_id','job_id'])
    df['prev_yr'] = grp['year'].shift(1)
    df['prev_lrw'] = grp['lrw'].shift(1)
    df['prev_ten'] = grp['tenure'].shift(1)
    df['prev_exp'] = grp['experience'].shift(1)

    fd = df[(df['prev_yr'].notna()) & (df['year'] - df['prev_yr'] == 1)].copy()
    fd['dlw'] = fd['lrw'] - fd['prev_lrw']
    fd['d_exp'] = fd['experience'] - fd['prev_exp']

    if dexp_filter:
        fd = fd[fd['d_exp'] == 1].copy()

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
    y = fd['dlw']
    valid = X.notna().all(axis=1) & y.notna()
    m = sm.OLS(y[valid], X[valid]).fit()

    b = m.params['d_ten']
    g2 = m.params['d_ten_sq']
    g3 = m.params['d_ten_cu']
    g4 = m.params['d_ten_qu']
    d2 = m.params['d_exp_sq']

    # Try approximate cumulative returns with beta_2 ≈ 0.055
    beta_2 = 0.055
    cum5 = beta_2*5 + g2*25 + g3*125 + g4*625
    cum10 = beta_2*10 + g2*100 + g3*1000 + g4*10000
    cum15 = beta_2*15 + g2*225 + g3*3375 + g4*50625
    cum20 = beta_2*20 + g2*400 + g3*8000 + g4*160000

    print(f'{label}: N={int(m.nobs)}')
    print(f'  b1+b2={b:.4f} ({m.bse["d_ten"]:.4f})')
    print(f'  g2*100={g2*100:.4f}, g3*1000={g3*1000:.4f}, g4*10000={g4*10000:.4f}')
    print(f'  d2*100={d2*100:.4f}')
    print(f'  Cum (b2=0.055): 5yr={cum5:.4f}, 10yr={cum10:.4f}, 15yr={cum15:.4f}, 20yr={cum20:.4f}')
    print()

print(f'Paper: N=8683, b1+b2=0.1258, g2=-0.4592, g3=0.1846, g4=-0.0245')
print(f'       d2=-0.4067')
print(f'       Cum: 5yr=0.1793, 10yr=0.2459, 15yr=0.2832, 20yr=0.3375')
