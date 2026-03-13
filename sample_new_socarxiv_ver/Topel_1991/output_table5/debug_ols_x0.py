"""Test OLS on X_0 (initial experience) instead of IV."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/psid_panel.csv')
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
df['lwcps'] = df['log_hourly_wage'] - np.log(df['year'].map(cps))

occ = df['occ_1digit'].copy()
m3 = occ > 9
three = occ[m3]
mapped = pd.Series(0, index=three.index, dtype=int)
mapped[(three>=1)&(three<=195)]=1; mapped[(three>=201)&(three<=245)]=2
mapped[(three>=260)&(three<=395)]=4; mapped[(three>=401)&(three<=580)]=5
mapped[(three>=601)&(three<=695)]=6; mapped[(three>=701)&(three<=785)]=7
mapped[(three>=801)&(three<=824)]=9; mapped[(three>=900)&(three<=965)]=8
occ[m3] = mapped; df['occ'] = occ

df = df.sort_values(['person_id','job_id','year'])
ju = df.groupby(['person_id','job_id'])['union_member'].agg(
    lambda x: (x.mean()>0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member':'job_union'})
df = df.merge(ju, on=['person_id','job_id'], how='left')

df = df[df['ten']>=1].dropna(subset=['log_hourly_wage','lrw']).copy()
df = df[~df['occ'].isin([0,3,9])].copy()
df = df[(df['self_employed']==0)|(df['self_employed'].isna())].copy()
df['x0'] = (df['exp'] - df['ten']).clip(lower=0)
for c in ['married','disabled','union_member','region_ne','region_nc','region_south']:
    df[c] = df[c].fillna(0)
yr = [f'year_{y}' for y in range(1969,1984)]

g2=-0.004592; g3=0.0001846; g4=-0.00000245
d2=-0.006051; d3=0.0002067; d4=-0.00000238

masks = {
    'PS': (df['occ'].isin([1,2,4,8]), 0.1309, ['union_member']),
    'BC_NU': ((df['occ'].isin([5,6,7]))&(df['job_union']==0), 0.1520, []),
    'BC_U': ((df['occ'].isin([5,6,7]))&(df['job_union']==1), 0.0992, []),
}

for name, (mask, b1b2, extra) in masks.items():
    sub = df[mask].copy()
    sub['ws'] = (sub['lrw'] - b1b2*sub['ten']
                 - g2*sub['ten']**2 - g3*sub['ten']**3 - g4*sub['ten']**4
                 - d2*sub['exp']**2 - d3*sub['exp']**3 - d4*sub['exp']**4)

    ct = ['ed','married','disabled','region_ne','region_nc','region_south'] + extra
    yr_use = [c for c in yr if c in sub.columns and sub[c].std() > 0]
    ctrls = [c for c in ct if c in sub.columns and sub[c].std() > 0] + yr_use

    # Method 1: OLS on X_0 (initial experience)
    X_ols = sm.add_constant(sub[ctrls + ['x0']])
    y = sub['ws']
    m_ols = sm.OLS(y, X_ols).fit()
    beta_1_ols = m_ols.params['x0']

    # Method 2: OLS on X (total experience)
    X_exp = sm.add_constant(sub[ctrls + ['exp']])
    m_exp = sm.OLS(y, X_exp).fit()
    beta_1_exp = m_exp.params['exp']

    # Method 3: IV - X instrumented by X_0
    C = sub[ctrls].values.astype(float)
    ones = np.ones(len(sub))
    x0_v = sub['x0'].values.astype(float)
    exp_v = sub['exp'].values.astype(float)
    y_v = sub['ws'].values.astype(float)

    yr_u = yr_use[:]
    while True:
        ctrl_c = [c for c in ct if c in sub.columns and sub[c].std() > 0] + yr_u
        C2 = sub[ctrl_c].values.astype(float)
        Z = np.column_stack([ones, C2, x0_v])
        if np.linalg.matrix_rank(Z) >= Z.shape[1]:
            break
        if not yr_u:
            break
        yr_u = yr_u[:-1]

    C2 = sub[ctrl_c].values.astype(float)
    Z = np.column_stack([ones, C2, x0_v])
    X_iv = np.column_stack([ones, C2, exp_v])

    pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
    exp_hat = Z @ pi
    X_hat = np.column_stack([ones, C2, exp_hat])
    b_iv = np.linalg.lstsq(X_hat, y_v, rcond=None)[0]
    beta_1_iv = b_iv[-1]

    # Method 4: IV - X_0 instrumented by X
    pi2 = np.linalg.lstsq(X_iv, x0_v, rcond=None)[0]
    x0_hat = X_iv @ pi2
    X_hat2 = np.column_stack([ones, C2, x0_hat])
    b_iv2 = np.linalg.lstsq(X_hat2, y_v, rcond=None)[0]
    beta_1_iv2 = b_iv2[-1]

    beta_2_ols = b1b2 - beta_1_ols
    beta_2_exp = b1b2 - beta_1_exp
    beta_2_iv = b1b2 - beta_1_iv
    beta_2_iv2 = b1b2 - beta_1_iv2

    print(f"\n{name}: N={len(sub)}")
    print(f"  OLS on X_0:  beta_1={beta_1_ols:.4f}, beta_2={beta_2_ols:.4f}")
    print(f"  OLS on X:    beta_1={beta_1_exp:.4f}, beta_2={beta_2_exp:.4f}")
    print(f"  IV (X by X_0): beta_1={beta_1_iv:.4f}, beta_2={beta_2_iv:.4f}")
    print(f"  IV (X_0 by X): beta_1={beta_1_iv2:.4f}, beta_2={beta_2_iv2:.4f}")
    print(f"  Paper: beta_1={'0.0707' if name=='PS' else ('0.1066' if name=='BC_NU' else '0.0592')}")

    # Also try: what if we use X with NO instrument (just OLS on w* regressed on X)?
    # This is what equation (7) says: y - TB_hat = X_0*beta_1 + e
    # With X_0 = X - T, this becomes: y - TB_hat = (X-T)*beta_1 + e
    # = X*beta_1 - T*beta_1 + e
    # So: w* + T*beta_1 = X*beta_1 + e
    # Or: w* = X*beta_1 - T*beta_1 + e = (X - T)*beta_1 + e = X_0*beta_1 + e
    # This is just OLS on X_0, same as Method 1
