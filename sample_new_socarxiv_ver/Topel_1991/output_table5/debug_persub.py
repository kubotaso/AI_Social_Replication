"""Test per-subsample step 1 with ALL parameters estimated per-subsample.
The paper says step 1 is estimated separately for each subsample.
Check if using OUR per-subsample step 1 (b1+b2 AND polynomial terms) gives differentiated beta_1.
"""
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

masks = {
    'PS': df['occ'].isin([1,2,4,8]),
    'BC_NU': (df['occ'].isin([5,6,7]))&(df['job_union']==0),
    'BC_U': (df['occ'].isin([5,6,7]))&(df['job_union']==1),
}

for name, mask in masks.items():
    sub = df[mask].copy()
    print(f"\n{'='*70}")
    print(f"{name}: N={len(sub)}")
    print(f"{'='*70}")

    # Step 1: per-subsample
    s = sub.sort_values(['person_id','job_id','year']).copy()
    s['within'] = ((s['person_id']==s['person_id'].shift(1))&
                   (s['job_id']==s['job_id'].shift(1))&
                   (s['year']-s['year'].shift(1)==1))
    s['dlw'] = s['lwcps'] - s['lwcps'].shift(1)
    for k in [2,3,4]:
        s[f'dt{k}'] = s['ten']**k - (s['ten']-1)**k
        s[f'dx{k}'] = s['exp']**k - (s['exp']-1)**k
    dyr = []
    for yc in yr:
        dc = f'd_{yc}'
        s[dc] = s[yc].astype(float) - s[yc].shift(1).astype(float)
        s[dc] = s[dc].fillna(0)
        dyr.append(dc)
    fd = s[s['within']].dropna(subset=['dlw']).copy()
    dyr_use = [c for c in dyr if fd[c].std()>1e-10]
    if len(dyr_use)>1: dyr_use = dyr_use[1:]
    xcols = ['dt2','dt3','dt4','dx2','dx3','dx4'] + dyr_use
    y1 = fd['dlw']; X1 = sm.add_constant(fd[xcols])
    v = y1.notna() & X1.notna().all(axis=1)
    m1 = sm.OLS(y1[v], X1[v]).fit()

    b1b2 = m1.params['const']
    g2 = m1.params.get('dt2', 0)
    g3 = m1.params.get('dt3', 0)
    g4 = m1.params.get('dt4', 0)
    d2 = m1.params.get('dx2', 0)
    d3 = m1.params.get('dx3', 0)
    d4 = m1.params.get('dx4', 0)
    b1b2_se = m1.bse['const']

    print(f"  Step 1: b1+b2={b1b2:.4f} ({b1b2_se:.4f})")
    print(f"    g2={g2:.6f}, g3={g3:.7f}, g4={g4:.9f}")
    print(f"    d2={d2:.6f}, d3={d3:.7f}, d4={d4:.9f}")
    print(f"    N_wj={v.sum()}")

    # Construct w* using per-subsample step 1
    sub['ws'] = (sub['lrw'] - b1b2*sub['ten']
                 - g2*sub['ten']**2 - g3*sub['ten']**3 - g4*sub['ten']**4
                 - d2*sub['exp']**2 - d3*sub['exp']**3 - d4*sub['exp']**4)

    ct = ['ed','married','disabled','region_ne','region_nc','region_south']
    if name == 'PS':
        ct.append('union_member')

    yr_use = [c for c in yr if c in sub.columns and sub[c].std() > 0]
    ctrls = [c for c in ct if c in sub.columns and sub[c].std() > 0] + yr_use

    while True:
        C = sub[ctrls].values.astype(float)
        Z = np.column_stack([np.ones(len(sub)), C, sub['x0'].values.astype(float)])
        if np.linalg.matrix_rank(Z) >= Z.shape[1]:
            break
        if not yr_use:
            break
        yr_use = yr_use[:-1]
        ctrls = [c for c in ct if c in sub.columns and sub[c].std() > 0] + yr_use

    C = sub[ctrls].values.astype(float)
    ones = np.ones(len(sub))
    x0_v = sub['x0'].values.astype(float)
    exp_v = sub['exp'].values.astype(float)
    y = sub['ws'].values.astype(float)

    Z = np.column_stack([ones, C, x0_v])
    X = np.column_stack([ones, C, exp_v])

    pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
    exp_hat = Z @ pi
    X_hat = np.column_stack([ones, C, exp_hat])
    b = np.linalg.lstsq(X_hat, y, rcond=None)[0]
    beta_1 = b[-1]
    beta_2 = b1b2 - beta_1

    resid = y - X @ b
    n, k = X.shape
    s2 = np.sum(resid**2) / (n - k)
    XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
    se_naive = np.sqrt(s2 * XhXh_inv[-1, -1])

    print(f"\n  beta_1={beta_1:.4f} (naive SE={se_naive:.4f})")
    print(f"  beta_2={beta_2:.4f}")
    print(f"  b1+b2={b1b2:.4f}")

    # Cumulative returns with per-subsample polynomial terms
    for Tv in [5, 10, 15, 20]:
        cum = beta_2 * Tv + g2 * Tv**2 + g3 * Tv**3 + g4 * Tv**4
        print(f"  cum{Tv}={cum:.4f}")

    print(f"  Paper beta_1: PS=0.0707, BC_NU=0.1066, BC_U=0.0592")
