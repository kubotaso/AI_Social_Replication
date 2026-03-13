"""Extract G_const from Murphy-Topel Jacobian for proper beta_2_se."""
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
occ[m3] = mapped
df['occ'] = occ

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

g2=-0.004592; g3=0.0001846; g4=-0.00000245
d2=-0.006051; d3=0.0002067; d4=-0.00000238

ct = ['ed','married','disabled','region_ne','region_nc','region_south']
yr = [f'year_{y}' for y in range(1969,1984)]

for name, mask, extra, b1b2, b1b2_se in [
    ('PS', df['occ'].isin([1,2,4,8]), ['union_member'], 0.1309, 0.0254),
    ('BC_NU', (df['occ'].isin([5,6,7]))&(df['job_union']==0), [], 0.1520, 0.0311),
    ('BC_U', (df['occ'].isin([5,6,7]))&(df['job_union']==1), [], 0.0992, 0.0297),
]:
    sub = df[mask].copy()
    sub['ws'] = (sub['lrw'] - b1b2*sub['ten']
                 - g2*sub['ten']**2 - g3*sub['ten']**3 - g4*sub['ten']**4
                 - d2*sub['exp']**2 - d3*sub['exp']**3 - d4*sub['exp']**4)

    # Run step 1 from data for V1
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

    s1_params = ['const','dt2','dt3','dt4','dx2','dx3','dx4']
    s1_present = [p for p in s1_params if p in m1.params.index]
    V1 = m1.cov_params().loc[s1_present, s1_present].values

    # Step 2: 2SLS
    ctrl_cols = [c for c in ct + extra if sub[c].std() > 0]
    yr_use = [c for c in yr if c in sub.columns and sub[c].std() > 0]
    while True:
        C = sub[ctrl_cols + yr_use].values.astype(float)
        Z = np.column_stack([np.ones(len(sub)), C, sub['x0'].values.astype(float)])
        if np.linalg.matrix_rank(Z) >= Z.shape[1]:
            break
        if not yr_use: break
        yr_use = yr_use[:-1]

    C = sub[ctrl_cols + yr_use].values.astype(float)
    ones = np.ones(len(sub))
    x0_v = sub['x0'].values.astype(float)
    exp_v = sub['exp'].values.astype(float)
    y = sub['ws'].values.astype(float)

    Z = np.column_stack([ones, C, x0_v])
    X = np.column_stack([ones, C, exp_v])
    n, k = X.shape

    pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
    exp_hat = Z @ pi
    X_hat = np.column_stack([ones, C, exp_hat])
    b = np.linalg.lstsq(X_hat, y, rcond=None)[0]
    resid = y - X @ b
    s2 = np.sum(resid**2) / (n - k)
    XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
    beta_1 = b[-1]

    # Murphy-Topel Jacobian
    T = sub['ten'].values.astype(float)
    Xp = sub['exp'].values.astype(float)

    J = np.zeros((n, len(s1_present)))
    for j, param in enumerate(s1_present):
        if param == 'const': J[:, j] = -T
        elif param == 'dt2': J[:, j] = -T**2
        elif param == 'dt3': J[:, j] = -T**3
        elif param == 'dt4': J[:, j] = -T**4
        elif param == 'dx2': J[:, j] = -Xp**2
        elif param == 'dx3': J[:, j] = -Xp**3
        elif param == 'dx4': J[:, j] = -Xp**4

    G = XhXh_inv @ (X_hat.T @ J)
    g_b1 = G[-1, :]  # beta_1 is last row

    # G_const is the gradient of beta_1 w.r.t. step 1 const (b1+b2)
    const_idx = s1_present.index('const')
    G_const = g_b1[const_idx]

    V_extra = g_b1 @ V1 @ g_b1
    V_naive = s2 * XhXh_inv[-1, -1]
    V_mt = V_naive + V_extra
    beta_1_se_mt = np.sqrt(max(0, V_mt))

    beta_2 = b1b2 - beta_1

    # Proper Var(beta_2) using G_const
    # Cov(b1+b2, beta_1_MT) = G_const * Var(b1+b2)
    cov = G_const * b1b2_se**2
    var_b2 = b1b2_se**2 + V_mt - 2 * cov
    if var_b2 < 0:
        var_b2 = abs(V_mt - b1b2_se**2)
    beta_2_se = np.sqrt(max(0, var_b2))

    print(f"\n{name}: beta_1={beta_1:.4f}, beta_2={beta_2:.4f}")
    print(f"  G_const = {G_const:.4f}")
    print(f"  V_naive(b1) = {V_naive:.8f}, V_extra = {V_extra:.8f}")
    print(f"  V_MT(b1) = {V_mt:.8f}, SE_MT(b1) = {beta_1_se_mt:.4f}")
    print(f"  Cov(b1b2, b1) = {cov:.8f}")
    print(f"  Var(b2) = {var_b2:.8f}, SE(b2) = {beta_2_se:.4f}")
    print(f"  G vector: {g_b1}")

    # Also compute cum_return SE using this approach
    for Tv in [5, 10, 15, 20]:
        cum = beta_2 * Tv + g2 * Tv**2 + g3 * Tv**3 + g4 * Tv**4
        cum_se = abs(Tv) * beta_2_se
        print(f"  cum{Tv} = {cum:.4f} ({cum_se:.4f})")

print("\nPaper targets:")
print("PS:    b1=0.0707 (0.0288), b2=0.0601 (0.0127)")
print("BC_NU: b1=0.1066 (0.0342), b2=0.0513 (0.0146)")
print("BC_U:  b1=0.0592 (0.0338), b2=0.0399 (0.0147)")
