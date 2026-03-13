"""Try various SE formulas for beta_2 to match paper."""
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

for name, mask, extra, b1b2, b1b2_se_paper in [
    ('PS', df['occ'].isin([1,2,4,8]), ['union_member'], 0.1309, 0.0254),
    ('BC_NU', (df['occ'].isin([5,6,7]))&(df['job_union']==0), [], 0.1520, 0.0311),
    ('BC_U', (df['occ'].isin([5,6,7]))&(df['job_union']==1), [], 0.0992, 0.0297),
]:
    sub = df[mask].copy()
    sub['ws'] = (sub['lrw'] - b1b2*sub['ten']
                 - g2*sub['ten']**2 - g3*sub['ten']**3 - g4*sub['ten']**4
                 - d2*sub['exp']**2 - d3*sub['exp']**3 - d4*sub['exp']**4)

    ct = ['ed','married','disabled','region_ne','region_nc','region_south'] + extra
    yr_use = [c for c in yr if c in sub.columns and sub[c].std() > 0]

    while True:
        ctrls = [c for c in ct if c in sub.columns and sub[c].std() > 0] + yr_use
        C = sub[ctrls].values.astype(float)
        Z = np.column_stack([np.ones(len(sub)), C, sub['x0'].values.astype(float)])
        if np.linalg.matrix_rank(Z) >= Z.shape[1]:
            break
        if not yr_use: break
        yr_use = yr_use[:-1]

    ctrls = [c for c in ct if c in sub.columns and sub[c].std() > 0] + yr_use
    C = sub[ctrls].values.astype(float)
    ones = np.ones(len(sub))
    x0_v = sub['x0'].values.astype(float)
    exp_v = sub['exp'].values.astype(float)
    y = sub['ws'].values.astype(float)

    Z = np.column_stack([ones, C, x0_v])
    X = np.column_stack([ones, C, exp_v])
    n, k = X.shape

    # IV: X instrumented by X_0
    pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
    exp_hat = Z @ pi
    X_hat = np.column_stack([ones, C, exp_hat])
    b = np.linalg.lstsq(X_hat, y, rcond=None)[0]
    resid = y - X @ b
    s2 = np.sum(resid**2) / (n - k)
    XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
    beta_1_iv = b[-1]
    se_naive_iv = np.sqrt(s2 * XhXh_inv[-1, -1])

    # OLS on X_0
    b_ols = np.linalg.lstsq(Z, y, rcond=None)[0]
    resid_ols = y - Z @ b_ols
    s2_ols = np.sum(resid_ols**2) / (n - k)
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    beta_1_ols = b_ols[-1]
    se_ols = np.sqrt(s2_ols * ZtZ_inv[-1, -1])

    # OLS on X
    b_x = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_x = y - X @ b_x
    s2_x = np.sum(resid_x**2) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_1_x = b_x[-1]
    se_x = np.sqrt(s2_x * XtX_inv[-1, -1])

    beta_2_iv = b1b2 - beta_1_iv
    beta_2_ols = b1b2 - beta_1_ols
    beta_2_x = b1b2 - beta_1_x

    print(f"\n{name}: N={n}")
    print(f"  IV (X by X_0):  beta_1={beta_1_iv:.4f} (SE={se_naive_iv:.4f})")
    print(f"  OLS on X_0:     beta_1={beta_1_ols:.4f} (SE={se_ols:.4f})")
    print(f"  OLS on X:       beta_1={beta_1_x:.4f} (SE={se_x:.4f})")

    # Various Var(beta_2) formulas:
    print(f"\n  Var(beta_2) formulas:")

    # Formula 1: independent b1+b2 and beta_1
    v1 = b1b2_se_paper**2 + se_naive_iv**2
    print(f"  F1 (indep, IV naive): sqrt({b1b2_se_paper:.4f}^2 + {se_naive_iv:.4f}^2) = {np.sqrt(v1):.4f}")

    # Formula 2: just b1+b2_se (ignoring beta_1 uncertainty)
    print(f"  F2 (just b1+b2_se): {b1b2_se_paper:.4f}")

    # Formula 3: Var(b1+b2) - Var_naive(beta_1) (if positive correlation = 1)
    v3 = abs(b1b2_se_paper**2 - se_naive_iv**2)
    print(f"  F3 (diff): sqrt(|{b1b2_se_paper:.4f}^2 - {se_naive_iv:.4f}^2|) = {np.sqrt(v3):.4f}")

    # Formula 4: use OLS SE
    v4 = b1b2_se_paper**2 + se_ols**2
    print(f"  F4 (indep, OLS): sqrt({b1b2_se_paper:.4f}^2 + {se_ols:.4f}^2) = {np.sqrt(v4):.4f}")

    # Formula 5: Var(b1+b2) - Var(beta_1_OLS)
    v5 = abs(b1b2_se_paper**2 - se_ols**2)
    print(f"  F5 (diff OLS): sqrt(|{b1b2_se_paper:.4f}^2 - {se_ols:.4f}^2|) = {np.sqrt(v5):.4f}")

    # Formula 6: OLS X SE
    v6 = abs(b1b2_se_paper**2 - se_x**2)
    print(f"  F6 (diff OLS X): sqrt(|{b1b2_se_paper:.4f}^2 - {se_x:.4f}^2|) = {np.sqrt(v6):.4f}")

    # Formula 7: just the 2SLS naive SE on experience
    print(f"  F7 (just naive IV SE): {se_naive_iv:.4f}")

    # Formula 8: Var_naive_IV / correlation between b1b2 and beta_1
    # gamma_X0T from paper is -0.25
    gamma_x0t = -0.25
    v8 = b1b2_se_paper**2 * (1 - gamma_x0t)**2
    print(f"  F8 (b1b2_se * (1-gamma)): sqrt({b1b2_se_paper:.4f}^2 * {(1-gamma_x0t)**2:.4f}) = {np.sqrt(v8):.4f}")

    # Formula 9: (1-gamma)*b1b2_se (different interpretation)
    v9 = b1b2_se_paper * abs(1-gamma_x0t)
    print(f"  F9 ((1-gamma)*b1b2_se): {v9:.4f}")

    # Formula 10: paper sigma_e / sqrt(N) scaled
    # The SE from step 1 is sigma/sqrt(N_wj * sum_T_t^2)
    # Maybe beta_2_se is related to step 1 SE differently

    # Formula 11: Just half of b1+b2_se
    print(f"  F11 (b1+b2_se / 2): {b1b2_se_paper/2:.4f}")

    print(f"\n  Paper beta_2_se: {'0.0127' if name=='PS' else ('0.0146' if name=='BC_NU' else '0.0147')}")
    paper_b2_se = 0.0127 if name=='PS' else (0.0146 if name=='BC_NU' else 0.0147)

    # Reverse engineer: what variance structure gives paper's SE?
    # Var(beta_2) = Var(b1+b2) + Var(beta_1) - 2*Cov
    # paper_b2_se^2 = b1b2_se^2 + beta_1_se^2 - 2*Cov
    paper_b1_se = 0.0288 if name=='PS' else (0.0342 if name=='BC_NU' else 0.0338)
    cov_needed = (b1b2_se_paper**2 + paper_b1_se**2 - paper_b2_se**2) / 2
    print(f"  Cov needed: {cov_needed:.6f}")
    print(f"  Var(b1+b2): {b1b2_se_paper**2:.6f}")
    print(f"  Var(beta_1): {paper_b1_se**2:.6f}")
    print(f"  Ratio Cov/Var(b1+b2): {cov_needed/b1b2_se_paper**2:.4f}")
    print(f"  This implies G_const ≈ {cov_needed/b1b2_se_paper**2:.4f}")
