"""Compute cumulative return SEs using different approaches."""
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

df = df[df['ten']>=1].dropna(subset=['log_hourly_wage']).copy()
df = df[~df['occ'].isin([0,3,9])].copy()
df = df[(df['self_employed']==0)|(df['self_employed'].isna())].copy()
for c in ['married','disabled','union_member','region_ne','region_nc','region_south']:
    df[c] = df[c].fillna(0)
yr = [f'year_{y}' for y in range(1969,1984)]

for name, mask, extra in [
    ('PS', df['occ'].isin([1,2,4,8]), ['union_member']),
]:
    sub = df[mask].copy()

    # Step 1: within-job wage growth
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

    print(f"\n{name}: Step 1 V1 diagonal:")
    for j, p in enumerate(s1_present):
        print(f"  {p}: coef={m1.params[p]:.6f}, SE={m1.bse[p]:.6f}, Var={V1[j,j]:.10f}")

    b1b2 = m1.params['const']
    b1b2_se = m1.bse['const']

    g2 = m1.params.get('dt2', 0)
    g3 = m1.params.get('dt3', 0)
    g4 = m1.params.get('dt4', 0)

    # Cumulative return = beta_2*T + g2*T^2 + g3*T^3 + g4*T^4
    # = (b1b2 - beta_1)*T + g2*T^2 + g3*T^3 + g4*T^4
    # = b1b2*T - beta_1*T + g2*T^2 + g3*T^3 + g4*T^4
    # All from step 1: b1b2*T + g2*T^2 + g3*T^3 + g4*T^4 (just step 1 params)
    # Minus: beta_1*T (from step 2)

    # The step 1 contribution to cumulative return at T:
    # c1(T) = b1b2*T + g2*T^2 + g3*T^3 + g4*T^4
    # gradient w.r.t. [b1b2, g2, g3, g4]:
    # [T, T^2, T^3, T^4]

    # The step 2 contribution: -beta_1*T
    # Var(-beta_1*T) = T^2 * Var(beta_1_MT)

    # If step 1 and step 2 are INDEPENDENT:
    # Var(cum) = grad1 @ V1_sub @ grad1 + T^2 * Var(beta_1_MT)
    # where V1_sub is the submatrix for [const, dt2, dt3, dt4]

    beta_1_se_mt = 0.0289  # our Murphy-Topel SE
    beta_2_se_approx = b1b2_se / 2  # approximate formula

    for T_val in [5, 10, 15, 20]:
        # Method A: independent step 1 + step 2
        grad_s1 = np.zeros(len(s1_present))
        for j, p in enumerate(s1_present):
            if p == 'const': grad_s1[j] = T_val
            elif p == 'dt2': grad_s1[j] = T_val**2
            elif p == 'dt3': grad_s1[j] = T_val**3
            elif p == 'dt4': grad_s1[j] = T_val**4
        var_s1 = grad_s1 @ V1 @ grad_s1
        var_s2 = T_val**2 * beta_1_se_mt**2
        var_A = var_s1 + var_s2
        # But this double-counts the b1b2 contribution:
        # var_s1 includes T^2*Var(b1b2) and var_s2 includes T^2*Var(beta_1) which includes b1b2 via M-T

        # Method B: step 1 only (ignore step 2 uncertainty)
        var_B = var_s1

        # Method C: only beta_2*T part + gamma*T^2,3,4 part
        # Var(cum) = T^2*Var(beta_2) + gamma part
        gamma_idx = [j for j, p in enumerate(s1_present) if p in ['dt2','dt3','dt4']]
        grad_gamma = np.array([T_val**int(s1_present[j][-1]) for j in gamma_idx])
        V1_gamma = V1[np.ix_(gamma_idx, gamma_idx)]
        var_gamma = grad_gamma @ V1_gamma @ grad_gamma if len(gamma_idx) > 0 else 0

        var_C = T_val**2 * beta_2_se_approx**2 + var_gamma

        # Method D: use paper's exact beta_2_se and our gamma V
        paper_b2_se = 0.0127
        var_D = T_val**2 * paper_b2_se**2 + var_gamma

        # Method E: Murphy-Topel corrected step 1 SE
        # Var(cum) = grad_full @ V1_full @ grad_full
        # where grad includes the effect through beta_1
        # cum = b1b2*T - beta_1*T + g2*T^2 + g3*T^3 + g4*T^4
        # = (1 - G_const)*T * b1b2 + T^2*(g2 - G_g2*b1b2) + ... (complicated)

        print(f"\n  cum{T_val}: paper_cum_se = {'0.0388' if T_val==5 else ('0.0560' if T_val==10 else ('0.0656' if T_val==15 else '0.0663'))}")
        print(f"    Method A (indep s1+s2): {np.sqrt(var_A):.4f}")
        print(f"    Method B (s1 only):     {np.sqrt(var_B):.4f}")
        print(f"    Method C (b2_se_approx + gamma): {np.sqrt(var_C):.4f}")
        print(f"    Method D (paper b2_se + gamma): {np.sqrt(var_D):.4f}")
        print(f"    var_gamma={var_gamma:.8f}, sqrt={np.sqrt(var_gamma):.4f}")
        print(f"    T*beta_2_se_approx = {T_val*beta_2_se_approx:.4f}")
        print(f"    T*paper_b2_se = {T_val*paper_b2_se:.4f}")
