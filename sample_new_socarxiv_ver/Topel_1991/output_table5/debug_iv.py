"""Debug IV2SLS for Table 5 subsamples - try manual 2SLS with rank fixes."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
from linearmodels.iv import IV2SLS

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

occ = df['occ_1digit'].copy()
m3 = occ > 9
three = occ[m3]
mapped = pd.Series(0, index=three.index, dtype=int)
mapped[(three>=1)&(three<=195)]=1
mapped[(three>=201)&(three<=245)]=2
mapped[(three>=260)&(three<=395)]=4
mapped[(three>=401)&(three<=580)]=5
mapped[(three>=601)&(three<=695)]=6
mapped[(three>=701)&(three<=785)]=7
mapped[(three>=801)&(three<=824)]=9
mapped[(three>=900)&(three<=965)]=8
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

g2 = -0.004592; g3 = 0.0001846; g4 = -0.00000245
d2 = -0.006051; d3 = 0.0002067; d4 = -0.00000238

ct = ['ed','married','disabled','region_ne','region_nc','region_south']

paper = {'PS': 0.1309, 'BC_NU': 0.1520, 'BC_U': 0.0992}

for name, mask, extra in [
    ('PS', df['occ'].isin([1,2,4,8]), ['union_member']),
    ('BC_NU', (df['occ'].isin([5,6,7]))&(df['job_union']==0), []),
    ('BC_U', (df['occ'].isin([5,6,7]))&(df['job_union']==1), []),
]:
    sub = df[mask].copy()
    b1b2 = paper[name]
    sub['ws'] = (sub['lrw'] - b1b2*sub['ten']
                 - g2*sub['ten']**2 - g3*sub['ten']**3 - g4*sub['ten']**4
                 - d2*sub['exp']**2 - d3*sub['exp']**3 - d4*sub['exp']**4)

    ctrl_use = [c for c in ct + extra if sub[c].std() > 0]

    print(f"\n{'='*60}")
    print(f"{name}: N={len(sub)}, b1b2={b1b2}")
    print(f"{'='*60}")

    # Manual 2SLS without year dummies
    y = sub['ws'].values.astype(float)
    exp_v = sub['exp'].values.astype(float)
    x0_v = sub['x0'].values.astype(float)
    C = sub[ctrl_use].values.astype(float)
    ones = np.ones(len(sub))

    Z = np.column_stack([ones, C, x0_v])
    X = np.column_stack([ones, C, exp_v])
    n, k = X.shape

    pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
    exp_hat = Z @ pi
    r2_fs = 1 - np.sum((exp_v-exp_hat)**2)/np.sum((exp_v-exp_v.mean())**2)

    X_hat = np.column_stack([ones, C, exp_hat])
    b = np.linalg.lstsq(X_hat, y, rcond=None)[0]
    resid = y - X @ b
    s2 = np.sum(resid**2)/(n-k)
    XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
    se = np.sqrt(s2 * XhXh_inv[-1,-1])
    b1 = b[-1]
    b2 = b1b2 - b1
    print(f"Manual 2SLS (no yr dum): beta_1={b1:.4f} (SE={se:.4f}), beta_2={b2:.4f}, FS_R2={r2_fs:.4f}")

    # Manual 2SLS WITH year dummies
    yr_cols = [f'year_{y}' for y in range(1969,1984)]
    yr_use = [c for c in yr_cols if c in sub.columns and sub[c].std() > 0]

    # Check rank with year dummies
    C_yr = sub[ctrl_use + yr_use].values.astype(float)
    Z_yr = np.column_stack([ones, C_yr, x0_v])
    X_yr = np.column_stack([ones, C_yr, exp_v])

    rank_Z = np.linalg.matrix_rank(Z_yr)
    rank_X = np.linalg.matrix_rank(X_yr)
    print(f"With year dums: rank_Z={rank_Z}/{Z_yr.shape[1]}, rank_X={rank_X}/{X_yr.shape[1]}")

    # Drop year dummies until full rank
    while np.linalg.matrix_rank(np.column_stack([ones, sub[ctrl_use + yr_use].values.astype(float), x0_v])) < len(ctrl_use) + len(yr_use) + 2:
        if len(yr_use) == 0:
            break
        yr_use = yr_use[:-1]

    C_yr2 = sub[ctrl_use + yr_use].values.astype(float)
    Z_yr2 = np.column_stack([ones, C_yr2, x0_v])
    X_yr2 = np.column_stack([ones, C_yr2, exp_v])
    n2, k2 = X_yr2.shape

    pi2 = np.linalg.lstsq(Z_yr2, exp_v, rcond=None)[0]
    exp_hat2 = Z_yr2 @ pi2
    r2_fs2 = 1 - np.sum((exp_v-exp_hat2)**2)/np.sum((exp_v-exp_v.mean())**2)

    X_hat2 = np.column_stack([ones, C_yr2, exp_hat2])
    b2_coef = np.linalg.lstsq(X_hat2, y, rcond=None)[0]
    resid2 = y - X_yr2 @ b2_coef
    s2_2 = np.sum(resid2**2)/(n2-k2)
    XhXh_inv2 = np.linalg.inv(X_hat2.T @ X_hat2)
    se2 = np.sqrt(s2_2 * XhXh_inv2[-1,-1])
    b1_2 = b2_coef[-1]
    b2_2 = b1b2 - b1_2
    print(f"Manual 2SLS (yr_use={len(yr_use)}): beta_1={b1_2:.4f} (SE={se2:.4f}), beta_2={b2_2:.4f}, FS_R2={r2_fs2:.4f}")

    # Also try linearmodels IV2SLS with reduced controls
    try:
        exog = sm.add_constant(sub[ctrl_use].astype(float))
        dep_s = sub['ws']
        endog_s = sub[['exp']]
        instr_s = sub[['x0']]
        iv = IV2SLS(dep_s, exog, endog_s, instr_s).fit(cov_type='unadjusted')
        print(f"linearmodels IV (no yr): beta_1={iv.params['exp']:.4f} (SE={iv.std_errors['exp']:.4f})")
    except Exception as e:
        print(f"linearmodels IV (no yr): FAILED - {e}")

    # Cumulative returns
    for T in [5, 10, 15, 20]:
        cum = b2 * T + g2 * T**2 + g3 * T**3 + g4 * T**4
        cum2 = b2_2 * T + g2 * T**2 + g3 * T**3 + g4 * T**4
        print(f"  cum{T}: no_yr={cum:.4f}, with_yr={cum2:.4f}")

print("\nPaper targets:")
print("PS:    beta_1=0.0707, beta_2=0.0601, cum5=0.1887, cum10=0.2400")
print("BC_NU: beta_1=0.1066, beta_2=0.0513, cum5=0.1577, cum10=0.2073")
print("BC_U:  beta_1=0.0592, beta_2=0.0399, cum5=0.1401, cum10=0.2033")
