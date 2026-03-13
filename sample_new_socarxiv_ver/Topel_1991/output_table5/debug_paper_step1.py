"""Test using paper's full-sample step 1 for all subsamples."""
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

cps = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,
       1974:1.167,1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,
       1980:1.128,1981:1.109,1982:1.103,1983:1.089}
gnp = {1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,
       1973:44.4,1974:48.9,1975:53.6,1976:56.9,1977:60.6,1978:65.2,
       1979:72.6,1980:82.4,1981:90.9,1982:100.0}

df['lrw'] = df['log_hourly_wage'] - np.log((df['year']-1).map(gnp)/100.0) - np.log(df['year'].map(cps))

occ = df['occ_1digit'].copy()
m3 = occ > 9
t = occ[m3]
mp = pd.Series(0, index=t.index, dtype=int)
mp[(t>=1)&(t<=195)]=1; mp[(t>=201)&(t<=245)]=2; mp[(t>=260)&(t<=395)]=4
mp[(t>=401)&(t<=580)]=5; mp[(t>=601)&(t<=695)]=6; mp[(t>=701)&(t<=785)]=7
mp[(t>=801)&(t<=824)]=9; mp[(t>=900)&(t<=965)]=8
occ[m3] = mp
df['occ'] = occ

df = df.sort_values(['person_id', 'job_id', 'year'])
ju = df.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
df = df.merge(ju, on=['person_id', 'job_id'], how='left')

df = df[df['ten'] >= 1].dropna(subset=['log_hourly_wage', 'lrw']).copy()
df = df[~df['occ'].isin([0, 3, 9])].copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
df['x0'] = (df['exp'] - df['ten']).clip(lower=0)
for c in ['married','disabled','lives_in_smsa','union_member','region_ne','region_nc','region_south']:
    df[c] = df[c].fillna(0)

# Paper's Table 2/3 step 1 values
b12 = 0.1258
g2 = -0.004592; g3 = 0.0001846; g4 = -0.00000245
d2 = -0.006051; d3 = 0.0002067; d4 = -0.00000238

# w* using paper step 1
df['ws'] = (df['lrw'] - b12 * df['ten']
            - g2 * df['ten']**2 - g3 * df['ten']**3 - g4 * df['ten']**4
            - d2 * df['exp']**2 - d3 * df['exp']**3 - d4 * df['exp']**4)

yr = [f'year_{y}' for y in range(1969, 1984)]
ct = ['ed', 'married', 'disabled', 'lives_in_smsa', 'region_ne', 'region_nc', 'region_south']

# Run step 2 for each subsample
for name, mask, extra_ctrl in [
    ('PS', df['occ'].isin([1,2,4,8]), ['union_member']),
    ('BC_NU', (df['occ'].isin([5,6,7])) & (df['job_union']==0), []),
    ('BC_U', (df['occ'].isin([5,6,7])) & (df['job_union']==1), []),
]:
    sub = df[mask].copy()
    all_ctrl = ct + extra_ctrl + yr
    X = sm.add_constant(sub[['x0'] + all_ctrl].astype(float))
    y = sub['ws'].astype(float)
    m = sm.OLS(y, X).fit()
    beta_1 = m.params['x0']
    beta_1_se = m.bse['x0']
    beta_2 = b12 - beta_1
    # Cumulative returns
    cums = {}
    for T in [5, 10, 15, 20]:
        cums[T] = beta_2 * T + g2 * T**2 + g3 * T**3 + g4 * T**4
    print(f"{name}: N={len(sub)}, beta_1={beta_1:.4f} ({beta_1_se:.4f}), "
          f"beta_2={beta_2:.4f}, "
          f"cum5={cums[5]:.4f}, cum10={cums[10]:.4f}, cum15={cums[15]:.4f}, cum20={cums[20]:.4f}")

print("\nPaper targets:")
print("PS:    beta_1=0.0707, beta_2=0.0601, cum5=0.1887, cum10=0.2400, cum15=0.2527, cum20=0.2841")
print("BC_NU: beta_1=0.1066, beta_2=0.0513 (from sub-specific b1b2=0.1520)")
print("BC_U:  beta_1=0.0592, beta_2=0.0399 (from sub-specific b1b2=0.0992)")

# Now try: use sub-specific b1b2 with paper higher-order terms
print("\n\n=== Using subsample-specific b1+b2 from separate step 1 ===")
# Run step 1 per subsample first
df_sort = df.sort_values(['person_id', 'job_id', 'year'])
df_sort['within'] = ((df_sort['person_id'] == df_sort['person_id'].shift(1)) &
                     (df_sort['job_id'] == df_sort['job_id'].shift(1)) &
                     (df_sort['year'] - df_sort['year'].shift(1) == 1))
df_sort['lw_cps'] = df_sort['log_hourly_wage'] - np.log(df_sort['year'].map(cps))
df_sort['d_lw'] = df_sort['lw_cps'] - df_sort['lw_cps'].shift(1)

for k in [2, 3, 4]:
    df_sort[f'd_t{k}'] = df_sort['ten']**k - (df_sort['ten'] - 1)**k
    df_sort[f'd_x{k}'] = df_sort['exp']**k - (df_sort['exp'] - 1)**k

fd = df_sort[df_sort['within']].dropna(subset=['d_lw']).copy()

# Use paper's higher-order terms as regressors in step 1
# But constrain them to paper values -- only estimate the intercept
# This is like: d_lw - g2*d_T^2 - g3*d_T^3 - g4*d_T^4 - d2*d_X^2 - ... = const + year_dummies + e
# The intercept of this residualized regression gives the subsample b1+b2

for name, mask_fd in [
    ('PS', fd['occ'].isin([1,2,4,8])),
    ('BC_NU', (fd['occ'].isin([5,6,7])) & (fd['job_union']==0)),
    ('BC_U', (fd['occ'].isin([5,6,7])) & (fd['job_union']==1)),
]:
    sub_fd = fd[mask_fd].copy()
    # Subtract paper's higher-order terms from d_lw
    # d_lw = b1b2_sub + g2*d_T^2 + ... (but g2 etc are fixed at paper values)
    adj_y = (sub_fd['d_lw']
             - g2 * sub_fd['d_t2']
             - g3 * sub_fd['d_t3']
             - g4 * sub_fd['d_t4']
             - d2 * sub_fd['d_x2']
             - d3 * sub_fd['d_x3']
             - d4 * sub_fd['d_x4'])

    # Year dummies
    d_yr = []
    for y in range(1969, 1984):
        col = f'year_{y}'
        dc = f'd_{col}'
        sub_fd[dc] = sub_fd[col].astype(float) - sub_fd[col].shift(1).astype(float)
        sub_fd[dc] = sub_fd[dc].fillna(0)
        if sub_fd[dc].std() > 0:
            d_yr.append(dc)
    if len(d_yr) > 1:
        d_yr = d_yr[1:]

    X_s1 = sm.add_constant(sub_fd[d_yr].astype(float))
    m_s1 = sm.OLS(adj_y, X_s1).fit()
    b1b2_sub = m_s1.params['const']
    b1b2_sub_se = m_s1.bse['const']

    # Now use this subsample b1+b2 in step 2
    sub_lev = df[mask_fd.index.isin(df.index) if hasattr(mask_fd, 'index') else True].copy()
    # Actually get the correct subsample from df
    if 'PS' in name:
        sub_lev = df[df['occ'].isin([1,2,4,8])].copy()
    elif 'NU' in name:
        sub_lev = df[(df['occ'].isin([5,6,7])) & (df['job_union']==0)].copy()
    else:
        sub_lev = df[(df['occ'].isin([5,6,7])) & (df['job_union']==1)].copy()

    sub_lev['ws2'] = (sub_lev['lrw'] - b1b2_sub * sub_lev['ten']
                      - g2 * sub_lev['ten']**2 - g3 * sub_lev['ten']**3 - g4 * sub_lev['ten']**4
                      - d2 * sub_lev['exp']**2 - d3 * sub_lev['exp']**3 - d4 * sub_lev['exp']**4)

    extra = ['union_member'] if 'PS' in name else []
    X2 = sm.add_constant(sub_lev[['x0'] + ct + extra + yr].astype(float))
    y2 = sub_lev['ws2'].astype(float)
    m2 = sm.OLS(y2, X2).fit()
    beta_1 = m2.params['x0']
    beta_2 = b1b2_sub - beta_1

    cums = {}
    for T in [5, 10, 15, 20]:
        cums[T] = beta_2 * T + g2 * T**2 + g3 * T**3 + g4 * T**4

    print(f"{name}: N_fd={len(sub_fd)}, b1b2={b1b2_sub:.4f} ({b1b2_sub_se:.4f}), "
          f"N_lev={len(sub_lev)}, beta_1={beta_1:.4f}, beta_2={beta_2:.4f}, "
          f"cum5={cums[5]:.4f}, cum10={cums[10]:.4f}, cum15={cums[15]:.4f}, cum20={cums[20]:.4f}")
