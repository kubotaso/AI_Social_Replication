"""Test subsample-specific b1+b2 with paper's higher-order terms."""
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

df['lw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(cps))
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

# Paper's higher-order terms
g2 = -0.004592; g3 = 0.0001846; g4 = -0.00000245
d2 = -0.006051; d3 = 0.0002067; d4 = -0.00000238

# Build FD data
df = df.sort_values(['person_id', 'job_id', 'year'])
df['within'] = ((df['person_id'] == df['person_id'].shift(1)) &
                (df['job_id'] == df['job_id'].shift(1)) &
                (df['year'] - df['year'].shift(1) == 1))
df['d_lw'] = df['lw_cps'] - df['lw_cps'].shift(1)

for k in [2, 3, 4]:
    df[f'd_t{k}'] = df['ten']**k - (df['ten'] - 1)**k
    df[f'd_x{k}'] = df['exp']**k - (df['exp'] - 1)**k

fd = df[df['within']].dropna(subset=['d_lw']).copy()

# Year dummies differenced
yr_cols = sorted([c for c in fd.columns if c.startswith('year_') and c != 'year'])
d_yr = []
for yc in yr_cols:
    dc = f'd_{yc}'
    fd[dc] = fd[yc].astype(float) - fd[yc].shift(1).astype(float)
    fd[dc] = fd[dc].fillna(0)
    if fd[dc].std() > 0:
        d_yr.append(dc)
if len(d_yr) > 1:
    d_yr = d_yr[1:]

yr = [f'year_{y}' for y in range(1969, 1984)]
ct = ['ed', 'married', 'disabled', 'lives_in_smsa', 'region_ne', 'region_nc', 'region_south']

print("=== Approach: Constrained Step 1 (paper HO terms) + OLS Step 2 ===\n")

for sub_name, fd_mask, lev_mask, extra in [
    ('PS',    fd['occ'].isin([1,2,4,8]),
              df['occ'].isin([1,2,4,8]),
              ['union_member']),
    ('BC_NU', (fd['occ'].isin([5,6,7])) & (fd['job_union']==0),
              (df['occ'].isin([5,6,7])) & (df['job_union']==0),
              []),
    ('BC_U',  (fd['occ'].isin([5,6,7])) & (fd['job_union']==1),
              (df['occ'].isin([5,6,7])) & (df['job_union']==1),
              []),
]:
    # Step 1: Constrained -- subtract paper HO terms, estimate only intercept
    sub_fd = fd[fd_mask].copy()
    adj_y = (sub_fd['d_lw']
             - g2 * sub_fd['d_t2']
             - g3 * sub_fd['d_t3']
             - g4 * sub_fd['d_t4']
             - d2 * sub_fd['d_x2']
             - d3 * sub_fd['d_x3']
             - d4 * sub_fd['d_x4'])

    # Regress adjusted FD on year dummies only -> intercept = b1+b2
    d_yr_sub = [c for c in d_yr if c in sub_fd.columns and sub_fd[c].std() > 0]
    X_s1 = sm.add_constant(sub_fd[d_yr_sub].astype(float))
    m_s1 = sm.OLS(adj_y, X_s1).fit()
    b1b2 = m_s1.params['const']
    b1b2_se = m_s1.bse['const']

    # Step 2: w* with sub-specific b1b2 + paper HO terms
    sub_lev = df[lev_mask].copy()
    sub_lev['ws'] = (sub_lev['lrw'] - b1b2 * sub_lev['ten']
                     - g2 * sub_lev['ten']**2 - g3 * sub_lev['ten']**3 - g4 * sub_lev['ten']**4
                     - d2 * sub_lev['exp']**2 - d3 * sub_lev['exp']**3 - d4 * sub_lev['exp']**4)

    X2 = sm.add_constant(sub_lev[['x0'] + ct + extra + yr].astype(float))
    y2 = sub_lev['ws'].astype(float)
    m2 = sm.OLS(y2, X2).fit()
    beta_1 = m2.params['x0']
    beta_1_se = m2.bse['x0']
    beta_2 = b1b2 - beta_1

    cums = {}
    for T in [5, 10, 15, 20]:
        cums[T] = beta_2 * T + g2 * T**2 + g3 * T**3 + g4 * T**4

    print(f"{sub_name}: N_fd={len(sub_fd)}, b1+b2={b1b2:.4f} ({b1b2_se:.4f})")
    print(f"  N={len(sub_lev)}, beta_1={beta_1:.4f} ({beta_1_se:.4f}), beta_2={beta_2:.4f}")
    print(f"  cum5={cums[5]:.4f}, cum10={cums[10]:.4f}, cum15={cums[15]:.4f}, cum20={cums[20]:.4f}")

print("\nPaper targets:")
print("PS:    b1b2=0.1309, beta_1=0.0707, beta_2=0.0601, N=4946")
print("       cum5=0.1887, cum10=0.2400, cum15=0.2527, cum20=0.2841")
print("BC_NU: b1b2=0.1520, beta_1=0.1066, beta_2=0.0513, N=2642")
print("       cum5=0.1577, cum10=0.2073, cum15=0.2480, cum20=0.3295")
print("BC_U:  b1b2=0.0992, beta_1=0.0592, beta_2=0.0399, N=2741")
print("       cum5=0.1401, cum10=0.2033, cum15=0.2384, cum20=0.2733")

# Now try: completely use paper's step 1 per subsample
print("\n\n=== Using PAPER'S subsample-specific b1+b2 values ===\n")
paper_b1b2 = {'PS': 0.1309, 'BC_NU': 0.1520, 'BC_U': 0.0992}

for sub_name, lev_mask, extra in [
    ('PS', df['occ'].isin([1,2,4,8]), ['union_member']),
    ('BC_NU', (df['occ'].isin([5,6,7])) & (df['job_union']==0), []),
    ('BC_U', (df['occ'].isin([5,6,7])) & (df['job_union']==1), []),
]:
    b1b2 = paper_b1b2[sub_name]
    sub_lev = df[lev_mask].copy()
    sub_lev['ws'] = (sub_lev['lrw'] - b1b2 * sub_lev['ten']
                     - g2 * sub_lev['ten']**2 - g3 * sub_lev['ten']**3 - g4 * sub_lev['ten']**4
                     - d2 * sub_lev['exp']**2 - d3 * sub_lev['exp']**3 - d4 * sub_lev['exp']**4)

    X2 = sm.add_constant(sub_lev[['x0'] + ct + extra + yr].astype(float))
    y2 = sub_lev['ws'].astype(float)
    m2 = sm.OLS(y2, X2).fit()
    beta_1 = m2.params['x0']
    beta_2 = b1b2 - beta_1

    cums = {}
    for T in [5, 10, 15, 20]:
        cums[T] = beta_2 * T + g2 * T**2 + g3 * T**3 + g4 * T**4

    print(f"{sub_name}: N={len(sub_lev)}, b1+b2={b1b2:.4f}, beta_1={beta_1:.4f}, "
          f"beta_2={beta_2:.4f}")
    print(f"  cum5={cums[5]:.4f}, cum10={cums[10]:.4f}, cum15={cums[15]:.4f}, cum20={cums[20]:.4f}")
