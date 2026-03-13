"""Debug step 1 for BC subsamples."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/psid_panel.csv')

# Prep
em = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17}
df['ed'] = df['education_clean'].map(em).fillna(12)
df['exp'] = (df['age'] - df['ed'] - 6).clip(lower=0)
df['ten'] = df['tenure_topel']

cps = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,
       1974:1.167,1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,
       1980:1.128,1981:1.109,1982:1.103,1983:1.089}
df['lw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(cps))

occ = df['occ_1digit'].copy()
m3 = occ > 9
t = occ[m3]
mp = pd.Series(0, index=t.index, dtype=int)
mp[(t>=1)&(t<=195)]=1; mp[(t>=201)&(t<=245)]=2; mp[(t>=260)&(t<=395)]=4
mp[(t>=401)&(t<=580)]=5; mp[(t>=601)&(t<=695)]=6; mp[(t>=701)&(t<=785)]=7
mp[(t>=801)&(t<=824)]=9; mp[(t>=900)&(t<=965)]=8
occ[m3] = mp
df['occ'] = occ

# Job-level union
df = df.sort_values(['person_id', 'job_id', 'year'])
ju = df.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
df = df.merge(ju, on=['person_id', 'job_id'], how='left')

df = df[df['ten'] >= 1].dropna(subset=['log_hourly_wage']).copy()
df = df[~df['occ'].isin([0, 3, 9])].copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

# Within-job first differences
df = df.sort_values(['person_id', 'job_id', 'year'])
df['within'] = ((df['person_id'] == df['person_id'].shift(1)) &
                (df['job_id'] == df['job_id'].shift(1)) &
                (df['year'] - df['year'].shift(1) == 1))
df['d_lw'] = df['lw_cps'] - df['lw_cps'].shift(1)
df['d_ten'] = df['ten'] - df['ten'].shift(1)

fd = df[df['within']].copy()
fd = fd.dropna(subset=['d_lw'])

print("Full sample FD stats:")
print(f"  N = {len(fd)}")
print(f"  Mean d_lw = {fd['d_lw'].mean():.4f}")
print(f"  d_ten distribution:")
print(fd['d_ten'].value_counts().sort_index().to_string())

# Filter to d_ten == 1 (standard within-job)
fd1 = fd[fd['d_ten'] == 1].copy()
print(f"\n  FD with d_ten==1: N={len(fd1)}, Mean d_lw={fd1['d_lw'].mean():.4f}")

# By subsample
for name, mask in [('PS (1,2,4,8)', fd1['occ'].isin([1,2,4,8])),
                    ('BC_NU (5,6,7, union==0)', (fd1['occ'].isin([5,6,7])) & (fd1['job_union']==0)),
                    ('BC_U (5,6,7, union==1)', (fd1['occ'].isin([5,6,7])) & (fd1['job_union']==1)),
                    ('BC all', fd1['occ'].isin([5,6,7]))]:
    sub = fd1[mask]
    print(f"\n  {name}: N_fd={len(sub)}, Mean d_lw={sub['d_lw'].mean():.4f}")

    # Quick step 1 regression
    T = sub['ten'].values; Tp = T - 1
    X = sub['exp'].values; Xp = X - 1

    dt2 = T**2 - Tp**2; dt3 = T**3 - Tp**3; dt4 = T**4 - Tp**4
    dx2 = X**2 - Xp**2; dx3 = X**3 - Xp**3; dx4 = X**4 - Xp**4

    rhs = pd.DataFrame({
        'const': 1, 'dt2': dt2, 'dt3': dt3, 'dt4': dt4,
        'dx2': dx2, 'dx3': dx3, 'dx4': dx4
    }, index=sub.index)

    m = sm.OLS(sub['d_lw'], rhs).fit()
    print(f"    b1+b2 (intercept) = {m.params['const']:.4f} (SE={m.bse['const']:.4f})")
    print(f"    g2={m.params['dt2']:.6f}")

# Check what the mean d_lw is by tenure level for BC
print("\n\nMean d_lw by tenure, BC nonunion:")
bc_nu = fd1[fd1['occ'].isin([5,6,7]) & (fd1['job_union']==0)]
for t in range(1, 15):
    sub_t = bc_nu[bc_nu['ten'] == t]
    if len(sub_t) > 5:
        print(f"  T={t}: N={len(sub_t)}, Mean d_lw={sub_t['d_lw'].mean():.4f}")

print("\nMean d_lw by tenure, BC union:")
bc_u = fd1[fd1['occ'].isin([5,6,7]) & (fd1['job_union']==1)]
for t in range(1, 15):
    sub_t = bc_u[bc_u['ten'] == t]
    if len(sub_t) > 5:
        print(f"  T={t}: N={len(sub_t)}, Mean d_lw={sub_t['d_lw'].mean():.4f}")

# Compare: what if we use log_hourly_wage instead of CPS-adjusted?
print("\n\nUsing log_hourly_wage directly:")
df['d_lhw'] = df['log_hourly_wage'] - df['log_hourly_wage'].shift(1)
fd_raw = df[df['within']].dropna(subset=['d_lhw']).copy()
fd_raw1 = fd_raw[fd_raw['d_ten'] == 1]
for name, mask in [('PS', fd_raw1['occ'].isin([1,2,4,8])),
                    ('BC_NU', (fd_raw1['occ'].isin([5,6,7])) & (fd_raw1['job_union']==0)),
                    ('BC_U', (fd_raw1['occ'].isin([5,6,7])) & (fd_raw1['job_union']==1))]:
    sub = fd_raw1[mask]
    print(f"  {name}: Mean d_lhw={sub['d_lhw'].mean():.4f}")

# What if we filter d_ten differently? Maybe some obs have d_ten > 1
print("\n\nFD with d_ten != 1:")
fd_other = fd[fd['d_ten'] != 1]
print(f"  N = {len(fd_other)}")
print(f"  d_ten values: {fd_other['d_ten'].value_counts().sort_index().to_string()}")
