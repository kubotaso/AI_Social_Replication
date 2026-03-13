"""Compute full-sample step 1 V1 for cumulative return SEs."""
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
df = df[df['ten']>=1].dropna(subset=['log_hourly_wage']).copy()
df = df[~df['occ'].isin([0,3,9])].copy()
df = df[(df['self_employed']==0)|(df['self_employed'].isna())].copy()
yr = [f'year_{y}' for y in range(1969,1984)]

# Full sample step 1
s = df.sort_values(['person_id','job_id','year']).copy()
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

print(f"Full-sample step 1: N={v.sum()}")
s1_params = ['const','dt2','dt3','dt4','dx2','dx3','dx4']
s1_present = [p for p in s1_params if p in m1.params.index]
V1_full = m1.cov_params().loc[s1_present, s1_present].values

print(f"\nFull-sample step 1 V1 diagonal:")
for j, p in enumerate(s1_present):
    print(f"  {p}: coef={m1.params[p]:.6f}, SE={m1.bse[p]:.6f}")

# Paper Table 2 col 3 values:
print(f"\nPaper Table 2 col 3:")
print(f"  const=0.1258, SE=0.0162")
print(f"  dt2=-0.4592e-2, SE=0.1080e-1")
print(f"  dt3=0.1846e-3, SE=0.0526e-2")
print(f"  dt4=-0.0245e-4, SE=0.0079e-3")
print(f"  dx2=-0.6051e-2, SE=0.1546e-1")
print(f"  dx3=0.2067e-3, SE=0.0517e-2")
print(f"  dx4=-0.0238e-4, SE=0.0058e-3")

# Cumulative return SEs using full-sample V1
# With G_const ≈ 1 assumption:
# grad_cum ≈ [0, T^2, T^3, T^4, 0, 0, 0]  (only gamma terms matter)
# But we also need the V_naive from step 2

b2_se = 0.0127  # paper value
b2_se_approx = m1.bse['const'] / 2  # our approximation

gamma_idx = [j for j, p in enumerate(s1_present) if p in ['dt2','dt3','dt4']]
V1_gamma = V1_full[np.ix_(gamma_idx, gamma_idx)]

print(f"\nFull-sample gamma V1:")
for i, j1 in enumerate(gamma_idx):
    for j, j2 in enumerate(gamma_idx):
        print(f"  V({s1_present[j1]},{s1_present[j2]}) = {V1_gamma[i,j]:.10f}")

print(f"\nCumulative return SEs:")
for T in [5, 10, 15, 20]:
    # Method 1: T^2*Var(beta_2) + gamma terms from full-sample V1
    grad_gamma = np.array([T**2, T**3, T**4])
    var_gamma = grad_gamma @ V1_gamma @ grad_gamma
    var_1 = T**2 * b2_se**2 + var_gamma

    # Method 2: same but with our approximation
    var_2 = T**2 * b2_se_approx**2 + var_gamma

    # Method 3: just gamma terms (G_const=1 means beta_2 part cancels)
    var_3 = var_gamma

    # Method 4: scale b1b2_se by sqrt(T)
    b1b2_se_full = m1.bse['const']
    var_4 = T * b1b2_se_full**2

    paper_se = {5: 0.0388, 10: 0.0560, 15: 0.0656, 20: 0.0663}[T]
    print(f"\n  T={T}: paper_SE = {paper_se}")
    print(f"    M1 (paper b2_se + full gamma): {np.sqrt(max(0,var_1)):.4f}")
    print(f"    M2 (approx b2_se + full gamma): {np.sqrt(max(0,var_2)):.4f}")
    print(f"    M3 (just full gamma): {np.sqrt(max(0,var_3)):.4f}")
    print(f"    M4 (sqrt(T)*b1b2_se): {np.sqrt(max(0,var_4)):.4f}")
    print(f"    var_gamma = {var_gamma:.8f}")
