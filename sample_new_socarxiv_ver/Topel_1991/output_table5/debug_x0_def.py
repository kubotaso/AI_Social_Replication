"""Test different definitions of initial experience X_0."""
import pandas as pd
import numpy as np
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
for c in ['married','disabled','union_member','region_ne','region_nc','region_south']:
    df[c] = df[c].fillna(0)

g2=-0.004592; g3=0.0001846; g4=-0.00000245
d2=-0.006051; d3=0.0002067; d4=-0.00000238

# Different X_0 definitions:
# 1. x0 = exp - ten (standard)
df['x0_1'] = (df['exp'] - df['ten']).clip(lower=0)

# 2. x0 = age - ed - 6 - ten (same thing, different route)
df['x0_2'] = (df['age'] - df['ed'] - 6 - df['ten']).clip(lower=0)

# 3. x0 = first observed experience on this job
first_exp = df.groupby(['person_id','job_id'])['exp'].first().reset_index()
first_exp.columns = ['person_id','job_id','x0_3']
df = df.merge(first_exp, on=['person_id','job_id'], how='left')
# Adjust: at job start, exp = x0 + 0 (ten=1 means started last year)
# So x0 = exp at first year of job - ten at first year + 1? No...
# Actually x0 is experience BEFORE starting the current job
# If tenure is 1, then x0 = exp - 1 (had exp-1 years before starting this 1-year-old job)
# First year x0 should be exp - ten = same as definition 1

# 4. x0 from the 'experience' column in original data (not our construction)
# Check if there's a separate experience column
print("Columns with 'exp':", [c for c in df.columns if 'exp' in c.lower()])
print(f"\nOur exp vs original 'experience':")
print(f"  Our exp: mean={df['exp'].mean():.2f}, std={df['exp'].std():.2f}")
if 'experience' in df.columns:
    print(f"  Original experience: mean={df['experience'].mean():.2f}, std={df['experience'].std():.2f}")
    print(f"  Correlation: {df['exp'].corr(df['experience']):.4f}")
    print(f"  Same values: {(df['exp'] == df['experience']).mean():.4f}")
    diff = df['exp'] - df['experience']
    print(f"  Diff: mean={diff.mean():.2f}, std={diff.std():.2f}, min={diff.min()}, max={diff.max()}")

    # Try using original experience column
    df['x0_orig'] = (df['experience'] - df['ten']).clip(lower=0)

    yr = [f'year_{y}' for y in range(1969,1984)]

    for name, mask, extra, b1b2 in [
        ('PS', df['occ'].isin([1,2,4,8]), ['union_member'], 0.1309),
        ('BC_NU', (df['occ'].isin([5,6,7]))&(df['job_union']==0), [], 0.1520),
        ('BC_U', (df['occ'].isin([5,6,7]))&(df['job_union']==1), [], 0.0992),
    ]:
        sub = df[mask].copy()
        sub['ws'] = (sub['lrw'] - b1b2*sub['ten']
                     - g2*sub['ten']**2 - g3*sub['ten']**3 - g4*sub['ten']**4
                     - d2*sub['exp']**2 - d3*sub['exp']**3 - d4*sub['exp']**4)

        ct = ['ed','married','disabled','region_ne','region_nc','region_south'] + extra
        yr_use = [c for c in yr if c in sub.columns and sub[c].std() > 0]
        ctrls = [c for c in ct if c in sub.columns and sub[c].std() > 0] + yr_use

        # Trim for rank
        while True:
            C = sub[ctrls].values.astype(float)
            Z = np.column_stack([np.ones(len(sub)), C, sub['x0_1'].values.astype(float)])
            if np.linalg.matrix_rank(Z) >= Z.shape[1]:
                break
            if not yr_use: break
            yr_use = yr_use[:-1]
            ctrls = [c for c in ct if c in sub.columns and sub[c].std() > 0] + yr_use

        C = sub[ctrls].values.astype(float)
        ones = np.ones(len(sub))
        y = sub['ws'].values.astype(float)

        results = {}
        for x0_name, x0_col in [('x0_standard', 'x0_1'), ('x0_orig', 'x0_orig')]:
            x0_v = sub[x0_col].values.astype(float)
            exp_v = sub['exp'].values.astype(float)

            Z = np.column_stack([ones, C, x0_v])
            X = np.column_stack([ones, C, exp_v])

            pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
            exp_hat = Z @ pi
            X_hat = np.column_stack([ones, C, exp_hat])
            b = np.linalg.lstsq(X_hat, y, rcond=None)[0]
            beta_1 = b[-1]
            results[x0_name] = beta_1

        print(f"\n{name} (N={len(sub)}):")
        for k, v in results.items():
            print(f"  {k}: beta_1={v:.4f}, beta_2={b1b2-v:.4f}")
        print(f"  Paper: beta_1={'0.0707' if name=='PS' else ('0.1066' if name=='BC_NU' else '0.0592')}")
