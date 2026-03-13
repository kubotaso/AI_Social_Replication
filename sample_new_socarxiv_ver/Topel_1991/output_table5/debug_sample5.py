"""First-year occupation + various sample restrictions to match paper N."""
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

# Occupation mapping
occ_raw = df['occ_1digit'].copy()
m3 = occ_raw > 9
three = occ_raw[m3]
mapped = pd.Series(0, index=three.index, dtype=int)
mapped[(three>=1)&(three<=195)]=1
mapped[(three>=201)&(three<=245)]=2
mapped[(three>=260)&(three<=395)]=4
mapped[(three>=401)&(three<=580)]=5
mapped[(three>=601)&(three<=695)]=6
mapped[(three>=701)&(three<=785)]=7
mapped[(three>=801)&(three<=824)]=9
mapped[(three>=900)&(three<=965)]=8
occ_raw[m3] = mapped
df['occ'] = occ_raw

# First-year occupation: use first observed year's occ for entire job
df = df.sort_values(['person_id','job_id','year'])
first_occ = df.groupby(['person_id','job_id'])['occ'].first().reset_index()
first_occ.columns = ['person_id','job_id','occ_first']
df = df.merge(first_occ, on=['person_id','job_id'], how='left')

# Job-level union
ju = df.groupby(['person_id','job_id'])['union_member'].agg(
    lambda x: (x.mean()>0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member':'job_union'})
df = df.merge(ju, on=['person_id','job_id'], how='left')

# Basic restrictions
df = df[df['ten']>=1].dropna(subset=['log_hourly_wage','lrw']).copy()
df = df[(df['self_employed']==0)|(df['self_employed'].isna())].copy()

df['x0'] = (df['exp'] - df['ten']).clip(lower=0)
for c in ['married','disabled','union_member','region_ne','region_nc','region_south']:
    df[c] = df[c].fillna(0)

# Try different occupation columns and union definitions
configs = [
    ("current_occ + job_union", 'occ', 'job_union'),
    ("first_occ + job_union", 'occ_first', 'job_union'),
    ("current_occ + union_member", 'occ', 'union_member'),
    ("first_occ + union_member", 'occ_first', 'union_member'),
]

for config_name, occ_col, union_col in configs:
    print(f"\n{'='*70}")
    print(f"Config: {config_name}")
    print(f"{'='*70}")

    sub = df[~df[occ_col].isin([0,3,9])].copy()

    ps_mask = sub[occ_col].isin([1,2,4,8])
    bc_mask = sub[occ_col].isin([5,6,7])

    ps = sub[ps_mask]
    bc = sub[bc_mask]
    bc_nu = bc[bc[union_col]==0]
    bc_u = bc[bc[union_col]==1]

    print(f"  PS: {len(ps)} (paper: 4946)")
    print(f"  BC_NU: {len(bc_nu)} (paper: 2642)")
    print(f"  BC_U: {len(bc_u)} (paper: 2741)")
    print(f"  Total: {len(ps)+len(bc_nu)+len(bc_u)} (paper: 10329)")

    # Quick 2SLS for each group to see if beta_1 differentiates
    for gname, gdata in [('PS', ps), ('BC_NU', bc_nu), ('BC_U', bc_u)]:
        if len(gdata) < 100:
            print(f"  {gname}: too few obs")
            continue

        g = gdata.copy()
        # Use paper b1+b2 for now
        if gname == 'PS': b1b2 = 0.1309
        elif gname == 'BC_NU': b1b2 = 0.1520
        else: b1b2 = 0.0992

        g2=-0.004592; g3=0.0001846; g4=-0.00000245
        d2=-0.006051; d3=0.0002067; d4=-0.00000238

        g['ws'] = (g['lrw'] - b1b2*g['ten']
                   - g2*g['ten']**2 - g3*g['ten']**3 - g4*g['ten']**4
                   - d2*g['exp']**2 - d3*g['exp']**3 - d4*g['exp']**4)

        ct = ['ed','married','disabled','region_ne','region_nc','region_south']
        if gname == 'PS':
            ct.append('union_member')

        yr = [f'year_{y}' for y in range(1969,1984) if f'year_{y}' in g.columns and g[f'year_{y}'].std()>0]
        ctrls = [c for c in ct if c in g.columns and g[c].std()>0] + yr

        # Trim for rank
        while True:
            C = g[ctrls].values.astype(float)
            Z = np.column_stack([np.ones(len(g)), C, g['x0'].values.astype(float)])
            if np.linalg.matrix_rank(Z) >= Z.shape[1]:
                break
            if not yr:
                break
            yr = yr[:-1]
            ctrls = [c for c in ct if c in g.columns and g[c].std()>0] + yr

        C = g[ctrls].values.astype(float)
        ones = np.ones(len(g))
        x0_v = g['x0'].values.astype(float)
        exp_v = g['exp'].values.astype(float)
        y = g['ws'].values.astype(float)

        Z = np.column_stack([ones, C, x0_v])
        X = np.column_stack([ones, C, exp_v])

        pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
        exp_hat = Z @ pi
        X_hat = np.column_stack([ones, C, exp_hat])
        b = np.linalg.lstsq(X_hat, y, rcond=None)[0]
        beta_1 = b[-1]
        beta_2 = b1b2 - beta_1

        print(f"  {gname}: N={len(g)}, beta_1={beta_1:.4f}, beta_2={beta_2:.4f}")

# Also check: what if occ=0 workers are included and treated as BC or PS?
print(f"\n\n{'='*70}")
print("What if occ=0 workers are included?")
print(f"{'='*70}")
occ0 = df[df['occ']==0]
print(f"  occ=0 workers: {len(occ0)}")
print(f"  union status: union={len(occ0[occ0['job_union']==1])}, nonunion={len(occ0[occ0['job_union']==0])}, NA={len(occ0[occ0['job_union'].isna()])}")
print(f"  education: mean={occ0['ed'].mean():.1f}")
print(f"  log_wage: mean={occ0['log_hourly_wage'].mean():.2f}")

# Compare with BC workers
bc_ref = df[df['occ'].isin([5,6,7])]
print(f"\n  BC workers: N={len(bc_ref)}")
print(f"  education: mean={bc_ref['ed'].mean():.1f}")
print(f"  log_wage: mean={bc_ref['log_hourly_wage'].mean():.2f}")

# Compare with PS workers
ps_ref = df[df['occ'].isin([1,2,4,8])]
print(f"\n  PS workers: N={len(ps_ref)}")
print(f"  education: mean={ps_ref['ed'].mean():.1f}")
print(f"  log_wage: mean={ps_ref['log_hourly_wage'].mean():.2f}")
