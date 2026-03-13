"""Debug sample sizes and union definitions."""
import pandas as pd
import numpy as np
df = pd.read_csv('data/psid_panel.csv')
occ = df['occ_1digit'].copy()
m3 = occ > 9
if m3.any():
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
df['ten'] = df['tenure_topel']
df = df[df['ten']>=1].copy()
df = df.dropna(subset=['log_hourly_wage']).copy()
df = df[~df['occ'].isin([0,3,9])].copy()
df = df[(df['self_employed']==0)|(df['self_employed'].isna())].copy()
df = df.sort_values(['person_id','job_id','year'])
ju = df.groupby(['person_id','job_id'])['union_member'].agg(
    lambda x: (x.mean()>0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member':'job_union'})
df = df.merge(ju, on=['person_id','job_id'], how='left')

ps = df['occ'].isin([1,2,4,8])
bc = df['occ'].isin([5,6,7])
bc_nu_job = bc & (df['job_union']==0)
bc_u_job = bc & (df['job_union']==1)
bc_nu_yr = bc & (df['union_member']==0)
bc_u_yr = bc & (df['union_member']==1)

print(f"Total after restrictions: {len(df)}")
print(f"PS: {ps.sum()}")
print(f"BC total: {bc.sum()}")
print(f"BC_NU (job-level): {bc_nu_job.sum()}, BC_U (job-level): {bc_u_job.sum()}")
print(f"BC_NU (year-level): {bc_nu_yr.sum()}, BC_U (year-level): {bc_u_yr.sum()}")
print(f"BC with NaN union_member: {(bc & df['union_member'].isna()).sum()}")
print(f"BC with NaN job_union: {(bc & df['job_union'].isna()).sum()}")
print()

# Try different occupation groupings
for codes_ps, codes_bc, label in [
    ([1,2,4,8], [5,6,7], 'Current: PS=1,2,4,8 BC=5,6,7'),
    ([1,2,8], [4,5,6,7], 'Alt1: PS=1,2,8 BC=4,5,6,7'),
    ([1,2,4], [5,6,7,8], 'Alt2: PS=1,2,4 BC=5,6,7,8'),
]:
    n_ps = df['occ'].isin(codes_ps).sum()
    n_bc = df['occ'].isin(codes_bc).sum()
    bc_mask = df['occ'].isin(codes_bc)
    n_bc_nu = (bc_mask & (df['union_member']==0)).sum()
    n_bc_u = (bc_mask & (df['union_member']==1)).sum()
    print(f"{label}: PS={n_ps}, BC_NU={n_bc_nu}, BC_U={n_bc_u}, Total={n_ps+n_bc_nu+n_bc_u}")

# Paper says N total = 10,685 in Table 3
# Table 5 Ns: PS=4946, BC_NU=2642, BC_U=2741
# Sum = 10,329 (less than 10685 by 356)
print(f"\nPaper Ns: PS=4946, BC_NU=2642, BC_U=2741, Sum=10329")
print(f"Table 3 N = 10685")
print(f"Difference = {10685-10329}")

# Try restricting to only persons who appear as household heads
if 'pnum' not in df.columns:
    df['pnum'] = df['person_id'] % 1000
print(f"\npnum value counts:")
print(df['pnum'].value_counts().head(10))

# Try heads only
heads = df[df['pnum'].isin([1, 170])].copy()
print(f"\nHeads only: {len(heads)}")
ps_h = heads['occ'].isin([1,2,4,8]).sum()
bc_h = heads['occ'].isin([5,6,7])
bc_nu_h = (bc_h & (heads['union_member']==0)).sum()
bc_u_h = (bc_h & (heads['union_member']==1)).sum()
print(f"PS={ps_h}, BC_NU={bc_nu_h}, BC_U={bc_u_h}")

# Check with lives_in_smsa as additional control
print(f"\nlives_in_smsa values:")
print(df['lives_in_smsa'].value_counts(dropna=False))
