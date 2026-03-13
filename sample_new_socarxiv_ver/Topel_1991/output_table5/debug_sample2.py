"""Debug sample sizes - investigate missing BC workers and alternate groupings."""
import pandas as pd
import numpy as np
df = pd.read_csv('data/psid_panel.csv')
print(f"Raw rows: {len(df)}")

# Check what occ codes we have before remapping
occ_raw = df['occ_1digit'].copy()
print(f"\nocc_1digit range: {occ_raw.min()}-{occ_raw.max()}")
print(f"NaN count: {occ_raw.isna().sum()}")

# Map 3-digit codes
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

# Check unmapped codes (mapped to 0)
unmapped = three[mapped == 0]
if len(unmapped) > 0:
    print(f"\nUnmapped 3-digit codes ({len(unmapped)} rows):")
    print(unmapped.value_counts().sort_index())

# Show occ distribution BEFORE sample restrictions
print(f"\nocc distribution (full data):")
print(df['occ'].value_counts().sort_index())

# Apply MINIMAL sample restrictions
df['ten'] = df['tenure_topel']
pre = df.copy()

# tenure >= 1
df = df[df['ten']>=1].copy()
print(f"\nAfter tenure>=1: {len(df)}")

# Drop missing wages
df = df.dropna(subset=['log_hourly_wage']).copy()
print(f"After drop missing wage: {len(df)}")

# Drop self-employed
df = df[(df['self_employed']==0)|(df['self_employed'].isna())].copy()
print(f"After drop self-employed: {len(df)}")

# What if we DON'T drop occ 0,3,9 yet?
print(f"\nocc distribution (after basic restrictions, before occ filter):")
print(df['occ'].value_counts().sort_index())

# The paper says "professional/service" and "blue-collar"
# Maybe the paper includes occ=0 as something?
# occ=0 could be "not available" or could be miscoded
# Let's check what occ=0 rows look like
occ0 = df[df['occ']==0]
print(f"\nocc=0: {len(occ0)} rows")
if len(occ0) > 0:
    print(f"  occ_1digit original values:")
    print(pre.loc[occ0.index, 'occ_1digit'].value_counts().sort_index().head(20))

# Drop only occ 3,9 (farm and self-employed), keep occ 0
df_keep0 = df[~df['occ'].isin([3,9])].copy()
print(f"\nAfter drop occ 3,9 only: {len(df_keep0)}")
ps0 = df_keep0['occ'].isin([0,1,2,4,8]).sum()
bc0 = df_keep0['occ'].isin([5,6,7]).sum()
print(f"PS (0,1,2,4,8)={ps0}, BC (5,6,7)={bc0}")

# Now drop occ 0 too
df = df[~df['occ'].isin([0,3,9])].copy()
print(f"\nAfter drop occ 0,3,9: {len(df)}")

# Sort and compute job union
df = df.sort_values(['person_id','job_id','year'])
ju = df.groupby(['person_id','job_id'])['union_member'].agg(
    lambda x: (x.mean()>0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member':'job_union'})
df = df.merge(ju, on=['person_id','job_id'], how='left')

# Try treating NaN union as nonunion
bc = df['occ'].isin([5,6,7])
bc_data = df[bc].copy()
print(f"\nBC workers total: {bc.sum()}")
print(f"BC union_member NaN: {bc_data['union_member'].isna().sum()}")
print(f"BC job_union NaN: {bc_data['job_union'].isna().sum()}")

# If we treat NaN as nonunion:
bc_nu_fill = (bc & ((df['job_union']==0) | df['job_union'].isna())).sum()
bc_u_fill = (bc & (df['job_union']==1)).sum()
print(f"\nBC_NU (NaN=nonunion): {bc_nu_fill}, BC_U: {bc_u_fill}")

# Try with year-level union and NaN=nonunion
bc_nu_yr = (bc & ((df['union_member']==0) | df['union_member'].isna())).sum()
bc_u_yr = (bc & (df['union_member']==1)).sum()
print(f"BC_NU (year, NaN=nonunion): {bc_nu_yr}, BC_U (year): {bc_u_yr}")

# What if we include lives_in_smsa restriction?
print(f"\nlives_in_smsa distribution:")
print(df['lives_in_smsa'].value_counts(dropna=False))

# What about govt_worker?
if 'govt_worker' in df.columns:
    print(f"\ngovt_worker distribution:")
    print(df['govt_worker'].value_counts(dropna=False))
    df_no_govt = df[(df['govt_worker']==0)|(df['govt_worker'].isna())].copy()
    print(f"After drop govt: {len(df_no_govt)}")

# What about agriculture?
if 'agriculture' in df.columns:
    print(f"\nagriculture distribution:")
    print(df['agriculture'].value_counts(dropna=False))
