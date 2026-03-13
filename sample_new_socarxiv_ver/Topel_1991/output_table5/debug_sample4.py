"""Deep investigation of sample selection and BC beta_1 differentiation."""
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
occ = df['occ_1digit'].copy()
print(f"Total rows: {len(df)}")
print(f"occ_1digit value counts:")
print(df['occ_1digit'].value_counts().sort_index().head(20))
print()

# How many have 3-digit codes?
m3 = occ > 9
print(f"1-digit codes (<=9): {(~m3).sum()}")
print(f"3-digit codes (>9): {m3.sum()}")
print()

# Map 3-digit
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

print("After mapping, occ value counts:")
print(df['occ'].value_counts().sort_index())
print()

# Check occ=0 - what are they?
zero_mask = df['occ'] == 0
print(f"occ=0: {zero_mask.sum()} rows")
if zero_mask.any():
    print(f"  occ_1digit values for occ=0:")
    print(f"  {df.loc[zero_mask, 'occ_1digit'].value_counts().sort_index().head(20)}")
    # Check if any of these have 3-digit codes that weren't mapped
    zero_3digit = df.loc[zero_mask & (df['occ_1digit'] > 9), 'occ_1digit']
    if len(zero_3digit) > 0:
        print(f"\n  3-digit codes that mapped to 0 (unmapped ranges):")
        print(f"  {zero_3digit.value_counts().sort_index().head(40)}")
        print(f"  Min: {zero_3digit.min()}, Max: {zero_3digit.max()}")
        # Check if 246-259 range exists (managers not in 201-245)
        print(f"\n  Codes 246-259: {((zero_3digit >= 246) & (zero_3digit <= 259)).sum()}")
        print(f"  Codes 580-600: {((zero_3digit >= 580) & (zero_3digit <= 600)).sum()}")
        print(f"  Codes 696-700: {((zero_3digit >= 696) & (zero_3digit <= 700)).sum()}")
        print(f"  Codes 786-799: {((zero_3digit >= 786) & (zero_3digit <= 799)).sum()}")
        print(f"  Codes 825-899: {((zero_3digit >= 825) & (zero_3digit <= 899)).sum()}")
        print(f"  Codes 966-999: {((zero_3digit >= 966) & (zero_3digit <= 999)).sum()}")

print()

# Now apply restrictions
df = df.sort_values(['person_id','job_id','year'])
ju = df.groupby(['person_id','job_id'])['union_member'].agg(
    lambda x: (x.mean()>0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member':'job_union'})
df = df.merge(ju, on=['person_id','job_id'], how='left')

# Basic restrictions
df_base = df[df['ten']>=1].dropna(subset=['log_hourly_wage','lrw']).copy()
df_base = df_base[(df_base['self_employed']==0)|(df_base['self_employed'].isna())].copy()

print(f"After tenure>=1, has wage, not self-employed: {len(df_base)}")
print()

# Paper says footnote 18: "Jobs were categorized as 'union' if the respondent indicated
# union membership in more than half of the years of the job."
# This is job-level union. Let's also try person-year level union.

# Check different union definitions
for uname, ucol in [('job_union', 'job_union'), ('union_member', 'union_member')]:
    print(f"\nUsing {uname} for union split:")

    # BC workers
    bc = df_base[df_base['occ'].isin([5,6,7])].copy()
    print(f"  BC total (occ 5,6,7): {len(bc)}")

    if ucol in bc.columns:
        bc_nu = bc[bc[ucol] == 0]
        bc_u = bc[bc[ucol] == 1]
        bc_na = bc[bc[ucol].isna()]
        print(f"  BC nonunion ({ucol}==0): {len(bc_nu)}")
        print(f"  BC union ({ucol}==1): {len(bc_u)}")
        print(f"  BC missing ({ucol} is NaN): {len(bc_na)}")

# Try using union_member directly (person-year) instead of job-level
# This might give different sample sizes

# Also check: what if we DON'T exclude occ=0,3,9?
print(f"\n\nCheck if including occ=3 (farmers) matters:")
df_all_occ = df_base.copy()
print(f"  occ=3: {(df_all_occ['occ']==3).sum()}")
print(f"  occ=9: {(df_all_occ['occ']==9).sum()}")
print(f"  occ=0 (unmapped): {(df_all_occ['occ']==0).sum()}")

# What if some occ=0 workers should be BC?
# Let's look at what 3-digit codes mapped to 0
zero_3d = df_base.loc[(df_base['occ']==0) & (df_base['occ_1digit']>9), 'occ_1digit']
print(f"\n  Unmapped 3-digit codes in base sample: {len(zero_3d)}")
if len(zero_3d) > 0:
    print(f"  Value counts:")
    vc = zero_3d.value_counts().sort_index()
    print(f"  {vc.to_string()}")

# Try using person-year union_member instead of job_union
print("\n\n=== TRY: person-year union with BC occupations ===")
bc_all = df_base[df_base['occ'].isin([5,6,7])].copy()
bc_nu_yr = bc_all[bc_all['union_member']==0]
bc_u_yr = bc_all[bc_all['union_member']==1]
print(f"BC nonunion (year-level): {len(bc_nu_yr)}")
print(f"BC union (year-level): {len(bc_u_yr)}")

# Compare with job-level
bc_nu_job = bc_all[bc_all['job_union']==0]
bc_u_job = bc_all[bc_all['job_union']==1]
print(f"BC nonunion (job-level): {len(bc_nu_job)}")
print(f"BC union (job-level): {len(bc_u_job)}")

# Try: what if we also include occ=0 workers in BC?
# Some codes like 246-259 might be clerical workers that belong in PS
# But codes in 580-600, 696-700, 786-799 might be BC-adjacent

# Let's see the occupation distribution with the 1-digit codes
print(f"\n\n=== 1-digit occ codes ===")
one_digit = df_base[df_base['occ_1digit'] <= 9]
print(f"1-digit occ in base sample: {len(one_digit)}")
print(one_digit['occ_1digit'].value_counts().sort_index())

# Paper says: "I finesse issues of promotion and the like by categorizing all periods
# of a job on the basis of the reported occupation in its first observed period."
# This means: for each (person, job), use the FIRST year's occupation for ALL years of that job
print("\n\n=== FIRST-YEAR OCCUPATION ASSIGNMENT ===")
first_occ = df.sort_values(['person_id','job_id','year']).groupby(['person_id','job_id'])['occ'].first()
first_occ.name = 'occ_first'
df_base2 = df_base.drop(columns=['occ'], errors='ignore').merge(
    first_occ.reset_index(), on=['person_id','job_id'], how='left')
df_base2['occ'] = df_base2['occ_first'].fillna(df_base['occ'].values if len(df_base) == len(df_base2) else 0)

# Actually let's be more careful
df_sorted = df.sort_values(['person_id','job_id','year'])
first_occ = df_sorted.groupby(['person_id','job_id']).first()[['occ']].rename(columns={'occ':'occ_first'})
df_base3 = df_base.merge(first_occ, on=['person_id','job_id'], how='left')
df_base3['occ_first'] = df_base3['occ_first'].fillna(df_base3['occ'])

print(f"With first-year occupation:")
for cat, codes in [('PS', [1,2,4,8]), ('BC', [5,6,7])]:
    n_current = df_base3[df_base3['occ'].isin(codes)].shape[0]
    n_first = df_base3[df_base3['occ_first'].isin(codes)].shape[0]
    print(f"  {cat}: current occ = {n_current}, first-year occ = {n_first}")

# Use first-year occupation
df_fo = df_base3.copy()
df_fo = df_fo[~df_fo['occ_first'].isin([0,3,9])].copy()
print(f"\nWith first-year occ, excluding 0,3,9: {len(df_fo)}")

bc_fo = df_fo[df_fo['occ_first'].isin([5,6,7])]
print(f"BC (first-year occ): {len(bc_fo)}")
bc_fo_nu = bc_fo[bc_fo['job_union']==0]
bc_fo_u = bc_fo[bc_fo['job_union']==1]
print(f"  Nonunion: {len(bc_fo_nu)}")
print(f"  Union: {len(bc_fo_u)}")

ps_fo = df_fo[df_fo['occ_first'].isin([1,2,4,8])]
print(f"PS (first-year occ): {len(ps_fo)}")
