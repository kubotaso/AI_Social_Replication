"""Test different union definitions to maximize BC_U sample size"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

def map_occ_split(occ):
    if occ <= 9: return str(occ)
    elif 1 <= occ <= 195: return '0'
    elif 201 <= occ <= 245: return '1'
    elif 260 <= occ <= 285: return '4'
    elif 301 <= occ <= 395: return '3'
    elif 401 <= occ <= 580: return '5'
    elif 601 <= occ <= 695: return '6'
    elif 701 <= occ <= 785: return 'L'
    elif 801 <= occ <= 824: return '8'
    elif 900 <= occ <= 965: return 'S'
    else: return '9'

df = pd.read_csv('data/psid_panel.csv')
df['education_years'] = df['education_clean'].map(EDUC_MAP).fillna(12)
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df['tenure'] = df['tenure_topel'].copy()
df = df[df['tenure'] >= 1].copy()
df = df.dropna(subset=['log_hourly_wage']).copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
df['occ_mapped'] = df['occ_1digit'].apply(map_occ_split)
df = df[~df['occ_mapped'].isin(['2', '8', '9'])].copy()

# Build job-level union
df = df.sort_values(['person_id', 'job_id', 'year'])
ju = df.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
df = df.merge(ju, on=['person_id', 'job_id'], how='left')

# Also build "ever union in this job" and "any union in this job"
ju_ever = df.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.max() == 1) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'ever_union'})
df = df.merge(ju_ever, on=['person_id', 'job_id'], how='left')

# Occupation grouping: PS=[1,3,4,7], BC=[0,5,6,L,S]
ps_codes = ['1', '3', '4', '7']
bc_codes = ['0', '5', '6', 'L', 'S']
bc = df[df['occ_mapped'].isin(bc_codes)]

# Test different union definitions for BC
print("Paper targets: BC_NU N=2642, BC_U N=2741")
print()

# 1. Obs-level union_member (current)
nu1 = bc[bc['union_member'] == 0]
u1 = bc[bc['union_member'] == 1]
print(f"1. Obs-level union_member: NU={len(nu1)}, U={len(u1)}, NA={len(bc)-len(nu1)-len(u1)}")
print(f"   NU err={abs(len(nu1)-2642)/2642:.1%}, U err={abs(len(u1)-2741)/2741:.1%}")

# 2. Job-level union (>50% of years)
nu2 = bc[bc['job_union'] == 0]
u2 = bc[bc['job_union'] == 1]
print(f"2. Job-level union (>50%): NU={len(nu2)}, U={len(u2)}, NA={len(bc)-len(nu2)-len(u2)}")
print(f"   NU err={abs(len(nu2)-2642)/2642:.1%}, U err={abs(len(u2)-2741)/2741:.1%}")

# 3. Ever union in job
nu3 = bc[bc['ever_union'] == 0]
u3 = bc[bc['ever_union'] == 1]
print(f"3. Ever union in job: NU={len(nu3)}, U={len(u3)}, NA={len(bc)-len(nu3)-len(u3)}")
print(f"   NU err={abs(len(nu3)-2642)/2642:.1%}, U err={abs(len(u3)-2741)/2741:.1%}")

# 4. Job-level but treating NaN union as nonunion
bc2 = bc.copy()
bc2['job_union_na0'] = bc2['job_union'].fillna(0)
nu4 = bc2[bc2['job_union_na0'] == 0]
u4 = bc2[bc2['job_union_na0'] == 1]
print(f"4. Job-level, NaN->nonunion: NU={len(nu4)}, U={len(u4)}")
print(f"   NU err={abs(len(nu4)-2642)/2642:.1%}, U err={abs(len(u4)-2741)/2741:.1%}")

# 5. Obs-level but treating NaN union as nonunion
bc3 = bc.copy()
bc3['union_na0'] = bc3['union_member'].fillna(0)
nu5 = bc3[bc3['union_na0'] == 0]
u5 = bc3[bc3['union_na0'] == 1]
print(f"5. Obs-level, NaN->nonunion: NU={len(nu5)}, U={len(u5)}")
print(f"   NU err={abs(len(nu5)-2642)/2642:.1%}, U err={abs(len(u5)-2741)/2741:.1%}")

# 6. First-observation union status (most recent year)
bc4 = bc.copy()
first_union = bc4.groupby(['person_id', 'job_id'])['union_member'].first().reset_index()
first_union = first_union.rename(columns={'union_member': 'first_union'})
bc4 = bc4.merge(first_union, on=['person_id', 'job_id'], how='left')
nu6 = bc4[bc4['first_union'] == 0]
u6 = bc4[bc4['first_union'] == 1]
print(f"6. First-obs union: NU={len(nu6)}, U={len(u6)}, NA={len(bc4)-len(nu6)-len(u6)}")
print(f"   NU err={abs(len(nu6)-2642)/2642:.1%}, U err={abs(len(u6)-2741)/2741:.1%}")

# What if we also try different PS codes that keep PS close to paper but allow more BC_U?
# Currently PS codes lose occ=0 Professional to BC (3306 rows).
# Maybe some of those should stay in PS?

# Check: what if BC excludes 'S' (service workers -> stays in PS)?
bc_noS = df[df['occ_mapped'].isin(['0', '5', '6', 'L'])]
nu7 = bc_noS[bc_noS['union_member'] == 0]
u7 = bc_noS[bc_noS['union_member'] == 1]
print(f"\n7. BC=[0,5,6,L] obs: NU={len(nu7)}, U={len(u7)}")

# What if BC uses [0,5,6,L,S] but with job_union?
nu8 = bc[bc['job_union'] == 0]
u8 = bc[bc['job_union'] == 1]
print(f"8. BC=[0,5,6,L,S] job: NU={len(nu8)}, U={len(u8)}")
print(f"   NU err={abs(len(nu8)-2642)/2642:.1%}, U err={abs(len(u8)-2741)/2741:.1%}")

# Key question: Using job_union for BC gives NU=3945, U=2537
# That gives U within 7.5%! But NU is 49.3% off.
# Is there a hybrid that gets better Ns?

# Try: use different codes for NU vs U - unlikely to help

# What about including occ='S' in BC AND PS?
# Or using a different PS grouping?

# Let me check: PS=[1,3,4,'7','S'] BC=[0,5,6,'L'] obs
ps2 = df[df['occ_mapped'].isin(['1', '3', '4', '7', 'S'])]
bc2 = df[df['occ_mapped'].isin(['0', '5', '6', 'L'])]
nu9 = bc2[bc2['union_member'] == 0]
u9 = bc2[bc2['union_member'] == 1]
print(f"\n9. PS=[1,3,4,7,S] BC=[0,5,6,L] obs: PS={len(ps2)}, NU={len(nu9)}, U={len(u9)}")
print(f"   PS err={abs(len(ps2)-4946)/4946:.1%}, NU err={abs(len(nu9)-2642)/2642:.1%}, U err={abs(len(u9)-2741)/2741:.1%}")

# PS=[0,1,3,4,7,S] BC=[5,6,L] obs
ps3 = df[df['occ_mapped'].isin(['0', '1', '3', '4', '7', 'S'])]
bc3 = df[df['occ_mapped'].isin(['5', '6', 'L'])]
nu10 = bc3[bc3['union_member'] == 0]
u10 = bc3[bc3['union_member'] == 1]
print(f"10. PS=[0,1,3,4,7,S] BC=[5,6,L] obs: PS={len(ps3)}, NU={len(nu10)}, U={len(u10)}")

# PS=[0,1,3,4,7,S] BC=[5,6,L] job
nu11 = bc3[bc3['job_union'] == 0]
u11 = bc3[bc3['job_union'] == 1]
print(f"11. PS=[0,1,3,4,7,S] BC=[5,6,L] job: PS={len(ps3)}, NU={len(nu11)}, U={len(u11)}")

# What about a hybrid: PS uses one definition, BC_NU uses obs, BC_U uses job?
# This doesn't make conceptual sense but let's see the Ns
print(f"\nHybrid: BC=[0,5,6,L,S]")
print(f"  obs NU={len(nu1)}, job U={len(u8)}")
print(f"  job NU={len(nu8)}, obs U={len(u1)}")

# Actually the paper says (footnote 18): "Jobs were categorized as 'union' if the
# respondent indicated union membership in more than half of the years of the job."
# So job_union is the correct definition. But with current grouping it gives NU=3945, U=2537.
# NU is too high, U is reasonable.

# Let's try: PS=[1,3,4,7], BC=[0,5,6,L,S], job_union but EXCLUDE NaN job_union
print(f"\n12. BC=[0,5,6,L,S] job (excl NaN): NU={len(nu8)}, U={len(u8)}")
print(f"    Total matched: {len(nu8)+len(u8)}")

# What if we try PS=[0,1,3,7,S] BC=[4,5,6,L] job?
# Paper says BC = "Craftsmen, Operatives, Laborers"
# PSID 4 = Sales - definitely not BC. But let's try for size match
ps4 = df[df['occ_mapped'].isin(['0', '1', '3', '7', 'S'])]
bc4 = df[df['occ_mapped'].isin(['4', '5', '6', 'L'])]
nu12 = bc4[bc4['job_union'] == 0]
u12 = bc4[bc4['job_union'] == 1]
print(f"\n13. PS=[0,1,3,7,S] BC=[4,5,6,L] job: PS={len(ps4)}, NU={len(nu12)}, U={len(u12)}")
print(f"    PS err={abs(len(ps4)-4946)/4946:.1%}, NU err={abs(len(nu12)-2642)/2642:.1%}, U err={abs(len(u12)-2741)/2741:.1%}")

# Try PS=[0,1,3,7,S] BC=[4,5,6,L] obs
nu13 = bc4[bc4['union_member'] == 0]
u13 = bc4[bc4['union_member'] == 1]
print(f"14. PS=[0,1,3,7,S] BC=[4,5,6,L] obs: PS={len(ps4)}, NU={len(nu13)}, U={len(u13)}")
print(f"    PS err={abs(len(ps4)-4946)/4946:.1%}, NU err={abs(len(nu13)-2642)/2642:.1%}, U err={abs(len(u13)-2741)/2741:.1%}")
