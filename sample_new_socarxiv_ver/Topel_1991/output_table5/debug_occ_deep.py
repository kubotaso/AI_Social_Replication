"""Deep investigation of occupation codes and missing observations"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')

# Check what columns we have for occupation
occ_cols = [c for c in df.columns if 'occ' in c.lower()]
print("Occupation columns:", occ_cols)

# Look at occ_1digit distribution
print("\nocc_1digit value counts:")
vc = df['occ_1digit'].value_counts().sort_index()
for idx, v in vc.items():
    print(f"  {idx}: {v}")

# The 2902 occ=0 rows - what are they?
# They could be 3-digit Census codes that aren't being mapped
print("\n=== occ=0 investigation ===")
occ0 = df[df['occ_1digit'] == 0]
print(f"occ=0: {len(occ0)} rows")

# Check if there's a 3-digit code column
for c in df.columns:
    if 'occ' in c.lower():
        print(f"\n  {c} in occ_1digit==0 rows:")
        vc2 = occ0[c].value_counts().head(20)
        for idx2, v2 in vc2.items():
            print(f"    {idx2}: {v2}")

# Check the 'occupation' column if it exists
if 'occupation' in df.columns:
    print("\n  'occupation' in full data, value counts:")
    vc_full = df['occupation'].value_counts().sort_index().head(30)
    for idx, v in vc_full.items():
        print(f"    {idx}: {v}")

# Check if occ_1digit might actually contain 3-digit codes mixed in
print("\n\nocc_1digit unique values > 9:")
big_occ = df[df['occ_1digit'] > 9]['occ_1digit'].value_counts().head(20)
for idx, v in big_occ.items():
    print(f"  {idx}: {v}")

# Paper notes say "white males". Check if we're filtering correctly.
# The paper's total N = 10,685 (Table 3) and Table 5 sums: 4946+2642+2741 = 10,329
# which is close but not equal, implying some obs dropped due to missing occ/union

# What's our total with tenure >= 1?
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
df['education_years'] = df['education_clean'].map(EDUC_MAP).fillna(12)
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
t1 = df[df['tenure_topel'] >= 1].copy()
t1 = t1.dropna(subset=['log_hourly_wage']).copy()
t1 = t1[(t1['self_employed'] == 0) | (t1['self_employed'].isna())].copy()
print(f"\nTotal with tenure>=1, non-self-employed, has log_wage: {len(t1)}")

# Without self-employed filter
t2 = df[df['tenure_topel'] >= 1].copy()
t2 = t2.dropna(subset=['log_hourly_wage']).copy()
print(f"Without self-employed filter: {len(t2)}")

# The occ=0 group is key. In PSID, occ code 0 can mean:
# - "NA; not in labor force" or
# - The actual Census occupation code 0XX range (professional/technical)
# Let's check if these people are valid workers
occ0_t1 = t1[t1['occ_1digit'] == 0]
print(f"\nocc=0 with tenure>=1, non-SE, has wage: {len(occ0_t1)}")
print(f"  Mean wage: {np.exp(occ0_t1['log_hourly_wage'].mean()):.2f}")
print(f"  Mean tenure: {occ0_t1['tenure_topel'].mean():.1f}")
print(f"  Mean age: {occ0_t1['age'].mean():.1f}")

# Check union status in occ=0
print(f"  union_member distribution:")
print(occ0_t1['union_member'].value_counts(dropna=False))

# If occ=0 are valid professional/technical workers (Census codes 001-195),
# adding them to PS would give us: 4624 + ~2902 = ~7526 (too many)
# But maybe occ=0 includes a MIX of occupations

# The real PSID 1-digit coding:
# 0 = Professional, Technical, and Kindred
# 1 = Managers, Officials, and Proprietors
# 2 = Self-Employed Businessmen
# 3 = Clerical and Kindred
# 4 = Sales Workers
# 5 = Craftsmen, Foremen, and Kindred
# 6 = Operatives and Kindred
# 7 = Laborers and Service Workers (exc farm)
# 8 = Farmers and Farm Managers
# 9 = Miscellaneous (farm laborers, etc)

# So in PSID, 0 = Professional/Technical!!!
# And 3 = Clerical, 8 = Farmers, 9 = Farm Laborers

# PS should be: 0 (Professional) + 1 (Managers) + 3 (Clerical) + 4 (Sales) + 7 (Service part)
# BC should be: 5 (Craftsmen) + 6 (Operatives) + 7 (Laborers part)

# Wait - PSID 1-digit codes might be:
# The issue is that code 7 in PSID = "Laborers and Service Workers"
# which mixes blue-collar (laborers) and service workers

# Let's try the correct PSID grouping:
# PS = 0 (professional) + 1 (managers) + 3 (clerical) + 4 (sales)
# BC = 5 (craftsmen) + 6 (operatives)
# And 7 needs to be split or assigned

# But wait - the paper says "Professional and Service" (column 1) and
# "Craftsmen, Operatives, Laborers" (columns 2-3)

# So with PSID coding:
# PS = 0 (professional) + 1 (managers) + 3 (clerical) + 4 (sales) + part of 7 (service)
# BC = 5 (craftsmen) + 6 (operatives) + part of 7 (laborers)

# Or more likely the paper's grouping is:
# PS = 0,1,2,3,4 (white collar + service workers from 7/8)
# BC = 5,6,7 or 5,6 with 7 being either PS or BC

# Let's try: PS = 0,1,3,4 (exclude 2=self-employed, 8=farmers, 9=misc)
# BC = 5,6,7
ps_with_0 = t1[t1['occ_1digit'].isin([0, 1, 3, 4])]
bc = t1[t1['occ_1digit'].isin([5, 6, 7])]
print(f"\nPS with occ=0: {len(ps_with_0)}")
print(f"BC (5,6,7): {len(bc)}")

# Now BC with union split using job-level
bc_s = bc.sort_values(['person_id', 'job_id', 'year'])
ju = bc_s.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
bc_s = bc_s.merge(ju, on=['person_id', 'job_id'], how='left')
bc_nu = bc_s[bc_s['job_union'] == 0]
bc_u = bc_s[bc_s['job_union'] == 1]
bc_na = bc_s[bc_s['job_union'].isna()]
print(f"BC job_union: NU={len(bc_nu)}, U={len(bc_u)}, NA={len(bc_na)}")

# Try obs-level union for BC
bc_nu_obs = bc[bc['union_member'] == 0]
bc_u_obs = bc[bc['union_member'] == 1]
bc_na_obs = bc[bc['union_member'].isna()]
print(f"BC obs_union: NU={len(bc_nu_obs)}, U={len(bc_u_obs)}, NA={len(bc_na_obs)}")

# Total with this grouping
total = len(ps_with_0) + len(bc_nu) + len(bc_u)
print(f"\nTotal (PS+BC_NU+BC_U): {total}")
print(f"Paper total (4946+2642+2741): {4946+2642+2741}")

# What if PS also includes occ=8 (service)?
# Paper says "Professional and Service" workers
# PSID 7 = "Laborers and Service Workers" - this is tricky
# PSID 8 = "Farmers" - probably excluded

# Let's see if including some of 7 in PS helps
# The 3-digit Census codes within 7 might distinguish laborers vs service
occ7 = t1[t1['occ_1digit'] == 7]
print(f"\nocc=7 total: {len(occ7)}")
print(f"  union_member distribution:")
print(occ7['union_member'].value_counts(dropna=False))

# Maybe we should try: PS = 0,1,3,4,8 and BC = 5,6,7,9
# No wait, 8 has only 2 obs. Let's try PS = 0,1,3,4 and BC = 5,6,7,9
bc2 = t1[t1['occ_1digit'].isin([5, 6, 7, 9])]
bc2_s = bc2.sort_values(['person_id', 'job_id', 'year'])
ju2 = bc2_s.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
bc2_s = bc2_s.merge(ju2, on=['person_id', 'job_id'], how='left')
bc2_nu = bc2_s[bc2_s['job_union'] == 0]
bc2_u = bc2_s[bc2_s['job_union'] == 1]
print(f"\nPS(0,1,3,4)={len(ps_with_0)}, BC(5,6,7,9) NU={len(bc2_nu)} U={len(bc2_u)}")
print(f"Total: {len(ps_with_0)+len(bc2_nu)+len(bc2_u)}")
