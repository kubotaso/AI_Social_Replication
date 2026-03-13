"""Map 3-digit Census codes to proper occupation groups"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')

# The occ_0 through occ_9 columns are occupation dummies
# occ_1digit contains the actual code (1-digit or 3-digit)
# For rows with occ_1digit > 9, we need to map 3-digit Census codes

# Census 1970 occupation codes:
# 001-195: Professional, Technical, and Kindred Workers -> PS
# 201-245: Managers and Administrators, exc Farm -> PS
# 260-395: Sales Workers + Clerical and Kindred Workers -> PS
# 401-580: Craftsmen and Kindred Workers -> BC
# 601-695: Operatives, exc Transport -> BC
# 701-715: Transport Equipment Operatives -> BC
# 740-785: Laborers, exc Farm -> BC
# 801-824: Farmers and Farm Managers -> exclude
# 901-965: Service Workers -> PS (the paper says "Professional and Service")

# PSID 1-digit codes (matching occ_0 thru occ_9):
# 0 = Professional, Technical, and Kindred -> PS
# 1 = Managers, Officials, and Proprietors -> PS
# 2 = Self-Employed Businessmen -> maybe exclude? or PS?
# 3 = Clerical and Kindred -> PS
# 4 = Sales Workers -> PS
# 5 = Craftsmen, Foremen, and Kindred -> BC
# 6 = Operatives and Kindred -> BC
# 7 = Laborers and Service Workers -> SPLIT! Laborers=BC, Service=PS
# 8 = Farmers and Farm Managers -> exclude
# 9 = Miscellaneous -> check

# Let's use the occ_0..occ_9 dummies to classify rows where occ_1digit > 9
# For 3-digit codes, map using Census classification

# First, let's look at what occ_X dummies are set for 3-digit codes
rows_3d = df[df['occ_1digit'] > 9].copy()
print(f"Rows with 3-digit codes: {len(rows_3d)}")

# Check which occ_X dummy is set
for i in range(10):
    col = f'occ_{i}'
    n = rows_3d[col].sum()
    if n > 0:
        print(f"  occ_{i}: {int(n)} rows")

# So the occ_X dummies tell us the 1-digit category for 3-digit codes!
# This means we can use the dummies directly

# Unified classification:
# PS (Professional/Service): occ_0 + occ_1 + occ_3 + occ_4 (+ part of occ_7?)
# BC: occ_5 + occ_6 + occ_7 (or just laborers from 7)
# Exclude: occ_2 (self-employed businessmen), occ_8 (farmers), occ_9 (misc)

# Actually, let me re-examine. The paper's Table 5 says:
# Column (1): Professional and Service workers
# Columns (2)/(3): Craftsmen, Operatives, and Laborers

# So Service workers are in PS, and Laborers are in BC.
# PSID code 7 = "Laborers and Service Workers" - this needs to be split!

# For 3-digit codes in the 700s:
# 701-715: Transport Equipment Operatives -> BC (operatives)
# 740-785: Laborers, exc Farm -> BC
# 801-824: Farmers -> exclude
# 901-965: Service Workers -> PS

# For PSID 1-digit code 7 (which lumps laborers + service):
# We can't split without 3-digit codes. But for rows WITH 3-digit codes we can.

# Let's first see how many rows we get with different groupings

# Method: Use occ_X dummies for all rows
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
df['education_years'] = df['education_clean'].map(EDUC_MAP).fillna(12)
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df = df[df['tenure_topel'] >= 1].copy()
df = df.dropna(subset=['log_hourly_wage']).copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

# Create proper occupation group using occ_X dummies
# PS: occ_0=1 OR occ_1=1 OR occ_3=1 OR occ_4=1
# BC: occ_5=1 OR occ_6=1 OR occ_7=1
# Exclude: occ_2=1 OR occ_8=1 OR occ_9=1

# For 3-digit codes, also use the Census code ranges
# Map 3-digit Census codes
def get_occ_group(row):
    occ = row['occ_1digit']
    if occ <= 9:
        # 1-digit code
        if occ in [0, 1, 3, 4]:
            return 'PS'
        elif occ in [5, 6]:
            return 'BC'
        elif occ == 7:
            # Laborers and Service Workers - ambiguous
            # Paper puts Service in PS, Laborers in BC
            # Without 3-digit code, assign based on...
            # Actually let's try both
            return 'BC'  # Default: 7 goes to BC (laborers)
        elif occ in [2, 8, 9]:
            return 'exclude'
        else:
            return 'exclude'
    else:
        # 3-digit Census code
        if 1 <= occ <= 195:
            return 'PS'  # Professional/Technical
        elif 201 <= occ <= 245:
            return 'PS'  # Managers
        elif 260 <= occ <= 395:
            return 'PS'  # Clerical/Sales
        elif 401 <= occ <= 580:
            return 'BC'  # Craftsmen
        elif 601 <= occ <= 695:
            return 'BC'  # Operatives
        elif 701 <= occ <= 785:
            return 'BC'  # Transport Operatives + Laborers
        elif 801 <= occ <= 824:
            return 'exclude'  # Farmers
        elif 900 <= occ <= 965:
            return 'PS'  # Service workers -> PS per paper title
        else:
            return 'exclude'
    return 'exclude'

df['occ_group'] = df.apply(get_occ_group, axis=1)
print("\nOccupation group counts:")
print(df['occ_group'].value_counts())

ps_new = df[df['occ_group'] == 'PS']
bc_new = df[df['occ_group'] == 'BC']
print(f"\nPS={len(ps_new)}, BC={len(bc_new)}")

# Union split for BC
bc_new = bc_new.sort_values(['person_id', 'job_id', 'year']).copy()
ju = bc_new.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
bc_new = bc_new.merge(ju, on=['person_id', 'job_id'], how='left')

bc_nu = bc_new[bc_new['job_union'] == 0]
bc_u = bc_new[bc_new['job_union'] == 1]
bc_na = bc_new[bc_new['job_union'].isna()]
print(f"BC job_union: NU={len(bc_nu)}, U={len(bc_u)}, NA={len(bc_na)}")
print(f"Total (PS+NU+U): {len(ps_new)+len(bc_nu)+len(bc_u)}")
print(f"Paper: PS=4946, BC_NU=2642, BC_U=2741, Total=10329")

# Obs-level union for BC
bc_nu_obs = bc_new[bc_new['union_member'] == 0]
bc_u_obs = bc_new[bc_new['union_member'] == 1]
bc_na_obs = bc_new[bc_new['union_member'].isna()]
print(f"\nBC obs_union: NU={len(bc_nu_obs)}, U={len(bc_u_obs)}, NA={len(bc_na_obs)}")
print(f"Total obs (PS+NU+U): {len(ps_new)+len(bc_nu_obs)+len(bc_u_obs)}")

# Alternative: Try putting occ=7 in PS (service workers)
def get_occ_group_v2(row):
    occ = row['occ_1digit']
    if occ <= 9:
        if occ in [0, 1, 3, 4, 7]:  # 7 goes to PS
            return 'PS'
        elif occ in [5, 6]:
            return 'BC'
        elif occ in [2, 8, 9]:
            return 'exclude'
        else:
            return 'exclude'
    else:
        if 1 <= occ <= 195:
            return 'PS'
        elif 201 <= occ <= 245:
            return 'PS'
        elif 260 <= occ <= 395:
            return 'PS'
        elif 401 <= occ <= 580:
            return 'BC'
        elif 601 <= occ <= 695:
            return 'BC'
        elif 701 <= occ <= 785:
            return 'BC'
        elif 900 <= occ <= 965:
            return 'PS'
        else:
            return 'exclude'

df['occ_group_v2'] = df.apply(get_occ_group_v2, axis=1)
ps_v2 = df[df['occ_group_v2'] == 'PS']
bc_v2 = df[df['occ_group_v2'] == 'BC']
bc_v2 = bc_v2.sort_values(['person_id', 'job_id', 'year']).copy()
ju2 = bc_v2.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
bc_v2 = bc_v2.merge(ju2, on=['person_id', 'job_id'], how='left')
bc_v2_nu = bc_v2[bc_v2['job_union'] == 0]
bc_v2_u = bc_v2[bc_v2['job_union'] == 1]
print(f"\nV2 (7=PS): PS={len(ps_v2)}, BC_NU={len(bc_v2_nu)}, BC_U={len(bc_v2_u)}")

# What about self-employed businessmen (occ=2)?
# In PSID, occ=2 is "Self-Employed Businessmen" - but we already filter self_employed
# Maybe we should NOT exclude occ=2 since they're not actually self-employed in the data?
occ2 = df[df['occ_1digit'] == 2]
print(f"\nocc=2 count: {len(occ2)}")
print(f"  self_employed distribution:")
print(occ2['self_employed'].value_counts(dropna=False))

# Actually check if occ=2 should be in PS (they're managers/proprietors who are NOT self-employed)
# since we already filter self_employed
# Let's try: PS = 0,1,2,3,4 and BC = 5,6,7
ps_v3 = df[df['occ_group'] == 'PS'].copy()
# Add occ=2 to PS
occ2_data = df[df['occ_1digit'] == 2].copy()
ps_v3 = pd.concat([ps_v3, occ2_data])
print(f"\nV3 (PS includes occ=2): PS={len(ps_v3)}")

# Now check occ=9. In PSID coding, what is 9?
occ9 = df[df['occ_1digit'] == 9]
print(f"\nocc=9 count: {len(occ9)}")

# Actually let's check what the occ_X dummies tell us
# For 3-digit codes, which occ_X dummy is set?
rows_3d = df[df['occ_1digit'] > 9].copy()
print(f"\n3-digit code rows: {len(rows_3d)}")

# What percent of each occ_X dummy do 3-digit codes have?
for i in range(10):
    col = f'occ_{i}'
    n_3d = rows_3d[col].sum()
    n_1d = df[df['occ_1digit'] == i][col].sum()
    print(f"  occ_{i}: 1-digit={int(n_1d)}, 3-digit={int(n_3d)}")

# So for 3-digit codes, which occ_X dummy do they have?
# This would tell us which 1-digit category they belong to
for i in range(10):
    col = f'occ_{i}'
    subset = rows_3d[rows_3d[col] == 1]
    if len(subset) > 0:
        codes = subset['occ_1digit'].value_counts().head(10)
        print(f"\n  occ_{i}=1 3-digit codes: ({len(subset)} rows)")
        for c, n in codes.items():
            print(f"    {c}: {n}")
