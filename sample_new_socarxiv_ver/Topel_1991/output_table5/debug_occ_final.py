"""Find the right occupation grouping to match paper sample sizes"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}

CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

df = pd.read_csv('data/psid_panel.csv')
df['education_years'] = df['education_clean'].map(EDUC_MAP).fillna(12)
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df = df[df['tenure_topel'] >= 1].copy()
df = df.dropna(subset=['log_hourly_wage']).copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

# Map ALL rows (1-digit AND 3-digit) to occupation groups
# Census 1970 occupation code ranges:
# 001-195: Professional, Technical -> PS
# 201-245: Managers, Administrators -> exclude (occ=2 is self-employed businessmen category)
# 260-395: Clerical + Sales -> PS
# 401-580: Craftsmen -> BC
# 601-695: Operatives -> BC
# 701-785: Transport + Laborers -> BC
# 801-824: Farmers -> exclude
# 901-965: Service Workers -> PS

def classify_occ(occ):
    """Classify occupation code to PS/BC/exclude"""
    if occ <= 9:
        # PSID 1-digit codes
        if occ in [0, 1, 4]:  # Professional, Managers, Sales
            return 'PS'
        elif occ == 3:  # Clerical
            return 'PS'
        elif occ in [5, 6]:  # Craftsmen, Operatives
            return 'BC'
        elif occ == 7:  # Laborers+Service - ambiguous
            return 'BC'  # laborers go to BC
        elif occ == 2:  # Self-Employed Businessmen
            return 'exclude'
        elif occ in [8, 9]:  # Farmers, Misc
            return 'exclude'
        else:
            return 'exclude'
    else:
        # 3-digit Census codes
        if 1 <= occ <= 195:
            return 'PS'  # Professional/Technical
        elif 201 <= occ <= 245:
            return 'PS'  # Managers
        elif 260 <= occ <= 395:
            return 'PS'  # Clerical + Sales
        elif 401 <= occ <= 580:
            return 'BC'  # Craftsmen
        elif 601 <= occ <= 695:
            return 'BC'  # Operatives
        elif 701 <= occ <= 785:
            return 'BC'  # Transport + Laborers
        elif 801 <= occ <= 824:
            return 'exclude'  # Farmers
        elif 900 <= occ <= 965:
            return 'PS'  # Service Workers -> PS
        else:
            return 'exclude'

df['occ_group'] = df['occ_1digit'].apply(classify_occ)

# Total by group
print("Occupation group counts:")
print(df['occ_group'].value_counts())

ps = df[df['occ_group'] == 'PS']
bc = df[df['occ_group'] == 'BC']

# Union split for BC - try both obs-level and job-level
bc = bc.sort_values(['person_id', 'job_id', 'year']).copy()
ju = bc.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
bc = bc.merge(ju, on=['person_id', 'job_id'], how='left')

for union_type in ['obs', 'job']:
    if union_type == 'obs':
        bc_nu = bc[bc['union_member'] == 0]
        bc_u = bc[bc['union_member'] == 1]
    else:
        bc_nu = bc[bc['job_union'] == 0]
        bc_u = bc[bc['job_union'] == 1]

    total = len(ps) + len(bc_nu) + len(bc_u)
    print(f"\n{union_type}-level union: PS={len(ps)}, BC_NU={len(bc_nu)}, BC_U={len(bc_u)}, Total={total}")
    print(f"Paper:                  PS=4946, BC_NU=2642, BC_U=2741, Total=10329")
    print(f"  PS diff: {len(ps)-4946} ({(len(ps)-4946)/4946*100:.1f}%)")
    print(f"  NU diff: {len(bc_nu)-2642} ({(len(bc_nu)-2642)/2642*100:.1f}%)")
    print(f"  U diff:  {len(bc_u)-2741} ({(len(bc_u)-2741)/2741*100:.1f}%)")

# The obs-level union total was 10336. Very close to 10329.
# PS=7171 is way too high vs 4946.
# Let's try excluding occ=2 from the totals but also see what occ codes give PS too large

# Count by 1-digit in PS group
print("\nPS breakdown by occ_1digit:")
for occ in sorted(ps['occ_1digit'].unique()):
    if occ <= 9:
        n = len(ps[ps['occ_1digit'] == occ])
        if n > 0:
            print(f"  occ={occ}: {n}")
# Also 3-digit codes that went to PS
ps_3d = ps[ps['occ_1digit'] > 9]
print(f"  3-digit codes in PS: {len(ps_3d)}")

# BC breakdown
print("\nBC breakdown by occ_1digit:")
for occ in sorted(bc['occ_1digit'].unique()):
    if occ <= 9:
        n = len(bc[bc['occ_1digit'] == occ])
        if n > 0:
            print(f"  occ={occ}: {n}")
bc_3d = bc[bc['occ_1digit'] > 9]
print(f"  3-digit codes in BC: {len(bc_3d)}")

# Hmm, PS=7171 is too high. Paper says 4946.
# Difference is 7171-4946=2225.
# occ=0 has 2902 rows. If occ=0 is NOT in PS, we'd have PS=7171-2902=4269 (too low).
# What if occ=2 IS included in PS? Then PS=7171+2128=9299 (way too high)

# Wait - let me reconsider. Paper says N=4946 for PS.
# Current PS has 7171. Let's check what happens if we EXCLUDE occ=3 (Clerical)
# and occ=7 goes to BC

# Try: PS = 0, 1, 4 + 3-digit PS codes
# Note: occ=3 (Clerical) has 440. Removing it: 7171-440=6731. Still too high.

# What if we DON'T include 3-digit codes in the classification?
# i.e., only use 1-digit codes and exclude all 3-digit code rows
print("\n=== WITHOUT 3-digit code rows ===")
df_1d = df[df['occ_1digit'] <= 9].copy()
print(f"Total with 1-digit codes only: {len(df_1d)}")

ps_1d = df_1d[df_1d['occ_group'] == 'PS']
bc_1d = df_1d[df_1d['occ_group'] == 'BC']
bc_1d = bc_1d.sort_values(['person_id', 'job_id', 'year']).copy()
ju_1d = bc_1d.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
bc_1d = bc_1d.merge(ju_1d, on=['person_id', 'job_id'], how='left')
bc_1d_nu = bc_1d[bc_1d['job_union'] == 0]
bc_1d_u = bc_1d[bc_1d['job_union'] == 1]
print(f"PS={len(ps_1d)}, BC_NU={len(bc_1d_nu)}, BC_U={len(bc_1d_u)}")

# Hmm, maybe the issue is that we need to map 3-digit codes to 1-digit first
# then use the 1-digit grouping. Let's create a proper 1-digit code for ALL rows

def to_1digit(occ):
    if occ <= 9:
        return occ
    elif 1 <= occ <= 195:
        return 0  # Professional
    elif 201 <= occ <= 245:
        return 1  # Managers (not 2=self-employed)
    elif 260 <= occ <= 285:
        return 4  # Sales
    elif 301 <= occ <= 395:
        return 3  # Clerical
    elif 401 <= occ <= 580:
        return 5  # Craftsmen
    elif 601 <= occ <= 695:
        return 6  # Operatives
    elif 701 <= occ <= 785:
        return 7  # Laborers + Transport
    elif 801 <= occ <= 824:
        return 8  # Farmers
    elif 900 <= occ <= 965:
        return 7  # Service -> put with laborers (PSID lumps them as occ=7)
    else:
        return 9  # Unknown/misc

df['occ_mapped'] = df['occ_1digit'].apply(to_1digit)
print("\n=== With mapped 1-digit codes ===")
print("occ_mapped distribution:")
for v in sorted(df['occ_mapped'].unique()):
    n = len(df[df['occ_mapped'] == v])
    print(f"  {v}: {n}")

# Now PS = 0,1,3,4 and BC = 5,6,7 (includes service mapped to 7)
ps_m = df[df['occ_mapped'].isin([0, 1, 3, 4])]
bc_m = df[df['occ_mapped'].isin([5, 6, 7])]
print(f"\nMapped: PS={len(ps_m)}, BC={len(bc_m)}")

# With union
bc_m = bc_m.sort_values(['person_id', 'job_id', 'year']).copy()
ju_m = bc_m.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
bc_m = bc_m.merge(ju_m, on=['person_id', 'job_id'], how='left')

for union_type in ['obs', 'job']:
    if union_type == 'obs':
        nu = bc_m[bc_m['union_member'] == 0]
        u = bc_m[bc_m['union_member'] == 1]
    else:
        nu = bc_m[bc_m['job_union'] == 0]
        u = bc_m[bc_m['job_union'] == 1]
    total = len(ps_m) + len(nu) + len(u)
    print(f"  {union_type}: PS={len(ps_m)}, NU={len(nu)}, U={len(u)}, Total={total}")

# The key insight: with 3-digit codes properly mapped, the totals should be closer
# but we need to find which grouping gives 4946/2642/2741

# Let's try: PS = 0,1,4 (exclude 3=clerical) and BC = 5,6 (exclude 7=laborers+service)
ps_no3 = df[df['occ_mapped'].isin([0, 1, 4])]
bc_no7 = df[df['occ_mapped'].isin([5, 6])]
print(f"\nPS(0,1,4)={len(ps_no3)}, BC(5,6)={len(bc_no7)}")

# PS = 0,1,3,4,7 (service part in PS) and BC = 5,6
ps_with7 = df[df['occ_mapped'].isin([0, 1, 3, 4, 7])]
bc_56 = df[df['occ_mapped'].isin([5, 6])]
print(f"PS(0,1,3,4,7)={len(ps_with7)}, BC(5,6)={len(bc_56)}")

# The paper's note says "Professional and Service" vs "Craftsmen, Operatives, Laborers"
# PSID occ=7 = "Laborers and Service"
# So "Service" should go to PS and "Laborers" to BC
# But we can't split occ=7 without 3-digit codes

# For 3-digit codes, we CAN split:
# 701-785: Transport Operatives + Laborers -> BC
# 900-965: Service Workers -> PS
# For 1-digit occ=7: these are mixed. Let's check how many are service vs laborers
# by looking at union rates (service is likely less unionized than laborers)

occ7 = df[df['occ_mapped'] == 7]
print(f"\nocc_mapped=7 breakdown:")
print(f"  Total: {len(occ7)}")
print(f"  With occ_1digit=7: {len(occ7[occ7['occ_1digit'] == 7])}")
print(f"  With 3d 700-785: {len(occ7[(occ7['occ_1digit'] >= 701) & (occ7['occ_1digit'] <= 785)])}")
print(f"  With 3d 900-965: {len(occ7[(occ7['occ_1digit'] >= 900) & (occ7['occ_1digit'] <= 965)])}")

# Actually, service workers (900-965) were mapped to occ=7. Let me remap properly
# where service goes to PS and laborers/transport go to BC
def classify_proper(occ):
    if occ <= 9:
        if occ in [0, 1, 3, 4]:
            return 'PS'
        elif occ in [5, 6]:
            return 'BC'
        elif occ == 7:
            return 'AMBIG'  # Can't split without 3-digit
        elif occ in [2, 8, 9]:
            return 'exclude'
        else:
            return 'exclude'
    else:
        if 1 <= occ <= 195:
            return 'PS'
        elif 201 <= occ <= 245:
            return 'PS'  # Managers
        elif 260 <= occ <= 285:
            return 'PS'  # Sales
        elif 301 <= occ <= 395:
            return 'PS'  # Clerical
        elif 401 <= occ <= 580:
            return 'BC'  # Craftsmen
        elif 601 <= occ <= 695:
            return 'BC'  # Operatives
        elif 701 <= occ <= 785:
            return 'BC'  # Transport + Laborers
        elif 801 <= occ <= 824:
            return 'exclude'
        elif 900 <= occ <= 965:
            return 'PS'  # Service -> PS
        else:
            return 'exclude'

df['occ_proper'] = df['occ_1digit'].apply(classify_proper)
print("\n=== Proper classification (service=PS, laborers=BC) ===")
print(df['occ_proper'].value_counts())

# AMBIG (occ=7 with 1-digit code) - try assigning to both
ambig = df[df['occ_proper'] == 'AMBIG']
print(f"\nAmbiguous (occ=7, 1-digit): {len(ambig)}")

# Try: AMBIG -> PS (they're service workers)
ps_p = df[df['occ_proper'].isin(['PS'])]
bc_p = df[df['occ_proper'].isin(['BC'])]
print(f"\nWithout AMBIG: PS={len(ps_p)}, BC={len(bc_p)}")

# AMBIG -> PS
ps_all = df[df['occ_proper'].isin(['PS', 'AMBIG'])]
print(f"AMBIG->PS: PS={len(ps_all)}, BC={len(bc_p)}")

# AMBIG -> BC
bc_all = df[df['occ_proper'].isin(['BC', 'AMBIG'])]
print(f"AMBIG->BC: PS={len(ps_p)}, BC={len(bc_all)}")

# For BC union split (without ambiguous)
bc_p = bc_p.sort_values(['person_id', 'job_id', 'year']).copy()
ju_p = bc_p.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
bc_p = bc_p.merge(ju_p, on=['person_id', 'job_id'], how='left')
nu_p = bc_p[bc_p['job_union'] == 0]
u_p = bc_p[bc_p['job_union'] == 1]
print(f"\nBC (without AMBIG) job union: NU={len(nu_p)}, U={len(u_p)}")
print(f"Total: PS={len(ps_p)}+NU={len(nu_p)}+U={len(u_p)}={len(ps_p)+len(nu_p)+len(u_p)}")

# AMBIG->BC union split
bc_all2 = bc_all.sort_values(['person_id', 'job_id', 'year']).copy()
ju_all = bc_all2.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
bc_all2 = bc_all2.merge(ju_all, on=['person_id', 'job_id'], how='left')
nu_all = bc_all2[bc_all2['job_union'] == 0]
u_all = bc_all2[bc_all2['job_union'] == 1]
print(f"\nBC (with AMBIG->BC) job union: NU={len(nu_all)}, U={len(u_all)}")
print(f"Total: PS={len(ps_p)}+NU={len(nu_all)}+U={len(u_all)}={len(ps_p)+len(nu_all)+len(u_all)}")

# Also try obs-level union
nu_obs = bc_all2[bc_all2['union_member'] == 0]
u_obs = bc_all2[bc_all2['union_member'] == 1]
print(f"\nBC (with AMBIG->BC) obs union: NU={len(nu_obs)}, U={len(u_obs)}")
print(f"Total obs: PS={len(ps_p)}+NU={len(nu_obs)}+U={len(u_obs)}={len(ps_p)+len(nu_obs)+len(u_obs)}")
