"""Detailed analysis of occupation codes in the data"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
df['education_years'] = df['education_clean'].map(EDUC_MAP).fillna(12)
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df['tenure'] = df['tenure_topel'].copy()
df = df[df['tenure'] >= 1].copy()
df = df.dropna(subset=['log_hourly_wage']).copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

# Look at the raw occ codes
occ = df['occ_1digit'].value_counts().sort_index()
print("occ_1digit distribution:")
print(occ.to_string())
print(f"\nTotal: {len(df)}")

# For 1-digit codes (0-9): how many rows?
one_digit = df[df['occ_1digit'] <= 9]
three_digit = df[df['occ_1digit'] > 9]
print(f"\n1-digit codes (0-9): {len(one_digit)} rows")
print(f"3-digit codes (>9): {len(three_digit)} rows")

# What unique 3-digit codes are there?
codes = three_digit['occ_1digit'].value_counts().sort_index()
print(f"\nUnique 3-digit codes: {len(codes)}")

# Check the ranges
ranges = [(1, 195, 'Prof/Tech'), (201, 245, 'Managers'), (246, 259, 'Gap1'),
          (260, 285, 'Sales'), (286, 300, 'Gap2'), (301, 395, 'Clerical'),
          (396, 400, 'Gap3'), (401, 580, 'Craftsmen'), (581, 600, 'Gap4'),
          (601, 695, 'Operatives'), (696, 700, 'Gap5'), (701, 785, 'Transport+Laborers'),
          (786, 800, 'Gap6'), (801, 824, 'Farmers'), (825, 899, 'Gap7'),
          (900, 965, 'Service')]

for lo, hi, name in ranges:
    mask = (three_digit['occ_1digit'] >= lo) & (three_digit['occ_1digit'] <= hi)
    n = mask.sum()
    if n > 0:
        u = three_digit.loc[mask, 'union_member']
        u_1 = (u == 1).sum()
        u_0 = (u == 0).sum()
        u_na = u.isna().sum()
        print(f"  {lo:3d}-{hi:3d} {name:20s}: {n:5d} rows (U={u_1}, NU={u_0}, NA={u_na})")

# The key question: Topel defines occupations as:
# "Professional and related" vs "Craftsmen, Operatives, and Laborers"
# Are service workers in PS or excluded?
# The paper says "workers in professional, managerial, sales, clerical, and service occupations" for PS
# and "craftsmen, operatives, and laborers" for BC

# Let me check: what does PSID code 7 contain?
# PSID 1-digit: 7 = "Laborers, except farm"
# But in census codes, 701-785 is "Transport Equipment Operatives and Laborers"
# PSID 7 might also include service workers

# Let me check if PSID code 7 rows are laborers or service or mixed
occ7 = df[df['occ_1digit'] == 7]
print(f"\nocc_1digit=7: {len(occ7)} rows")
print(f"  union=0: {(occ7['union_member']==0).sum()}")
print(f"  union=1: {(occ7['union_member']==1).sum()}")
print(f"  union=NA: {occ7['union_member'].isna().sum()}")

# What about the paper's definition of PS vs BC?
# Paper p.158: "workers in professional, managerial, sales, clerical, and service occupations" = PS
# Paper p.158: "craftsmen, operatives, and laborers" = BC
#
# PSID 1-digit mapping:
# 0 = Professional/Technical -> PS
# 1 = Managers -> PS
# 2 = Self-employed -> EXCLUDED
# 3 = Clerical -> PS
# 4 = Sales -> PS
# 5 = Craftsmen -> BC
# 6 = Operatives -> BC
# 7 = Laborers -> BC (but PSID code 7 might be "Laborers, except farm" which is BC)
# 8 = Farmers -> EXCLUDED
# 9 = Service -> PS
#
# Wait! Code 9 in PSID = Service workers! These should be PS!

# But in our mapping, we exclude occ_mapped='9' which maps to PSID code 9 AND unmapped 3-digit codes

# Let me check what happens if we use PSID mapping:
# PS = [0, 1, 3, 4, 9]  (Prof, Mgr, Clerical, Sales, Service)
# BC = [5, 6, 7]  (Craftsmen, Operatives, Laborers)
# Excluded = [2, 8]

# But we also have 3-digit codes that need mapping...
# The issue is that code 9 could be PSID-9 (service) or could be a 3-digit code > 9

# Actually the function map_occ_split handles this:
# if occ <= 9: return str(occ)  -- so occ=9 returns '9'
# 900-965 returns 'S'
# Everything else > 9 gets mapped to ranges

# So PSID code 9 (occ_1digit=9) is mapped to '9', which we exclude!
# But service workers should be in PS!

print("\n\nCRITICAL FINDING:")
print(f"PSID code 9 (service workers): {len(df[df['occ_1digit']==9])} rows")
print("These should be in PS but we EXCLUDE them!")
print("Let me include them and see what happens...")

def map_occ_v2(occ):
    """Include PSID code 9 as service workers (mapped to 'S' for PS)"""
    if occ == 9: return 'S'  # PSID service -> PS
    if occ <= 8: return str(occ)
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

df['occ_v2'] = df['occ_1digit'].apply(map_occ_v2)
df_v2 = df[~df['occ_v2'].isin(['2', '8', '9'])].copy()

# PS = [0, 1, 3, 4, 7, S]  (Prof, Mgr, Clerical, Sales, Laborers(PSID), Service(both))
# Wait, PSID 7 = Laborers. Are laborers BC?
# Paper: BC = "craftsmen, operatives, and laborers"
# So 7 = Laborers should be BC, not PS!

# Let me try the standard paper definition:
# PS = [0, 1, 3, 4, S]  (Prof, Mgr, Clerical, Sales, Service)
# BC = [5, 6, 7, L]  (Craftsmen, Operatives, Laborers(PSID), Laborers(3-digit))
# Note: 3-digit service (S, 900-965) goes to PS
# Note: PSID 7 + 3-digit 701-785 (L) both go to BC

ps_v2 = df_v2[df_v2['occ_v2'].isin(['0', '1', '3', '4', 'S'])]
bc_v2 = df_v2[df_v2['occ_v2'].isin(['5', '6', '7', 'L'])]
bc_nu_v2 = bc_v2[bc_v2['union_member'] == 0]
bc_u_v2 = bc_v2[bc_v2['union_member'] == 1]

print(f"\nPS=[0,1,3,4,S], BC=[5,6,7,L]:")
print(f"  PS={len(ps_v2)}, BC_NU={len(bc_nu_v2)}, BC_U={len(bc_u_v2)}")
for n, t, l in [(len(ps_v2), 4946, 'PS'), (len(bc_nu_v2), 2642, 'NU'), (len(bc_u_v2), 2741, 'U')]:
    re = abs(n - t) / t
    pts = 5 if re <= 0.05 else (3 if re <= 0.10 else (1 if re <= 0.20 else 0))
    print(f"  {l}: {n} vs {t} ({re:.1%}) -> {pts}/5")

# Alternative: What if PSID 7 includes BOTH laborers and service workers?
# PSID codebook: 7 = "Laborers, except farm"  -- nope, it's laborers
# But wait, some years might code differently

# Let me try: PS = [0, 1, 3, 4, S] (no 7), BC = [5, 6, 7, L] (include 7)
# This is the "standard" paper interpretation

# Also try with 9 included in PS
ps_v3 = df_v2[df_v2['occ_v2'].isin(['0', '1', '3', '4', 'S'])]
print(f"\nPS=[0,1,3,4,S] (no PSID-7, with PSID-9 as S):")
print(f"  PS N={len(ps_v3)} vs 4946 ({abs(len(ps_v3)-4946)/4946:.1%})")

# What if we keep current best PS=[1,3,4,7] but add PSID-9 as service to PS?
# PS = [1, 3, 4, 7, S] where S includes both PSID-9 and 3-digit 900-965
ps_v4 = df_v2[df_v2['occ_v2'].isin(['1', '3', '4', '7', 'S'])]
print(f"\nPS=[1,3,4,7,S] (add PSID-9 as S to current best):")
print(f"  PS N={len(ps_v4)} vs 4946 ({abs(len(ps_v4)-4946)/4946:.1%})")

# What about BC with this?
bc_v4 = df_v2[df_v2['occ_v2'].isin(['0', '5', '6', 'L'])]
bc_nu_v4 = bc_v4[bc_v4['union_member'] == 0]
bc_u_v4 = bc_v4[bc_v4['union_member'] == 1]
print(f"  BC=[0,5,6,L]: NU={len(bc_nu_v4)}, U={len(bc_u_v4)}")

# The key issue: In the current best, we have PSID 7 in PS and PSID 0+S in BC
# But according to the paper, laborers should be BC and professional/service should be PS
# Let me try the paper-faithful mapping

# Paper-faithful:
# PS = Professional(0), Managers(1), Clerical(3), Sales(4), Service(9/S)
# BC = Craftsmen(5), Operatives(6), Laborers(7/L)
# Excluded = Self-employed(2), Farmers(8)

ps_paper = df_v2[df_v2['occ_v2'].isin(['0', '1', '3', '4', 'S'])]
bc_paper = df_v2[df_v2['occ_v2'].isin(['5', '6', '7', 'L'])]
bc_nu_paper = bc_paper[bc_paper['union_member'] == 0]
bc_u_paper = bc_paper[bc_paper['union_member'] == 1]

print(f"\nPaper-faithful PS=[0,1,3,4,S], BC=[5,6,7,L]:")
print(f"  PS={len(ps_paper)} vs 4946 ({abs(len(ps_paper)-4946)/4946:.1%})")
print(f"  NU={len(bc_nu_paper)} vs 2642 ({abs(len(bc_nu_paper)-2642)/2642:.1%})")
print(f"  U={len(bc_u_paper)} vs 2741 ({abs(len(bc_u_paper)-2741)/2741:.1%})")
for n, t, l in [(len(ps_paper), 4946, 'PS'), (len(bc_nu_paper), 2642, 'NU'), (len(bc_u_paper), 2741, 'U')]:
    re = abs(n - t) / t
    pts = 5 if re <= 0.05 else (3 if re <= 0.10 else (1 if re <= 0.20 else 0))
    print(f"  {l}: {n} vs {t} ({re:.1%}) -> {pts}/5")
