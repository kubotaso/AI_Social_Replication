"""Try to find the exact grouping that gives paper's sample sizes"""
import pandas as pd
import numpy as np
import itertools

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

df = pd.read_csv('data/psid_panel.csv')
df['education_years'] = df['education_clean'].map(EDUC_MAP).fillna(12)
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df = df[df['tenure_topel'] >= 1].copy()
df = df.dropna(subset=['log_hourly_wage']).copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

# Map 3-digit codes to 1-digit
def to_1digit(occ):
    if occ <= 9:
        return occ
    elif 1 <= occ <= 195:
        return 0
    elif 201 <= occ <= 245:
        return 1
    elif 260 <= occ <= 285:
        return 4  # Sales
    elif 301 <= occ <= 395:
        return 3  # Clerical
    elif 401 <= occ <= 580:
        return 5  # Craftsmen
    elif 601 <= occ <= 695:
        return 6  # Operatives
    elif 701 <= occ <= 785:
        return 7  # Transport + Laborers
    elif 801 <= occ <= 824:
        return 8
    elif 900 <= occ <= 965:
        return 7  # Service -> 7 (ambiguous with laborers)
    else:
        return 9

df['occ_m'] = df['occ_1digit'].apply(to_1digit)

# Build union measures
df = df.sort_values(['person_id', 'job_id', 'year'])
ju = df.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
df = df.merge(ju, on=['person_id', 'job_id'], how='left')

# Counts by occ_m
print("occ_m counts:")
for v in sorted(df['occ_m'].unique()):
    print(f"  {v}: {len(df[df['occ_m']==v])}")

# Paper targets: PS=4946, BC_NU=2642, BC_U=2741
# Total PS+BC_NU+BC_U = 10329

# Strategy: try all combinations of which 1-digit codes go to PS vs BC
# PS candidates: 0, 1, 3, 4, 7
# BC candidates: 5, 6, 7
# Excluded: 2, 8, 9

# For union: try both obs and job level
# Let's brute-force search

ps_candidates = [0, 1, 3, 4, 7]
bc_fixed = [5, 6]  # These always go to BC

best_configs = []

for r in range(1, len(ps_candidates) + 1):
    for ps_combo in itertools.combinations(ps_candidates, r):
        # BC gets: fixed BC codes + any PS candidate NOT in ps_combo + optionally 7
        bc_from_remaining = [x for x in ps_candidates if x not in ps_combo and x not in bc_fixed]
        # BC = bc_fixed + bc_from_remaining
        bc_combo = list(bc_fixed) + bc_from_remaining

        ps_data = df[df['occ_m'].isin(list(ps_combo))]
        bc_data = df[df['occ_m'].isin(bc_combo)]

        ps_n = len(ps_data)

        for union_type in ['obs', 'job']:
            if union_type == 'obs':
                nu = bc_data[bc_data['union_member'] == 0]
                u = bc_data[bc_data['union_member'] == 1]
            else:
                nu = bc_data[bc_data['job_union'] == 0]
                u = bc_data[bc_data['job_union'] == 1]

            nu_n = len(nu)
            u_n = len(u)
            total = ps_n + nu_n + u_n

            # Check closeness to paper
            ps_err = abs(ps_n - 4946) / 4946
            nu_err = abs(nu_n - 2642) / 2642
            u_err = abs(u_n - 2741) / 2741
            total_err = abs(total - 10329) / 10329

            if ps_err < 0.3 and total_err < 0.15:
                score = ps_err + nu_err + u_err + total_err
                best_configs.append({
                    'ps': list(ps_combo),
                    'bc': bc_combo,
                    'union': union_type,
                    'ps_n': ps_n,
                    'nu_n': nu_n,
                    'u_n': u_n,
                    'total': total,
                    'score': score
                })

best_configs.sort(key=lambda x: x['score'])
print("\nTop configurations (closest to paper Ns):")
for c in best_configs[:15]:
    print(f"  PS={c['ps']}, BC={c['bc']}, union={c['union']}")
    print(f"    PS={c['ps_n']}, NU={c['nu_n']}, U={c['u_n']}, Total={c['total']}")
    print(f"    Errors: PS={abs(c['ps_n']-4946)/4946:.1%}, NU={abs(c['nu_n']-2642)/2642:.1%}, U={abs(c['u_n']-2741)/2741:.1%}")

# Also try with 7 split: service(7) -> PS, laborers(7) -> BC
# But we can't split 1-digit 7. Let's try with service codes in PS
def to_1digit_v2(occ):
    """Version 2: service workers (900-965) go to PS category (mapped as 77)"""
    if occ <= 9:
        return occ
    elif 1 <= occ <= 195:
        return 0
    elif 201 <= occ <= 245:
        return 1
    elif 260 <= occ <= 285:
        return 4
    elif 301 <= occ <= 395:
        return 3
    elif 401 <= occ <= 580:
        return 5
    elif 601 <= occ <= 695:
        return 6
    elif 701 <= occ <= 785:
        return 77  # Laborers -> BC (distinct code)
    elif 801 <= occ <= 824:
        return 8
    elif 900 <= occ <= 965:
        return 88  # Service -> PS (distinct code)
    else:
        return 9

df['occ_v2'] = df['occ_1digit'].apply(to_1digit_v2)

# Now try with service split
# PS = {0, 1, 3, 4, 88} and maybe parts of 7 (ambiguous)
# BC = {5, 6, 77} and maybe parts of 7 (ambiguous)

# The 1-digit occ=7 can't be split
# Try: 1-digit 7 -> BC (as "laborers")
ps_v2_data = df[df['occ_v2'].isin([0, 1, 3, 4, 88])]
bc_v2_data = df[df['occ_v2'].isin([5, 6, 7, 77])]
print(f"\nSplit service: PS(0,1,3,4,service)={len(ps_v2_data)}, BC(5,6,7,laborers)={len(bc_v2_data)}")

bc_v2_data = bc_v2_data.sort_values(['person_id', 'job_id', 'year']).copy()
ju_v2 = bc_v2_data.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
bc_v2_data = bc_v2_data.merge(ju_v2, on=['person_id', 'job_id'], how='left')

for ut in ['obs', 'job']:
    if ut == 'obs':
        nu = bc_v2_data[bc_v2_data['union_member'] == 0]
        u = bc_v2_data[bc_v2_data['union_member'] == 1]
    else:
        nu = bc_v2_data[bc_v2_data['job_union'] == 0]
        u = bc_v2_data[bc_v2_data['job_union'] == 1]
    print(f"  {ut}: NU={len(nu)}, U={len(u)}, Total={len(ps_v2_data)+len(nu)+len(u)}")

# Try: 1-digit 7 -> PS (as "service workers")
ps_v3 = df[df['occ_v2'].isin([0, 1, 3, 4, 7, 88])]
bc_v3 = df[df['occ_v2'].isin([5, 6, 77])]
print(f"\n7->PS: PS(0,1,3,4,7,service)={len(ps_v3)}, BC(5,6,laborers)={len(bc_v3)}")

bc_v3 = bc_v3.sort_values(['person_id', 'job_id', 'year']).copy()
ju_v3 = bc_v3.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
bc_v3 = bc_v3.merge(ju_v3, on=['person_id', 'job_id'], how='left')

for ut in ['obs', 'job']:
    if ut == 'obs':
        nu = bc_v3[bc_v3['union_member'] == 0]
        u = bc_v3[bc_v3['union_member'] == 1]
    else:
        nu = bc_v3[bc_v3['job_union'] == 0]
        u = bc_v3[bc_v3['job_union'] == 1]
    print(f"  {ut}: NU={len(nu)}, U={len(u)}, Total={len(ps_v3)+len(nu)+len(u)}")

# Maybe the paper excludes some specific codes. Let's check:
# What if PS excludes occ=0? (Maybe 0 means "unknown")
ps_no0 = df[df['occ_v2'].isin([1, 3, 4, 88])]
bc_with07 = df[df['occ_v2'].isin([5, 6, 7, 77])]
print(f"\nPS without 0: PS={len(ps_no0)}, BC(5,6,7,lab)={len(bc_with07)}")
