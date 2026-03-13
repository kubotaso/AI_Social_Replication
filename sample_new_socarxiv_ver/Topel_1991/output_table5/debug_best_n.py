"""Find the grouping that maximizes N score (closest to paper Ns)"""
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

# Map 3-digit codes
def to_1digit(occ):
    if occ <= 9:
        return occ
    elif 1 <= occ <= 195: return 0
    elif 201 <= occ <= 245: return 1
    elif 260 <= occ <= 285: return 4
    elif 301 <= occ <= 395: return 3
    elif 401 <= occ <= 580: return 5
    elif 601 <= occ <= 695: return 6
    elif 701 <= occ <= 785: return 7
    elif 801 <= occ <= 824: return 8
    elif 900 <= occ <= 965: return 7  # Service/Laborers
    else: return 9

df['occ_m'] = df['occ_1digit'].apply(to_1digit)
df = df.sort_values(['person_id', 'job_id', 'year'])
ju = df.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
df = df.merge(ju, on=['person_id', 'job_id'], how='left')

# Score: points for N matching
# 5 pts if within 5%, 3 pts if within 10%, 1 pt if within 15%
def n_score(gen, true):
    re = abs(gen - true) / true
    if re <= 0.05: return 5
    elif re <= 0.10: return 3
    elif re <= 0.15: return 1
    elif re <= 0.20: return 1
    return 0

# Try ALL combinations
all_codes = [0, 1, 3, 4, 5, 6, 7]
bc_must = [5, 6]  # These are always BC

best = []
for ps_size in range(1, len(all_codes)):
    for ps_combo in itertools.combinations(all_codes, ps_size):
        ps_codes = list(ps_combo)
        if 5 in ps_codes or 6 in ps_codes:
            continue  # 5,6 always BC
        bc_codes = [c for c in all_codes if c not in ps_codes]
        if not bc_codes:
            continue

        ps_data = df[df['occ_m'].isin(ps_codes)]
        bc_data = df[df['occ_m'].isin(bc_codes)]
        ps_n = len(ps_data)

        for ut in ['obs', 'job']:
            if ut == 'obs':
                nu = bc_data[bc_data['union_member'] == 0]
                u = bc_data[bc_data['union_member'] == 1]
            else:
                nu = bc_data[bc_data['job_union'] == 0]
                u = bc_data[bc_data['job_union'] == 1]

            nu_n = len(nu)
            u_n = len(u)

            s = n_score(ps_n, 4946) + n_score(nu_n, 2642) + n_score(u_n, 2741)
            if s > 0:
                best.append({
                    'ps': ps_codes, 'bc': bc_codes, 'union': ut,
                    'ps_n': ps_n, 'nu_n': nu_n, 'u_n': u_n,
                    'score': s,
                    'ps_err': abs(ps_n-4946)/4946,
                    'nu_err': abs(nu_n-2642)/2642,
                    'u_err': abs(u_n-2741)/2741,
                })

best.sort(key=lambda x: (-x['score'], x['ps_err']+x['nu_err']+x['u_err']))
print("Top N-score configurations:")
for i, c in enumerate(best[:20]):
    print(f"  {i+1}. PS={c['ps']}, BC={c['bc']}, union={c['union']}")
    print(f"     PS={c['ps_n']} ({c['ps_err']:.1%}), NU={c['nu_n']} ({c['nu_err']:.1%}), U={c['u_n']} ({c['u_err']:.1%})")
    print(f"     N score: {c['score']}/15")
