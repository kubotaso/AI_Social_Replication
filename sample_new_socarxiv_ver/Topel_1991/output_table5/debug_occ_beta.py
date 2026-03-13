"""Test whether better occupation groupings fix the BC beta_1 differentiation"""
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
PAPER_G2 = -0.004592
PAPER_G3 = 0.0001846
PAPER_G4 = -0.00000245
PAPER_D2 = -0.006051
PAPER_D3 = 0.0002067
PAPER_D4 = -0.00000238

df = pd.read_csv('data/psid_panel.csv')
df['education_years'] = df['education_clean'].map(EDUC_MAP).fillna(12)
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df['tenure'] = df['tenure_topel'].copy()
df = df[df['tenure'] >= 1].copy()
df = df.dropna(subset=['log_hourly_wage']).copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
df['gnp_defl'] = (df['year'] - 1).map(GNP_DEFLATOR)
df['log_real_wage'] = (df['log_hourly_wage']
                       - np.log(df['gnp_defl'] / 100.0)
                       - np.log(df['cps_index']))
df['log_wage_cps'] = df['log_hourly_wage'] - np.log(df['cps_index'])
df['initial_experience'] = (df['experience'] - df['tenure']).clip(lower=0)

for c in ['married', 'disabled', 'region_ne', 'region_nc', 'region_south']:
    if c in df.columns:
        df[c] = df[c].fillna(0)
for y in range(1969, 1984):
    col = f'year_{y}'
    if col not in df.columns:
        df[col] = (df['year'] == y).astype(int)

# Map all occupation codes to 1-digit
def to_1digit(occ):
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
        return 7  # Transport + Laborers
    elif 801 <= occ <= 824:
        return 8
    elif 900 <= occ <= 965:
        return 7  # Service -> map to 7 (will handle separately)
    else:
        return 9

df['occ_m'] = df['occ_1digit'].apply(to_1digit)

# Build job-level union
df = df.sort_values(['person_id', 'job_id', 'year'])
ju = df.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
df = df.merge(ju, on=['person_id', 'job_id'], how='left')


def quick_2step(sub, b1b2, name=""):
    """Quick 2-step to get beta_1"""
    sub = sub.copy()
    sub['w_star'] = (sub['log_real_wage']
                     - b1b2 * sub['tenure']
                     - PAPER_G2 * sub['tenure']**2
                     - PAPER_G3 * sub['tenure']**3
                     - PAPER_G4 * sub['tenure']**4
                     - PAPER_D2 * sub['experience']**2
                     - PAPER_D3 * sub['experience']**3
                     - PAPER_D4 * sub['experience']**4)

    ctrls = ['education_years', 'married', 'disabled',
             'region_ne', 'region_nc', 'region_south']
    yr_use = [f'year_{y}' for y in range(1969, 1984)
              if f'year_{y}' in sub.columns and sub[f'year_{y}'].std() > 0]

    step2 = sub.dropna(subset=['w_star', 'experience', 'initial_experience']).copy()
    ctrls_use = [c for c in ctrls if c in step2.columns and step2[c].std() > 0]

    y_vals = step2['w_star'].values.astype(float)
    exp_v = step2['experience'].values.astype(float)
    x0_v = step2['initial_experience'].values.astype(float)
    ones = np.ones(len(step2))

    ctrl_cols = ctrls_use + yr_use
    while True:
        C = step2[ctrl_cols].values.astype(float)
        Z = np.column_stack([ones, C, x0_v])
        if np.linalg.matrix_rank(Z) >= Z.shape[1]:
            break
        if len(yr_use) == 0:
            break
        yr_use = yr_use[:-1]
        ctrl_cols = ctrls_use + yr_use

    C = step2[ctrl_cols].values.astype(float)
    Z = np.column_stack([ones, C, x0_v])
    X = np.column_stack([ones, C, exp_v])

    pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
    exp_hat = Z @ pi
    X_hat = np.column_stack([ones, C, exp_hat])
    b = np.linalg.lstsq(X_hat, y_vals, rcond=None)[0]

    beta_1 = b[-1]
    beta_2 = b1b2 - beta_1
    return beta_1, beta_2, len(step2)


# Test promising configurations
configs = [
    ("Current: PS=[1,2,4,8], BC=[5,6,7], job", [1, 2, 4, 8], [5, 6, 7], 'job'),
    ("PS=[0,1,3,4], BC=[5,6,7], job", [0, 1, 3, 4], [5, 6, 7], 'job'),
    ("PS=[0,1,3,4], BC=[5,6,7], obs", [0, 1, 3, 4], [5, 6, 7], 'obs'),
    ("PS=[0,1,3], BC=[5,6,4,7], obs", [0, 1, 3], [5, 6, 4, 7], 'obs'),
    ("PS=[1,3,4,7], BC=[5,6,0], obs", [1, 3, 4, 7], [5, 6, 0], 'obs'),
    ("PS=[0,3], BC=[5,6,1,4,7], obs", [0, 3], [5, 6, 1, 4, 7], 'obs'),
    ("PS=[0,1], BC=[5,6,3,4,7], obs", [0, 1], [5, 6, 3, 4, 7], 'obs'),
    ("PS=[0,1,3,4,7], BC=[5,6], job", [0, 1, 3, 4, 7], [5, 6], 'job'),
    ("PS=[0,1,3,4,7], BC=[5,6], obs", [0, 1, 3, 4, 7], [5, 6], 'obs'),
]

print("Paper targets: PS beta_1=0.0707, BC_NU beta_1=0.1066, BC_U beta_1=0.0592")
print("Paper Ns:      PS=4946, BC_NU=2642, BC_U=2741\n")

for label, ps_codes, bc_codes, union_type in configs:
    ps_data = df[df['occ_m'].isin(ps_codes)]
    bc_data = df[df['occ_m'].isin(bc_codes)]

    if union_type == 'obs':
        bc_nu = bc_data[bc_data['union_member'] == 0].copy()
        bc_u = bc_data[bc_data['union_member'] == 1].copy()
    else:
        bc_nu = bc_data[bc_data['job_union'] == 0].copy()
        bc_u = bc_data[bc_data['job_union'] == 1].copy()

    ps_b1, ps_b2, ps_n = quick_2step(ps_data, 0.1309, "PS")
    nu_b1, nu_b2, nu_n = quick_2step(bc_nu, 0.1520, "BC_NU")
    u_b1, u_b2, u_n = quick_2step(bc_u, 0.0992, "BC_U")

    print(f"{label}")
    print(f"  PS: N={ps_n:5d}, b1={ps_b1:.4f} (target 0.0707)")
    print(f"  NU: N={nu_n:5d}, b1={nu_b1:.4f} (target 0.1066)")
    print(f"  U:  N={u_n:5d}, b1={u_b1:.4f} (target 0.0592)")
    print(f"  b1_diff(NU-U)={nu_b1-u_b1:.4f} (target 0.0474)")
    print()
