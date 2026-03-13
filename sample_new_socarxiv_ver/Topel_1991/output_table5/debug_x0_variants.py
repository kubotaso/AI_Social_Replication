"""Test different initial_experience constructions"""
import pandas as pd
import numpy as np
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
PAPER_G2 = -0.004592; PAPER_G3 = 0.0001846; PAPER_G4 = -0.00000245
PAPER_D2 = -0.006051; PAPER_D3 = 0.0002067; PAPER_D4 = -0.00000238

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
df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
df['gnp_defl'] = (df['year'] - 1).map(GNP_DEFLATOR)
df['log_real_wage'] = df['log_hourly_wage'] - np.log(df['gnp_defl']/100.0) - np.log(df['cps_index'])
for c in ['married', 'disabled', 'region_ne', 'region_nc', 'region_south']:
    if c in df.columns: df[c] = df[c].fillna(0)
for y in range(1969, 1984):
    col = f'year_{y}'
    if col not in df.columns: df[col] = (df['year'] == y).astype(int)

# Different x0 constructions
# 1. Current: x0 = (experience - tenure).clip(0) = (age - edu - 6 - tenure).clip(0)
df['x0_current'] = (df['experience'] - df['tenure']).clip(lower=0)

# 2. x0 = experience at first observation of this job
df = df.sort_values(['person_id', 'job_id', 'year'])
first_exp = df.groupby(['person_id', 'job_id'])['experience'].first().reset_index()
first_exp.columns = ['person_id', 'job_id', 'x0_first_obs']
df = df.merge(first_exp, on=['person_id', 'job_id'], how='left')

# 3. x0 = age_at_job_start - edu - 6 (use first year of job to compute age at start)
first_year = df.groupby(['person_id', 'job_id'])['year'].first().reset_index()
first_year.columns = ['person_id', 'job_id', 'job_start_year']
first_age = df.groupby(['person_id', 'job_id'])['age'].first().reset_index()
first_age.columns = ['person_id', 'job_id', 'age_at_start']
df = df.merge(first_year, on=['person_id', 'job_id'], how='left')
df = df.merge(first_age, on=['person_id', 'job_id'], how='left')
df['x0_age_start'] = (df['age_at_start'] - df['education_years'] - 6).clip(lower=0)

bc = df[df['occ_mapped'].isin(['0', '5', '6', 'L', 'S'])]
bc_nu = bc[bc['union_member'] == 0].copy()
bc_u = bc[bc['union_member'] == 1].copy()
ps = df[df['occ_mapped'].isin(['1', '3', '4', '7'])]

def q2s(sub, b1b2, x0_col):
    sub = sub.copy()
    sub['w_star'] = (sub['log_real_wage']
                     - b1b2 * sub['tenure']
                     - PAPER_G2 * sub['tenure']**2 - PAPER_G3 * sub['tenure']**3 - PAPER_G4 * sub['tenure']**4
                     - PAPER_D2 * sub['experience']**2 - PAPER_D3 * sub['experience']**3 - PAPER_D4 * sub['experience']**4)
    ctrls = ['education_years', 'married', 'disabled', 'region_ne', 'region_nc', 'region_south']
    yr_use = [f'year_{y}' for y in range(1969, 1984) if f'year_{y}' in sub.columns and sub[f'year_{y}'].std() > 0]
    step2 = sub.dropna(subset=['w_star', 'experience', x0_col]).copy()
    ctrls_use = [c for c in ctrls if c in step2.columns and step2[c].std() > 0]
    y_v = step2['w_star'].values.astype(float)
    exp_v = step2['experience'].values.astype(float)
    x0_v = step2[x0_col].values.astype(float)
    ones = np.ones(len(step2))
    ctrl_cols = ctrls_use + yr_use
    while True:
        C = step2[ctrl_cols].values.astype(float)
        Z = np.column_stack([ones, C, x0_v])
        if np.linalg.matrix_rank(Z) >= Z.shape[1]: break
        if len(yr_use) == 0: break
        yr_use = yr_use[:-1]
        ctrl_cols = ctrls_use + yr_use
    C = step2[ctrl_cols].values.astype(float)
    Z = np.column_stack([ones, C, x0_v])
    X = np.column_stack([ones, C, exp_v])
    pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
    exp_hat = Z @ pi
    X_hat = np.column_stack([ones, C, exp_hat])
    b = np.linalg.lstsq(X_hat, y_v, rcond=None)[0]
    return b[-1]

print("x0 variant effect on beta_1:")
print(f"Paper: PS=0.0707, NU=0.1066, U=0.0592\n")
for name, x0_col in [("x0 = exp - tenure (current)", "x0_current"),
                       ("x0 = exp at first obs", "x0_first_obs"),
                       ("x0 = age_at_start - edu - 6", "x0_age_start")]:
    ps_b1 = q2s(ps, 0.1309, x0_col)
    nu_b1 = q2s(bc_nu, 0.1520, x0_col)
    u_b1 = q2s(bc_u, 0.0992, x0_col)
    print(f"{name:35s}: PS={ps_b1:.4f}, NU={nu_b1:.4f}, U={u_b1:.4f}, diff={nu_b1-u_b1:.4f}")
