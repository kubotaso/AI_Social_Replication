"""Try different instrument strategies to improve beta_1 differentiation"""
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
df['log_wage_cps'] = df['log_hourly_wage'] - np.log(df['cps_index'])
df['log_real_wage'] = df['log_hourly_wage'] - np.log(df['gnp_defl']/100.0) - np.log(df['cps_index'])
df['initial_experience'] = (df['experience'] - df['tenure']).clip(lower=0)
for c in ['married', 'disabled', 'region_ne', 'region_nc', 'region_south']:
    if c in df.columns: df[c] = df[c].fillna(0)
for y in range(1969, 1984):
    col = f'year_{y}'
    if col not in df.columns: df[col] = (df['year'] == y).astype(int)

bc = df[df['occ_mapped'].isin(['0', '5', '6', 'L', 'S'])]
bc_nu = bc[bc['union_member'] == 0].copy()
bc_u = bc[bc['union_member'] == 1].copy()

def run_2step(sub, b1b2, label, add_x0_sq=False, use_education_iv=False):
    sub = sub.copy()
    sub['w_star'] = (sub['log_real_wage']
                     - b1b2 * sub['tenure']
                     - PAPER_G2 * sub['tenure']**2 - PAPER_G3 * sub['tenure']**3 - PAPER_G4 * sub['tenure']**4
                     - PAPER_D2 * sub['experience']**2 - PAPER_D3 * sub['experience']**3 - PAPER_D4 * sub['experience']**4)

    ctrls = ['education_years', 'married', 'disabled', 'region_ne', 'region_nc', 'region_south']
    yr_use = [f'year_{y}' for y in range(1969, 1984) if f'year_{y}' in sub.columns and sub[f'year_{y}'].std() > 0]
    step2 = sub.dropna(subset=['w_star', 'experience', 'initial_experience']).copy()
    ctrls_use = [c for c in ctrls if c in step2.columns and step2[c].std() > 0]

    y = step2['w_star'].values.astype(float)
    exp_v = step2['experience'].values.astype(float)
    x0_v = step2['initial_experience'].values.astype(float)
    ones = np.ones(len(step2))
    ctrl_cols = ctrls_use + yr_use

    while True:
        C = step2[ctrl_cols].values.astype(float)
        if add_x0_sq:
            x0_sq = x0_v**2
            Z = np.column_stack([ones, C, x0_v, x0_sq])
        elif use_education_iv:
            educ = step2['education_years'].values.astype(float)
            Z = np.column_stack([ones, C, x0_v, educ * x0_v])
        else:
            Z = np.column_stack([ones, C, x0_v])
        if np.linalg.matrix_rank(Z) >= Z.shape[1]: break
        if len(yr_use) == 0: break
        yr_use = yr_use[:-1]
        ctrl_cols = ctrls_use + yr_use

    C = step2[ctrl_cols].values.astype(float)
    if add_x0_sq:
        x0_sq = x0_v**2
        Z = np.column_stack([ones, C, x0_v, x0_sq])
    elif use_education_iv:
        educ = step2['education_years'].values.astype(float)
        Z = np.column_stack([ones, C, x0_v, educ * x0_v])
    else:
        Z = np.column_stack([ones, C, x0_v])
    X = np.column_stack([ones, C, exp_v])

    pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
    exp_hat = Z @ pi
    X_hat = np.column_stack([ones, C, exp_hat])
    b = np.linalg.lstsq(X_hat, y, rcond=None)[0]
    beta_1 = b[-1]
    beta_2 = b1b2 - beta_1
    return beta_1, beta_2

# Test different instrument strategies
print("Testing different instrument strategies for BC groups:")
print(f"Paper: BC_NU beta_1=0.1066, BC_U beta_1=0.0592")
print(f"Diff = {0.1066-0.0592:.4f}")
print()

strategies = [
    ("Baseline (x0 only)", False, False),
    ("x0 + x0^2", True, False),
    ("x0 + educ*x0", False, True),
]

for name, add_sq, use_edu in strategies:
    nu_b1, nu_b2 = run_2step(bc_nu, 0.1520, 'BC_NU', add_x0_sq=add_sq, use_education_iv=use_edu)
    u_b1, u_b2 = run_2step(bc_u, 0.0992, 'BC_U', add_x0_sq=add_sq, use_education_iv=use_edu)
    print(f"{name}:")
    print(f"  BC_NU: b1={nu_b1:.4f} (paper 0.1066), b2={nu_b2:.4f} (paper 0.0513)")
    print(f"  BC_U:  b1={u_b1:.4f} (paper 0.0592), b2={u_b2:.4f} (paper 0.0399)")
    print(f"  Diff:  {nu_b1 - u_b1:.4f} (paper {0.1066-0.0592:.4f})")
    print()

# Also try: what if we use tenure as an additional control in step 2?
# This is non-standard but let's see
print("\nAdditional tests:")

# Try including initial tenure (=1 for everyone due to tenure>=1 filter) -- not useful
# Try different experience measure
# Try restricting sample to first observation per person

for label, sub, b1b2, paper_b1 in [('BC_NU', bc_nu, 0.1520, 0.1066), ('BC_U', bc_u, 0.0992, 0.0592)]:
    # First obs per person
    first_obs = sub.groupby('person_id').first().reset_index()
    first_obs = first_obs.dropna(subset=['log_real_wage', 'experience', 'initial_experience'])
    b1, b2 = run_2step(first_obs, b1b2, label)
    print(f"{label} first-obs only (N={len(first_obs)}): b1={b1:.4f} (paper {paper_b1})")

# What about using age as instrument instead of initial_experience?
print("\nAge as instrument:")
for label, sub, b1b2, paper_b1 in [('BC_NU', bc_nu, 0.1520, 0.1066), ('BC_U', bc_u, 0.0992, 0.0592)]:
    sub = sub.copy()
    sub['w_star'] = (sub['log_real_wage']
                     - b1b2 * sub['tenure']
                     - PAPER_G2 * sub['tenure']**2 - PAPER_G3 * sub['tenure']**3 - PAPER_G4 * sub['tenure']**4
                     - PAPER_D2 * sub['experience']**2 - PAPER_D3 * sub['experience']**3 - PAPER_D4 * sub['experience']**4)
    ctrls = ['education_years', 'married', 'disabled', 'region_ne', 'region_nc', 'region_south']
    yr_use = [f'year_{y}' for y in range(1969, 1984) if f'year_{y}' in sub.columns and sub[f'year_{y}'].std() > 0]
    step2 = sub.dropna(subset=['w_star', 'experience', 'initial_experience']).copy()
    ctrls_use = [c for c in ctrls if c in step2.columns and step2[c].std() > 0]
    y_v = step2['w_star'].values.astype(float)
    exp_v = step2['experience'].values.astype(float)
    age_v = step2['age'].values.astype(float)
    x0_v = step2['initial_experience'].values.astype(float)
    ones = np.ones(len(step2))
    ctrl_cols = ctrls_use + yr_use
    C = step2[ctrl_cols].values.astype(float)
    # Instrument with age instead
    Z = np.column_stack([ones, C, age_v])
    while np.linalg.matrix_rank(Z) < Z.shape[1]:
        yr_use = yr_use[:-1]
        ctrl_cols = ctrls_use + yr_use
        C = step2[ctrl_cols].values.astype(float)
        Z = np.column_stack([ones, C, age_v])
    X = np.column_stack([ones, C, exp_v])
    pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
    exp_hat = Z @ pi
    X_hat = np.column_stack([ones, C, exp_hat])
    b = np.linalg.lstsq(X_hat, y_v, rcond=None)[0]
    beta_1 = b[-1]
    beta_2 = b1b2 - beta_1
    print(f"  {label}: b1={beta_1:.4f} (paper {paper_b1})")
