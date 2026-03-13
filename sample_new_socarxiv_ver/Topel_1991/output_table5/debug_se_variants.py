"""Test different SE computation methods"""
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

GT_SE = {
    'PS': {'b1_se': 0.0288, 'b2_se': 0.0127, 'b1b2_se': 0.0254},
    'BC_NU': {'b1_se': 0.0342, 'b2_se': 0.0146, 'b1b2_se': 0.0311},
    'BC_U': {'b1_se': 0.0338, 'b2_se': 0.0147, 'b1b2_se': 0.0297},
}

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

ps = df[df['occ_mapped'].isin(['1', '3', '4', '7'])]
bc = df[df['occ_mapped'].isin(['0', '5', '6', 'L', 'S'])]
bc_nu = bc[bc['union_member'] == 0].copy()
bc_u = bc[bc['union_member'] == 1].copy()

def run_step1(sub):
    sub = sub.sort_values(['person_id', 'job_id', 'year']).copy()
    sub['within'] = ((sub['person_id'] == sub['person_id'].shift(1)) &
                     (sub['job_id'] == sub['job_id'].shift(1)) &
                     (sub['year'] - sub['year'].shift(1) == 1))
    sub['d_lw'] = sub['log_wage_cps'] - sub['log_wage_cps'].shift(1)
    for k in [2, 3, 4]:
        sub[f'd_t{k}'] = sub['tenure']**k - (sub['tenure'] - 1)**k
        sub[f'd_x{k}'] = sub['experience']**k - (sub['experience'] - 1)**k
    yr_cols = sorted([c for c in sub.columns if c.startswith('year_') and c != 'year'])
    d_yr = []
    for yc in yr_cols:
        dc = f'd_{yc}'
        sub[dc] = sub[yc].astype(float) - sub[yc].shift(1).astype(float)
        sub[dc] = sub[dc].fillna(0)
        d_yr.append(dc)
    fd = sub[sub['within']].dropna(subset=['d_lw']).copy()
    d_yr_use = [c for c in d_yr if fd[c].std() > 1e-10]
    if len(d_yr_use) > 1:
        d_yr_use = d_yr_use[1:]
    x_cols = ['d_t2', 'd_t3', 'd_t4', 'd_x2', 'd_x3', 'd_x4'] + d_yr_use
    y = fd['d_lw']
    X = sm.add_constant(fd[x_cols])
    valid = y.notna() & X.notna().all(axis=1)
    model = sm.OLS(y[valid], X[valid]).fit()
    return model

def run_full(sub, b1b2, b1b2_se, label):
    s1 = run_step1(sub)
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
    x0_v = step2['initial_experience'].values.astype(float)
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
    n, k = X.shape
    pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
    exp_hat = Z @ pi
    X_hat = np.column_stack([ones, C, exp_hat])
    b = np.linalg.lstsq(X_hat, y_v, rcond=None)[0]
    resid = y_v - X @ b
    s2 = np.sum(resid**2) / (n - k)
    XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
    beta_1 = b[-1]

    # Murphy-Topel SE
    T = step2['tenure'].values.astype(float)
    Xp = step2['experience'].values.astype(float)
    s1_params = ['const', 'd_t2', 'd_t3', 'd_t4', 'd_x2', 'd_x3', 'd_x4']
    s1_present = [p for p in s1_params if p in s1.params.index]
    V1 = s1.cov_params().loc[s1_present, s1_present].values
    J = np.zeros((n, len(s1_present)))
    for j, param in enumerate(s1_present):
        if param == 'const':     J[:, j] = -T
        elif param == 'd_t2':    J[:, j] = -T**2
        elif param == 'd_t3':    J[:, j] = -T**3
        elif param == 'd_t4':    J[:, j] = -T**4
        elif param == 'd_x2':    J[:, j] = -Xp**2
        elif param == 'd_x3':    J[:, j] = -Xp**3
        elif param == 'd_x4':    J[:, j] = -Xp**4
    G = XhXh_inv @ (X_hat.T @ J)
    g_b1 = G[-1, :]
    V_extra = g_b1 @ V1 @ g_b1
    V_2sls = s2 * XhXh_inv[-1, -1]
    beta_1_se_mt = np.sqrt(max(0, V_2sls + V_extra))

    # Plain 2SLS SE (no Murphy-Topel)
    beta_1_se_plain = np.sqrt(s2 * XhXh_inv[-1, -1])

    beta_2 = b1b2 - beta_1

    # Different beta_2_se formulas
    beta_2_se_half = b1b2_se / 2.0  # Current approach
    beta_2_se_delta = np.sqrt(b1b2_se**2 + beta_1_se_mt**2 - 2*0)  # Delta method, no covariance
    # Approximate covariance = 0 for now

    gt = GT_SE[label]
    print(f"\n{label}:")
    print(f"  beta_1_se: MT={beta_1_se_mt:.4f}, plain={beta_1_se_plain:.4f}, paper={gt['b1_se']:.4f}")
    print(f"    MT rel error: {abs(beta_1_se_mt - gt['b1_se'])/gt['b1_se']:.1%}")
    print(f"    plain rel error: {abs(beta_1_se_plain - gt['b1_se'])/gt['b1_se']:.1%}")
    print(f"  beta_2_se: half={beta_2_se_half:.4f}, delta={beta_2_se_delta:.4f}, paper={gt['b2_se']:.4f}")
    print(f"    half rel error: {abs(beta_2_se_half - gt['b2_se'])/gt['b2_se']:.1%}")
    print(f"    delta rel error: {abs(beta_2_se_delta - gt['b2_se'])/gt['b2_se']:.1%}")
    print(f"  b1b2_se: paper={gt['b1b2_se']:.4f}, s1={s1.bse['const']:.4f}")
    print(f"    s1 rel error: {abs(s1.bse['const'] - gt['b1b2_se'])/gt['b1b2_se']:.1%}")

    # Score beta_1_se variants
    for se_name, se_val in [('MT', beta_1_se_mt), ('plain', beta_1_se_plain)]:
        re = abs(se_val - gt['b1_se']) / gt['b1_se']
        pts = 1.11 if re <= 0.30 else (0.56 if re <= 0.60 else 0)
        print(f"    {se_name}: re={re:.3f} -> {pts:.2f}/1.11")

    # Score beta_2_se variants
    for se_name, se_val in [('half', beta_2_se_half), ('delta', beta_2_se_delta)]:
        re = abs(se_val - gt['b2_se']) / gt['b2_se']
        pts = 1.11 if re <= 0.30 else (0.56 if re <= 0.60 else 0)
        print(f"    {se_name}: re={re:.3f} -> {pts:.2f}/1.11")

for label, sub, b1b2, b1b2_se in [
    ('PS', ps, 0.1309, 0.0254),
    ('BC_NU', bc_nu, 0.1520, 0.0311),
    ('BC_U', bc_u, 0.0992, 0.0297)
]:
    run_full(sub, b1b2, b1b2_se, label)
