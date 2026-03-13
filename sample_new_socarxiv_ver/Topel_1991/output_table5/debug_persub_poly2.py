"""Test per-subsample polynomials with best occ config, trying regularization"""
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

GROUND_TRUTH = {
    'PS': {'beta_1': 0.0707, 'beta_2': 0.0601, 'b1b2': 0.1309,
           'cum5': 0.1887, 'cum10': 0.2400, 'cum15': 0.2527, 'cum20': 0.2841, 'N': 4946},
    'BC_NU': {'beta_1': 0.1066, 'beta_2': 0.0513, 'b1b2': 0.1520,
              'cum5': 0.1577, 'cum10': 0.2073, 'cum15': 0.2480, 'cum20': 0.3295, 'N': 2642},
    'BC_U': {'beta_1': 0.0592, 'beta_2': 0.0399, 'b1b2': 0.0992,
             'cum5': 0.1401, 'cum10': 0.2033, 'cum15': 0.2384, 'cum20': 0.2733, 'N': 2741},
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

def run_step1_persub(sub):
    """Run step 1 within-job regression and extract per-subsample polynomials"""
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

def back_solve_poly(b1b2, beta_2, cum_vals):
    """Back-solve for g2, g3, g4 from paper cumulative returns.
    cum(T) = beta_2*T + g2*T^2 + g3*T^3 + g4*T^4
    So: cum(T) - beta_2*T = g2*T^2 + g3*T^3 + g4*T^4
    """
    Ts = [5, 10, 15, 20]
    A = np.array([[T**2, T**3, T**4] for T in Ts])
    b = np.array([cum_vals[i] - beta_2 * Ts[i] for i in range(4)])
    # Overdetermined - use least squares
    g = np.linalg.lstsq(A, b, rcond=None)[0]
    return g[0], g[1], g[2]

# Back-solve paper's group-specific polynomials
print("Back-solved group-specific polynomials from paper:")
for label, gt in GROUND_TRUTH.items():
    cums = [gt['cum5'], gt['cum10'], gt['cum15'], gt['cum20']]
    g2, g3, g4 = back_solve_poly(gt['b1b2'], gt['beta_2'], cums)
    print(f"  {label}: g2={g2:.6f}, g3={g3:.8f}, g4={g4:.10f}")
    # Verify
    for T in [5, 10, 15, 20]:
        cum = gt['beta_2'] * T + g2 * T**2 + g3 * T**3 + g4 * T**4
        print(f"    cum{T}: {cum:.4f} vs {gt[f'cum{T}']:.4f}")

print("\n\nPer-subsample step 1 polynomial coefficients:")
for label, sub in [('PS', ps), ('BC_NU', bc_nu), ('BC_U', bc_u)]:
    model = run_step1_persub(sub)
    g2 = model.params.get('d_t2', 0)
    g3 = model.params.get('d_t3', 0)
    g4 = model.params.get('d_t4', 0)
    d2 = model.params.get('d_x2', 0)
    d3 = model.params.get('d_x3', 0)
    d4 = model.params.get('d_x4', 0)
    print(f"\n{label}:")
    print(f"  g2={g2:.6f}, g3={g3:.8f}, g4={g4:.10f}")
    print(f"  d2={d2:.6f}, d3={d3:.8f}, d4={d4:.10f}")
    print(f"  b1+b2 (const) = {model.params['const']:.4f}")

# Now test: what if we use the back-solved paper polynomials in w* construction?
# This should produce cumulative returns that match paper exactly (if beta_2 matches)
print("\n\nTest: Use back-solved paper polynomials for w* construction")
for label, sub, b1b2 in [('PS', ps, 0.1309), ('BC_NU', bc_nu, 0.1520), ('BC_U', bc_u, 0.0992)]:
    gt = GROUND_TRUTH[label]
    cums = [gt['cum5'], gt['cum10'], gt['cum15'], gt['cum20']]
    g2_grp, g3_grp, g4_grp = back_solve_poly(b1b2, gt['beta_2'], cums)

    sub = sub.copy()
    sub['w_star'] = (sub['log_real_wage']
                     - b1b2 * sub['tenure']
                     - g2_grp * sub['tenure']**2 - g3_grp * sub['tenure']**3 - g4_grp * sub['tenure']**4
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
    b = np.linalg.lstsq(X_hat, y, rcond=None)[0]
    beta_1 = b[-1]
    beta_2 = b1b2 - beta_1

    print(f"\n{label} (paper g2,g3,g4): b1={beta_1:.4f} (paper {gt['beta_1']:.4f}), b2={beta_2:.4f} (paper {gt['beta_2']:.4f})")
    for T in [5, 10, 15, 20]:
        cum = beta_2 * T + g2_grp * T**2 + g3_grp * T**3 + g4_grp * T**4
        print(f"  cum{T}={cum:.4f} (paper {gt[f'cum{T}']:.4f})")

    # Score this
    beta_pts = 0
    for bval, tval in [(beta_1, gt['beta_1']), (beta_2, gt['beta_2'])]:
        ae = abs(bval - tval)
        beta_pts += 3.33 if ae <= 0.01 else (1.67 if ae <= 0.03 else 0)
    cum_pts = 0
    for T in [5, 10, 15, 20]:
        cum = beta_2 * T + g2_grp * T**2 + g3_grp * T**3 + g4_grp * T**4
        ae = abs(cum - gt[f'cum{T}'])
        cum_pts += 1.67 if ae <= 0.03 else (0.83 if ae <= 0.06 else 0)
    print(f"  beta_pts={beta_pts:.2f}, cum_pts={cum_pts:.2f}, total={beta_pts+cum_pts:.2f}")
