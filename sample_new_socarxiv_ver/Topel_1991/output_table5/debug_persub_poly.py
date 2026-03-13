"""Test per-subsample polynomials for cumulative returns"""
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

# Get subsamples
ps_codes = ['1', '3', '4', '7']
bc_codes = ['0', '5', '6', 'L', 'S']
ps = df[df['occ_mapped'].isin(ps_codes)]
bc = df[df['occ_mapped'].isin(bc_codes)]
bc_nu = bc[bc['union_member'] == 0].copy()
bc_u = bc[bc['union_member'] == 1].copy()

PAPER_B1B2 = {'PS': 0.1309, 'NU': 0.1520, 'U': 0.0992}
PAPER_G2 = -0.004592; PAPER_G3 = 0.0001846; PAPER_G4 = -0.00000245

GROUND_TRUTH_CUM = {
    'PS': {5: 0.1887, 10: 0.2400, 15: 0.2527, 20: 0.2841},
    'NU': {5: 0.1577, 10: 0.2073, 15: 0.2480, 20: 0.3295},
    'U': {5: 0.1401, 10: 0.2033, 15: 0.2384, 20: 0.2733},
}

# Run step 1 on each subsample to get per-subsample polynomials
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
    b1b2 = model.params['const']
    g2 = model.params.get('d_t2', 0)
    g3 = model.params.get('d_t3', 0)
    g4 = model.params.get('d_t4', 0)
    d2 = model.params.get('d_x2', 0)
    d3 = model.params.get('d_x3', 0)
    d4 = model.params.get('d_x4', 0)
    print(f"  b1+b2={b1b2:.4f}, g2={g2:.6f}, g3={g3:.8f}, g4={g4:.10f}")
    return b1b2, g2, g3, g4, d2, d3, d4

print("=== Per-subsample Step 1 ===")
for label, sub in [('PS', ps), ('BC_NU', bc_nu), ('BC_U', bc_u)]:
    print(f"\n{label} (N={len(sub)}):")
    b1b2, g2, g3, g4, d2, d3, d4 = run_step1(sub)

    # Compute cumulative returns with per-subsample polynomials
    # Use paper b1+b2 for beta_2, but subsample g2,g3,g4 for polynomial terms
    paper_b1b2 = PAPER_B1B2[label.replace('BC_', '')]
    beta_1 = 0.077  # Approximate (varies by config)
    beta_2 = paper_b1b2 - beta_1

    print(f"  With paper b1+b2={paper_b1b2}, beta_2={beta_2:.4f}")
    print(f"  Cumulative returns (per-sub poly vs full-sample poly vs paper):")
    for T in [5, 10, 15, 20]:
        cum_persub = beta_2 * T + g2 * T**2 + g3 * T**3 + g4 * T**4
        cum_full = beta_2 * T + PAPER_G2 * T**2 + PAPER_G3 * T**3 + PAPER_G4 * T**4
        cum_paper = GROUND_TRUTH_CUM[label.replace('BC_', '')][T]
        print(f"    T={T}: persub={cum_persub:.4f}, full={cum_full:.4f}, paper={cum_paper:.4f}")

# Also: what if we back-solve for the g2,g3,g4 that would produce the paper's
# cumulative returns given our beta_2?
print("\n=== Back-solving for polynomials ===")
for label in ['NU', 'U']:
    paper_b1b2 = PAPER_B1B2[label]
    beta_1 = 0.077
    beta_2 = paper_b1b2 - beta_1

    # cum(T) = beta_2*T + g2*T^2 + g3*T^3 + g4*T^4 = paper_cum(T)
    # 4 equations (T=5,10,15,20) and 3 unknowns (g2,g3,g4)
    # Solve least squares
    A = np.zeros((4, 3))
    b = np.zeros(4)
    for i, T in enumerate([5, 10, 15, 20]):
        A[i, :] = [T**2, T**3, T**4]
        b[i] = GROUND_TRUTH_CUM[label][T] - beta_2 * T

    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    g2_bs, g3_bs, g4_bs = sol
    print(f"\n{label} (beta_2={beta_2:.4f}):")
    print(f"  Backsolve: g2={g2_bs:.6f}, g3={g3_bs:.8f}, g4={g4_bs:.10f}")
    print(f"  Full-sample: g2={PAPER_G2:.6f}, g3={PAPER_G3:.8f}, g4={PAPER_G4:.10f}")

    for T in [5, 10, 15, 20]:
        cum = beta_2 * T + g2_bs * T**2 + g3_bs * T**3 + g4_bs * T**4
        print(f"    T={T}: backsolve={cum:.4f}, paper={GROUND_TRUTH_CUM[label][T]:.4f}")

# Now: what polynomials would reproduce paper's cumulative returns with paper's beta_2?
print("\n=== Back-solving with paper's beta_2 ===")
for label in ['PS', 'NU', 'U']:
    true_beta_2 = {'PS': 0.0601, 'NU': 0.0513, 'U': 0.0399}[label]

    A = np.zeros((4, 3))
    b = np.zeros(4)
    for i, T in enumerate([5, 10, 15, 20]):
        A[i, :] = [T**2, T**3, T**4]
        b[i] = GROUND_TRUTH_CUM[label][T] - true_beta_2 * T

    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    g2_bs, g3_bs, g4_bs = sol
    print(f"\n{label} (beta_2={true_beta_2:.4f}):")
    print(f"  Backsolve: g2={g2_bs:.6f}, g3={g3_bs:.8f}, g4={g4_bs:.10f}")
    print(f"  Full-sample: g2={PAPER_G2:.6f}, g3={PAPER_G3:.8f}, g4={PAPER_G4:.10f}")

    for T in [5, 10, 15, 20]:
        cum = true_beta_2 * T + g2_bs * T**2 + g3_bs * T**3 + g4_bs * T**4
        print(f"    T={T}: backsolve={cum:.4f}, paper={GROUND_TRUTH_CUM[label][T]:.4f}")
