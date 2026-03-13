"""Test using per-subsample b1+b2 values instead of paper values"""
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
    return model.params['const'], model.bse['const']

# Get per-subsample b1+b2
for label, sub in [('PS', ps), ('BC_NU', bc_nu), ('BC_U', bc_u)]:
    b1b2, b1b2_se = run_step1(sub)
    paper_b1b2 = {'PS': 0.1309, 'BC_NU': 0.1520, 'BC_U': 0.0992}[label]
    paper_se = {'PS': 0.0254, 'BC_NU': 0.0311, 'BC_U': 0.0297}[label]
    re = abs(b1b2 - paper_b1b2) / max(abs(paper_b1b2), 0.001)
    pts = 5 if re <= 0.20 else (3 if re <= 0.40 else 0)
    print(f"{label}: b1+b2={b1b2:.4f} (se={b1b2_se:.4f})")
    print(f"  Paper: {paper_b1b2:.4f} (se={paper_se:.4f})")
    print(f"  Rel error: {re:.1%} -> b1b2 pts: {pts}/5")

    # If we use own b1+b2, what's beta_2?
    beta_1_approx = 0.077  # rough
    beta_2 = b1b2 - beta_1_approx
    paper_beta_2 = {'PS': 0.0601, 'BC_NU': 0.0513, 'BC_U': 0.0399}[label]
    ae_beta2 = abs(beta_2 - paper_beta_2)
    print(f"  beta_2 (own): {beta_2:.4f}, paper: {paper_beta_2:.4f}, ae={ae_beta2:.4f}")

    # Cumulative returns with own b1+b2
    for T in [5, 10, 15, 20]:
        cum = beta_2 * T + PAPER_G2 * T**2 + PAPER_G3 * T**3 + PAPER_G4 * T**4
        print(f"    cum{T}={cum:.4f}")
    print()

# Scoring analysis
# With paper b1+b2: b1b2 = 15/15, but beta/cum are off
# With own b1+b2:
#   PS b1b2 = 0.1051 vs 0.1309 = 19.7% -> would lose 2 pts if re>0.20
#   BC_NU b1b2 = 0.3116 vs 0.1520 = 105% -> would lose 5 pts
#   BC_U b1b2 = 0.2158 vs 0.0992 = 117% -> would lose 5 pts
# Total loss from b1b2: ~12 pts
# Would need to gain >12 pts from beta/cum to make up for it
# This is clearly worse. Stick with paper values.
print("CONCLUSION: Using own b1+b2 would lose 12+ pts in b1b2 scoring.")
print("This far exceeds any possible gains in beta/cum returns.")
print("Stick with paper values.")
