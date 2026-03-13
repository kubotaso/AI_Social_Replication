"""Test different tenure filters and their effect on N and betas"""
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
df = df.dropna(subset=['log_hourly_wage']).copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
df['occ_mapped'] = df['occ_1digit'].apply(map_occ_split)
df = df[~df['occ_mapped'].isin(['2', '8', '9'])].copy()
df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
df['gnp_defl'] = (df['year'] - 1).map(GNP_DEFLATOR)
df['log_real_wage'] = df['log_hourly_wage'] - np.log(df['gnp_defl']/100.0) - np.log(df['cps_index'])
df['initial_experience'] = (df['experience'] - df['tenure']).clip(lower=0)
for c in ['married', 'disabled', 'region_ne', 'region_nc', 'region_south']:
    if c in df.columns: df[c] = df[c].fillna(0)
for y in range(1969, 1984):
    col = f'year_{y}'
    if col not in df.columns: df[col] = (df['year'] == y).astype(int)

for min_t in [0, 0.5, 1, 2]:
    df_t = df[df['tenure'] >= min_t].copy()
    ps = df_t[df_t['occ_mapped'].isin(['1', '3', '4', '7'])]
    bc = df_t[df_t['occ_mapped'].isin(['0', '5', '6', 'L', 'S'])]
    bc_nu = bc[bc['union_member'] == 0]
    bc_u = bc[bc['union_member'] == 1]
    print(f"tenure >= {min_t}: PS={len(ps)}, NU={len(bc_nu)}, U={len(bc_u)}")
    for n, t, l in [(len(ps), 4946, 'PS'), (len(bc_nu), 2642, 'NU'), (len(bc_u), 2741, 'U')]:
        re = abs(n - t) / t
        pts = 5 if re <= 0.05 else (3 if re <= 0.10 else (1 if re <= 0.20 else 0))
        print(f"  {l}: {n} vs {t} ({re:.1%}) -> {pts}/5")
    print()

# What about tenure >= 1 but including tenure_topel <= 0 values?
print("tenure_topel value distribution (first 20):")
print(df['tenure'].describe())
print(f"\ntenure < 0: {(df['tenure'] < 0).sum()}")
print(f"tenure == 0: {(df['tenure'] == 0).sum()}")
print(f"tenure between 0 and 1: {((df['tenure'] > 0) & (df['tenure'] < 1)).sum()}")
print(f"tenure >= 1: {(df['tenure'] >= 1).sum()}")

# Try tenure > 0
df_t = df[df['tenure'] > 0].copy()
ps = df_t[df_t['occ_mapped'].isin(['1', '3', '4', '7'])]
bc = df_t[df_t['occ_mapped'].isin(['0', '5', '6', 'L', 'S'])]
bc_nu = bc[bc['union_member'] == 0]
bc_u = bc[bc['union_member'] == 1]
print(f"\ntenure > 0: PS={len(ps)}, NU={len(bc_nu)}, U={len(bc_u)}")
for n, t, l in [(len(ps), 4946, 'PS'), (len(bc_nu), 2642, 'NU'), (len(bc_u), 2741, 'U')]:
    re = abs(n - t) / t
    pts = 5 if re <= 0.05 else (3 if re <= 0.10 else (1 if re <= 0.20 else 0))
    print(f"  {l}: {n} vs {t} ({re:.1%}) -> {pts}/5")
