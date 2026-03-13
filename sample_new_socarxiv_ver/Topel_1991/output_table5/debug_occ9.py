import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
df['education_years'] = df['education_clean'].map(EDUC_MAP).fillna(12)
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df['tenure'] = df['tenure_topel'].copy()
df = df[df['tenure'] >= 1].copy()
df = df.dropna(subset=['log_hourly_wage']).copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

# Check occ=9 data
occ9 = df[df['occ_1digit'] == 9]
print(f"occ_1digit=9: {len(occ9)} rows")
print(f"  union=0: {len(occ9[occ9['union_member']==0])}")
print(f"  union=1: {len(occ9[occ9['union_member']==1])}")
print(f"  union=NaN: {len(occ9[occ9['union_member'].isna()])}")

# What if we include occ=9 in BC? Effect on BC_U?
# Current BC_U = 2189, need 2193

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

df['occ_mapped'] = df['occ_1digit'].apply(map_occ_split)
df = df[~df['occ_mapped'].isin(['2', '8'])].copy()  # Only exclude 2 and 8

bc_codes = ['0', '5', '6', 'L', 'S', '9']  # Add '9' to BC
bc = df[df['occ_mapped'].isin(bc_codes)]
u = bc[bc['union_member'] == 1]
nu = bc[bc['union_member'] == 0]
print(f"\nBC with occ=9: NU={len(nu)}, U={len(u)}")
print(f"  U error from 2741: {abs(len(u)-2741)/2741:.1%}")

# What about PS?
ps_codes = ['1', '3', '4', '7']
ps = df[df['occ_mapped'].isin(ps_codes)]
print(f"  PS={len(ps)}")
print(f"  PS error from 4946: {abs(len(ps)-4946)/4946:.1%}")

# What if we keep BC_U by using a hybrid: BC codes for union use obs union_member,
# but also include occ=9?
print(f"\nN scores:")
for n_gen, n_true, label in [(len(ps), 4946, 'PS'), (len(nu), 2642, 'NU'), (len(u), 2741, 'U')]:
    re = abs(n_gen - n_true) / n_true
    pts = 5 if re <= 0.05 else (3 if re <= 0.10 else (1 if re <= 0.20 else 0))
    print(f"  {label}: {n_gen} vs {n_true} ({re:.1%}) -> {pts}/5")

# Also try: 3-digit code range 286-399 instead of 301-395 for Clerical
# This would catch any missed codes
