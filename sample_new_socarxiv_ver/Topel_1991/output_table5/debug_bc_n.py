import pandas as pd
import numpy as np
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

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

bc_codes = ['5', '6', 'L', '0', 'S']
bc = df[df['occ_mapped'].isin(bc_codes)]
print(f'BC total: {len(bc)}')
print(f'BC union_member==0: {len(bc[bc["union_member"]==0])}')
print(f'BC union_member==1: {len(bc[bc["union_member"]==1])}')
print(f'BC union_member NaN: {len(bc[bc["union_member"].isna()])}')

for code in bc_codes:
    sub = bc[bc['occ_mapped']==code]
    nu = len(sub[sub['union_member']==0])
    u = len(sub[sub['union_member']==1])
    na = len(sub[sub['union_member'].isna()])
    print(f'  occ={code}: total={len(sub)}, NU={nu}, U={u}, NA={na}')

# The search script used a different mapping (to_1digit which maps S->7)
# Check what the search actually computed
def to_1digit(occ):
    if occ <= 9: return occ
    elif 1 <= occ <= 195: return 0
    elif 201 <= occ <= 245: return 1
    elif 260 <= occ <= 285: return 4
    elif 301 <= occ <= 395: return 3
    elif 401 <= occ <= 580: return 5
    elif 601 <= occ <= 695: return 6
    elif 701 <= occ <= 785: return 7
    elif 801 <= occ <= 824: return 8
    elif 900 <= occ <= 965: return 7
    else: return 9

df2 = pd.read_csv('data/psid_panel.csv')
df2['education_years'] = df2['education_clean'].map(EDUC_MAP).fillna(12)
df2['experience'] = (df2['age'] - df2['education_years'] - 6).clip(lower=0)
df2['tenure'] = df2['tenure_topel'].copy()
df2 = df2[df2['tenure'] >= 1].copy()
df2 = df2.dropna(subset=['log_hourly_wage']).copy()
df2 = df2[(df2['self_employed'] == 0) | (df2['self_employed'].isna())].copy()
df2['occ_m'] = df2['occ_1digit'].apply(to_1digit)

# Config from non-split search: PS=[1,3,4,7], BC=[0,5,6], obs
bc_search = df2[df2['occ_m'].isin([0, 5, 6])]
print(f'\nSearch config BC=[0,5,6]: total={len(bc_search)}')
print(f'  NU={len(bc_search[bc_search["union_member"]==0])}')
print(f'  U={len(bc_search[bc_search["union_member"]==1])}')
print(f'  NA={len(bc_search[bc_search["union_member"].isna()])}')

# The discrepancy: search says BC=[0,5,6] with obs union gives NU=2673, U=2028
# But my split config BC=[5,6,L,0,S] includes MORE codes than BC=[0,5,6]
# because L maps additional 701-785 codes and S maps 900-965 codes
# In the search, those 3-digit codes were mapped to 7, which went to PS

# So the correct split config should be: BC=[0,5,6] (1-digit only)
# Plus the 3-digit codes that map to these categories
# But NOT 7 (which is PS) and NOT L or S

# Wait -- in the split search, PS=['1','3','4','7'] includes both
# 1-digit 7 and 3-digit service (900-965) since 7 includes ambiguous 1-digit
# But the BC codes in split search were ['5','6','L','0','S']
# which includes 3-digit laborers (L) AND 3-digit service (S)

# The issue is that split PS has code '7' = 1-digit occ=7 (907 rows)
# while split BC has 'L' = 3-digit 701-785 (290 rows) and 'S' = 3-digit 900-965 (63 rows)
# Plus '0' = 1-digit occ=0 (2902 rows) + 3-digit professional (some)

# Check: what codes does '0' include?
occ0 = df[df['occ_mapped']=='0']
print(f'\nocc_mapped=0 breakdown:')
print(f'  occ_1digit<=9: {len(occ0[occ0["occ_1digit"]<=9])}')
print(f'  occ_1digit>9 (3-digit professional): {len(occ0[occ0["occ_1digit"]>9])}')
