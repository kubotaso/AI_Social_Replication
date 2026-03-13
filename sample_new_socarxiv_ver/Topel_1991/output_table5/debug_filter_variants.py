"""Test filter variations to squeeze out a few more BC_U observations"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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

# How many rows BEFORE filters?
print(f"Raw data: {len(df)} rows")

# Apply filters one by one
df1 = df[df['tenure'] >= 1].copy()
print(f"After tenure >= 1: {len(df1)}")
df2 = df1.dropna(subset=['log_hourly_wage']).copy()
print(f"After drop log_hourly_wage NA: {len(df2)}")
df3 = df2[(df2['self_employed'] == 0) | (df2['self_employed'].isna())].copy()
print(f"After self_employed filter: {len(df3)}")
df3['occ_mapped'] = df3['occ_1digit'].apply(map_occ_split)
df4 = df3[~df3['occ_mapped'].isin(['2', '8', '9'])].copy()
print(f"After occ exclusions: {len(df4)}")

# How many self_employed == 1 in BC union?
bc_codes = ['0', '5', '6', 'L', 'S']
df_pre_se = df2.copy()
df_pre_se['occ_mapped'] = df_pre_se['occ_1digit'].apply(map_occ_split)
df_pre_se = df_pre_se[~df_pre_se['occ_mapped'].isin(['2', '8', '9'])]
bc_pre = df_pre_se[df_pre_se['occ_mapped'].isin(bc_codes)]
bc_u_pre = bc_pre[bc_pre['union_member'] == 1]
se_in_bc_u = bc_u_pre[bc_u_pre['self_employed'] == 1]
print(f"\nSelf-employed==1 in BC union: {len(se_in_bc_u)}")
print(f"Self-employed values in BC union: {bc_u_pre['self_employed'].value_counts(dropna=False).to_dict()}")

# What if we don't exclude self_employed for BC union?
# How many extra would that give?
bc_all_u = df2.copy()
bc_all_u['occ_mapped'] = bc_all_u['occ_1digit'].apply(map_occ_split)
bc_all_u = bc_all_u[~bc_all_u['occ_mapped'].isin(['2', '8', '9'])]
bc_all_u = bc_all_u[bc_all_u['occ_mapped'].isin(bc_codes)]
bc_all_u_union = bc_all_u[bc_all_u['union_member'] == 1]
print(f"\nBC union without self-employed filter: {len(bc_all_u_union)}")
print(f"BC union with self-employed filter: {len(bc_u_pre[bc_u_pre['self_employed'] != 1])}")

# Check: what's the dropna doing? Are there log_hourly_wage NAs in BC union?
df_raw = df[df['tenure'] >= 1].copy()
df_raw['occ_mapped'] = df_raw['occ_1digit'].apply(map_occ_split)
df_raw = df_raw[~df_raw['occ_mapped'].isin(['2', '8', '9'])]
bc_raw = df_raw[df_raw['occ_mapped'].isin(bc_codes)]
bc_u_raw = bc_raw[bc_raw['union_member'] == 1]
print(f"\nBC union before log_hourly_wage filter: {len(bc_u_raw)}")
print(f"  with log_hourly_wage NA: {bc_u_raw['log_hourly_wage'].isna().sum()}")
print(f"  with self_employed==1: {(bc_u_raw['self_employed']==1).sum()}")

# What about extending the occupation ranges?
# Could Census codes 786-899 include anyone?
occ_gap = df_raw[(df_raw['occ_1digit'] > 785) & (df_raw['occ_1digit'] < 900) & (df_raw['occ_1digit'] > 824)]
print(f"\nOcc codes 825-899: {len(occ_gap)} rows")

# Check the 999 code
occ_999 = df_raw[df_raw['occ_1digit'] == 999]
print(f"Occ code 999: {len(occ_999)} rows")
if len(occ_999) > 0:
    print(f"  union: {occ_999['union_member'].value_counts(dropna=False).to_dict()}")

# What about occ code 246-259 gap?
occ_gap2 = df_raw[(df_raw['occ_1digit'] >= 246) & (df_raw['occ_1digit'] <= 259)]
print(f"Occ codes 246-259: {len(occ_gap2)} rows")

# What about codes 396-400?
occ_gap3 = df_raw[(df_raw['occ_1digit'] >= 396) & (df_raw['occ_1digit'] <= 400)]
print(f"Occ codes 396-400: {len(occ_gap3)} rows")

# Codes 581-600
occ_gap4 = df_raw[(df_raw['occ_1digit'] >= 581) & (df_raw['occ_1digit'] <= 600)]
print(f"Occ codes 581-600: {len(occ_gap4)} rows")

# Codes 696-700
occ_gap5 = df_raw[(df_raw['occ_1digit'] >= 696) & (df_raw['occ_1digit'] <= 700)]
print(f"Occ codes 696-700: {len(occ_gap5)} rows")

# How about including log_real_wage dropna in the step2 function?
# The current code drops NAs in step 2 which could lose rows
# Let me check what GNP deflator and CPS index coverage looks like
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

df4['cps_index'] = df4['year'].map(CPS_WAGE_INDEX)
df4['gnp_defl'] = (df4['year'] - 1).map(GNP_DEFLATOR)
df4['log_real_wage'] = df4['log_hourly_wage'] - np.log(df4['gnp_defl']/100.0) - np.log(df4['cps_index'])
print(f"\nlog_real_wage NAs: {df4['log_real_wage'].isna().sum()}")
print(f"cps_index NAs: {df4['cps_index'].isna().sum()}")
print(f"gnp_defl NAs: {df4['gnp_defl'].isna().sum()}")
print(f"Year range: {df4['year'].min()} to {df4['year'].max()}")
print(f"Year distribution: {df4['year'].value_counts().sort_index().to_dict()}")
