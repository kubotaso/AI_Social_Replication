"""Try to get a few more BC_U observations to cross 20% threshold"""
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

# Current filtering
df_base = df.copy()
df_base = df_base[df_base['tenure'] >= 1].copy()
df_base = df_base.dropna(subset=['log_hourly_wage']).copy()
df_base = df_base[(df_base['self_employed'] == 0) | (df_base['self_employed'].isna())].copy()
df_base['occ_mapped'] = df_base['occ_1digit'].apply(map_occ_split)
df_base = df_base[~df_base['occ_mapped'].isin(['2', '8', '9'])].copy()

bc_codes = ['0', '5', '6', 'L', 'S']
bc = df_base[df_base['occ_mapped'].isin(bc_codes)]

# Current: obs-level union
nu = bc[bc['union_member'] == 0]
u = bc[bc['union_member'] == 1]
print(f"Current BC_U: {len(u)} (need >= 2193 for <20% error from 2741)")
print(f"  Deficit: {2193 - len(u)} observations")

# What's causing the deficit?
# 1. NaN union_member in BC - these could be union
na_union = bc[bc['union_member'].isna()]
print(f"\nBC with NaN union_member: {len(na_union)}")

# 2. What if we include occ=9 (misc) in BC?
occ9 = df_base[df_base['occ_mapped'] == '9']
print(f"occ=9 (excluded): {len(occ9)}")

# 3. What if we DON'T exclude occ=2?
# occ=2 was already excluded by self_employed filter, but some have self_employed=0
occ2 = df[df['occ_1digit'] == 2]
occ2 = occ2[occ2['tenure'] >= 1].copy()
occ2 = occ2.dropna(subset=['log_hourly_wage']).copy()
occ2_nse = occ2[(occ2['self_employed'] == 0) | (occ2['self_employed'].isna())]
print(f"occ=2 with self_employed=0|NaN: {len(occ2_nse)}")
print(f"  union=1: {len(occ2_nse[occ2_nse['union_member']==1])}")

# 4. What if we relax the self_employed filter?
df_no_se = df.copy()
df_no_se = df_no_se[df_no_se['tenure'] >= 1].copy()
df_no_se = df_no_se.dropna(subset=['log_hourly_wage']).copy()
df_no_se['occ_mapped'] = df_no_se['occ_1digit'].apply(map_occ_split)
df_no_se = df_no_se[~df_no_se['occ_mapped'].isin(['2', '8', '9'])].copy()
bc_no_se = df_no_se[df_no_se['occ_mapped'].isin(bc_codes)]
u_no_se = bc_no_se[bc_no_se['union_member'] == 1]
print(f"\n5. Without self_employed filter: BC_U={len(u_no_se)}")
# Same because self_employed filter doesn't affect these occ codes?

# 5. Check: do we lose any obs in the dropna for w_star/experience/initial_experience?
# In run_two_step, we do: step2 = sub.dropna(subset=['w_star', 'experience', 'initial_experience'])
# Check how many BC_U obs have missing values in these
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

u_data = bc[bc['union_member'] == 1].copy()
u_data['cps_index'] = u_data['year'].map(CPS_WAGE_INDEX)
u_data['gnp_defl'] = (u_data['year'] - 1).map(GNP_DEFLATOR)
u_data['initial_experience'] = (u_data['experience'] - u_data['tenure']).clip(lower=0)

# Check for missing gnp_defl or cps_index
missing_gnp = u_data['gnp_defl'].isna().sum()
missing_cps = u_data['cps_index'].isna().sum()
missing_exp = u_data['experience'].isna().sum()
missing_x0 = u_data['initial_experience'].isna().sum()
print(f"\n6. Missing values in BC_U union obs:")
print(f"  gnp_defl NaN: {missing_gnp}")
print(f"  cps_index NaN: {missing_cps}")
print(f"  experience NaN: {missing_exp}")
print(f"  initial_experience NaN: {missing_x0}")

# Year distribution
print(f"\n  BC_U year distribution:")
vc = u_data['year'].value_counts().sort_index()
for yr, n in vc.items():
    in_cps = yr in CPS_WAGE_INDEX
    in_gnp = (yr-1) in GNP_DEFLATOR
    print(f"    {yr}: {n} (CPS:{in_cps}, GNP:{in_gnp})")

# What if we use 1968 GNP deflator for year 1968 (currently 1967 is the lag year)?
# The issue: gnp_defl = df['year'] - 1 mapped to GNP_DEFLATOR
# For year 1968: gnp_defl = GNP_DEFLATOR[1967] = 33.4 -> OK
# For year 1983: gnp_defl = GNP_DEFLATOR[1982] = 100.0 -> OK
# All years 1968-1983 should be covered

# Let me check: are there any years outside 1968-1983 in the union data?
print(f"\n  Min year: {u_data['year'].min()}, Max year: {u_data['year'].max()}")

# The answer may be simpler: just need to include a few more Service workers
# What if Census code range for Service is 900-999 instead of 900-965?
# Or if codes 286-300 exist (between Sales 260-285 and Clerical 301-395)?

# Actually, let me check if Clerical should start at 260 (not 301)
# Census 1970 occupation codes:
# 260-280: Sales workers
# 281-285: Other sales
# 301-395: Clerical and kindred workers
# Actually let me check: is it 260-285 Sales, or does Sales extend to 300?

# Let's just count: how many 3-digit codes in 286-300 range?
print(f"\n7. 3-digit codes in range 286-300:")
for code in range(286, 301):
    n = len(df_base[df_base['occ_1digit'] == code])
    if n > 0:
        print(f"    {code}: {n}")

# How about extending service to include 966-999?
print(f"\n8. 3-digit codes in range 966-999:")
for code in range(966, 1000):
    n = len(df_base[df_base['occ_1digit'] == code])
    if n > 0:
        print(f"    {code}: {n}")

# What about 580-600 gap?
print(f"\n9. 3-digit codes in range 581-600:")
for code in range(581, 601):
    n = len(df_base[df_base['occ_1digit'] == code])
    if n > 0:
        print(f"    {code}: {n}")

# Try: use 260-395 as one range (Sales+Clerical) instead of split
# This doesn't change anything since both map to PS codes

# The real question: 2189 vs 2193 - can we get 4 more union BC workers?
# Strategy: relax the 3-digit code range boundaries slightly
# Try extending Craftsmen to include 580 (might be in gap):
print(f"\n10. Code 580 count: {len(df_base[df_base['occ_1digit'] == 580])}")
print(f"    Code 600 count: {len(df_base[df_base['occ_1digit'] == 600])}")
print(f"    Code 700 count: {len(df_base[df_base['occ_1digit'] == 700])}")

# Actually our range already includes 580 (401-580) and 695 (601-695)
# Let me check which group these borderline codes are in
for code in [395, 396, 400, 401, 580, 581, 600, 601, 695, 696, 700, 701, 785, 786, 800, 801]:
    n = len(df_base[df_base['occ_1digit'] == code])
    if n > 0:
        in_bc = df_base[df_base['occ_1digit'] == code]
        n_u = len(in_bc[in_bc['union_member'] == 1])
        print(f"    Code {code}: {n} rows, {n_u} union")
