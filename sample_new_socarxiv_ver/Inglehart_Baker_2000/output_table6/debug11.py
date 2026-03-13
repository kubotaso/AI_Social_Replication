"""Debug11: Investigate remaining partial matches and Italy."""
import pandas as pd
import numpy as np

MONTHLY_VALS = [1, 2, 3]
VALID_8PT = [1, 2, 3, 4, 5, 6, 7, 8]

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS', 'S003', 'S020', 'F028', 'X048WVS'],
                   low_memory=False)
evs = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False,
                     columns=['c_abrv', 'country1', 'q336', 'year'])

# === FINLAND EVS 1990: 11% (paper=13%) ===
print("=== FINLAND ===")
fin_evs = evs[evs['c_abrv'] == 'FI']
print(f"Finland EVS n={len(fin_evs)}")
print("q336 distribution:")
print(fin_evs['q336'].value_counts().sort_index())
valid = fin_evs[fin_evs['q336'].isin(VALID_8PT)]
monthly = fin_evs[fin_evs['q336'].isin(MONTHLY_VALS)]
print(f"8pt: {len(monthly)}/{len(valid)} = {round(len(monthly)/len(valid)*100)}%")
# Check if WVS wave 1 Finland gives 13%
fin_w1 = wvs[(wvs['S003'] == 246) & (wvs['S002VS'] == 1)]
valid_w1 = fin_w1[fin_w1['F028'].isin(VALID_8PT)]
monthly_w1 = fin_w1[fin_w1['F028'].isin(MONTHLY_VALS)]
print(f"WVS W1: {len(monthly_w1)}/{len(valid_w1)} = {round(len(monthly_w1)/len(valid_w1)*100)}%")
# The paper may use WVS wave 1 (13%) for 1981 AND EVS for 1990 (11%).
# Paper says Finland 1990=13... let me check WVS wave 3 Finland
fin_w3 = wvs[(wvs['S003'] == 246) & (wvs['S002VS'] == 3)]
valid_w3 = fin_w3[fin_w3['F028'].isin(VALID_8PT)]
monthly_w3 = fin_w3[fin_w3['F028'].isin(MONTHLY_VALS)]
print(f"WVS W3: {len(monthly_w3)}/{len(valid_w3)} = {round(len(monthly_w3)/len(valid_w3)*100)}%")

# === ITALY EVS 1990: 51% (paper=47%) ===
print("\n=== ITALY ===")
ita_evs = evs[evs['c_abrv'] == 'IT']
print(f"Italy EVS n={len(ita_evs)}")
print("q336 distribution:")
print(ita_evs['q336'].value_counts().sort_index())
valid = ita_evs[ita_evs['q336'].isin(VALID_8PT)]
monthly = ita_evs[ita_evs['q336'].isin(MONTHLY_VALS)]
print(f"8pt: {len(monthly)}/{len(valid)} = {len(monthly)/len(valid)*100:.1f}%")
# Check with 9pt (include -1?)
all_vals = sorted(ita_evs['q336'].dropna().unique())
print(f"All q336 values: {all_vals}")
# What if some values are missing/skipped in the denominator?
# Try including NaN or negative values
for denom in [[1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7], [1,2,3,4,5,6,7,8,9]]:
    valid = ita_evs[ita_evs['q336'].isin(denom)]
    monthly = ita_evs[ita_evs['q336'].isin(MONTHLY_VALS)]
    if len(valid) > 0:
        print(f"  denom={denom}: {len(monthly)}/{len(valid)} = {round(len(monthly)/len(valid)*100)}%")

# === SWITZERLAND W3: 28% (paper=25%) ===
print("\n=== SWITZERLAND ===")
swi_w3 = wvs[(wvs['S003'] == 756) & (wvs['S002VS'] == 3)]
print(f"Switzerland WVS W3 n={len(swi_w3)}")
print("F028 distribution:")
print(swi_w3['F028'].value_counts().sort_index())
valid = swi_w3[swi_w3['F028'].isin(VALID_8PT)]
monthly = swi_w3[swi_w3['F028'].isin(MONTHLY_VALS)]
print(f"8pt: {len(monthly)}/{len(valid)} = {round(len(monthly)/len(valid)*100)}%")
# Try with -2
valid_neg2 = swi_w3[swi_w3['F028'].isin(VALID_8PT + [-2])]
print(f"8pt+neg2: {len(monthly)}/{len(valid_neg2)} = {round(len(monthly)/len(valid_neg2)*100)}%")
# Switzerland W2
swi_w2 = wvs[(wvs['S003'] == 756) & (wvs['S002VS'] == 2)]
print(f"\nSwitzerland WVS W2 n={len(swi_w2)}")
print("F028 distribution:")
print(swi_w2['F028'].value_counts().sort_index())

# === USA W3: 57% (paper=55%) ===
print("\n=== USA ===")
usa_w3 = wvs[(wvs['S003'] == 840) & (wvs['S002VS'] == 3)]
print(f"USA WVS W3 n={len(usa_w3)}")
print("F028 distribution:")
print(usa_w3['F028'].value_counts().sort_index())
valid = usa_w3[usa_w3['F028'].isin(VALID_8PT)]
monthly = usa_w3[usa_w3['F028'].isin(MONTHLY_VALS)]
print(f"8pt: {len(monthly)}/{len(valid)} = {len(monthly)/len(valid)*100:.1f}%")
# With -2
valid_neg2 = usa_w3[usa_w3['F028'].isin(VALID_8PT + [-2])]
print(f"8pt+neg2: {len(monthly)}/{len(valid_neg2)} = {len(monthly)/len(valid_neg2)*100:.1f}%")

# === ARGENTINA W3: 43% (paper=41%) ===
print("\n=== ARGENTINA W3 ===")
arg_w3 = wvs[(wvs['S003'] == 32) & (wvs['S002VS'] == 3)]
print(f"Argentina WVS W3 n={len(arg_w3)}")
print("F028 distribution:")
print(arg_w3['F028'].value_counts().sort_index())
valid = arg_w3[arg_w3['F028'].isin(VALID_8PT)]
monthly = arg_w3[arg_w3['F028'].isin(MONTHLY_VALS)]
print(f"8pt: {len(monthly)}/{len(valid)} = {len(monthly)/len(valid)*100:.1f}%")
# With -2
valid_neg2 = arg_w3[arg_w3['F028'].isin(VALID_8PT + [-2])]
if len(valid_neg2) > len(valid):
    print(f"8pt+neg2: {len(monthly)}/{len(valid_neg2)} = {len(monthly)/len(valid_neg2)*100:.1f}%")

# === NIGERIA W3: 90% (paper=87%) ===
print("\n=== NIGERIA W3 ===")
nig_w3 = wvs[(wvs['S003'] == 566) & (wvs['S002VS'] == 3)]
print(f"Nigeria WVS W3 n={len(nig_w3)}")
print("F028 distribution:")
print(nig_w3['F028'].value_counts().sort_index())
valid = nig_w3[nig_w3['F028'].isin(VALID_8PT)]
monthly = nig_w3[nig_w3['F028'].isin(MONTHLY_VALS)]
print(f"8pt: {len(monthly)}/{len(valid)} = {len(monthly)/len(valid)*100:.1f}%")

# === SOUTH KOREA W1: 27% (paper=29%) ===
print("\n=== S.KOREA W1 ===")
kor_w1 = wvs[(wvs['S003'] == 410) & (wvs['S002VS'] == 1)]
print("F028 distribution:")
print(kor_w1['F028'].value_counts().sort_index())
# Already know: 8pt+neg2 = 27%, 8pt = 45%
# Paper says 29%. Neither matches exactly.
# Try: 8pt, treat -2 differently
total = len(kor_w1)
monthly_n = len(kor_w1[kor_w1['F028'].isin(MONTHLY_VALS)])
print(f"Monthly count: {monthly_n}")
# What if we use total respondents (including all invalid)?
print(f"Monthly/Total: {monthly_n}/{total} = {round(monthly_n/total*100)}%")
# Try including -2 but as separate analysis
for thresh in [VALID_8PT, VALID_8PT + [-2], [1,2,3,4,5,6,7,8,-2,-1]]:
    valid = kor_w1[kor_w1['F028'].isin(thresh)]
    monthly = kor_w1[kor_w1['F028'].isin(MONTHLY_VALS)]
    if len(valid) > 0:
        pct = len(monthly) / len(valid) * 100
        print(f"  denom={thresh}: {monthly_n}/{len(valid)} = {pct:.1f}% -> {round(pct)}%")

# === EVS: Check if Finland 1990 data differs with different years ===
print("\n=== FINLAND EVS year detail ===")
fin_evs = evs[evs['c_abrv'] == 'FI']
print(f"Finland EVS years: {sorted(fin_evs['year'].unique())}")
