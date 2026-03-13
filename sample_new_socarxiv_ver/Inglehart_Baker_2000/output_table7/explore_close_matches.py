"""Try to convert close matches to exact by exploring different approaches."""
import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F063','S017','G006'], low_memory=False)
evs = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False,
                     columns=['c_abrv','country1','q365','weight_s','weight_g','year'])

# Close matches from attempt 8:
# BRA 1990: 82.50% -> 82 (paper=83) - WVS wave 2
# DEU_WEST 1990: 13.04% -> 13 (paper=14) - EVS country1=900
# DEU_WEST 1995: 15.33% -> 15 (paper=16) - WVS wave 3 G006=[1,4]
# ESP 1990: 17.13% -> 17 (paper=18) - EVS
# IND 1995: 53.93% -> 54 (paper=56) - WVS wave 3
# JPN 1995: 5.82% -> 6 (paper=5) - WVS wave 3
# MEX 1995: 49.50% -> 49 (paper=50) - WVS wave 3
# NGA 1995: 86.19% -> 86 (paper=87) - WVS wave 3 weighted
# NLD 1990: 11.96% -> 12 (paper=11) - EVS
# RUS 1995: 18.42% -> 18 (paper=19) - WVS wave 3
# ZAF 1990: 73.13% -> 73 (paper=74) - WVS wave 2 weighted
# ZAF 1995: 70.50% -> 70 (paper=71) - WVS wave 3 weighted

print("=== EXPLORING ALTERNATIVES FOR CLOSE MATCHES ===\n")

# For each close match, try:
# 1. Different rounding (floor vs round vs ceil)
# 2. Different weight thresholds
# 3. Check if WVS wave 2 data exists (for 1990 EVS cells) with different values
# 4. Check raw % with different missing data handling

# BRA wave 2
print("BRA wave 2 (paper=83):")
bra = wvs[(wvs['COUNTRY_ALPHA']=='BRA') & (wvs['S002VS']==2)]
v = bra[(bra['F063']>=1)&(bra['F063']<=10)]
print(f"  Unweighted: {(v['F063']==10).mean()*100:.4f}% -> {round((v['F063']==10).mean()*100)}")
w = v['S017']
print(f"  Weighted: {((v['F063']==10).astype(float)*w).sum()/w.sum()*100:.4f}%")
# Try including F063==0 as valid non-10 response
v2 = bra[(bra['F063']>=0)&(bra['F063']<=10)]
print(f"  Including F063=0: {(v2['F063']==10).mean()*100:.4f}% n={len(v2)}")
# Floor
print(f"  Floor: {int((v['F063']==10).mean()*100)}")

# DEU_WEST wave 3 (paper=16)
print("\nDEU_WEST wave 3 (paper=16):")
deu = wvs[(wvs['COUNTRY_ALPHA']=='DEU')&(wvs['S002VS']==3)]
# Try different G006 combos
for label, codes in [('[1]', [1]), ('[4]', [4]), ('[1,4]', [1,4]),
                      ('[1,2]', [1,2]), ('[1,-1]', [1,-1])]:
    sub = deu[deu['G006'].isin(codes)]
    valid = sub[(sub['F063']>=1)&(sub['F063']<=10)]
    if len(valid) > 0:
        pct = (valid['F063']==10).mean()*100
        print(f"  G006={label}: {pct:.2f}% -> {round(pct)} (n={len(valid)})")

# MEX wave 3 (paper=50)
print("\nMEX wave 3 (paper=50):")
mex = wvs[(wvs['COUNTRY_ALPHA']=='MEX')&(wvs['S002VS']==3)]
v = mex[(mex['F063']>=1)&(mex['F063']<=10)]
pct = (v['F063']==10).mean()*100
print(f"  Unweighted: {pct:.4f}% -> round={round(pct)}")
# Python round(49.5) = 50 (banker's rounding, but 49.50 rounds to 50 since 49.5 rounds to even)
# Actually Python: round(49.5) = 50, round(0.5) = 0 (banker's). But 49.4990 rounds to 49.
import math
print(f"  math.floor: {math.floor(pct)}")
print(f"  int(pct+0.5): {int(pct+0.5)}")  # Simple round up at .5
print(f"  round(): {round(pct)}")

# NGA wave 3 (paper=87)
print("\nNGA wave 3 (paper=87):")
nga = wvs[(wvs['COUNTRY_ALPHA']=='NGA')&(wvs['S002VS']==3)]
v = nga[(nga['F063']>=1)&(nga['F063']<=10)]
pct_uw = (v['F063']==10).mean()*100
w = v['S017']
pct_w = ((v['F063']==10).astype(float)*w).sum()/w.sum()*100
print(f"  Unweighted: {pct_uw:.4f}% -> {round(pct_uw)} (paper=87)")
print(f"  Weighted: {pct_w:.4f}% -> {round(pct_w)}")
print(f"  Weight std: {w.std():.4f}")

# RUS wave 3 (paper=19)
print("\nRUS wave 3 (paper=19):")
rus = wvs[(wvs['COUNTRY_ALPHA']=='RUS')&(wvs['S002VS']==3)]
v = rus[(rus['F063']>=1)&(rus['F063']<=10)]
pct = (v['F063']==10).mean()*100
print(f"  Unweighted: {pct:.4f}% -> {round(pct)}")
print(f"  Weighted: {((v['F063']==10).astype(float)*v['S017']).sum()/v['S017'].sum()*100:.4f}%")

# ZAF wave 2 (paper=74)
print("\nZAF wave 2 (paper=74):")
zaf = wvs[(wvs['COUNTRY_ALPHA']=='ZAF')&(wvs['S002VS']==2)]
v = zaf[(zaf['F063']>=1)&(zaf['F063']<=10)]
pct_uw = (v['F063']==10).mean()*100
w = v['S017']
pct_w = ((v['F063']==10).astype(float)*w).sum()/w.sum()*100
print(f"  Unweighted: {pct_uw:.4f}% -> {round(pct_uw)}")
print(f"  Weighted: {pct_w:.4f}% -> {round(pct_w)}")
print(f"  Weight stats: mean={w.mean():.4f}, std={w.std():.4f}")

# ZAF wave 3 (paper=71)
print("\nZAF wave 3 (paper=71):")
zaf3 = wvs[(wvs['COUNTRY_ALPHA']=='ZAF')&(wvs['S002VS']==3)]
v = zaf3[(zaf3['F063']>=1)&(zaf3['F063']<=10)]
pct_uw = (v['F063']==10).mean()*100
w = v['S017']
pct_w = ((v['F063']==10).astype(float)*w).sum()/w.sum()*100
print(f"  Unweighted: {pct_uw:.4f}% -> {round(pct_uw)}")
print(f"  Weighted: {pct_w:.4f}% -> {round(pct_w)}")
print(f"  Weight stats: mean={w.mean():.4f}, std={w.std():.4f}")

# NLD EVS (paper=11)
print("\nNLD EVS (paper=11):")
nld = evs[evs['c_abrv']=='NL']
v = nld[(nld['q365']>=1)&(nld['q365']<=10)]
pct_uw = (v['q365']==10).mean()*100
w = v['weight_s']
pct_w = ((v['q365']==10).astype(float)*w).sum()/w.sum()*100
print(f"  Unweighted: {pct_uw:.4f}% -> {round(pct_uw)}")
print(f"  Weighted: {pct_w:.4f}% -> {round(pct_w)}")

# JPN wave 3 (paper=5)
print("\nJPN wave 3 (paper=5):")
jpn = wvs[(wvs['COUNTRY_ALPHA']=='JPN')&(wvs['S002VS']==3)]
v = jpn[(jpn['F063']>=1)&(jpn['F063']<=10)]
pct = (v['F063']==10).mean()*100
print(f"  Unweighted: {pct:.4f}% -> round={round(pct)}, floor={int(pct)}")
# Paper says 5, we get 5.82 -> 6. Floor would give 5.

# ESP EVS (paper=18)
print("\nESP EVS (paper=18):")
esp = evs[evs['c_abrv']=='ES']
v = esp[(esp['q365']>=1)&(esp['q365']<=10)]
pct_uw = (v['q365']==10).mean()*100
w_s = v['weight_s']
w_g = v['weight_g']
pct_ws = ((v['q365']==10).astype(float)*w_s).sum()/w_s.sum()*100
pct_wg = ((v['q365']==10).astype(float)*w_g).sum()/w_g.sum()*100
print(f"  Unweighted: {pct_uw:.4f}% -> {round(pct_uw)}")
print(f"  weight_s: {pct_ws:.4f}% -> {round(pct_ws)}")
print(f"  weight_g: {pct_wg:.4f}% -> {round(pct_wg)}")
# Also check WVS wave 2 Spain
esp_wvs = wvs[(wvs['COUNTRY_ALPHA']=='ESP')&(wvs['S002VS']==2)]
v2 = esp_wvs[(esp_wvs['F063']>=1)&(esp_wvs['F063']<=10)]
if len(v2) > 0:
    pct2 = (v2['F063']==10).mean()*100
    print(f"  WVS wave 2: {pct2:.4f}% -> {round(pct2)} (n={len(v2)})")

# DEU_WEST EVS country1=900 (paper=14)
print("\nDEU_WEST EVS (paper=14):")
deu_evs = evs[(evs['c_abrv']=='DE')&(evs['country1']==900)]
v = deu_evs[(deu_evs['q365']>=1)&(deu_evs['q365']<=10)]
pct_uw = (v['q365']==10).mean()*100
w = v['weight_s']
pct_w = ((v['q365']==10).astype(float)*w).sum()/w.sum()*100
print(f"  Unweighted: {pct_uw:.4f}% -> {round(pct_uw)}")
print(f"  Weighted: {pct_w:.4f}% -> {round(pct_w)}")
# Also try using EVS CSV with G006=[1,2] for West
evs_csv = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)
deu_csv = evs_csv[(evs_csv['COUNTRY_ALPHA']=='DEU')]
west_csv = deu_csv[deu_csv['G006'].isin([1,2])]
v_csv = west_csv[(west_csv['A006']>=1)&(west_csv['A006']<=10)]
if len(v_csv) > 0:
    pct_csv = (v_csv['A006']==10).mean()*100
    print(f"  EVS CSV G006=[1,2]: {pct_csv:.4f}% -> {round(pct_csv)} (n={len(v_csv)})")

# IND wave 3 (paper=56)
print("\nIND wave 3 (paper=56):")
ind = wvs[(wvs['COUNTRY_ALPHA']=='IND')&(wvs['S002VS']==3)]
v = ind[(ind['F063']>=1)&(ind['F063']<=10)]
pct = (v['F063']==10).mean()*100
print(f"  Unweighted: {pct:.4f}% -> {round(pct)}")
print(f"  N={len(v)}")
# Check if including all non-negative values changes things
v_all = ind[ind['F063']>=0]
print(f"  F063 distribution: {v['F063'].value_counts().sort_index().to_dict()}")
