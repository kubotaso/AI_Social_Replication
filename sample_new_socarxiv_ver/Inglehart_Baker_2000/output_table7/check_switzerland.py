"""
Check where Switzerland 1990-1991 = 26 is coming from.
"""
import pandas as pd
import math

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv', low_memory=False,
                  usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'S020', 'G006', 'S017'])
evs = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False,
                    columns=['c_abrv', 'country1', 'q365', 'weight_s', 'year'])
evs_csv = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)

def std_round(x):
    return math.floor(x + 0.5)

print("=== SWITZERLAND SOURCES ===")

# WVS wave 2
che_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'CHE') & (wvs['S002VS'] == 2)]
print(f"WVS CHE wave 2: N={len(che_w2)}")
if len(che_w2) > 0:
    valid = che_w2[(che_w2['F063'] >= 1) & (che_w2['F063'] <= 10)]
    print(f"Valid F063={len(valid)}")
    if len(valid) > 0:
        pct = (valid['F063'] == 10).mean() * 100
        w = che_w2.loc[valid.index, 'S017']
        pct_w = ((valid['F063'] == 10) * w).sum() / w.sum() * 100 if w.gt(0).all() else pct
        print(f"Unweighted %10: {pct:.4f}%->{std_round(pct)}, weighted: {pct_w:.4f}%->{std_round(pct_w)}")
        print(f"Paper=26")
    print(f"Years: {che_w2['S020'].value_counts().sort_index()}")

# EVS ZA4460 for CH
che_evs = evs[evs['c_abrv'] == 'CH']
print(f"\nEVS ZA4460 CH: N={len(che_evs)}")
if len(che_evs) > 0:
    valid = che_evs[(che_evs['q365'] >= 1) & (che_evs['q365'] <= 10)]
    print(f"Valid q365={len(valid)}")
    if len(valid) > 0:
        pct = (valid['q365'] == 10).mean() * 100
        print(f"q365 %10: {pct:.4f}% -> {std_round(pct)}")
    print(f"Years: {che_evs['year'].value_counts().sort_index()}")

# EVS CSV for CHE
che_csv = evs_csv[evs_csv['COUNTRY_ALPHA'] == 'CHE']
print(f"\nEVS CSV CHE: N={len(che_csv)}")
if len(che_csv) > 0:
    valid = che_csv[(che_csv['A006'] >= 1) & (che_csv['A006'] <= 10)]
    print(f"Valid A006={len(valid)}")
    if len(valid) > 0:
        pct = (valid['A006'] == 10).mean() * 100
        print(f"A006 %10: {pct:.4f}% -> {std_round(pct)}")
    print(f"Years: {che_csv['S020'].value_counts().sort_index()}")

print("\n=== FINLAND WAVE 2 SOURCES ===")
fin_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'FIN') & (wvs['S002VS'] == 2)]
print(f"WVS FIN wave 2: N={len(fin_w2)}")
fin_evs = evs[evs['c_abrv'] == 'FI']
print(f"EVS ZA4460 FI: N={len(fin_evs)}, valid q365={len(fin_evs[(fin_evs['q365']>=1)&(fin_evs['q365']<=10)])}")

print("\n=== HUNGARY WAVE 2 SOURCES ===")
hun_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'HUN') & (wvs['S002VS'] == 2)]
print(f"WVS HUN wave 2: N={len(hun_w2)}")
hun_evs = evs[evs['c_abrv'] == 'HU']
valid = hun_evs[(hun_evs['q365'] >= 1) & (hun_evs['q365'] <= 10)]
print(f"EVS ZA4460 HU: N={len(hun_evs)}, valid q365={len(valid)}")
if len(valid) > 0:
    pct = (valid['q365'] == 10).mean() * 100
    print(f"Hungary EVS q365 %10: {pct:.4f}% -> {std_round(pct)} (paper=22)")

print("\n=== BULGARIA WAVE 2 SOURCES ===")
bgr_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'BGR') & (wvs['S002VS'] == 2)]
print(f"WVS BGR wave 2: N={len(bgr_w2)}")
bgr_evs = evs[evs['c_abrv'] == 'BG']
valid = bgr_evs[(bgr_evs['q365'] >= 1) & (bgr_evs['q365'] <= 10)]
print(f"EVS ZA4460 BG: N={len(bgr_evs)}, valid q365={len(valid)}")
if len(valid) > 0:
    pct = (valid['q365'] == 10).mean() * 100
    print(f"Bulgaria EVS q365 %10: {pct:.4f}% -> {std_round(pct)} (paper=7)")

print("\n=== SLOVENIA WAVE 2 SOURCES ===")
svn_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'SVN') & (wvs['S002VS'] == 2)]
print(f"WVS SVN wave 2: N={len(svn_w2)}")
svn_evs = evs[evs['c_abrv'] == 'SI']
valid = svn_evs[(svn_evs['q365'] >= 1) & (svn_evs['q365'] <= 10)]
print(f"EVS ZA4460 SI: N={len(svn_evs)}, valid q365={len(valid)}")
if len(valid) > 0:
    pct = (valid['q365'] == 10).mean() * 100
    print(f"Slovenia EVS q365 %10: {pct:.4f}% -> {std_round(pct)} (paper=14)")
