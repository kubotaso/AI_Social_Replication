"""
Investigation script for Table 7 data issues.
Focus on: India 1990, East Germany 1995, South Africa 1981, Korea, close matches
"""
import pandas as pd
import numpy as np

wvs_path = "data/WVS_Time_Series_1981-2022_csv_v5_0.csv"
evs_dta_path = "data/ZA4460_v3-0-0.dta"
evs_csv_path = "data/EVS_1990_wvs_format.csv"

print("=== LOADING WVS DATA ===")
wvs = pd.read_csv(wvs_path,
                   usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'A006', 'S020', 'G006', 'S017'],
                   low_memory=False)

print(f"WVS shape: {wvs.shape}")

# Check India Wave 2
print("\n=== INDIA WAVE 2 (1990-1991) ===")
ind_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'IND') & (wvs['S002VS'] == 2)]
print(f"India W2 rows: {len(ind_w2)}")
print(f"F063 range (valid 1-10): {ind_w2['F063'].value_counts().sort_index().head(20)}")
print(f"A006 range: {ind_w2['A006'].value_counts().sort_index().head(20)}")

f063_valid = ind_w2[ind_w2['F063'].between(1, 10)]
a006_valid = ind_w2[ind_w2['A006'].between(1, 10)]
print(f"\nF063 valid N: {len(f063_valid)}, % = 10: {(f063_valid['F063'] == 10).mean()*100:.2f}%")
print(f"A006 valid N: {len(a006_valid)}, % = 10: {(a006_valid['A006'] == 10).mean()*100:.2f}%")

# Check East Germany WVS wave 3
print("\n=== EAST GERMANY WAVE 3 (1995-1998) ===")
deu_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'DEU') & (wvs['S002VS'] == 3)]
print(f"DEU W3 rows: {len(deu_w3)}")
print(f"G006 value counts: {deu_w3['G006'].value_counts().sort_index()}")
deu_east_w3 = deu_w3[deu_w3['G006'].isin([2, 3])]
deu_west_w3 = deu_w3[deu_w3['G006'].isin([1, 4])]
print(f"East DEU W3 rows: {len(deu_east_w3)}")
print(f"West DEU W3 rows: {len(deu_west_w3)}")

f063_e = deu_east_w3[deu_east_w3['F063'].between(1, 10)]
f063_w = deu_west_w3[deu_west_w3['F063'].between(1, 10)]
print(f"East DEU W3 F063 valid N: {len(f063_e)}, % = 10: {(f063_e['F063'] == 10).mean()*100:.2f}%")
print(f"West DEU W3 F063 valid N: {len(f063_w)}, % = 10: {(f063_w['F063'] == 10).mean()*100:.2f}%")

# Check ZA4460 EVS for East Germany 1995
print("\n=== CHECKING ZA4460 FOR EAST GERMANY (if available for wave 3) ===")
evs = pd.read_stata(evs_dta_path, convert_categoricals=False)
print(f"EVS shape: {evs.shape}")
print(f"EVS columns related to country: {[c for c in evs.columns if 'country' in c.lower() or 'c_abrv' in c.lower()]}")
print(f"EVS c_abrv unique: {sorted(evs['c_abrv'].unique())}")
print(f"EVS year range: {evs['year'].describe()}")

# Check Germany EVS
deu_evs = evs[evs['c_abrv'] == 'DE']
print(f"\nEVS Germany rows: {len(deu_evs)}")
print(f"EVS Germany years: {sorted(deu_evs['year'].unique())}")
print(f"EVS Germany country1: {sorted(deu_evs['country1'].unique())}")

# Check q365 for East Germany
if 'q365' in evs.columns:
    east_evs = deu_evs[deu_evs['country1'] == 901]
    valid_east = east_evs[east_evs['q365'].between(1, 10)]
    print(f"EVS East Germany q365 valid N: {len(valid_east)}, % = 10: {(valid_east['q365'] == 10).mean()*100:.2f}%")

# Check South Africa 1981
print("\n=== SOUTH AFRICA WAVE 1 (1981) ===")
zaf_w1 = wvs[(wvs['COUNTRY_ALPHA'] == 'ZAF') & (wvs['S002VS'] == 1)]
print(f"ZAF W1 rows: {len(zaf_w1)}")
f063_zaf = zaf_w1[zaf_w1['F063'].between(1, 10)]
a006_zaf = zaf_w1[zaf_w1['A006'].between(1, 10)]
print(f"F063 valid N: {len(f063_zaf)}, % = 10: {(f063_zaf['F063'] == 10).mean()*100:.2f}%")
print(f"A006 valid N: {len(a006_zaf)}, % = 10: {(a006_zaf['A006'] == 10).mean()*100:.2f}%")
print(f"F063 values: {f063_zaf['F063'].value_counts().sort_index()}")

# Check Korea
print("\n=== KOREA ===")
kor = wvs[(wvs['COUNTRY_ALPHA'] == 'KOR')]
print(f"KOR rows: {len(kor)}")
print(f"KOR waves: {kor['S002VS'].value_counts().sort_index()}")
f063_kor = kor[kor['F063'].between(1, 10)]
print(f"KOR F063 valid: {len(f063_kor)}")
if len(f063_kor) > 0:
    for wave, g in f063_kor.groupby('S002VS'):
        print(f"  Wave {wave}: N={len(g)}, %10={( g['F063']==10).mean()*100:.2f}%")

# Check USA 1981
print("\n=== USA WAVE 1 (1981) ===")
usa_w1 = wvs[(wvs['COUNTRY_ALPHA'] == 'USA') & (wvs['S002VS'] == 1)]
print(f"USA W1 rows: {len(usa_w1)}")
f063_usa = usa_w1[usa_w1['F063'].between(1, 10)]
print(f"F063 valid N: {len(f063_usa)}, % = 10: {(f063_usa['F063'] == 10).mean()*100:.2f}%")

# Check 1981 countries available
print("\n=== WAVE 1 COUNTRIES ===")
w1 = wvs[wvs['S002VS'] == 1]
w1_valid = w1[w1['F063'].between(1, 10)]
for country, g in w1_valid.groupby('COUNTRY_ALPHA'):
    pct = (g['F063'] == 10).mean() * 100
    print(f"  {country}: N={len(g)}, %10={pct:.2f}%")

# Check close matches more carefully with different strategies
print("\n=== CLOSE MATCH ANALYSIS ===")

# Brazil 1990 (close: 82 vs 83)
bra_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'BRA') & (wvs['S002VS'] == 2)]
f063_bra = bra_w2[bra_w2['F063'].between(1, 10)]
print(f"\nBrazil W2: N={len(f063_bra)}, raw %10={f063_bra['F063'].eq(10).mean()*100:.4f}%")
w = bra_w2.loc[f063_bra.index, 'S017']
print(f"  S017 stats: mean={w.mean():.4f}, std={w.std():.4f}")
pct_weighted = (f063_bra['F063'].eq(10) * bra_w2.loc[f063_bra.index, 'S017']).sum() / bra_w2.loc[f063_bra.index, 'S017'].sum() * 100
print(f"  Weighted %10: {pct_weighted:.4f}%")

# Netherlands 1990 (close: 12 vs 11) - EVS
nld_evs = evs[evs['c_abrv'] == 'NL']
nld_valid = nld_evs[nld_evs['q365'].between(1, 10)]
if len(nld_valid) > 0:
    pct_nld = (nld_valid['q365'] == 10).mean() * 100
    pct_nld_w = (nld_valid['q365'].eq(10) * nld_evs.loc[nld_valid.index, 'weight_s']).sum() / nld_evs.loc[nld_valid.index, 'weight_s'].sum() * 100
    print(f"\nNetherlands EVS: N={len(nld_valid)}, unweighted %10={pct_nld:.4f}%, weighted %10={pct_nld_w:.4f}%")

# West Germany 1990 (close: 13 vs 14)
west_evs = deu_evs[deu_evs['country1'] == 900]
west_valid = west_evs[west_evs['q365'].between(1, 10)]
pct_west_uw = (west_valid['q365'] == 10).mean() * 100
w_west = deu_evs.loc[west_valid.index, 'weight_s']
pct_west_w = (west_valid['q365'].eq(10) * w_west).sum() / w_west.sum() * 100
print(f"\nWest Germany EVS: N={len(west_valid)}, unweighted %10={pct_west_uw:.4f}%, weighted %10={pct_west_w:.4f}%")

# Spain 1990 (close: 17 vs 18)
esp_evs = evs[evs['c_abrv'] == 'ES']
esp_valid = esp_evs[esp_evs['q365'].between(1, 10)]
pct_esp_uw = (esp_valid['q365'] == 10).mean() * 100
pct_esp_w = (esp_valid['q365'].eq(10) * evs.loc[esp_valid.index, 'weight_s']).sum() / evs.loc[esp_valid.index, 'weight_s'].sum() * 100
print(f"\nSpain EVS: N={len(esp_valid)}, unweighted %10={pct_esp_uw:.4f}%, weighted %10={pct_esp_w:.4f}%")

# Russia 1995 (close: 18 vs 19)
rus_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'RUS') & (wvs['S002VS'] == 3)]
f063_rus = rus_w3[rus_w3['F063'].between(1, 10)]
print(f"\nRussia W3: N={len(f063_rus)}, raw %10={f063_rus['F063'].eq(10).mean()*100:.4f}%")
w_rus = wvs.loc[f063_rus.index, 'S017']
print(f"  S017 stats: mean={w_rus.mean():.4f}, std={w_rus.std():.4f}")
pct_rus_w = (f063_rus['F063'].eq(10) * w_rus).sum() / w_rus.sum() * 100
print(f"  Weighted %10: {pct_rus_w:.4f}%")

# India 1995
ind_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'IND') & (wvs['S002VS'] == 3)]
f063_ind3 = ind_w3[ind_w3['F063'].between(1, 10)]
print(f"\nIndia W3: N={len(f063_ind3)}, raw %10={f063_ind3['F063'].eq(10).mean()*100:.4f}%")
w_ind3 = wvs.loc[f063_ind3.index, 'S017']
print(f"  S017 stats: mean={w_ind3.mean():.4f}, std={w_ind3.std():.4f}")
pct_ind3_w = (f063_ind3['F063'].eq(10) * w_ind3).sum() / w_ind3.sum() * 100
print(f"  Weighted %10: {pct_ind3_w:.4f}%")

# Japan 1995
jpn_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'JPN') & (wvs['S002VS'] == 3)]
f063_jpn3 = jpn_w3[jpn_w3['F063'].between(1, 10)]
print(f"\nJapan W3: N={len(f063_jpn3)}, raw %10={f063_jpn3['F063'].eq(10).mean()*100:.4f}%")
w_jpn3 = wvs.loc[f063_jpn3.index, 'S017']
print(f"  S017 stats: mean={w_jpn3.mean():.4f}, std={w_jpn3.std():.4f}")
pct_jpn3_w = (f063_jpn3['F063'].eq(10) * w_jpn3).sum() / w_jpn3.sum() * 100
print(f"  Weighted %10: {pct_jpn3_w:.4f}%")

# Mexico 1995
mex_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'MEX') & (wvs['S002VS'] == 3)]
f063_mex3 = mex_w3[mex_w3['F063'].between(1, 10)]
print(f"\nMexico W3: N={len(f063_mex3)}, raw %10={f063_mex3['F063'].eq(10).mean()*100:.4f}%")
w_mex3 = wvs.loc[f063_mex3.index, 'S017']
print(f"  S017 stats: mean={w_mex3.mean():.4f}, std={w_mex3.std():.4f}")
pct_mex3_w = (f063_mex3['F063'].eq(10) * w_mex3).sum() / w_mex3.sum() * 100
print(f"  Weighted %10: {pct_mex3_w:.4f}%")

# Nigeria 1995
nga_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'NGA') & (wvs['S002VS'] == 3)]
f063_nga3 = nga_w3[nga_w3['F063'].between(1, 10)]
print(f"\nNigeria W3: N={len(f063_nga3)}, raw %10={f063_nga3['F063'].eq(10).mean()*100:.4f}%")
w_nga3 = wvs.loc[f063_nga3.index, 'S017']
print(f"  S017 stats: mean={w_nga3.mean():.4f}, std={w_nga3.std():.4f}")
pct_nga3_w = (f063_nga3['F063'].eq(10) * w_nga3).sum() / w_nga3.sum() * 100
print(f"  Weighted %10: {pct_nga3_w:.4f}%")

# South Africa wave 2 and 3
print("\n=== SOUTH AFRICA WAVES ===")
for wave in [1, 2, 3]:
    sub = wvs[(wvs['COUNTRY_ALPHA'] == 'ZAF') & (wvs['S002VS'] == wave)]
    valid = sub[sub['F063'].between(1, 10)]
    if len(valid) > 0:
        w = sub.loc[valid.index, 'S017']
        pct_uw = (valid['F063'] == 10).mean() * 100
        pct_w = (valid['F063'].eq(10) * w).sum() / w.sum() * 100
        print(f"ZAF wave {wave}: N={len(valid)}, unweighted={pct_uw:.4f}%, weighted={pct_w:.4f}%")
        print(f"  S017 stats: mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}")

# Check BEL 1981 in EVS
print("\n=== EVS 1981 DATA CHECK ===")
evs_csv = pd.read_csv(evs_csv_path, low_memory=False)
print(f"EVS CSV shape: {evs_csv.shape}")
print(f"EVS CSV countries: {sorted(evs_csv['COUNTRY_ALPHA'].unique())}")
print(f"EVS CSV S020 (year) values: {sorted(evs_csv['S020'].unique())}")
print(f"EVS CSV S002VS (wave) values: {sorted(evs_csv['S002VS'].unique())}")
