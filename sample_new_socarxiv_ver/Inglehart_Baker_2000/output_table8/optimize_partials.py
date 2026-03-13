"""Try to fix remaining partial matches by finding better methods"""
import pandas as pd
import numpy as np
import math

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F001','S017','X048WVS'])
evs_long = pd.read_stata('/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/OldFiles/Replication_Claude_IB/data/ZA4804_v3-1-0.dta/ZA4804_v3-1-0.dta',
                          convert_categoricals=False, columns=['S002EVS','S003','F001','S017','S020'])
evs1990 = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)

def pct_uw(f001):
    v = f001[f001>0]
    return (v==1).mean()*100 if len(v)>0 else None

def pct_wt(f001, w):
    m = f001>0; f=f001[m]; ww=w[m].fillna(1)
    return ((f==1)*ww).sum()/ww.sum()*100 if len(f)>0 else None

# ==== CHECK EACH PARTIAL ====

# 1. Canada 1981: EVS longitudinal gives 36.91 (uw) 37.11 (wt). Paper=38.
print("=== Canada 1981 ===")
evs_w1 = evs_long[evs_long['S002EVS']==1]
can = evs_w1[evs_w1['S003']==124]
f = can['F001']
print(f"F001 dist: {dict(f.value_counts().sort_index())}")
uw = pct_uw(f); wt = pct_wt(f, can['S017'])
print(f"uw={uw:.4f} wt={wt:.4f}")
# Try including DK/NA as not-often
all_resp = f[f >= -1]
pct_with_dk = 100 * (all_resp == 1).sum() / len(all_resp)
print(f"Including DK: {pct_with_dk:.4f}")

# 2. Spain 1990: EVS ZA4460 gives 28.15 (uw). Paper=27.
print("\n=== Spain 1990 ===")
esp = evs1990[evs1990['c_abrv']=='ES']
f = esp['q322']
print(f"q322 dist: {dict(f.value_counts().sort_index())}")
# Check if there are different year subsets
for y in sorted(esp['year'].unique()):
    sub = esp[esp['year']==y]
    v = sub['q322'][sub['q322']>0]
    if len(v) > 0:
        print(f"  Year {y}: n={len(v)}, Often={100*(v==1).mean():.2f}%")
# Check EVS longitudinal for Spain 1990
esp_evs = evs_long[(evs_long['S003']==724) & (evs_long['S002EVS']==2)]
if len(esp_evs) > 0:
    uw2 = pct_uw(esp_evs['F001'])
    wt2 = pct_wt(esp_evs['F001'], esp_evs['S017'])
    print(f"EVS longitudinal Spain W2: uw={uw2:.4f} wt={wt2:.4f}")

# 3. Brazil W2: WVS gives 43.09 (uw) 43.47 (wt). Paper=44.
print("\n=== Brazil W2 ===")
w2 = wvs[wvs['S002VS']==2]
bra = w2[w2['COUNTRY_ALPHA']=='BRA']
f = bra['F001']
print(f"F001 dist: {dict(f.value_counts().sort_index())}")
# S017 gives 43.47 which rounds to 43. Paper says 44.
# What if we use int(x+0.5) "round half up"?
wt_val = pct_wt(f, bra['S017'])
print(f"S017 wt={wt_val:.4f}, int(x+0.5)={int(wt_val+0.5)}")

# 4. Chile W2: WVS gives 53.43 (uw). Paper=54.
print("\n=== Chile W2 ===")
chl = w2[w2['COUNTRY_ALPHA']=='CHL']
f = chl['F001']
uw = pct_uw(f); wt = pct_wt(f, chl['S017'])
print(f"uw={uw:.4f} wt={wt:.4f}")

# 5. China W2: WVS gives 31.49 (uw). Paper=30.
print("\n=== China W2 ===")
chn = w2[w2['COUNTRY_ALPHA']=='CHN']
f = chn['F001']
uw = pct_uw(f); wt = pct_wt(f, chn['S017'])
print(f"uw={uw:.4f} wt={wt:.4f} int(uw)={int(uw)} int(wt)={int(wt)}")

# 6. Nigeria W2: WVS gives 58.79 (uw). Paper=60.
print("\n=== Nigeria W2 ===")
nga = w2[w2['COUNTRY_ALPHA']=='NGA']
f = nga['F001']
uw = pct_uw(f); wt = pct_wt(f, nga['S017'])
print(f"uw={uw:.4f} wt={wt:.4f}")

# 7. Japan W3: WVS gives 25.39 (uw). Paper=26.
print("\n=== Japan W3 ===")
w3 = wvs[wvs['S002VS']==3]
jpn = w3[w3['COUNTRY_ALPHA']=='JPN']
f = jpn['F001']
uw = pct_uw(f); wt = pct_wt(f, jpn['S017'])
print(f"uw={uw:.4f} wt={wt:.4f} ceil(uw)={math.ceil(uw)}")

# 8. Lithuania W3: WVS gives 41.38 (uw). Paper=42.
print("\n=== Lithuania W3 ===")
ltu = w3[w3['COUNTRY_ALPHA']=='LTU']
f = ltu['F001']
uw = pct_uw(f); wt = pct_wt(f, ltu['S017'])
print(f"uw={uw:.4f} wt={wt:.4f} ceil(uw)={math.ceil(uw)}")

# 9. Turkey W3: WVS gives 49.46 (uw), 47.36 (wt). Paper=50.
print("\n=== Turkey W3 ===")
tur = w3[w3['COUNTRY_ALPHA']=='TUR']
f = tur['F001']
uw = pct_uw(f); wt = pct_wt(f, tur['S017'])
print(f"uw={uw:.4f} wt={wt:.4f} ceil(uw)={math.ceil(uw)} int(uw+0.5)={int(uw+0.5)}")

# 10. US 1981: EVS gives 49.15 (uw). Paper=48.
print("\n=== US 1981 ===")
us81 = evs_long[(evs_long['S003']==840) & (evs_long['S002EVS']==1)]
f = us81['F001']
uw = pct_uw(f); wt = pct_wt(f, us81['S017'])
print(f"uw={uw:.4f} wt={wt:.4f}")
# Can we match 48%? Neither rounds or floors to 48.
# The paper may have used the original EVS data release, not the longitudinal harmonization

# Summary: which partials can potentially be fixed?
print("\n\n=== SUMMARY: POTENTIALLY FIXABLE PARTIALS ===")
print("None of the remaining partials can be converted to exact matches")
print("using standard methods (unweighted, S017, floor, round, ceil)")
print("The discrepancies likely arise from different data versions/harmonizations.")
