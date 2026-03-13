"""Investigate every remaining discrepancy to find optimal approach"""
import pandas as pd
import numpy as np
import math

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','S003','COUNTRY_ALPHA','F001','S020','S017','S018','X048WVS'])
evs = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)

def try_all_methods(f001, s017, s018=None):
    """Try unweighted, S017, S018, and different rounding methods"""
    results = {}
    valid = f001[f001 > 0]
    if len(valid) == 0:
        return results

    # Unweighted
    raw_uw = (valid == 1).mean() * 100
    results['uw'] = raw_uw
    results['uw_round'] = round(raw_uw)
    results['uw_floor'] = int(raw_uw)

    # S017 weighted
    mask = f001 > 0
    f2 = f001[mask]; w2 = s017[mask].fillna(1)
    raw_wt = ((f2 == 1) * w2).sum() / w2.sum() * 100
    results['s017'] = raw_wt
    results['s017_round'] = round(raw_wt)
    results['s017_floor'] = int(raw_wt)

    # S018 weighted if available
    if s018 is not None:
        w3 = s018[mask].fillna(1)
        if w3.sum() > 0:
            raw_s18 = ((f2 == 1) * w3).sum() / w3.sum() * 100
            results['s018'] = raw_s18
            results['s018_round'] = round(raw_s18)
            results['s018_floor'] = int(raw_s18)

    results['n'] = len(valid)
    return results

# ============ SYSTEMATIC CHECK ============
# For each country-period, find which method matches paper

paper_vals = {
    # 1981
    ('Australia','1981'): 34, ('Finland','1981'): 32, ('Hungary','1981'): 44,
    ('Japan','1981'): 21, ('South Korea','1981'): 29, ('Mexico','1981'): 32,
    ('Argentina','1981'): 30, ('South Africa','1981'): 38,
    # 1990-1991 EVS
    ('Belgium','1990'): 29, ('Canada','1990'): 43, ('Finland','1990'): 38,
    ('France','1990'): 39, ('Great Britain','1990'): 36, ('Iceland','1990'): 36,
    ('Ireland','1990'): 34, ('Northern Ireland','1990'): 33, ('Italy','1990'): 48,
    ('Netherlands','1990'): 31, ('Norway','1990'): 31, ('Spain','1990'): 27,
    ('Sweden','1990'): 24, ('United States','1990'): 48, ('Hungary','1990'): 45,
    ('Bulgaria','1990'): 44, ('Estonia','1990'): 35, ('Latvia','1990'): 36,
    ('Lithuania','1990'): 41, ('Slovenia','1990'): 37,
    ('East Germany','1990'): 40, ('West Germany','1990'): 30,
    # 1990-1991 WVS W2
    ('South Korea','1990w'): 39, ('Japan','1990w'): 21, ('Switzerland','1990w'): 44,
    ('Argentina','1990w'): 57, ('Brazil','1990w'): 44, ('Chile','1990w'): 54,
    ('China','1990w'): 30, ('India','1990w'): 28, ('Mexico','1990w'): 40,
    ('Nigeria','1990w'): 60, ('South Africa','1990w'): 57, ('Turkey','1990w'): 38,
    ('Belarus','1990w'): 35, ('Russia','1990w'): 41,
    # 1995-1998
    ('Australia','W3'): 44, ('Finland','W3'): 40, ('Japan','W3'): 26,
    ('Norway','W3'): 32, ('Spain','W3'): 24, ('Sweden','W3'): 28,
    ('Switzerland','W3'): 43, ('United States','W3'): 46,
    ('Belarus','W3'): 47, ('Bulgaria','W3'): 33, ('China','W3'): 26,
    ('Estonia','W3'): 39, ('Latvia','W3'): 43, ('Lithuania','W3'): 42,
    ('Russia','W3'): 45, ('Slovenia','W3'): 33,
    ('Argentina','W3'): 51, ('Brazil','W3'): 37, ('Chile','W3'): 50,
    ('India','W3'): 23, ('Mexico','W3'): 39, ('Nigeria','W3'): 50,
    ('South Africa','W3'): 46, ('Turkey','W3'): 50,
    ('East Germany','W3'): 47, ('West Germany','W3'): 41,
}

print("="*120)
print("SYSTEMATIC METHOD COMPARISON")
print("="*120)

# 1981 - WVS Wave 1
print("\n--- 1981 (WVS Wave 1) ---")
w1 = wvs[wvs['S002VS']==1]
for name, code in [('Australia','AUS'),('Finland','FIN'),('Hungary','HUN'),
                    ('Japan','JPN'),('South Korea','KOR'),('Mexico','MEX'),
                    ('Argentina','ARG'),('South Africa','ZAF')]:
    sub = w1[w1['COUNTRY_ALPHA']==code]
    r = try_all_methods(sub['F001'], sub['S017'], sub.get('S018'))
    paper = paper_vals.get((name,'1981'))
    best = None
    for method in ['uw_round','uw_floor','s017_round','s017_floor','s018_round','s018_floor']:
        if method in r and r[method] == paper:
            best = method
            break
    print(f'{name:20s}: uw={r.get("uw",0):.2f} s017={r.get("s017",0):.2f} s018={r.get("s018","N/A")} paper={paper} BEST={best or "NONE"}')

# 1990-1991 EVS
print("\n--- 1990-1991 (EVS) ---")
for name, code in [('Belgium','BE'),('Canada','CA'),('Finland','FI'),('France','FR'),
                    ('Great Britain','GB-GBN'),('Iceland','IS'),('Ireland','IE'),
                    ('Northern Ireland','GB-NIR'),('Italy','IT'),('Netherlands','NL'),
                    ('Norway','NO'),('Spain','ES'),('Sweden','SE'),('United States','US'),
                    ('Hungary','HU'),('Bulgaria','BG'),('Estonia','EE'),
                    ('Latvia','LV'),('Lithuania','LT'),('Slovenia','SI')]:
    sub = evs[evs['c_abrv']==code]
    valid = sub['q322'][sub['q322']>0]
    uw_raw = (valid==1).mean()*100 if len(valid)>0 else 0
    mask = sub['q322']>0
    f2 = sub['q322'][mask]; wg = sub['weight_g'][mask].fillna(1); ws = sub['weight_s'][mask].fillna(1)
    wg_raw = ((f2==1)*wg).sum()/wg.sum()*100 if len(f2)>0 else 0
    ws_raw = ((f2==1)*ws).sum()/ws.sum()*100 if len(f2)>0 else 0
    paper = paper_vals.get((name,'1990'))

    best = None
    for val, method in [(round(uw_raw),'uw_round'),(int(uw_raw),'uw_floor'),
                         (round(wg_raw),'wg_round'),(int(wg_raw),'wg_floor'),
                         (round(ws_raw),'ws_round'),(int(ws_raw),'ws_floor')]:
        if val == paper:
            best = method
            break
    print(f'{name:20s}: uw={uw_raw:.2f} wg={wg_raw:.2f} ws={ws_raw:.2f} paper={paper} n={len(valid)} BEST={best or "NONE"}')

# East/West Germany EVS
for name, code in [('East Germany','DE-E'),('West Germany','DE-W')]:
    sub = evs[evs['c_abrv1']==code]
    valid = sub['q322'][sub['q322']>0]
    uw_raw = (valid==1).mean()*100 if len(valid)>0 else 0
    mask = sub['q322']>0
    f2 = sub['q322'][mask]; wg = sub['weight_g'][mask].fillna(1)
    wg_raw = ((f2==1)*wg).sum()/wg.sum()*100 if len(f2)>0 else 0
    paper = paper_vals.get((name,'1990'))
    best = None
    for val, method in [(round(uw_raw),'uw_round'),(int(uw_raw),'uw_floor'),
                         (round(wg_raw),'wg_round'),(int(wg_raw),'wg_floor')]:
        if val == paper:
            best = method
            break
    print(f'{name:20s}: uw={uw_raw:.2f} wg={wg_raw:.2f} paper={paper} n={len(valid)} BEST={best or "NONE"}')

# WVS Wave 2
print("\n--- 1990-1991 (WVS W2) ---")
w2 = wvs[wvs['S002VS']==2]
for name, code in [('South Korea','KOR'),('Japan','JPN'),('Switzerland','CHE'),
                    ('Argentina','ARG'),('Brazil','BRA'),('Chile','CHL'),
                    ('China','CHN'),('India','IND'),('Mexico','MEX'),
                    ('Nigeria','NGA'),('South Africa','ZAF'),('Turkey','TUR'),
                    ('Belarus','BLR'),('Russia','RUS')]:
    sub = w2[w2['COUNTRY_ALPHA']==code]
    r = try_all_methods(sub['F001'], sub['S017'], sub.get('S018'))
    paper = paper_vals.get((name,'1990w'))
    best = None
    for method in ['uw_round','uw_floor','s017_round','s017_floor','s018_round','s018_floor']:
        if method in r and r[method] == paper:
            best = method
            break
    print(f'{name:20s}: uw={r.get("uw",0):.2f} s017={r.get("s017",0):.2f} s018={r.get("s018","N/A")} paper={paper} n={r.get("n",0)} BEST={best or "NONE"}')

# WVS Wave 3
print("\n--- 1995-1998 (WVS W3) ---")
w3 = wvs[wvs['S002VS']==3]
for name, code in [('Australia','AUS'),('Finland','FIN'),('Japan','JPN'),
                    ('Norway','NOR'),('Spain','ESP'),('Sweden','SWE'),
                    ('Switzerland','CHE'),('United States','USA'),
                    ('Belarus','BLR'),('Bulgaria','BGR'),('China','CHN'),
                    ('Estonia','EST'),('Latvia','LVA'),('Lithuania','LTU'),
                    ('Russia','RUS'),('Slovenia','SVN'),
                    ('Argentina','ARG'),('Brazil','BRA'),('Chile','CHL'),
                    ('India','IND'),('Mexico','MEX'),('Nigeria','NGA'),
                    ('South Africa','ZAF'),('Turkey','TUR')]:
    sub = w3[w3['COUNTRY_ALPHA']==code]
    r = try_all_methods(sub['F001'], sub['S017'], sub.get('S018'))
    paper = paper_vals.get((name,'W3'))
    best = None
    for method in ['uw_round','uw_floor','s017_round','s017_floor','s018_round','s018_floor']:
        if method in r and r[method] == paper:
            best = method
            break
    print(f'{name:20s}: uw={r.get("uw",0):.2f} s017={r.get("s017",0):.2f} s018={r.get("s018","N/A")} paper={paper} n={r.get("n",0)} BEST={best or "NONE"}')

# Germany W3 split
print("\n--- Germany W3 ---")
deu_w3 = wvs[(wvs['COUNTRY_ALPHA']=='DEU') & (wvs['S002VS']==3)]
west_codes = [276001,276002,276003,276004,276005,276006,276007,276008,276009,276010,276019]
east_codes = [276012,276013,276014,276015,276016]
east_codes_berlin = [276012,276013,276014,276015,276016,276020]

for label, codes, paper_key in [('West Germany', west_codes, ('West Germany','W3')),
                                 ('East Germany', east_codes, ('East Germany','W3')),
                                 ('East Germany+Berlin', east_codes_berlin, ('East Germany','W3'))]:
    sub = deu_w3[deu_w3['X048WVS'].isin(codes)]
    r = try_all_methods(sub['F001'], sub['S017'], sub.get('S018'))
    paper = paper_vals.get(paper_key)
    best = None
    for method in ['uw_round','uw_floor','s017_round','s017_floor','s018_round','s018_floor']:
        if method in r and r[method] == paper:
            best = method
            break
    print(f'{label:25s}: uw={r.get("uw",0):.2f} s017={r.get("s017",0):.2f} paper={paper} n={r.get("n",0)} BEST={best or "NONE"}')
