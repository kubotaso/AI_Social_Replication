"""Debug script to investigate all value discrepancies for Table 8"""
import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','S003','COUNTRY_ALPHA','F001','S020','S017','X048WVS'])
evs = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)

gt = {
    'Australia':        {'1981': 34, '1995-1998': 44, 'net_change': 10},
    'Belgium':          {'1981': 22, '1990-1991': 29, 'net_change': 7},
    'Canada':           {'1981': 38, '1990-1991': 43, 'net_change': 5},
    'Finland':          {'1981': 32, '1990-1991': 38, '1995-1998': 40, 'net_change': 8},
    'France':           {'1981': 36, '1990-1991': 39, 'net_change': 3},
    'East Germany':     {'1990-1991': 40, '1995-1998': 47, 'net_change': 7},
    'West Germany':     {'1981': 29, '1990-1991': 30, '1995-1998': 41, 'net_change': 12},
    'Great Britain':    {'1981': 34, '1990-1991': 36, 'net_change': 2},
    'Iceland':          {'1981': 39, '1990-1991': 36, 'net_change': -3},
    'Ireland':          {'1981': 25, '1990-1991': 34, 'net_change': 9},
    'Northern Ireland': {'1981': 29, '1990-1991': 33, 'net_change': 4},
    'South Korea':      {'1981': 29, '1990-1991': 39, 'net_change': 10},
    'Italy':            {'1981': 36, '1990-1991': 48, 'net_change': 12},
    'Japan':            {'1981': 21, '1990-1991': 21, '1995-1998': 26, 'net_change': 5},
    'Netherlands':      {'1981': 21, '1990-1991': 31, 'net_change': 10},
    'Norway':           {'1981': 26, '1990-1991': 31, '1995-1998': 32, 'net_change': 6},
    'Spain':            {'1981': 24, '1990-1991': 27, '1995-1998': 24, 'net_change': 0},
    'Sweden':           {'1981': 20, '1990-1991': 24, '1995-1998': 28, 'net_change': 8},
    'Switzerland':      {'1990-1991': 44, '1995-1998': 43, 'net_change': -1},
    'United States':    {'1981': 48, '1990-1991': 48, '1995-1998': 46, 'net_change': -2},
    'Belarus':          {'1990-1991': 35, '1995-1998': 47, 'net_change': 12},
    'Bulgaria':         {'1990-1991': 44, '1995-1998': 33, 'net_change': -11},
    'China':            {'1990-1991': 30, '1995-1998': 26, 'net_change': -4},
    'Estonia':          {'1990-1991': 35, '1995-1998': 39, 'net_change': 4},
    'Hungary':          {'1981': 44, '1990-1991': 45, 'net_change': 1},
    'Latvia':           {'1990-1991': 36, '1995-1998': 43, 'net_change': 7},
    'Lithuania':        {'1990-1991': 41, '1995-1998': 42, 'net_change': 1},
    'Russia':           {'1990-1991': 41, '1995-1998': 45, 'net_change': 4},
    'Slovenia':         {'1990-1991': 37, '1995-1998': 33, 'net_change': -4},
    'Argentina':        {'1981': 30, '1990-1991': 57, '1995-1998': 51, 'net_change': 21},
    'Brazil':           {'1990-1991': 44, '1995-1998': 37, 'net_change': -7},
    'Chile':            {'1990-1991': 54, '1995-1998': 50, 'net_change': -4},
    'India':            {'1990-1991': 28, '1995-1998': 23, 'net_change': -5},
    'Mexico':           {'1981': 32, '1990-1991': 40, '1995-1998': 39, 'net_change': 7},
    'Nigeria':          {'1990-1991': 60, '1995-1998': 50, 'net_change': -10},
    'South Africa':     {'1981': 38, '1990-1991': 57, '1995-1998': 46, 'net_change': 8},
    'Turkey':           {'1990-1991': 38, '1995-1998': 50, 'net_change': 12},
}

def pct_uw(series):
    valid = series[series > 0]
    if len(valid)==0: return None, 0
    return (valid==1).mean()*100, len(valid)

def pct_wt(f, w):
    mask = f > 0
    f2 = f[mask]; w2 = w[mask].fillna(1)
    if len(f2)==0: return None, 0
    return ((f2==1)*w2).sum() / w2.sum() * 100, len(f2)

print("="*100)
print("DETAILED COMPARISON: All computable values vs paper")
print("="*100)

# 1981 - WVS Wave 1
print("\n--- 1981 (WVS Wave 1) ---")
w1 = wvs[wvs['S002VS']==1]
for name, code in [('Australia','AUS'),('Finland','FIN'),('Hungary','HUN'),
                    ('Japan','JPN'),('South Korea','KOR'),('Mexico','MEX'),
                    ('Argentina','ARG'),('South Africa','ZAF')]:
    sub = w1[w1['COUNTRY_ALPHA']==code]
    uw, n_uw = pct_uw(sub['F001'])
    wt, n_wt = pct_wt(sub['F001'], sub['S017'])
    gt_val = gt.get(name,{}).get('1981')
    if uw is not None and gt_val is not None:
        print(f'{name:20s} 1981: uw={uw:5.1f}(r={round(uw):2d}) wt={wt:5.1f}(r={round(wt):2d}) paper={gt_val:2d} n={n_uw}')

# 1990-1991 - EVS (ZA4460)
print("\n--- 1990-1991 (EVS ZA4460) ---")
for name, code in [('Belgium','BE'),('Canada','CA'),('Finland','FI'),('France','FR'),
                    ('Great Britain','GB-GBN'),('Iceland','IS'),('Ireland','IE'),
                    ('Northern Ireland','GB-NIR'),('Italy','IT'),('Netherlands','NL'),
                    ('Norway','NO'),('Spain','ES'),('Sweden','SE'),('United States','US'),
                    ('Hungary','HU'),('Bulgaria','BG'),('Estonia','EE'),
                    ('Latvia','LV'),('Lithuania','LT'),('Slovenia','SI')]:
    sub = evs[evs['c_abrv']==code]
    uw, n_uw = pct_uw(sub['q322'])
    wt_g, _ = pct_wt(sub['q322'], sub['weight_g'])
    wt_s, _ = pct_wt(sub['q322'], sub['weight_s'])
    gt_val = gt.get(name,{}).get('1990-1991')
    if uw is not None and gt_val is not None:
        print(f'{name:20s} 1990: uw={uw:5.1f}(r={round(uw):2d}) wt_g={wt_g:5.1f}(r={round(wt_g):2d}) wt_s={wt_s:5.1f}(r={round(wt_s):2d}) paper={gt_val:2d} n={n_uw}')

# East/West Germany EVS 1990
for name, code in [('East Germany','DE-E'),('West Germany','DE-W')]:
    sub = evs[evs['c_abrv1']==code]
    uw, n_uw = pct_uw(sub['q322'])
    wt_g, _ = pct_wt(sub['q322'], sub['weight_g'])
    wt_s, _ = pct_wt(sub['q322'], sub['weight_s'])
    gt_val = gt.get(name,{}).get('1990-1991')
    if uw is not None and gt_val is not None:
        print(f'{name:20s} 1990: uw={uw:5.1f}(r={round(uw):2d}) wt_g={wt_g:5.1f}(r={round(wt_g):2d}) wt_s={wt_s:5.1f}(r={round(wt_s):2d}) paper={gt_val:2d} n={n_uw}')

# 1990-1991 - WVS Wave 2
print("\n--- 1990-1991 (WVS Wave 2) ---")
w2 = wvs[wvs['S002VS']==2]
for name, code in [('South Korea','KOR'),('Japan','JPN'),('Switzerland','CHE'),
                    ('Argentina','ARG'),('Brazil','BRA'),('Chile','CHL'),
                    ('China','CHN'),('India','IND'),('Mexico','MEX'),
                    ('Nigeria','NGA'),('South Africa','ZAF'),('Turkey','TUR'),
                    ('Belarus','BLR'),('Russia','RUS')]:
    sub = w2[w2['COUNTRY_ALPHA']==code]
    uw, n_uw = pct_uw(sub['F001'])
    wt, _ = pct_wt(sub['F001'], sub['S017'])
    gt_val = gt.get(name,{}).get('1990-1991')
    if uw is not None and gt_val is not None:
        print(f'{name:20s} W2: uw={uw:5.1f}(r={round(uw):2d}) wt={wt:5.1f}(r={round(wt):2d}) paper={gt_val:2d} n={n_uw}')

# 1995-1998 - WVS Wave 3
print("\n--- 1995-1998 (WVS Wave 3) ---")
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
    uw, n_uw = pct_uw(sub['F001'])
    wt, _ = pct_wt(sub['F001'], sub['S017'])
    gt_val = gt.get(name,{}).get('1995-1998')
    if uw is not None and gt_val is not None:
        print(f'{name:20s} W3: uw={uw:5.1f}(r={round(uw):2d}) wt={wt:5.1f}(r={round(wt):2d}) paper={gt_val:2d} n={n_uw}')

# Germany W3 split
print("\n--- Germany W3 split ---")
deu_w3 = wvs[(wvs['COUNTRY_ALPHA']=='DEU') & (wvs['S002VS']==3)]
west_codes = [276001,276002,276003,276004,276005,276006,276007,276008,276009,276010,276019]
east_codes = [276012,276013,276014,276015,276016]
east_codes_berlin = [276012,276013,276014,276015,276016,276020]

w = deu_w3[deu_w3['X048WVS'].isin(west_codes)]
e = deu_w3[deu_w3['X048WVS'].isin(east_codes)]
e_b = deu_w3[deu_w3['X048WVS'].isin(east_codes_berlin)]

for label, sub in [('West (no Berlin-E)', w), ('East (no Berlin-E)', e), ('East (with Berlin-E)', e_b)]:
    uw, n = pct_uw(sub['F001'])
    wt, _ = pct_wt(sub['F001'], sub['S017'])
    print(f'{label:25s}: uw={uw:.1f}(r={round(uw)}) wt={wt:.1f}(r={round(wt)}) n={n}')

# Now check if using floor instead of round helps for any borderline values
print("\n\n=== BORDERLINE VALUES (where floor vs round matters) ===")
print("Checking all values with fractional part near 0.5:")
