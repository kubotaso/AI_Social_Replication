"""
Investigate if different rounding approaches or weight choices can fix partial matches.
For each partial, try: round, floor, ceil, and different weights.
"""
import pandas as pd
import numpy as np
import os
import math

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, 'data', 'WVS_Time_Series_1981-2022_csv_v5_0.csv')
evs_path = os.path.join(base, 'data', 'ZA4460_v3-0-0.dta')

wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'COUNTRY_ALPHA', 'F001', 'S020', 'S017', 'X048WVS'])
evs = pd.read_stata(evs_path, convert_categoricals=False)

def calc_pct(f001, weights=None):
    """Return exact float percentage"""
    mask = f001 > 0
    f = f001[mask]
    if len(f) == 0: return None
    if weights is not None:
        w = weights[mask].fillna(1)
        return ((f == 1) * w).sum() / w.sum() * 100
    return (f == 1).mean() * 100

# Test all partial cases with different rounding
cases = []

# WVS Wave 3 countries
w3 = wvs[wvs['S002VS'] == 3]
for country, code, paper_val, period in [
    ('Japan', 'JPN', 26, '1995-1998'),
    ('Lithuania', 'LTU', 42, '1995-1998'),
    ('Nigeria', 'NGA', 50, '1995-1998'),
    ('India', 'IND', 23, '1995-1998'),
    ('Turkey', 'TUR', 50, '1995-1998'),
]:
    sub = w3[w3['COUNTRY_ALPHA'] == code]
    pct_w = calc_pct(sub['F001'], sub['S017'])
    pct_u = calc_pct(sub['F001'])
    cases.append((country, period, paper_val, pct_w, pct_u))

# WVS Wave 2 countries
w2 = wvs[wvs['S002VS'] == 2]
for country, code, paper_val, period in [
    ('China', 'CHN', 30, '1990-1991'),
    ('Brazil', 'BRA', 44, '1990-1991'),
    ('Chile', 'CHL', 54, '1990-1991'),
    ('Nigeria', 'NGA', 60, '1990-1991'),
    ('India', 'IND', 28, '1990-1991'),
]:
    sub = w2[w2['COUNTRY_ALPHA'] == code]
    pct_w = calc_pct(sub['F001'], sub['S017'])
    pct_u = calc_pct(sub['F001'])
    cases.append((country, period, paper_val, pct_w, pct_u))

# WVS Wave 1
w1 = wvs[wvs['S002VS'] == 1]
for country, code, paper_val, period in [
    ('South Africa', 'ZAF', 38, '1981'),
]:
    sub = w1[w1['COUNTRY_ALPHA'] == code]
    pct_w = calc_pct(sub['F001'], sub['S017'])
    pct_u = calc_pct(sub['F001'])
    cases.append((country, period, paper_val, pct_w, pct_u))

# EVS 1990 countries
for country, code, paper_val, period in [
    ('Spain', 'ES', 27, '1990-1991'),
]:
    sub = evs[evs['c_abrv'] == code]
    pct_u = calc_pct(sub['q322'])
    pct_wg = calc_pct(sub['q322'], sub['weight_g']) if 'weight_g' in evs.columns else None
    pct_ws = calc_pct(sub['q322'], sub['weight_s']) if 'weight_s' in evs.columns else None
    cases.append((country, period, paper_val, pct_wg, pct_u))
    if pct_ws is not None:
        print(f"  {country} EVS weight_s: {pct_ws:.4f}")

print(f"\n{'Country':<18} {'Period':<12} {'Paper':>6} {'Weighted':>10} {'Unweighted':>12} {'rnd(W)':>7} {'rnd(U)':>7} {'floor(W)':>9} {'ceil(W)':>8} {'Match?':>8}")
print("-" * 110)

for country, period, paper_val, pct_w, pct_u in cases:
    rw = round(pct_w) if pct_w else '-'
    ru = round(pct_u) if pct_u else '-'
    fw = math.floor(pct_w) if pct_w else '-'
    cw = math.ceil(pct_w) if pct_w else '-'
    fu = math.floor(pct_u) if pct_u else '-'
    cu = math.ceil(pct_u) if pct_u else '-'

    match = 'EXACT' if (rw == paper_val or ru == paper_val) else \
            'FLOOR' if (fw == paper_val or fu == paper_val) else \
            'CEIL' if (cw == paper_val or cu == paper_val) else 'MISS'

    pw_str = f"{pct_w:.4f}" if pct_w else "N/A"
    pu_str = f"{pct_u:.4f}" if pct_u else "N/A"
    print(f"{country:<18} {period:<12} {paper_val:>6} {pw_str:>10} {pu_str:>12} {rw:>7} {ru:>7} {fw:>9} {cw:>8} {match:>8}")

# EVS weight_s investigation
print("\n\nEVS weight_s investigation:")
for country, code, paper_val in [('Spain', 'ES', 27), ('Belgium', 'BE', 29), ('France', 'FR', 39),
                                  ('Great Britain', 'GB-GBN', 36), ('Iceland', 'IS', 36),
                                  ('Ireland', 'IE', 34), ('Northern Ireland', 'GB-NIR', 33),
                                  ('Italy', 'IT', 48), ('Netherlands', 'NL', 31),
                                  ('Norway', 'NO', 31), ('Sweden', 'SE', 24),
                                  ('Finland', 'FI', 38), ('Hungary', 'HU', 45),
                                  ('Bulgaria', 'BG', 44), ('Estonia', 'EE', 35),
                                  ('Latvia', 'LV', 36), ('Lithuania', 'LT', 41),
                                  ('Slovenia', 'SI', 37)]:
    sub = evs[evs['c_abrv'] == code]
    if len(sub) > 0 and 'q322' in evs.columns:
        pct_u = calc_pct(sub['q322'])
        pct_wg = calc_pct(sub['q322'], sub['weight_g']) if 'weight_g' in evs.columns else None
        pct_ws = calc_pct(sub['q322'], sub['weight_s']) if 'weight_s' in evs.columns else None
        matches = []
        if pct_u and round(pct_u) == paper_val: matches.append('unw')
        if pct_wg and round(pct_wg) == paper_val: matches.append('wg')
        if pct_ws and round(pct_ws) == paper_val: matches.append('ws')
        u_str = f"{pct_u:.2f}" if pct_u else "N/A"
        wg_str = f"{pct_wg:.2f}" if pct_wg else "N/A"
        ws_str = f"{pct_ws:.2f}" if pct_ws else "N/A"
        match_str = ','.join(matches) if matches else 'NONE'
        print(f"  {country:<22}: unw={u_str:>8}, wg={wg_str:>8}, ws={ws_str:>8}, paper={paper_val}, match={match_str}")

# East/West Germany EVS
print("\nEVS East/West Germany:")
for code_field, code, name in [('c_abrv1', 'DE-E', 'East Germany'), ('c_abrv1', 'DE-W', 'West Germany')]:
    sub = evs[evs[code_field] == code]
    if len(sub) > 0:
        pct_u = calc_pct(sub['q322'])
        pct_wg = calc_pct(sub['q322'], sub['weight_g']) if 'weight_g' in evs.columns else None
        pct_ws = calc_pct(sub['q322'], sub['weight_s']) if 'weight_s' in evs.columns else None
        u_str = f"{pct_u:.2f}" if pct_u else "N/A"
        wg_str = f"{pct_wg:.2f}" if pct_wg else "N/A"
        ws_str = f"{pct_ws:.2f}" if pct_ws else "N/A"
        print(f"  {name:<22}: unw={u_str:>8}, wg={wg_str:>8}, ws={ws_str:>8}")

# US via EVS
print("\nUS via EVS:")
sub = evs[evs['c_abrv'] == 'US']
if len(sub) > 0:
    pct_u = calc_pct(sub['q322'])
    pct_wg = calc_pct(sub['q322'], sub['weight_g']) if 'weight_g' in evs.columns else None
    print(f"  US EVS: unw={pct_u:.2f}, wg={pct_wg:.2f}" if pct_wg else f"  US EVS: unw={pct_u:.2f}")

# Canada via EVS
sub = evs[evs['c_abrv'] == 'CA']
if len(sub) > 0:
    pct_u = calc_pct(sub['q322'])
    pct_wg = calc_pct(sub['q322'], sub['weight_g']) if 'weight_g' in evs.columns else None
    print(f"  Canada EVS: unw={pct_u:.2f}, wg={pct_wg:.2f}" if pct_wg else f"  Canada EVS: unw={pct_u:.2f}")
