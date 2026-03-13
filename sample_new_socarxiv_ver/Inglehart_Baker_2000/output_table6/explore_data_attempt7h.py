"""
Final checks: Poland, Norway wave 3, and all current status
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
evs_stata_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")

evs = pd.read_stata(evs_stata_path, convert_categoricals=False,
                     columns=['c_abrv', 'country1', 'q336', 'year', 'weight_g', 'weight_s'])
wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'S020', 'F028', 'S017'], low_memory=False)

# Poland EVS check (paper=85%)
print("=== Poland 1990-1991 ===")
pol = evs[evs['c_abrv'] == 'PL']
print("Poland EVS rows:", len(pol))
print("q336 dist:", dict(pol['q336'].value_counts().sort_index()))
valid_8 = pol[pol['q336'].isin([1,2,3,4,5,6,7,8])]
valid_7 = pol[pol['q336'].isin([1,2,3,4,5,6,7])]
monthly = pol[pol['q336'].isin([1,2,3])]
print(f"  8pt unweighted: {len(monthly)}/{len(valid_8)} = {len(monthly)/len(valid_8)*100:.3f}% -> rounds to {round(len(monthly)/len(valid_8)*100)}")
print(f"  7pt unweighted: {len(monthly)}/{len(valid_7)} = {len(monthly)/len(valid_7)*100:.3f}% -> rounds to {round(len(monthly)/len(valid_7)*100)}")

# Try WVS Poland wave 2
w2 = wvs[wvs['S002VS'] == 2]
pol2 = w2[w2['S003'] == 616]
print(f"\nPoland WVS wave 2 rows: {len(pol2)}")

# Try WVS Poland wave 3
w3 = wvs[wvs['S002VS'] == 3]
pol3 = w3[w3['S003'] == 616]
print(f"Poland WVS wave 3 rows: {len(pol3)}")
if len(pol3) > 0:
    print("F028 dist:", dict(pol3['F028'].value_counts().sort_index()))
    valid = pol3[pol3['F028'].isin([1,2,3,4,5,6,7,8])]
    monthly_p3 = pol3[pol3['F028'].isin([1,2,3])]
    print(f"  Poland W3: {round(len(monthly_p3)/len(valid)*100)}% (paper=74%)")
    pct_w = round(monthly_p3['S017'].sum()/valid['S017'].sum()*100) if valid['S017'].sum() > 0 else None
    print(f"  Poland W3 weighted: {pct_w}%")

# EVS Poland with weights
for wcol in ['weight_g', 'weight_s']:
    if wcol in evs.columns:
        valid = pol[pol['q336'].isin([1,2,3,4,5,6,7,8])]
        monthly_pw = pol[pol['q336'].isin([1,2,3])]
        wpct = round(monthly_pw[wcol].sum() / valid[wcol].sum() * 100)
        print(f"  Poland EVS {wcol}: {wpct}%")

# Check Norway wave 3 (paper=13%)
print("\n=== Norway WVS wave 3 (paper=13%) ===")
nor3 = w3[w3['S003'] == 578]
print("Norway wave 3 rows:", len(nor3))
if len(nor3) > 0:
    print("F028 dist:", dict(nor3['F028'].value_counts().sort_index()))
    valid = nor3[nor3['F028'].isin([1,2,3,4,5,6,7,8])]
    monthly_n = nor3[nor3['F028'].isin([1,2,3])]
    print(f"  Norway W3 unweighted: {round(len(monthly_n)/len(valid)*100)}%")
    pct_w = round(monthly_n['S017'].sum()/valid['S017'].sum()*100) if valid['S017'].sum() > 0 else None
    print(f"  Norway W3 weighted: {pct_w}%")

# Complete current status check
print("\n\n=== ESTIMATED NEW SCORE with Finland fix ===")
print("Previously (attempt 6): 61 FULL + 3 PARTIAL + 2 MISS + 14 MISSING")
print("New fix: Finland 1990-1991 from PARTIAL (11 vs 13) -> FULL (13 vs 13)")
print("Korea W2 was already a FULL match (60% correct)")
print()
print("NEW ESTIMATED: 62 FULL + 2 PARTIAL + 2 MISS + 14 MISSING")
print()

# Compute score with new numbers
total_cells = 80
matched = 62
partial = 2
missed = 2
missing = 14

present = total_cells - missing
categories_score = present/total_cells * 20
values_numerator = matched * 1.0 + partial * 0.7 + missed * 0.2
values_score = (values_numerator / total_cells) * 40
ordering_score = 10
net_change_score = present/total_cells * 20
column_score = 10

total_score = categories_score + values_score + ordering_score + net_change_score + column_score
print(f"Categories (66/80 present): {categories_score:.2f}/20")
print(f"Values (62 full, 2 partial, 2 miss): {values_score:.2f}/40")
print(f"Ordering: {ordering_score}/10")
print(f"Net change (66/80): {net_change_score:.2f}/20")
print(f"Column: {column_score}/10")
print(f"TOTAL: {total_score:.2f}/100 -> {round(total_score)}")
