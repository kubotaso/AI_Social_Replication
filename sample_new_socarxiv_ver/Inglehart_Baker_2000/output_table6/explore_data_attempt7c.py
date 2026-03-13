"""
Exploration: EVS CSV F028 binary variable analysis
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
evs_stata_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")
evs_csv_path = os.path.join(base, "data", "EVS_1990_wvs_format.csv")

# Load EVS CSV
evs_csv = pd.read_csv(evs_csv_path, low_memory=False)
print("=== EVS CSV F028 binary analysis ===")
print("F028 is binary: 0=less than monthly, 1=at least monthly")
print()

# Known results from paper for comparison
paper_values = {
    'BEL': 35, 'CAN': 40, 'FIN': 13, 'FRA': 17,
    'GBR': 25, 'ISL': 9, 'IRL': 88, 'NIR': 69,
    'ITA': 47, 'NLD': 31, 'NOR': 13, 'ESP': 40,
    'SWE': 10, 'USA': 59,
}

# Country map
country_map = {
    'AUT': 'Austria', 'BEL': 'Belgium', 'BGR': 'Bulgaria', 'CAN': 'Canada',
    'CZE': 'Czech Republic', 'DEU': 'Germany', 'DNK': 'Denmark', 'ESP': 'Spain',
    'EST': 'Estonia', 'FIN': 'Finland', 'FRA': 'France', 'GBR': 'Great Britain',
    'HUN': 'Hungary', 'IRL': 'Ireland', 'ISL': 'Iceland', 'ITA': 'Italy',
    'LTU': 'Lithuania', 'LVA': 'Latvia', 'MLT': 'Malta', 'NIR': 'Northern Ireland',
    'NLD': 'Netherlands', 'NOR': 'Norway', 'POL': 'Poland', 'PRT': 'Portugal',
    'ROU': 'Romania', 'SVK': 'Slovakia', 'SVN': 'Slovenia', 'SWE': 'Sweden',
    'USA': 'United States'
}

print(f"{'Country':<25} {'Total':>8} {'Attending':>10} {'% F028=1':>10} {'Paper':>8} {'Diff':>6}")
print("-" * 75)
for alpha in sorted(evs_csv['COUNTRY_ALPHA'].unique()):
    sub = evs_csv[evs_csv['COUNTRY_ALPHA'] == alpha]
    total = len(sub)
    attending = (sub['F028'] == 1).sum()
    pct_binary = round(attending/total*100) if total > 0 else None
    paper = paper_values.get(alpha, None)
    diff = pct_binary - paper if (pct_binary is not None and paper is not None) else None
    country_name = country_map.get(alpha, alpha)
    print(f"  {country_name:<23} {total:>8} {attending:>10} {pct_binary:>9}% {paper if paper else '':>8} {diff if diff is not None else '':>6}")

print()
print("=== HUNGARY SPECIAL CHECK ===")
hun = evs_csv[evs_csv['COUNTRY_ALPHA'] == 'HUN']
print("Hungary total rows:", len(hun))
print("F028 dist:", dict(hun['F028'].value_counts().sort_index()))

# Compare with EVS Stata Hungary
evs = pd.read_stata(evs_stata_path, convert_categoricals=False,
                     columns=['c_abrv', 'country1', 'q336', 'year'])
hun_evs = evs[evs['c_abrv'] == 'HU']
print("Hungary EVS Stata rows:", len(hun_evs))
print("q336 dist:", dict(hun_evs['q336'].value_counts().sort_index()))
# Paper Hungary 1990 = 34%
valid = hun_evs[hun_evs['q336'].isin([1,2,3,4,5,6,7])]
monthly = hun_evs[hun_evs['q336'].isin([1,2,3])]
print(f"Hungary q336 1-3 / 1-7: {round(len(monthly)/len(valid)*100)}%")

# Also check Switzerland in EVS CSV (not in country list above)
print()
print("=== Switzerland check ===")
# CHE not in EVS CSV apparently...
print("Countries in EVS CSV:", sorted(evs_csv['COUNTRY_ALPHA'].unique()))

# Check WVS for Switzerland wave 3
wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'S020', 'F028', 'S017'], low_memory=False)
w3 = wvs[wvs['S002VS'] == 3]
che = w3[w3['S003'] == 756]
print("Switzerland WVS wave 3 rows:", len(che))
print("F028 dist:", dict(che['F028'].value_counts().sort_index()))
valid_che = che[che['F028'].isin([1,2,3,4,5,6,7,8])]
monthly_che = che[che['F028'].isin([1,2,3])]
print(f"Switzerland unweighted: {round(len(monthly_che)/len(valid_che)*100)}%")
# Paper says 25% - our attempt 6 got 25% with weights
weighted = round(monthly_che['S017'].sum() / valid_che['S017'].sum() * 100) if valid_che['S017'].sum() > 0 else None
print(f"Switzerland weighted: {weighted}%")

# Now check the EVS CSV for specific known countries vs paper
print()
print("=== Key comparisons: EVS CSV F028 binary vs paper ===")
for alpha, paper_val in paper_values.items():
    sub = evs_csv[evs_csv['COUNTRY_ALPHA'] == alpha]
    total = len(sub)
    attending = (sub['F028'] == 1).sum()
    pct = round(attending/total*100) if total > 0 else 'N/A'
    print(f"  {alpha}: EVS_CSV={pct}%, paper={paper_val}%, diff={pct-paper_val if isinstance(pct, int) else 'N/A'}")

# Also check Norway in both sources
print()
print("=== Norway in WVS vs paper ===")
for wave_num in [1, 2, 3]:
    wave_data = wvs[wvs['S002VS'] == wave_num]
    nor = wave_data[wave_data['S003'] == 578]
    print(f"Norway WVS wave {wave_num}: {len(nor)} rows")
    if len(nor) > 0:
        valid = nor[nor['F028'].isin([1,2,3,4,5,6,7,8])]
        monthly = nor[nor['F028'].isin([1,2,3])]
        if len(valid) > 0:
            print(f"  Unweighted: {round(len(monthly)/len(valid)*100)}%")
