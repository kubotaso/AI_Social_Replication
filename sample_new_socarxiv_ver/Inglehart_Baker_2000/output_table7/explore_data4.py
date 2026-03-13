"""Final exploration: EVS Germany G006 split and optimal strategy."""
import pandas as pd
import numpy as np

evs_csv = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)

# EVS Germany G006 split
print("EVS CSV Germany G006 split for A006 (God importance):")
deu_evs = evs_csv[evs_csv['COUNTRY_ALPHA'] == 'DEU']
for g_codes, label in [([1,2], 'West(1,2)'), ([3,4], 'East(3,4)'),
                        ([1], 'West(1)'), ([2], 'West(2)'), ([3], 'East(3)'), ([4], 'East(4)')]:
    sub = deu_evs[deu_evs['G006'].isin(g_codes)]
    valid = sub[(sub['A006'] >= 1) & (sub['A006'] <= 10)]
    if len(valid) > 0:
        pct = (valid['A006'] == 10).mean() * 100
        print(f"  {label}: pct10={pct:.2f}% ({round(pct)}), n={len(valid)}")

# Now check ZA4460 Germany with country1 split for better E/W
print("\nZA4460 Germany country1 + q365:")
evs_dta = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
deu_dta = evs_dta[evs_dta['c_abrv'] == 'DE']

# Check if studynoc distinguishes E/W
print(f"  studynoc unique: {deu_dta['studynoc'].unique()}")

# country1: 900=West, 901=East most likely
# Paper says East=13, West=14
# Unweighted: East(901)=12.55%(13), West(900)=13.04%(13)
# The paper has West=14 which doesn't match either way

# Let me try the EVS CSV G006 split with paper's expectations:
# Paper: East=13, West=14
# EVS G006: West(1,2)=14.18%(14), East(3,4)=10.47%(10)
# EVS G006: West(1)=18.14%(18), West(2)=12.46%(12)
# EVS G006: East(3)=10.00%(10), East(4)=11.61%(12)
# So West(1,2)=14 matches paper perfectly for West!
# But East(3,4)=10 doesn't match paper's 13

# What about country1 from ZA4460 mapped to EVS CSV?
# Let me check if EVS CSV has the same data as ZA4460

# Check ZA4460 Germany country1 split more carefully with different weights
print("\n\nZA4460 Germany detailed:")
for c1 in sorted(deu_dta['country1'].unique()):
    sub = deu_dta[deu_dta['country1'] == c1]
    valid = sub[(sub['q365'] >= 1) & (sub['q365'] <= 10)]
    is_10 = (valid['q365'] == 10).astype(float)
    print(f"  country1={c1}: n={len(valid)}")
    print(f"    q365 distribution: {valid['q365'].value_counts().sort_index().to_dict()}")
    # Also check number choosing 10
    n_10 = (valid['q365'] == 10).sum()
    print(f"    n_10={n_10}, pct={n_10/len(valid)*100:.2f}%")

# What if the original Inglehart/Baker used a slightly different dataset version?
# Let me check if the percentages change if we use int() (floor) instead of round()
print("\n\nFLOOR vs ROUND comparison:")
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F063','S017','G006'], low_memory=False)

problem_cells = [
    # Country, wave, source, paper_val
    ('BRA', 2, 'WVS', 83),      # 82.55% -> round=83, floor=82 -- round matches
    ('GBR', None, 'EVS', 16),   # 17.28% unw, 16.00% wtd -> wtd matches
    ('DEU_EAST', None, 'EVS_DTA', 13),  # country1=901: 12.55% unw -> 13
    ('DEU_WEST', None, 'EVS_DTA', 14),  # country1=900: 13.04% unw -> 13
    ('NLD', None, 'EVS', 11),   # 11.96% -> round=12, floor=11 -- floor matches!
    ('ESP', None, 'EVS', 18),   # 17.13% unw -> 17; 18.60% wtd -> 19. Neither matches exactly
    ('USA', None, 'EVS', 48),   # 48.69% unw -> 49; 47.52% wtd -> 48 -- wtd matches
    ('JPN', 3, 'WVS', 5),       # 5.82% -> round=6, floor=5 -- floor matches!
    ('MEX', 3, 'WVS', 50),      # 49.50% -> round=49 or 50 (banker's round=50!)
    ('NGA', 3, 'WVS', 87),      # 86.82% -> round=87
    ('RUS', 3, 'WVS', 19),      # 18.42% -> round=18. Need different approach
    ('IND', 2, 'WVS', 44),      # 37.43% -> way off. Data version difference
    ('IND', 3, 'WVS', 56),      # 53.93% -> round=54
    ('ZAF', 1, 'WVS', 50),      # 52.78% -> round=53
    ('ZAF', 2, 'WVS', 74),      # 70.84% unw, 73.13% wtd -> wtd closer
    ('ZAF', 3, 'WVS', 71),      # 71.55% unw, 70.50% wtd -> wtd=71 matches!
    ('USA', 3, 'WVS', 50),      # 51.48% unw, 50.34% wtd -> wtd=50 matches!
    ('ARG', 3, 'WVS', 57),      # 57.81% unw -> 58; 56.83% wtd -> 57 matches!
    ('HUN', 1, 'WVS', 21),      # 20.23% unw -> 20; 21.36% wtd -> 21 matches!
    ('AUS', 1, 'WVS', 25),      # 25.54% unw -> 26; 25.08% wtd -> 25 matches!
    ('MEX', 1, 'WVS', 60),      # 59.47% unw -> 59; 60.32% wtd -> 60 matches!
]

# Conclusion: Need WEIGHTED for some, UNWEIGHTED for others
# The paper authors likely used S017 weights consistently
# Let me check which approach gives better overall accuracy

# Strategy: Use S017 weights when available and > 0, for ALL cells
print("\nSTRATEGY COMPARISON: Always weighted vs Always unweighted")

# Check all WVS cells with weights
paper_values = {
    ('ARG', 1): 32, ('AUS', 1): 25, ('FIN', 1): 14,
    ('HUN', 1): 21, ('JPN', 1): 6, ('MEX', 1): 60, ('ZAF', 1): 50,
    ('ARG', 2): 49, ('BRA', 2): 83, ('CHL', 2): 61,
    ('IND', 2): 44, ('MEX', 2): 44, ('NGA', 2): 87,
    ('ZAF', 2): 74, ('TUR', 2): 71, ('BLR', 2): 8, ('RUS', 2): 10,
    ('ARG', 3): 57, ('BRA', 3): 87, ('CHL', 3): 58,
    ('IND', 3): 56, ('MEX', 3): 50, ('NGA', 3): 87,
    ('ZAF', 3): 71, ('TUR', 3): 81, ('BLR', 3): 20,
    ('BGR', 3): 10, ('RUS', 3): 19, ('LVA', 3): 17,
    ('SVN', 3): 15, ('JPN', 3): 5, ('NOR', 3): 12,
    ('SWE', 3): 8, ('ESP', 3): 26, ('USA', 3): 50,
}

exact_uw = 0
exact_wt = 0
total = 0

for (country, wave), paper_val in sorted(paper_values.items()):
    sub = wvs[(wvs['COUNTRY_ALPHA'] == country) & (wvs['S002VS'] == wave)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
    if len(valid) == 0:
        continue
    total += 1
    is_10 = (valid['F063'] == 10).astype(float)
    w = valid['S017']
    pct_uw = is_10.mean() * 100
    pct_w = (is_10 * w).sum() / w.sum() * 100 if w.sum() > 0 else pct_uw

    r_uw = round(pct_uw)
    r_w = round(pct_w)

    if r_uw == paper_val:
        exact_uw += 1
    if r_w == paper_val:
        exact_wt += 1

print(f"  Unweighted exact: {exact_uw}/{total}")
print(f"  Weighted exact: {exact_wt}/{total}")

# Now check EVS with weights
evs_paper = {
    'BEL': 13, 'CAN': 28, 'FIN': 12, 'FRA': 10, 'GBR': 16,
    'ISL': 17, 'IRL': 40, 'NIR': 41, 'ITA': 29, 'NLD': 11,
    'NOR': 15, 'ESP': 18, 'SWE': 8, 'USA': 48, 'HUN': 22,
    'BGR': 7, 'SVN': 14
}

za_to_alpha = {
    'US': 'USA', 'GB-GBN': 'GBR', 'GB-NIR': 'NIR', 'IE': 'IRL',
    'BE': 'BEL', 'FR': 'FRA', 'SE': 'SWE', 'NL': 'NLD', 'NO': 'NOR',
    'FI': 'FIN', 'IS': 'ISL', 'ES': 'ESP', 'IT': 'ITA',
    'CA': 'CAN', 'HU': 'HUN', 'BG': 'BGR', 'SI': 'SVN'
}

evs_uw = 0
evs_wt = 0
evs_total = 0

for za_code, alpha in sorted(za_to_alpha.items(), key=lambda x: x[1]):
    paper_val = evs_paper.get(alpha)
    if paper_val is None:
        continue
    sub = evs_dta[evs_dta['c_abrv'] == za_code]
    valid = sub[(sub['q365'] >= 1) & (sub['q365'] <= 10)]
    if len(valid) == 0:
        continue
    evs_total += 1
    is_10 = (valid['q365'] == 10).astype(float)
    pct_uw = is_10.mean() * 100
    pct_w = (is_10 * valid['weight_s']).sum() / valid['weight_s'].sum() * 100

    if round(pct_uw) == paper_val:
        evs_uw += 1
    if round(pct_w) == paper_val:
        evs_wt += 1

print(f"\nEVS Unweighted exact: {evs_uw}/{evs_total}")
print(f"EVS Weighted exact: {evs_wt}/{evs_total}")

# Detailed check: which EVS countries benefit from weights?
print("\nEVS cells that change with weights:")
for za_code, alpha in sorted(za_to_alpha.items(), key=lambda x: x[1]):
    paper_val = evs_paper.get(alpha)
    if paper_val is None:
        continue
    sub = evs_dta[evs_dta['c_abrv'] == za_code]
    valid = sub[(sub['q365'] >= 1) & (sub['q365'] <= 10)]
    if len(valid) == 0:
        continue
    is_10 = (valid['q365'] == 10).astype(float)
    pct_uw = is_10.mean() * 100
    pct_w = (is_10 * valid['weight_s']).sum() / valid['weight_s'].sum() * 100
    r_uw = round(pct_uw)
    r_w = round(pct_w)
    if r_uw != r_w:
        print(f"  {alpha}: paper={paper_val}, unw={r_uw}, wtd={r_w}")
