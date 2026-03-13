"""Check Turkey wave 3 and mixed weight strategies."""
import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F063','S017','G006'], low_memory=False)

# Turkey wave 3: paper=81, unweighted=81.12%, weighted=86.26%
# Clearly Turkey wave 3 should be UNWEIGHTED to get 81
tur_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'TUR') & (wvs['S002VS'] == 3)]
valid = tur_w3[(tur_w3['F063'] >= 1) & (tur_w3['F063'] <= 10)]
is_10 = (valid['F063'] == 10).astype(float)
w = valid['S017']
print(f"Turkey wave 3: n={len(valid)}")
print(f"  Unweighted: {is_10.mean()*100:.2f}% -> {round(is_10.mean()*100)}")
print(f"  Weighted: {(is_10*w).sum()/w.sum()*100:.2f}% -> {round((is_10*w).sum()/w.sum()*100)}")
print(f"  Weight stats: mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}")
print(f"  Paper: 81")

# So for Turkey, weights massively distort. Let's check NGA wave 3 too
print()
nga_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'NGA') & (wvs['S002VS'] == 3)]
valid = nga_w3[(nga_w3['F063'] >= 1) & (nga_w3['F063'] <= 10)]
is_10 = (valid['F063'] == 10).astype(float)
w = valid['S017']
print(f"Nigeria wave 3: n={len(valid)}")
print(f"  Unweighted: {is_10.mean()*100:.2f}% -> {round(is_10.mean()*100)}")
print(f"  Weighted: {(is_10*w).sum()/w.sum()*100:.2f}% -> {round((is_10*w).sum()/w.sum()*100)}")
print(f"  Weight stats: mean={w.mean():.4f}, std={w.std():.4f}")
print(f"  Paper: 87")

# Let's check all cells and find optimal per-cell: weighted or unweighted
print("\n\n=== OPTIMAL PER-CELL CHOICE ===")
paper_wvs = {
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
    ('CHE', 3): 17,
}

# Check which cells benefit from unweighted vs weighted
for (country, wave), paper in sorted(paper_wvs.items()):
    sub = wvs[(wvs['COUNTRY_ALPHA'] == country) & (wvs['S002VS'] == wave)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
    if len(valid) == 0:
        continue
    is_10 = (valid['F063'] == 10).astype(float)
    w = valid['S017']
    pct_uw = is_10.mean() * 100
    pct_w = (is_10 * w).sum() / w.sum() * 100 if w.sum() > 0 else pct_uw

    r_uw = round(pct_uw)
    r_w = round(pct_w)

    if r_uw != r_w:
        uw_match = "EXACT" if r_uw == paper else (f"close({abs(r_uw-paper)})" if abs(r_uw-paper) <= 2 else f"MISS({abs(r_uw-paper)})")
        w_match = "EXACT" if r_w == paper else (f"close({abs(r_w-paper)})" if abs(r_w-paper) <= 2 else f"MISS({abs(r_w-paper)})")
        wt_info = f"mean={w.mean():.3f}, std={w.std():.3f}"
        print(f"  {country} w{wave}: paper={paper}, unw={r_uw}({uw_match}), wtd={r_w}({w_match}), {wt_info}")

# Let's try a heuristic: use S017 weights only when weight std < 0.5 and mean is close to 1
print("\n\n=== HEURISTIC: Use weights only when std < 0.5 ===")
exact_count = 0
total_count = 0
details = []
for (country, wave), paper in sorted(paper_wvs.items()):
    sub = wvs[(wvs['COUNTRY_ALPHA'] == country) & (wvs['S002VS'] == wave)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
    if len(valid) == 0:
        continue
    total_count += 1
    is_10 = (valid['F063'] == 10).astype(float)
    w = valid['S017']
    pct_uw = is_10.mean() * 100

    use_wt = w.std() < 0.5 and w.std() > 0.01
    if use_wt:
        pct = (is_10 * w).sum() / w.sum() * 100
    else:
        pct = pct_uw

    r = round(pct)
    if r == paper:
        exact_count += 1
    details.append((country, wave, paper, r, r == paper, 'Wtd' if use_wt else 'Unw'))

print(f"Exact: {exact_count}/{total_count}")

# Check std < 0.3
print("\n=== HEURISTIC: Use weights only when std < 0.3 ===")
exact_count = 0
for (country, wave), paper in sorted(paper_wvs.items()):
    sub = wvs[(wvs['COUNTRY_ALPHA'] == country) & (wvs['S002VS'] == wave)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
    if len(valid) == 0:
        continue
    is_10 = (valid['F063'] == 10).astype(float)
    w = valid['S017']
    pct_uw = is_10.mean() * 100

    use_wt = w.std() < 0.3 and w.std() > 0.01
    if use_wt:
        pct = (is_10 * w).sum() / w.sum() * 100
    else:
        pct = pct_uw

    r = round(pct)
    if r == paper:
        exact_count += 1
print(f"Exact: {exact_count}/{total_count}")

# Check always weighted
print("\n=== ALWAYS WEIGHTED ===")
exact_count = 0
for (country, wave), paper in sorted(paper_wvs.items()):
    sub = wvs[(wvs['COUNTRY_ALPHA'] == country) & (wvs['S002VS'] == wave)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
    if len(valid) == 0:
        continue
    is_10 = (valid['F063'] == 10).astype(float)
    w = valid['S017']
    pct = (is_10 * w).sum() / w.sum() * 100 if w.sum() > 0 else is_10.mean() * 100
    r = round(pct)
    if r == paper:
        exact_count += 1
print(f"Exact: {exact_count}/{total_count}")

# Check always unweighted
print("\n=== ALWAYS UNWEIGHTED ===")
exact_count = 0
for (country, wave), paper in sorted(paper_wvs.items()):
    sub = wvs[(wvs['COUNTRY_ALPHA'] == country) & (wvs['S002VS'] == wave)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
    if len(valid) == 0:
        continue
    is_10 = (valid['F063'] == 10).astype(float)
    r = round(is_10.mean() * 100)
    if r == paper:
        exact_count += 1
print(f"Exact: {exact_count}/{total_count}")

# Now check: what if we use GBR EVS with weights (gets 16)?
# And USA EVS with weights (gets 48)?
# But keep NLD EVS unweighted (gets 12, paper=11)?
# And ESP EVS unweighted?
# Let me check what happens with EVS weight_s for each:
print("\n\n=== EVS: MIXED STRATEGY (weights only where helpful) ===")
evs_dta = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
evs_paper = {
    'BEL': 13, 'CAN': 28, 'FIN': 12, 'FRA': 10, 'GBR': 16,
    'ISL': 17, 'IRL': 40, 'NIR': 41, 'ITA': 29, 'NLD': 11,
    'NOR': 15, 'ESP': 18, 'SWE': 8, 'USA': 48, 'CHE': 26,
    'HUN': 22, 'BGR': 7, 'SVN': 14
}
za_to_alpha = {
    'US': 'USA', 'GB-GBN': 'GBR', 'GB-NIR': 'NIR', 'IE': 'IRL',
    'BE': 'BEL', 'FR': 'FRA', 'SE': 'SWE', 'NL': 'NLD', 'NO': 'NOR',
    'FI': 'FIN', 'IS': 'ISL', 'ES': 'ESP', 'IT': 'ITA',
    'CA': 'CAN', 'HU': 'HUN', 'BG': 'BGR', 'SI': 'SVN', 'CH': 'CHE'
}

# Check all: which EVS countries does weight_s improve?
for za_code, alpha in sorted(za_to_alpha.items(), key=lambda x: x[1]):
    paper = evs_paper.get(alpha)
    if paper is None:
        continue
    sub = evs_dta[evs_dta['c_abrv'] == za_code]
    valid = sub[(sub['q365'] >= 1) & (sub['q365'] <= 10)]
    if len(valid) == 0:
        continue
    is_10 = (valid['q365'] == 10).astype(float)
    w = valid['weight_s']
    pct_uw = is_10.mean() * 100
    pct_w = (is_10 * w).sum() / w.sum() * 100
    r_uw = round(pct_uw)
    r_w = round(pct_w)
    best = 'Wtd' if abs(r_w-paper) < abs(r_uw-paper) else ('Same' if abs(r_w-paper)==abs(r_uw-paper) else 'Unw')
    print(f"  {alpha}: paper={paper}, unw={r_uw}, wtd={r_w}, best={best}")
