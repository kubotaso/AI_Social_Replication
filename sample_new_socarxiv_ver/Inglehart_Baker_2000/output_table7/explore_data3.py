"""Deep dive into ZA4460 weights and Germany split."""
import pandas as pd
import numpy as np

evs_dta = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)

# Check Germany country1 split (900 vs 901)
print("="*60)
print("GERMANY SPLIT BY country1")
print("="*60)
deu = evs_dta[evs_dta['c_abrv'] == 'DE']
for c1 in sorted(deu['country1'].unique()):
    sub = deu[deu['country1'] == c1]
    valid = sub[(sub['q365'] >= 1) & (sub['q365'] <= 10)]
    if len(valid) > 0:
        pct = (valid['q365'] == 10).mean() * 100
        pct_w = ((valid['q365'] == 10).astype(float) * valid['weight_s']).sum() / valid['weight_s'].sum() * 100
        print(f"  country1={c1}: n={len(valid)}, pct10_uw={pct:.2f}%, pct10_w={pct_w:.2f}%")
    else:
        print(f"  country1={c1}: no valid q365 data")

# Check paper values for E/W Germany 1990: East=13, West=14
# country1=900 is likely West Germany, 901 is likely East Germany

# Now let's check ALL countries with weight_s vs unweighted
print("\n\n" + "="*60)
print("ALL COUNTRIES: weight_s vs unweighted for q365")
print("="*60)

paper_1990 = {
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

print(f"{'Country':<8} {'Paper':>6} {'Unw%':>8} {'Unw_R':>6} {'Wtd%':>8} {'Wtd_R':>6} {'Best':>6}")
print("-" * 55)

for za_code, alpha in sorted(za_to_alpha.items(), key=lambda x: x[1]):
    paper_val = paper_1990.get(alpha, None)
    if paper_val is None:
        continue

    sub = evs_dta[evs_dta['c_abrv'] == za_code]
    valid = sub[(sub['q365'] >= 1) & (sub['q365'] <= 10)].copy()
    if len(valid) == 0:
        print(f"{alpha:<8} {paper_val:>6} {'N/A':>8} {'N/A':>6} {'N/A':>8} {'N/A':>6}")
        continue

    is_10 = (valid['q365'] == 10).astype(float)
    pct_uw = is_10.mean() * 100
    pct_w = (is_10 * valid['weight_s']).sum() / valid['weight_s'].sum() * 100

    r_uw = round(pct_uw)
    r_w = round(pct_w)

    # Which is closer?
    d_uw = abs(r_uw - paper_val)
    d_w = abs(r_w - paper_val)
    best = "Wtd" if d_w < d_uw else ("Unw" if d_uw < d_w else "Same")

    print(f"{alpha:<8} {paper_val:>6} {pct_uw:>7.2f}% {r_uw:>6} {pct_w:>7.2f}% {r_w:>6} {best:>6}")

# Check WVS S017 weights for key countries
print("\n\n" + "="*60)
print("WVS S017 WEIGHTS FOR KEY COUNTRIES")
print("="*60)
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F063','S017','G006'], low_memory=False)

# For countries where we use WVS data (not EVS)
wvs_checks = [
    ('ARG', 2, 49), ('BRA', 2, 83), ('CHL', 2, 61),
    ('IND', 2, 44), ('MEX', 2, 44), ('NGA', 2, 87),
    ('ZAF', 2, 74), ('TUR', 2, 71),
    ('BLR', 2, 8), ('RUS', 2, 10),
    # Wave 3
    ('ARG', 3, 57), ('BRA', 3, 87), ('CHL', 3, 58),
    ('IND', 3, 56), ('MEX', 3, 50), ('NGA', 3, 87),
    ('ZAF', 3, 71), ('TUR', 3, 81),
    ('BLR', 3, 20), ('BGR', 3, 10), ('RUS', 3, 19),
    ('LVA', 3, 17), ('SVN', 3, 15),
    ('JPN', 3, 5), ('NOR', 3, 12), ('SWE', 3, 8),
    ('ESP', 3, 26), ('USA', 3, 50),
    # Wave 1
    ('ARG', 1, 32), ('AUS', 1, 25), ('FIN', 1, 14),
    ('HUN', 1, 21), ('JPN', 1, 6), ('MEX', 1, 60),
    ('ZAF', 1, 50),
]

print(f"{'Country':<6} {'Wave':>4} {'Paper':>6} {'Unw%':>8} {'Unw_R':>6} {'Wtd%':>8} {'Wtd_R':>6}")
print("-" * 50)

for country, wave, paper_val in wvs_checks:
    sub = wvs[(wvs['COUNTRY_ALPHA'] == country) & (wvs['S002VS'] == wave)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
    if len(valid) == 0:
        print(f"{country:<6} {wave:>4} {paper_val:>6} {'N/A':>8} {'N/A':>6} {'N/A':>8} {'N/A':>6}")
        continue

    is_10 = (valid['F063'] == 10).astype(float)
    w = valid['S017']
    pct_uw = is_10.mean() * 100

    if w.notna().all() and w.sum() > 0 and w.std() > 0.01:
        pct_w = (is_10 * w).sum() / w.sum() * 100
    else:
        pct_w = pct_uw

    r_uw = round(pct_uw)
    r_w = round(pct_w)

    d_uw = abs(r_uw - paper_val)
    d_w = abs(r_w - paper_val)

    print(f"{country:<6} {wave:>4} {paper_val:>6} {pct_uw:>7.2f}% {r_uw:>6} {pct_w:>7.2f}% {r_w:>6}")

# Check East Germany EVS 1990 with country1=901
print("\n\nEast Germany EVS 1990 (country1=901) with weights:")
deu_east = evs_dta[(evs_dta['c_abrv'] == 'DE') & (evs_dta['country1'] == 901)]
valid = deu_east[(deu_east['q365'] >= 1) & (deu_east['q365'] <= 10)]
is_10 = (valid['q365'] == 10).astype(float)
pct_uw = is_10.mean() * 100
pct_w = (is_10 * valid['weight_s']).sum() / valid['weight_s'].sum() * 100
print(f"  Unweighted: {pct_uw:.2f}% ({round(pct_uw)}), Weighted: {pct_w:.2f}% ({round(pct_w)})")
print(f"  Paper: 13")

# West Germany EVS 1990
print("\nWest Germany EVS 1990 (country1=900) with weights:")
deu_west = evs_dta[(evs_dta['c_abrv'] == 'DE') & (evs_dta['country1'] == 900)]
valid = deu_west[(deu_west['q365'] >= 1) & (deu_west['q365'] <= 10)]
is_10 = (valid['q365'] == 10).astype(float)
pct_uw = is_10.mean() * 100
pct_w = (is_10 * valid['weight_s']).sum() / valid['weight_s'].sum() * 100
print(f"  Unweighted: {pct_uw:.2f}% ({round(pct_uw)}), Weighted: {pct_w:.2f}% ({round(pct_w)})")
print(f"  Paper: 14")

# WVS East Germany wave 3 with different G006 combos and weights
print("\n\nWVS East Germany wave 3 with S017 weights:")
deu_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'DEU') & (wvs['S002VS'] == 3)]
for label, codes in [('East(2,3)', [2,3]), ('East(3)', [3]), ('East(2)', [2]),
                      ('East(3,4)', [3,4])]:
    sub = deu_w3[deu_w3['G006'].isin(codes)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
    if len(valid) == 0:
        continue
    is_10 = (valid['F063'] == 10).astype(float)
    w = valid['S017']
    pct_uw = is_10.mean() * 100
    pct_w = (is_10 * w).sum() / w.sum() * 100 if w.sum() > 0 else pct_uw
    print(f"  {label}: unw={pct_uw:.2f}% ({round(pct_uw)}), wtd={pct_w:.2f}% ({round(pct_w)}), paper=6")

print("\nWVS West Germany wave 3 with S017 weights:")
for label, codes in [('West(1,4)', [1,4]), ('West(1)', [1]), ('West(4)', [4]),
                      ('West(1,2)', [1,2])]:
    sub = deu_w3[deu_w3['G006'].isin(codes)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
    if len(valid) == 0:
        continue
    is_10 = (valid['F063'] == 10).astype(float)
    w = valid['S017']
    pct_uw = is_10.mean() * 100
    pct_w = (is_10 * w).sum() / w.sum() * 100 if w.sum() > 0 else pct_uw
    print(f"  {label}: unw={pct_uw:.2f}% ({round(pct_uw)}), wtd={pct_w:.2f}% ({round(pct_w)}), paper=16")
