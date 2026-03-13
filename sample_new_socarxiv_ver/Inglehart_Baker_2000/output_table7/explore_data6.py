"""Show detailed results for Strategy C (WVS weighted + EVS unweighted)."""
import pandas as pd
import numpy as np

evs_dta = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F063','S017','G006'], low_memory=False)

# ALL paper values (77 cells)
paper_values = {
    ('AUS', '1981'): 25, ('AUS', '1995-1998'): 21,
    ('BEL', '1981'): 9, ('BEL', '1990-1991'): 13,
    ('CAN', '1981'): 36, ('CAN', '1990-1991'): 28,
    ('FIN', '1981'): 14, ('FIN', '1990-1991'): 12,
    ('FRA', '1981'): 10, ('FRA', '1990-1991'): 10,
    ('DEU_EAST', '1990-1991'): 13, ('DEU_EAST', '1995-1998'): 6,
    ('DEU_WEST', '1981'): 16, ('DEU_WEST', '1990-1991'): 14, ('DEU_WEST', '1995-1998'): 16,
    ('GBR', '1981'): 20, ('GBR', '1990-1991'): 16,
    ('ISL', '1981'): 22, ('ISL', '1990-1991'): 17,
    ('IRL', '1981'): 29, ('IRL', '1990-1991'): 40,
    ('NIR', '1981'): 38, ('NIR', '1990-1991'): 41,
    ('KOR', '1981'): 29, ('KOR', '1990-1991'): 39,
    ('ITA', '1981'): 31, ('ITA', '1990-1991'): 29,
    ('JPN', '1981'): 6, ('JPN', '1990-1991'): 6, ('JPN', '1995-1998'): 5,
    ('NLD', '1981'): 11, ('NLD', '1990-1991'): 11,
    ('NOR', '1981'): 19, ('NOR', '1990-1991'): 15, ('NOR', '1995-1998'): 12,
    ('ESP', '1981'): 18, ('ESP', '1990-1991'): 18, ('ESP', '1995-1998'): 26,
    ('SWE', '1981'): 9, ('SWE', '1990-1991'): 8, ('SWE', '1995-1998'): 8,
    ('CHE', '1990-1991'): 26, ('CHE', '1995-1998'): 17,
    ('USA', '1981'): 50, ('USA', '1990-1991'): 48, ('USA', '1995-1998'): 50,
    ('BLR', '1990-1991'): 8, ('BLR', '1995-1998'): 20,
    ('BGR', '1990-1991'): 7, ('BGR', '1995-1998'): 10,
    ('HUN', '1981'): 21, ('HUN', '1990-1991'): 22,
    ('LVA', '1990-1991'): 9, ('LVA', '1995-1998'): 17,
    ('RUS', '1990-1991'): 10, ('RUS', '1995-1998'): 19,
    ('SVN', '1990-1991'): 14, ('SVN', '1995-1998'): 15,
    ('ARG', '1981'): 32, ('ARG', '1990-1991'): 49, ('ARG', '1995-1998'): 57,
    ('BRA', '1990-1991'): 83, ('BRA', '1995-1998'): 87,
    ('CHL', '1990-1991'): 61, ('CHL', '1995-1998'): 58,
    ('IND', '1990-1991'): 44, ('IND', '1995-1998'): 56,
    ('MEX', '1981'): 60, ('MEX', '1990-1991'): 44, ('MEX', '1995-1998'): 50,
    ('NGA', '1990-1991'): 87, ('NGA', '1995-1998'): 87,
    ('ZAF', '1981'): 50, ('ZAF', '1990-1991'): 74, ('ZAF', '1995-1998'): 71,
    ('TUR', '1990-1991'): 71, ('TUR', '1995-1998'): 81,
}

za_to_alpha = {
    'US': 'USA', 'GB-GBN': 'GBR', 'GB-NIR': 'NIR', 'IE': 'IRL',
    'BE': 'BEL', 'FR': 'FRA', 'SE': 'SWE', 'NL': 'NLD', 'NO': 'NOR',
    'FI': 'FIN', 'IS': 'ISL', 'ES': 'ESP', 'IT': 'ITA',
    'CA': 'CAN', 'HU': 'HUN', 'BG': 'BGR', 'SI': 'SVN', 'CH': 'CHE'
}

# Generate results with Strategy C
results = {}

# WVS data with S017 weights
for wave, period in [(1, '1981'), (2, '1990-1991'), (3, '1995-1998')]:
    for country in wvs['COUNTRY_ALPHA'].unique():
        if country == 'DEU':
            continue  # Handle separately
        sub = wvs[(wvs['COUNTRY_ALPHA'] == country) & (wvs['S002VS'] == wave)]
        valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
        if len(valid) == 0:
            continue
        is_10 = (valid['F063'] == 10).astype(float)
        w = valid['S017']
        if w.notna().all() and w.sum() > 0:
            pct = (is_10 * w).sum() / w.sum() * 100
        else:
            pct = is_10.mean() * 100
        results[(country, period)] = (pct, round(pct), len(valid))

# WVS Germany wave 3 split
deu_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'DEU') & (wvs['S002VS'] == 3)]
for label, codes in [('DEU_EAST', [2,3]), ('DEU_WEST', [1,4])]:
    sub = deu_w3[deu_w3['G006'].isin(codes)]
    valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
    if len(valid) > 0:
        is_10 = (valid['F063'] == 10).astype(float)
        w = valid['S017']
        pct = (is_10 * w).sum() / w.sum() * 100 if w.sum() > 0 else is_10.mean() * 100
        results[(label, '1995-1998')] = (pct, round(pct), len(valid))

# EVS data unweighted (from ZA4460 q365)
evs_priority = ['BEL', 'CAN', 'FIN', 'FRA', 'GBR', 'NIR', 'ISL', 'IRL',
                'ITA', 'NLD', 'NOR', 'ESP', 'SWE', 'USA', 'CHE',
                'HUN', 'BGR', 'SVN', 'LVA']

for za_code, alpha in za_to_alpha.items():
    sub = evs_dta[evs_dta['c_abrv'] == za_code]
    valid = sub[(sub['q365'] >= 1) & (sub['q365'] <= 10)].copy()
    if len(valid) == 0:
        continue
    is_10 = (valid['q365'] == 10).astype(float)
    pct = is_10.mean() * 100

    if alpha in evs_priority:
        results[(alpha, '1990-1991')] = (pct, round(pct), len(valid))
    elif (alpha, '1990-1991') not in results:
        results[(alpha, '1990-1991')] = (pct, round(pct), len(valid))

# EVS Germany E/W via country1
deu_evs = evs_dta[evs_dta['c_abrv'] == 'DE']
for c1, label in [(900, 'DEU_WEST'), (901, 'DEU_EAST')]:
    sub = deu_evs[deu_evs['country1'] == c1]
    valid = sub[(sub['q365'] >= 1) & (sub['q365'] <= 10)].copy()
    if len(valid) > 0:
        is_10 = (valid['q365'] == 10).astype(float)
        pct = is_10.mean() * 100
        results[(label, '1990-1991')] = (pct, round(pct), len(valid))

# Print detailed comparison
print(f"{'Country':<20} {'Period':<12} {'Paper':>6} {'Raw%':>8} {'Round':>6} {'Status':<10}")
print("-" * 65)

exact = close = miss = missing = 0
close_details = []
miss_details = []

for key in sorted(paper_values.keys()):
    country, period = key
    paper_val = paper_values[key]
    gen_data = results.get(key)

    if gen_data is None:
        status = "MISSING"
        missing += 1
        print(f"{country:<20} {period:<12} {paper_val:>6} {'---':>8} {'---':>6} {status:<10}")
    else:
        raw_pct, rounded, n = gen_data
        diff = abs(rounded - paper_val)
        if diff == 0:
            status = "EXACT"
            exact += 1
        elif diff <= 2:
            status = f"CLOSE({diff})"
            close += 1
            close_details.append((country, period, paper_val, raw_pct, rounded))
        else:
            status = f"MISS({diff})"
            miss += 1
            miss_details.append((country, period, paper_val, raw_pct, rounded))
        print(f"{country:<20} {period:<12} {paper_val:>6} {raw_pct:>7.2f}% {rounded:>6} {status:<10}")

total = exact + close + miss + missing
print(f"\nTotal: {total} cells")
print(f"Exact: {exact} ({exact/total*100:.0f}%)")
print(f"Close: {close} ({close/total*100:.0f}%)")
print(f"Miss: {miss} ({miss/total*100:.0f}%)")
print(f"Missing: {missing} ({missing/total*100:.0f}%)")

print(f"\n=== CLOSE MATCHES (potential improvements) ===")
for country, period, paper, raw, rounded in close_details:
    print(f"  {country} {period}: paper={paper}, raw={raw:.2f}%, round={rounded}")

print(f"\n=== MISSES (hard to fix) ===")
for country, period, paper, raw, rounded in miss_details:
    print(f"  {country} {period}: paper={paper}, raw={raw:.2f}%, round={rounded}")

# Check if rounding to nearest integer differently could help some close matches
print(f"\n=== ROUNDING EXPERIMENTS FOR CLOSE MATCHES ===")
for country, period, paper, raw, rounded in close_details:
    floor_val = int(raw)
    ceil_val = int(raw) + 1
    # Python's round() uses banker's rounding for .5
    # What if the authors used simple rounding (always round .5 up)?
    simple_round = int(raw + 0.5)
    trunc = int(raw)
    print(f"  {country} {period}: paper={paper}, raw={raw:.4f}%, round={rounded}, floor={floor_val}, ceil={ceil_val}, simple_round={simple_round}")
