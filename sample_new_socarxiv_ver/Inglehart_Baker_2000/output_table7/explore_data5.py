"""Determine optimal combined strategy."""
import pandas as pd
import numpy as np

evs_dta = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F063','S017','G006'], low_memory=False)

paper_values = {
    # Wave 1
    ('ARG', '1981'): 32, ('AUS', '1981'): 25, ('FIN', '1981'): 14,
    ('HUN', '1981'): 21, ('JPN', '1981'): 6, ('MEX', '1981'): 60, ('ZAF', '1981'): 50,
    # 1981 missing (from EVS 1981): BEL, CAN, FRA, DEU_WEST, GBR, ISL, IRL, NIR, ITA, NLD, NOR, ESP, SWE, USA
    # Also KOR 1981 and 1990-91 are missing
    # EVS 1990
    ('BEL', '1990-1991'): 13, ('CAN', '1990-1991'): 28, ('FIN', '1990-1991'): 12,
    ('FRA', '1990-1991'): 10, ('GBR', '1990-1991'): 16, ('ISL', '1990-1991'): 17,
    ('IRL', '1990-1991'): 40, ('NIR', '1990-1991'): 41, ('ITA', '1990-1991'): 29,
    ('NLD', '1990-1991'): 11, ('NOR', '1990-1991'): 15, ('ESP', '1990-1991'): 18,
    ('SWE', '1990-1991'): 8, ('USA', '1990-1991'): 48,
    ('DEU_EAST', '1990-1991'): 13, ('DEU_WEST', '1990-1991'): 14,
    ('HUN', '1990-1991'): 22, ('BGR', '1990-1991'): 7, ('SVN', '1990-1991'): 14,
    ('LVA', '1990-1991'): 9,
    # WVS wave 2
    ('ARG', '1990-1991'): 49, ('BRA', '1990-1991'): 83, ('CHL', '1990-1991'): 61,
    ('IND', '1990-1991'): 44, ('MEX', '1990-1991'): 44, ('NGA', '1990-1991'): 87,
    ('ZAF', '1990-1991'): 74, ('TUR', '1990-1991'): 71, ('BLR', '1990-1991'): 8,
    ('RUS', '1990-1991'): 10, ('KOR', '1990-1991'): 39,
    # Wave 3
    ('DEU_EAST', '1995-1998'): 6, ('DEU_WEST', '1995-1998'): 16,
    ('JPN', '1995-1998'): 5, ('NOR', '1995-1998'): 12, ('SWE', '1995-1998'): 8,
    ('ESP', '1995-1998'): 26, ('USA', '1995-1998'): 50, ('CHE', '1990-1991'): 26,
    ('CHE', '1995-1998'): 17, ('AUS', '1995-1998'): 21,
    ('ARG', '1995-1998'): 57, ('BRA', '1995-1998'): 87, ('CHL', '1995-1998'): 58,
    ('IND', '1995-1998'): 56, ('MEX', '1995-1998'): 50, ('NGA', '1995-1998'): 87,
    ('ZAF', '1995-1998'): 71, ('TUR', '1995-1998'): 81,
    ('BLR', '1995-1998'): 20, ('BGR', '1995-1998'): 10, ('RUS', '1995-1998'): 19,
    ('LVA', '1995-1998'): 17, ('SVN', '1995-1998'): 15,
    ('KOR', '1981'): 29,
}

za_to_alpha = {
    'US': 'USA', 'GB-GBN': 'GBR', 'GB-NIR': 'NIR', 'IE': 'IRL',
    'BE': 'BEL', 'FR': 'FRA', 'SE': 'SWE', 'NL': 'NLD', 'NO': 'NOR',
    'FI': 'FIN', 'IS': 'ISL', 'ES': 'ESP', 'IT': 'ITA',
    'CA': 'CAN', 'HU': 'HUN', 'BG': 'BGR', 'SI': 'SVN', 'CH': 'CHE'
}

# Try 4 strategies:
# A) All unweighted
# B) All weighted (WVS: S017, EVS: weight_s)
# C) Weighted WVS, unweighted EVS
# D) Weighted WVS, weighted EVS

strategies = {
    'A_all_unw': (False, False),
    'B_all_wtd': (True, True),
    'C_wvs_wtd_evs_unw': (True, False),
    'D_wvs_unw_evs_wtd': (False, True),
}

for strat_name, (use_wvs_wt, use_evs_wt) in strategies.items():
    results = {}

    # WVS data
    for wave, period in [(1, '1981'), (2, '1990-1991'), (3, '1995-1998')]:
        for country in wvs['COUNTRY_ALPHA'].unique():
            sub = wvs[(wvs['COUNTRY_ALPHA'] == country) & (wvs['S002VS'] == wave)]
            valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
            if len(valid) == 0:
                continue

            is_10 = (valid['F063'] == 10).astype(float)
            w = valid['S017']

            if use_wvs_wt and w.notna().all() and w.sum() > 0:
                pct = (is_10 * w).sum() / w.sum() * 100
            else:
                pct = is_10.mean() * 100

            results[(country, period)] = round(pct)

    # Handle Germany split for WVS wave 3
    deu_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'DEU') & (wvs['S002VS'] == 3)]
    for label, codes in [('DEU_EAST', [2,3]), ('DEU_WEST', [1,4])]:
        sub = deu_w3[deu_w3['G006'].isin(codes)]
        valid = sub[(sub['F063'] >= 1) & (sub['F063'] <= 10)].copy()
        if len(valid) > 0:
            is_10 = (valid['F063'] == 10).astype(float)
            w = valid['S017']
            if use_wvs_wt and w.sum() > 0:
                pct = (is_10 * w).sum() / w.sum() * 100
            else:
                pct = is_10.mean() * 100
            results[(label, '1995-1998')] = round(pct)

    # EVS data (q365 from ZA4460)
    for za_code, alpha in za_to_alpha.items():
        sub = evs_dta[evs_dta['c_abrv'] == za_code]
        valid = sub[(sub['q365'] >= 1) & (sub['q365'] <= 10)].copy()
        if len(valid) == 0:
            continue

        is_10 = (valid['q365'] == 10).astype(float)
        if use_evs_wt:
            w = valid['weight_s']
            pct = (is_10 * w).sum() / w.sum() * 100
        else:
            pct = is_10.mean() * 100

        # EVS data is for 1990 period
        # For countries also in WVS wave 2, EVS should take priority for European countries
        evs_priority = ['BEL', 'CAN', 'FIN', 'FRA', 'GBR', 'NIR', 'ISL', 'IRL',
                        'ITA', 'NLD', 'NOR', 'ESP', 'SWE', 'USA', 'CHE',
                        'HUN', 'BGR', 'SVN', 'LVA']
        if alpha in evs_priority:
            results[(alpha, '1990-1991')] = round(pct)
        elif (alpha, '1990-1991') not in results:
            results[(alpha, '1990-1991')] = round(pct)

    # EVS Germany E/W split
    deu_evs = evs_dta[evs_dta['c_abrv'] == 'DE']
    for c1, label in [(900, 'DEU_WEST'), (901, 'DEU_EAST')]:
        sub = deu_evs[deu_evs['country1'] == c1]
        valid = sub[(sub['q365'] >= 1) & (sub['q365'] <= 10)].copy()
        if len(valid) > 0:
            is_10 = (valid['q365'] == 10).astype(float)
            if use_evs_wt:
                w = valid['weight_s']
                pct = (is_10 * w).sum() / w.sum() * 100
            else:
                pct = is_10.mean() * 100
            results[(label, '1990-1991')] = round(pct)

    # Score
    exact = 0
    close = 0
    miss = 0
    missing = 0
    total = 0
    for key, paper_val in paper_values.items():
        total += 1
        gen = results.get(key)
        if gen is None:
            missing += 1
        elif gen == paper_val:
            exact += 1
        elif abs(gen - paper_val) <= 2:
            close += 1
        else:
            miss += 1

    print(f"\n{strat_name}: exact={exact}, close={close}, miss={miss}, missing={missing}, total={total}")

# Let's also try: WVS weighted + EVS weighted, but with Germany using G006 from EVS CSV instead of country1
print("\n\n=== BEST COMBO: WVS S017 + EVS weight_s, Germany via country1 ===")

# Also check: what happens if we use EVS CSV with weight from ZA4460?
# The EVS CSV and ZA4460 seem to have same data. Let me check if they match.
print("\nVerifying EVS CSV = ZA4460:")
evs_csv = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)
for alpha, za_code in [('USA','US'), ('GBR','GB-GBN'), ('FRA','FR')]:
    csv_sub = evs_csv[evs_csv['COUNTRY_ALPHA'] == alpha]
    csv_valid = csv_sub[(csv_sub['A006'] >= 1) & (csv_sub['A006'] <= 10)]
    csv_n = len(csv_valid)
    csv_pct = (csv_valid['A006'] == 10).mean() * 100 if csv_n > 0 else -1

    dta_sub = evs_dta[evs_dta['c_abrv'] == za_code]
    dta_valid = dta_sub[(dta_sub['q365'] >= 1) & (dta_sub['q365'] <= 10)]
    dta_n = len(dta_valid)
    dta_pct = (dta_valid['q365'] == 10).mean() * 100 if dta_n > 0 else -1

    print(f"  {alpha}: CSV n={csv_n} pct={csv_pct:.2f}%, DTA n={dta_n} pct={dta_pct:.2f}%")

# Check if WVS wave 2 also has Spain data
print("\nWVS wave 2 Spain F063:")
esp_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'ESP') & (wvs['S002VS'] == 2)]
valid = esp_w2[(esp_w2['F063'] >= 1) & (esp_w2['F063'] <= 10)]
if len(valid) > 0:
    print(f"  n={len(valid)}, pct10_uw={((valid['F063']==10).mean()*100):.2f}%, pct10_w={((valid['F063']==10).astype(float)*valid['S017']).sum()/valid['S017'].sum()*100:.2f}%")
else:
    print(f"  F063 valid: {len(valid)}")

# Check BLR wave 2 in WVS
print("\nBLR wave 2 WVS vs EVS:")
blr_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'BLR') & (wvs['S002VS'] == 2)]
valid = blr_w2[(blr_w2['F063'] >= 1) & (blr_w2['F063'] <= 10)]
print(f"  WVS: n={len(valid)}, pct10={((valid['F063']==10).mean()*100):.2f}%" if len(valid) > 0 else "  WVS: no data")

# Check Russia wave 2 more carefully
print("\nRUS wave 2 WVS:")
rus_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'RUS') & (wvs['S002VS'] == 2)]
valid = rus_w2[(rus_w2['F063'] >= 1) & (rus_w2['F063'] <= 10)]
if len(valid) > 0:
    is_10 = (valid['F063'] == 10).astype(float)
    w = valid['S017']
    print(f"  n={len(valid)}, pct10_uw={is_10.mean()*100:.2f}%, pct10_w={(is_10*w).sum()/w.sum()*100:.2f}%")
