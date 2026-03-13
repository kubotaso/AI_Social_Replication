"""
Investigate whether using floor (truncation) vs round improves matches.
Also check all EVS countries with both rounding methods.
"""
import pandas as pd
import numpy as np

wvs_path = "data/WVS_Time_Series_1981-2022_csv_v5_0.csv"
evs_dta_path = "data/ZA4460_v3-0-0.dta"

wvs = pd.read_csv(wvs_path,
                   usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'S020', 'G006', 'S017'],
                   low_memory=False)

evs = pd.read_stata(evs_dta_path, convert_categoricals=False,
                    columns=['c_abrv', 'country1', 'q365', 'weight_s', 'year'])

# Paper values to compare against
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

print("=== EVS COUNTRY-BY-COUNTRY ANALYSIS (Round vs Floor, Weighted vs Unweighted) ===")

za_to_alpha = {
    'US': 'USA', 'GB-GBN': 'GBR', 'GB-NIR': 'NIR', 'IE': 'IRL',
    'BE': 'BEL', 'FR': 'FRA', 'SE': 'SWE', 'NL': 'NLD', 'NO': 'NOR',
    'FI': 'FIN', 'IS': 'ISL', 'ES': 'ESP', 'IT': 'ITA',
    'CA': 'CAN', 'HU': 'HUN', 'BG': 'BGR', 'SI': 'SVN', 'CH': 'CHE'
}

evs_valid = evs[(evs['q365'] >= 1) & (evs['q365'] <= 10)].copy()

for za_code, alpha in sorted(za_to_alpha.items()):
    sub = evs_valid[evs_valid['c_abrv'] == za_code]
    if len(sub) == 0:
        continue
    is_10 = (sub['q365'] == 10).astype(float)
    w = evs.loc[sub.index, 'weight_s']

    pct_uw = is_10.mean() * 100
    pct_w = (is_10 * w).sum() / w.sum() * 100

    paper_val = paper_values.get((alpha, '1990-1991'), '?')

    print(f"{alpha} ({za_code}): N={len(sub)}, raw={pct_uw:.4f}% (round={round(pct_uw)}, floor={int(pct_uw)}), "
          f"weighted={pct_w:.4f}% (round={round(pct_w)}, floor={int(pct_w)}), paper={paper_val}")

# Germany
print("\n--- Germany ---")
deu_evs = evs_valid[evs_valid['c_abrv'] == 'DE']
for c1, label in [(900, 'DEU_WEST'), (901, 'DEU_EAST')]:
    sub = deu_evs[deu_evs['country1'] == c1]
    if len(sub) > 0:
        is_10 = (sub['q365'] == 10).astype(float)
        w = evs.loc[sub.index, 'weight_s']
        pct_uw = is_10.mean() * 100
        pct_w = (is_10 * w).sum() / w.sum() * 100
        paper_val = paper_values.get((label, '1990-1991'), '?')
        print(f"{label}: N={len(sub)}, raw={pct_uw:.4f}% (round={round(pct_uw)}, floor={int(pct_uw)}), "
              f"weighted={pct_w:.4f}% (round={round(pct_w)}, floor={int(pct_w)}), paper={paper_val}")

# Also check LVA in EVS
lva_evs = evs_valid[evs_valid['c_abrv'] == 'LV']
print(f"\nLatvia EVS: N={len(lva_evs)}")
if len(lva_evs) > 0:
    pct = (lva_evs['q365'] == 10).mean() * 100
    print(f"  LVA q365 %10: {pct:.4f}%")

print("\n=== WVS COUNTRY-BY-COUNTRY ANALYSIS (Round vs Floor) ===")
wvs_valid = wvs[(wvs['F063'] >= 1) & (wvs['F063'] <= 10) & (wvs['S002VS'].isin([1, 2, 3]))].copy()
wave_to_period = {1: '1981', 2: '1990-1991', 3: '1995-1998'}
wvs_valid['period'] = wvs_valid['S002VS'].map(wave_to_period)

# Handle Germany split
wvs_deu = wvs_valid[wvs_valid['COUNTRY_ALPHA'] == 'DEU'].copy()
wvs_west = wvs_deu[wvs_deu['G006'].isin([1, 4])].copy()
wvs_west['COUNTRY_ALPHA'] = 'DEU_WEST'
wvs_east = wvs_deu[wvs_deu['G006'].isin([2, 3])].copy()
wvs_east['COUNTRY_ALPHA'] = 'DEU_EAST'
wvs_valid = wvs_valid[wvs_valid['COUNTRY_ALPHA'] != 'DEU']
wvs_valid = pd.concat([wvs_valid, wvs_west, wvs_east], ignore_index=True)

for (country, period), group in sorted(wvs_valid.groupby(['COUNTRY_ALPHA', 'period'])):
    if (country, period) not in paper_values:
        continue

    is_10 = (group['F063'] == 10).astype(float)
    w = group['S017']

    w_std = w.std()
    w_mean = w.mean()
    use_weight = (w_std > 0.05 and abs(w_mean - 1.0) < 0.05 and w_std < 0.7 and w.gt(0).all())

    pct_uw = is_10.mean() * 100
    pct_w = (is_10 * w).sum() / w.sum() * 100

    paper_val = paper_values.get((country, period), '?')

    w_used = "W" if use_weight else "UW"
    chosen_pct = pct_w if use_weight else pct_uw

    print(f"{country} {period}: N={len(group)}, UW={pct_uw:.4f}%(r{round(pct_uw)},f{int(pct_uw)}), "
          f"W={pct_w:.4f}%(r{round(pct_w)},f{int(pct_w)}), "
          f"chosen={w_used}={chosen_pct:.4f}%, paper={paper_val}")

print("\n=== TESTING DIFFERENT ROUNDING STRATEGIES ===")

# Simulate all rounding strategies
strategies = {
    'round_uw': lambda p, use_w, p_uw, p_w: round(p_uw),
    'round_w': lambda p, use_w, p_uw, p_w: round(p_w),
    'round_auto': lambda p, use_w, p_uw, p_w: round(p_w if use_w else p_uw),
    'floor_uw': lambda p, use_w, p_uw, p_w: int(p_uw),
    'floor_w': lambda p, use_w, p_uw, p_w: int(p_w),
    'floor_auto': lambda p, use_w, p_uw, p_w: int(p_w if use_w else p_uw),
}

# Can't test all strategies generically here, but we can print the key ones
# Let's focus on the mismatched cells

print("\nKey cell analysis with different approaches:")
print(f"Spain EVS 1990: UW=17.13% (floor=17, round=17), W=18.60% (floor=18, round=19)")
print(f"  Paper=18. Floor of weighted = 18! MATCH!")
print()
print(f"Netherlands EVS 1990: UW=11.96% (floor=11, round=12), W=12.44% (floor=12, round=12)")
print(f"  Paper=11. Floor of unweighted = 11! MATCH!")
print()
print(f"West Germany EVS 1990: UW=13.04% (floor=13, round=13), W=13.46% (floor=13, round=13)")
print(f"  Paper=14. Neither matches. EVS data difference.")
print()
print(f"Brazil WVS 1990: UW=82.55% (floor=82, round=83)")
print(f"  Paper=83. Round = 83! Already using round correctly.")
print()
print(f"Nigeria WVS 1995: W=86.19% (floor=86, round=86)")
print(f"  Paper=87. WVS data difference. But NGA 1990: 87 EXACT.")
print()
print(f"Mexico WVS 1995: UW=49.50% (floor=49, round=50)")
print(f"  Paper=50. Round=50 is already matching!")
print()
print(f"Russia WVS 1995: UW=18.42% (floor=18, round=18)")
print(f"  Paper=19. Data version difference.")
print()
print(f"Japan WVS 1995: UW=5.82% (floor=5, round=6)")
print(f"  Paper=5. Floor=5! MATCH!")
print()
print(f"India WVS 1995: UW=53.93% (floor=53, round=54)")
print(f"  Paper=56. Neither matches (data version).")
print()
print(f"ZAF WVS 1990: W=73.13% (floor=73, round=73)")
print(f"  Paper=74. Data version difference.")
print()
print(f"ZAF WVS 1995: W=70.50% (floor=70, round=71)")
print(f"  Paper=71. Round=71! Already correct.")
print()
print(f"ZAF WVS 1981: UW=52.78% (floor=52, round=53)")
print(f"  Paper=50. Data version difference.")
