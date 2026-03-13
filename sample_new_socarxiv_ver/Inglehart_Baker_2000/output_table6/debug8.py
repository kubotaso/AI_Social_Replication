"""Debug8: EVS Germany East/West split and recompute with correct denominator."""
import pandas as pd
import numpy as np

evs_orig = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)

MONTHLY_VALS = [1, 2, 3]
VALID_7PT = [1, 2, 3, 4, 5, 6, 7]  # Excluding 8 = "practically never"
VALID_8PT = [1, 2, 3, 4, 5, 6, 7, 8]  # Including 8

# === GERMANY: country1 = 900 (West) vs 901 (East) ===
print("=== GERMANY East/West via country1 ===")
deu = evs_orig[evs_orig['c_abrv'] == 'DE']
for c1_val in sorted(deu['country1'].unique()):
    sub = deu[deu['country1'] == c1_val]
    for vname, vvals in [('7pt', VALID_7PT), ('8pt', VALID_8PT)]:
        valid = sub[sub['q336'].isin(vvals)]
        monthly = sub[sub['q336'].isin(MONTHLY_VALS)]
        if len(valid) > 0:
            pct = round(len(monthly) / len(valid) * 100)
            print(f"  country1={c1_val}, {vname}: attend% = {pct}% (n_valid={len(valid)}, n_monthly={len(monthly)})")
        else:
            print(f"  country1={c1_val}, {vname}: no valid data")

# === RECOMPUTE ALL COUNTRIES with 7-point scale ===
print("\n\n=== All countries, q336, 7-point denominator (excluding 'practically never') ===")
paper_vals = {
    'AT': ('Austria', None),
    'BE': ('Belgium', 35),
    'BG': ('Bulgaria', 9),
    'CA': ('Canada', 40),
    'CZ': ('Czech Republic', None),
    'DE': ('Germany combined', None),
    'DK': ('Denmark', None),
    'EE': ('Estonia', None),
    'ES': ('Spain', 40),
    'FI': ('Finland', 13),
    'FR': ('France', 17),
    'GB-GBN': ('Great Britain', 25),
    'GB-NIR': ('Northern Ireland', 69),
    'HU': ('Hungary', 34),
    'IE': ('Ireland', 88),
    'IS': ('Iceland', 9),
    'IT': ('Italy', 47),
    'LT': ('Lithuania', None),
    'LV': ('Latvia', 9),
    'MT': ('Malta', None),
    'NL': ('Netherlands', 31),
    'NO': ('Norway', 13),
    'PL': ('Poland', 85),
    'PT': ('Portugal', None),
    'RO': ('Romania', None),
    'SE': ('Sweden', 10),
    'SI': ('Slovenia', 35),
    'SK': ('Slovakia', None),
    'US': ('United States', 59),
}

for alpha in sorted(evs_orig['c_abrv'].dropna().unique()):
    sub = evs_orig[evs_orig['c_abrv'] == alpha]
    name, paper_val = paper_vals.get(alpha, (alpha, None))

    for vname, vvals in [('7pt', VALID_7PT), ('8pt', VALID_8PT)]:
        valid = sub[sub['q336'].isin(vvals)]
        monthly = sub[sub['q336'].isin(MONTHLY_VALS)]
        if len(valid) > 0:
            pct = round(len(monthly) / len(valid) * 100)
            match = ''
            if paper_val is not None:
                diff = abs(pct - paper_val)
                if diff <= 1:
                    match = ' MATCH'
                elif diff <= 3:
                    match = f' CLOSE (diff={diff})'
                else:
                    match = f' MISS (diff={diff})'
            print(f"  {alpha:<10} {name:<22} {vname}: {pct:>3}% (n={len(valid):>4}) paper={paper_val}{match}")

# === Check what happens with WVS 8-point vs 7-point for key countries ===
print("\n\n=== WVS countries: 8pt vs 7pt denominator ===")
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS', 'S003', 'F028'],
                   low_memory=False)

wvs_countries = {
    (32, 'Argentina', 2, 55),
    (32, 'Argentina', 3, 41),
    (76, 'Brazil', 2, 50),
    (76, 'Brazil', 3, 54),
    (152, 'Chile', 2, 47),
    (152, 'Chile', 3, 44),
    (356, 'India', 2, 71),
    (356, 'India', 3, 54),
    (484, 'Mexico', 2, 63),
    (484, 'Mexico', 3, 65),
    (566, 'Nigeria', 2, 88),
    (566, 'Nigeria', 3, 87),
    (710, 'South Africa', 2, None),
    (710, 'South Africa', 3, 70),
    (792, 'Turkey', 2, 38),
    (792, 'Turkey', 3, 44),
    (112, 'Belarus', 2, 6),
    (112, 'Belarus', 3, 14),
    (643, 'Russia', 2, 6),
    (643, 'Russia', 3, 8),
    (410, 'South Korea', 2, 60),
    (410, 'South Korea', 3, 27),
    (392, 'Japan', 2, 14),
    (392, 'Japan', 3, 11),
    (578, 'Norway', 3, 13),
    (752, 'Sweden', 3, 11),
    (826, 'Great Britain', 3, None),
    (840, 'United States', 3, 55),
}

for s003, name, wave, paper_val in sorted(wvs_countries):
    sub = wvs[(wvs['S003'] == s003) & (wvs['S002VS'] == wave)]
    if len(sub) == 0:
        continue
    for vname, vvals in [('7pt', VALID_7PT), ('8pt', VALID_8PT)]:
        valid = sub[sub['F028'].isin(vvals)]
        monthly = sub[sub['F028'].isin(MONTHLY_VALS)]
        if len(valid) > 0:
            pct = round(len(monthly) / len(valid) * 100)
            match = ''
            if paper_val is not None:
                diff = abs(pct - paper_val)
                if diff <= 1:
                    match = ' MATCH'
                elif diff <= 3:
                    match = f' CLOSE (diff={diff})'
                else:
                    match = f' MISS (diff={diff})'
            print(f"  {name:<20} W{wave} {vname}: {pct:>3}% (n={len(valid):>4}) paper={paper_val}{match}")
        # Also try with -2 included
        valid_neg2 = sub[sub['F028'].isin(vvals + [-2])]
        monthly_neg2 = sub[sub['F028'].isin(MONTHLY_VALS)]
        if len(valid_neg2) > len(valid):
            pct_neg2 = round(len(monthly_neg2) / len(valid_neg2) * 100)
            match_neg2 = ''
            if paper_val is not None:
                diff = abs(pct_neg2 - paper_val)
                if diff <= 1:
                    match_neg2 = ' MATCH'
                elif diff <= 3:
                    match_neg2 = f' CLOSE (diff={diff})'
                else:
                    match_neg2 = f' MISS (diff={diff})'
            print(f"  {name:<20} W{wave} {vname}+neg2: {pct_neg2:>3}% (n={len(valid_neg2):>4}) paper={paper_val}{match_neg2}")
