"""Debug13: Comprehensive weighted check for ALL WVS countries."""
import pandas as pd
import numpy as np

MONTHLY = [1, 2, 3]
VALID = [1, 2, 3, 4, 5, 6, 7, 8]

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS', 'S003', 'S017', 'F028', 'X048WVS'],
                   low_memory=False)

# Ground truth
ground_truth = {
    ('Argentina', 32, 1, '1981'): 56,
    ('Argentina', 32, 2, '1990-1991'): 55,
    ('Argentina', 32, 3, '1995-1998'): 41,
    ('Australia', 36, 1, '1981'): 40,
    ('Australia', 36, 3, '1995-1998'): 25,
    ('Belarus', 112, 2, '1990-1991'): 6,
    ('Belarus', 112, 3, '1995-1998'): 14,
    ('Brazil', 76, 2, '1990-1991'): 50,
    ('Brazil', 76, 3, '1995-1998'): 54,
    ('Bulgaria', 100, 3, '1995-1998'): 15,
    ('Chile', 152, 2, '1990-1991'): 47,
    ('Chile', 152, 3, '1995-1998'): 44,
    ('Hungary', 348, 1, '1981'): 16,
    ('Hungary', 348, 3, '1995-1998'): None,  # Not in paper table
    ('India', 356, 2, '1990-1991'): 71,
    ('India', 356, 3, '1995-1998'): 54,
    ('Japan', 392, 1, '1981'): 12,
    ('Japan', 392, 2, '1990-1991'): 14,
    ('Japan', 392, 3, '1995-1998'): 11,
    ('South Korea', 410, 1, '1981'): 29,
    ('South Korea', 410, 2, '1990-1991'): 60,
    ('South Korea', 410, 3, '1995-1998'): 27,
    ('Latvia', 428, 3, '1995-1998'): 16,
    ('Mexico', 484, 1, '1981'): 74,
    ('Mexico', 484, 2, '1990-1991'): 63,
    ('Mexico', 484, 3, '1995-1998'): 65,
    ('Nigeria', 566, 2, '1990-1991'): 88,
    ('Nigeria', 566, 3, '1995-1998'): 87,
    ('Norway', 578, 3, '1995-1998'): 13,
    ('Poland', 616, 3, '1995-1998'): 74,
    ('Russia', 643, 2, '1990-1991'): 6,
    ('Russia', 643, 3, '1995-1998'): 8,
    ('Slovenia', 705, 3, '1995-1998'): 33,
    ('South Africa', 710, 1, '1981'): 61,
    ('South Africa', 710, 3, '1995-1998'): 70,
    ('Spain', 724, 2, '1990-1991'): 40,
    ('Spain', 724, 3, '1995-1998'): 38,
    ('Sweden', 752, 3, '1995-1998'): 11,
    ('Switzerland', 756, 2, '1990-1991'): 43,
    ('Switzerland', 756, 3, '1995-1998'): 25,
    ('Turkey', 792, 2, '1990-1991'): 38,
    ('Turkey', 792, 3, '1995-1998'): 44,
    ('Great Britain', 826, 3, '1995-1998'): None,
    ('United States', 840, 3, '1995-1998'): 55,
}

# Countries where -2 should be included in denominator
WAVE1_NEG2 = {348, 392, 410, 484, 710}
WAVE2_NEG2 = {410}

print(f"{'Country':<22} {'Wave':<5} {'Paper':>6} {'Unwt':>6} {'Wt':>6} {'Unwt+neg2':>10} {'Wt+neg2':>8}")
print("-" * 75)

for (name, s003, wave, wlabel), paper_val in sorted(ground_truth.items()):
    sub = wvs[(wvs['S003'] == s003) & (wvs['S002VS'] == wave)]
    if len(sub) == 0:
        continue

    # Handle Germany separately
    if s003 == 276:
        continue

    include_neg2 = False
    if wave == 1 and s003 in WAVE1_NEG2:
        include_neg2 = True
    elif wave == 2 and s003 in WAVE2_NEG2:
        include_neg2 = True

    # Standard 8pt
    valid = sub[sub['F028'].isin(VALID)]
    monthly = valid[valid['F028'].isin(MONTHLY)]
    pct_unwt = round(len(monthly)/len(valid)*100) if len(valid) > 0 else None

    # Weighted
    if valid['S017'].notna().any() and valid['S017'].sum() > 0:
        pct_wt = round(monthly['S017'].sum() / valid['S017'].sum() * 100)
    else:
        pct_wt = None

    # With neg2
    if include_neg2:
        valid_neg2 = sub[sub['F028'].isin(VALID + [-2])]
        monthly_neg2 = valid_neg2[valid_neg2['F028'].isin(MONTHLY)]
        pct_unwt_neg2 = round(len(monthly_neg2)/len(valid_neg2)*100) if len(valid_neg2) > 0 else None
        if valid_neg2['S017'].notna().any() and valid_neg2['S017'].sum() > 0:
            pct_wt_neg2 = round(monthly_neg2['S017'].sum() / valid_neg2['S017'].sum() * 100)
        else:
            pct_wt_neg2 = None
    else:
        pct_unwt_neg2 = pct_unwt
        pct_wt_neg2 = pct_wt

    paper_str = str(paper_val) if paper_val else '-'
    def fmt(v):
        return str(v) if v is not None else 'N/A'
    print(f"{name:<22} W{wave:<4} {paper_str:>6} {fmt(pct_unwt):>6} {fmt(pct_wt):>6} {fmt(pct_unwt_neg2):>10} {fmt(pct_wt_neg2):>8}")

# === Germany West/East with weights ===
print("\n=== Germany East/West with weights ===")
deu_w3 = wvs[(wvs['S003'] == 276) & (wvs['S002VS'] == 3)].copy()
deu_w3['state'] = deu_w3['X048WVS'] % 1000

for label, states in [('West', [1,2,3,4,5,6,7,8,9,10]), ('East', [12,13,14,15,16,19,20])]:
    sub = deu_w3[deu_w3['state'].isin(states)]
    valid = sub[sub['F028'].isin(VALID)]
    monthly = valid[valid['F028'].isin(MONTHLY)]
    pct_unwt = round(len(monthly)/len(valid)*100) if len(valid) > 0 else None
    if valid['S017'].notna().any() and valid['S017'].sum() > 0:
        pct_wt = round(monthly['S017'].sum() / valid['S017'].sum() * 100)
    else:
        pct_wt = None
    print(f"  {label} Germany W3: unwt={pct_unwt}%, wt={pct_wt}%")

# === EVS with weights ===
print("\n=== EVS key countries with original weights ===")
evs = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False,
                     columns=['c_abrv', 'country1', 'q336', 'weight_g', 'weight_s'])

evs_gt = {
    'BE': ('Belgium', 35),
    'BG': ('Bulgaria', 9),
    'CA': ('Canada', 40),
    'ES': ('Spain', 40),
    'FI': ('Finland', 13),
    'FR': ('France', 17),
    'GB-GBN': ('Great Britain', 25),
    'GB-NIR': ('Northern Ireland', 69),
    'HU': ('Hungary', 34),
    'IE': ('Ireland', 88),
    'IS': ('Iceland', 9),
    'IT': ('Italy', 47),
    'LV': ('Latvia', 9),
    'NL': ('Netherlands', 31),
    'NO': ('Norway', 13),
    'PL': ('Poland', 85),
    'SE': ('Sweden', 10),
    'SI': ('Slovenia', 35),
    'US': ('United States', 59),
}

for alpha, (name, paper_val) in sorted(evs_gt.items()):
    sub = evs[evs['c_abrv'] == alpha]
    if len(sub) == 0:
        continue

    # Hungary special: use 7pt
    if alpha == 'HU':
        valid_vals = [1,2,3,4,5,6,7]
    else:
        valid_vals = VALID

    valid = sub[sub['q336'].isin(valid_vals)]
    monthly = valid[valid['q336'].isin(MONTHLY)]

    pct_unwt = round(len(monthly)/len(valid)*100) if len(valid) > 0 else None

    for wt_col in ['weight_g', 'weight_s']:
        if valid[wt_col].notna().any() and valid[wt_col].sum() > 0:
            pct_wt = round(monthly[wt_col].sum() / valid[wt_col].sum() * 100)
        else:
            pct_wt = None
        match = abs(pct_wt - paper_val) if pct_wt else None
        marker = ' MATCH' if match is not None and match <= 1 else ''
        print(f"  {alpha:<10} {name:<22} {wt_col}: unwt={pct_unwt}%, wt={pct_wt}% (paper={paper_val}){marker}")

# Germany EVS with weights
print("\n=== Germany EVS East/West with weights ===")
deu_evs = evs[evs['c_abrv'] == 'DE']
for c1, label in [(900, 'West'), (901, 'East')]:
    sub = deu_evs[deu_evs['country1'] == c1]
    valid = sub[sub['q336'].isin(VALID)]
    monthly = valid[valid['q336'].isin(MONTHLY)]
    pct_unwt = round(len(monthly)/len(valid)*100) if len(valid) > 0 else None
    for wt_col in ['weight_g', 'weight_s']:
        if valid[wt_col].notna().any() and valid[wt_col].sum() > 0:
            pct_wt = round(monthly[wt_col].sum() / valid[wt_col].sum() * 100)
        else:
            pct_wt = None
        print(f"  {label} Germany {wt_col}: unwt={pct_unwt}%, wt={pct_wt}%")
