"""Debug9: Wave 1 (1981) countries and remaining issues."""
import pandas as pd
import numpy as np

MONTHLY_VALS = [1, 2, 3]
VALID_8PT = [1, 2, 3, 4, 5, 6, 7, 8]

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS', 'S003', 'F028', 'X048WVS'],
                   low_memory=False)

# === WAVE 1 COUNTRIES - check all denominator approaches ===
print("=== WAVE 1 (1981) COUNTRIES ===")
wave1_paper = {
    32: ('Argentina', 56),
    36: ('Australia', 40),
    246: ('Finland', 13),
    348: ('Hungary', 16),
    392: ('Japan', 12),
    410: ('South Korea', 29),
    484: ('Mexico', 74),
    710: ('South Africa', 61),
}

for s003, (name, paper_val) in sorted(wave1_paper.items()):
    sub = wvs[(wvs['S003'] == s003) & (wvs['S002VS'] == 1)]
    if len(sub) == 0:
        print(f"  {name}: NO DATA")
        continue
    print(f"\n  {name} (S003={s003}):")
    print(f"    F028 dist: {sub['F028'].value_counts().sort_index().to_dict()}")

    for include_neg2 in [False, True]:
        for scale in ['8pt', '7pt']:
            if scale == '8pt':
                valid_vals = VALID_8PT
            else:
                valid_vals = [1, 2, 3, 4, 5, 6, 7]

            if include_neg2:
                denom = sub[sub['F028'].isin(valid_vals + [-2])]
            else:
                denom = sub[sub['F028'].isin(valid_vals)]
            numer = sub[sub['F028'].isin(MONTHLY_VALS)]

            if len(denom) > 0:
                pct = round(len(numer) / len(denom) * 100)
                diff = abs(pct - paper_val)
                match = 'MATCH' if diff <= 1 else f'diff={diff}'
                print(f"    {scale} neg2={include_neg2}: {pct}% (n={len(denom)}) {match}")

# === WVS Wave 2 detailed for all relevant countries ===
print("\n\n=== WVS WAVE 2 COUNTRIES ===")
wave2_paper = {
    32: ('Argentina', 55),
    76: ('Brazil', 50),
    112: ('Belarus', 6),
    152: ('Chile', 47),
    356: ('India', 71),
    392: ('Japan', 14),
    410: ('South Korea', 60),
    484: ('Mexico', 63),
    566: ('Nigeria', 88),
    643: ('Russia', 6),
    703: ('Slovakia', None),
    710: ('South Africa', None),
    724: ('Spain', 40),
    756: ('Switzerland', 43),
    792: ('Turkey', 38),
}

for s003, (name, paper_val) in sorted(wave2_paper.items()):
    sub = wvs[(wvs['S003'] == s003) & (wvs['S002VS'] == 2)]
    if len(sub) == 0:
        print(f"  {name}: NO DATA in WVS W2")
        continue

    for include_neg2 in [False, True]:
        denom = sub[sub['F028'].isin(VALID_8PT + ([-2] if include_neg2 else []))]
        numer = sub[sub['F028'].isin(MONTHLY_VALS)]
        if len(denom) > 0:
            pct = round(len(numer) / len(denom) * 100)
            match = ''
            if paper_val:
                diff = abs(pct - paper_val)
                match = 'MATCH' if diff <= 1 else f'diff={diff}'
            print(f"  {name:<20} neg2={include_neg2}: {pct:>3}% (n={len(denom):>4}) {match}")

# === Italy: check WVS wave 2 ===
print("\n\n=== ITALY WVS Wave 2 ===")
ita_w2 = wvs[(wvs['S003'] == 380) & (wvs['S002VS'] == 2)]
print(f"Italy S003=380 W2: n={len(ita_w2)}")
# Italy might have different code
for code in [380, 381, 382]:
    sub = wvs[(wvs['S003'] == code)]
    if len(sub) > 0:
        print(f"  S003={code}: n={len(sub)}, waves={sorted(sub['S002VS'].unique())}")

# === West Germany 1981: check EVS for 1981 data ===
# The EVS file only has 1990 data, so we can't get 1981 European data

# === Germany Wave 3: verify East/West split ===
print("\n\n=== GERMANY Wave 3 East/West ===")
deu_w3 = wvs[(wvs['S003'] == 276) & (wvs['S002VS'] == 3)].copy()
deu_w3['state'] = deu_w3['X048WVS'] % 1000

west_states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
east_states = [12, 13, 14, 15, 16]

for region, states in [('West', west_states), ('East', east_states), ('Berlin(11)', [11]), ('Other(19,20)', [19, 20])]:
    sub = deu_w3[deu_w3['state'].isin(states)]
    if len(sub) == 0:
        print(f"  {region}: NO DATA")
        continue
    valid = sub[sub['F028'].isin(VALID_8PT)]
    monthly = sub[sub['F028'].isin(MONTHLY_VALS)]
    if len(valid) > 0:
        pct = round(len(monthly) / len(valid) * 100)
        print(f"  {region}: {pct}% (n_valid={len(valid)})")

# Check what paper says for West Germany W3
# Paper: West=25, East=9
# Our West (states 1-10): 25%  -- MATCHES!
# Our East (states 12-16): 10% -- close to 9
# Let's also try including state 11 (Berlin) in East
for label, states_list in [('East+Berlin', [11, 12, 13, 14, 15, 16]),
                            ('East+19+20', [12, 13, 14, 15, 16, 19, 20]),
                            ('East only', [12, 13, 14, 15, 16])]:
    sub = deu_w3[deu_w3['state'].isin(states_list)]
    valid = sub[sub['F028'].isin(VALID_8PT)]
    monthly = sub[sub['F028'].isin(MONTHLY_VALS)]
    if len(valid) > 0:
        pct = round(len(monthly) / len(valid) * 100)
        print(f"  {label}: {pct}% (n={len(valid)})")
