"""Debug3: Deep dive into specific discrepancies."""
import pandas as pd
import numpy as np

MONTHLY_VALS = [1, 2, 3]
VALID_8PT = [1, 2, 3, 4, 5, 6, 7, 8]

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS', 'S003', 'S020', 'F028', 'S024', 'X048WVS'],
                   low_memory=False)
evs = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)

# === 1. GERMANY WAVE 3: Check if all data is West Germany ===
print("=== GERMANY Wave 3 - X048WVS distribution ===")
deu_w3 = wvs[(wvs['S003'] == 276) & (wvs['S002VS'] == 3)]
x048_vals = deu_w3['X048WVS'].value_counts().sort_index()
print(x048_vals)
print()

# Extract state numbers (last 3 digits of X048WVS)
deu_w3 = deu_w3.copy()
deu_w3['state'] = deu_w3['X048WVS'] % 1000
state_dist = deu_w3['state'].value_counts().sort_index()
print("State distribution:")
print(state_dist)
print()

# Compute attend% for all Germany W3 vs West only vs East only
west_states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
east_states = [12, 13, 14, 15, 16]

all_valid = deu_w3[deu_w3['F028'].isin(VALID_8PT)]
all_monthly = deu_w3[deu_w3['F028'].isin(MONTHLY_VALS)]
print(f"All Germany W3: {round(len(all_monthly)/len(all_valid)*100)}% (n={len(all_valid)})")

west = deu_w3[deu_w3['state'].isin(west_states)]
west_valid = west[west['F028'].isin(VALID_8PT)]
west_monthly = west[west['F028'].isin(MONTHLY_VALS)]
print(f"West Germany W3: {round(len(west_monthly)/len(west_valid)*100)}% (n={len(west_valid)})")

east = deu_w3[deu_w3['state'].isin(east_states)]
if len(east) > 0:
    east_valid = east[east['F028'].isin(VALID_8PT)]
    east_monthly = east[east['F028'].isin(MONTHLY_VALS)]
    if len(east_valid) > 0:
        print(f"East Germany W3: {round(len(east_monthly)/len(east_valid)*100)}% (n={len(east_valid)})")
    else:
        print("East Germany W3: no valid data")
else:
    print("East Germany W3: no data at all")

# Check state 11 (Berlin) and 19
for s in [11, 19, 20]:
    sub = deu_w3[deu_w3['state'] == s]
    if len(sub) > 0:
        sv = sub[sub['F028'].isin(VALID_8PT)]
        sm = sub[sub['F028'].isin(MONTHLY_VALS)]
        if len(sv) > 0:
            print(f"State {s}: {round(len(sm)/len(sv)*100)}% (n={len(sv)})")

# === 2. HUNGARY: Deep investigation ===
print("\n\n=== HUNGARY ===")
# Wave 1
hun_w1 = wvs[(wvs['S003'] == 348) & (wvs['S002VS'] == 1)]
print("Hungary W1 F028:")
print(hun_w1['F028'].value_counts().sort_index())
# Different denominator treatments
for include_neg2 in [False, True]:
    if include_neg2:
        denom = hun_w1[hun_w1['F028'].isin(VALID_8PT + [-2])]
    else:
        denom = hun_w1[hun_w1['F028'].isin(VALID_8PT)]
    numer = hun_w1[hun_w1['F028'].isin(MONTHLY_VALS)]
    if len(denom) > 0:
        print(f"  Include -2={include_neg2}: {round(len(numer)/len(denom)*100)}% (n_valid={len(denom)}, n_monthly={len(numer)})")

# EVS Hungary
hun_evs = evs[evs['COUNTRY_ALPHA'] == 'HUN']
print("\nHungary EVS F063:")
print(hun_evs['F063'].value_counts().sort_index())
valid = hun_evs[hun_evs['F063'].isin(VALID_8PT)]
monthly = hun_evs[hun_evs['F063'].isin(MONTHLY_VALS)]
print(f"EVS Hungary: {round(len(monthly)/len(valid)*100)}% (n={len(valid)})")

# Wave 3
hun_w3 = wvs[(wvs['S003'] == 348) & (wvs['S002VS'] == 3)]
print("\nHungary W3 F028:")
print(hun_w3['F028'].value_counts().sort_index())
valid = hun_w3[hun_w3['F028'].isin(VALID_8PT)]
monthly = hun_w3[hun_w3['F028'].isin(MONTHLY_VALS)]
if len(valid) > 0:
    print(f"Hungary W3: {round(len(monthly)/len(valid)*100)}% (n={len(valid)})")

# === 3. SOUTH KOREA wave 1 and 2 ===
print("\n\n=== SOUTH KOREA ===")
for w in [1, 2, 3]:
    kor = wvs[(wvs['S003'] == 410) & (wvs['S002VS'] == w)]
    if len(kor) == 0:
        print(f"S.Korea W{w}: no data")
        continue
    print(f"\nS.Korea W{w} F028:")
    print(kor['F028'].value_counts().sort_index())
    for include_neg2 in [False, True]:
        if include_neg2:
            denom = kor[kor['F028'].isin(VALID_8PT + [-2])]
        else:
            denom = kor[kor['F028'].isin(VALID_8PT)]
        numer = kor[kor['F028'].isin(MONTHLY_VALS)]
        if len(denom) > 0:
            print(f"  Include -2={include_neg2}: {round(len(numer)/len(denom)*100)}% (n_valid={len(denom)}, n_monthly={len(numer)})")

# === 4. Brazil wave 2 and 3 ===
print("\n\n=== BRAZIL ===")
for w in [2, 3]:
    bra = wvs[(wvs['S003'] == 76) & (wvs['S002VS'] == w)]
    if len(bra) == 0:
        print(f"Brazil W{w}: no data")
        continue
    print(f"\nBrazil W{w} F028:")
    print(bra['F028'].value_counts().sort_index())
    valid = bra[bra['F028'].isin(VALID_8PT)]
    monthly = bra[bra['F028'].isin(MONTHLY_VALS)]
    if len(valid) > 0:
        print(f"  Pct: {round(len(monthly)/len(valid)*100)}% (n={len(valid)})")
    # Try different valid values
    valid_no5 = bra[bra['F028'].isin([1, 2, 3, 4, 6, 7, 8])]
    monthly2 = bra[bra['F028'].isin(MONTHLY_VALS)]
    if len(valid_no5) > 0:
        print(f"  Pct (excl 5): {round(len(monthly2)/len(valid_no5)*100)}% (n={len(valid_no5)})")
    # What if we include -2 in denominator?
    valid_with_neg2 = bra[bra['F028'].isin(VALID_8PT + [-2])]
    if len(valid_with_neg2) > 0:
        print(f"  Pct (incl -2): {round(len(monthly2)/len(valid_with_neg2)*100)}% (n={len(valid_with_neg2)})")

# === 5. EVS Italy detailed ===
print("\n\n=== ITALY EVS ===")
ita = evs[evs['COUNTRY_ALPHA'] == 'ITA']
print("F063 detailed:")
print(ita['F063'].value_counts().sort_index())
# Different valid sets
for include_neg in [False, True]:
    if include_neg:
        neg_vals = sorted(ita['F063'][ita['F063'] < 0].unique())
        denom = ita[ita['F063'].isin(VALID_8PT + list(neg_vals))]
    else:
        denom = ita[ita['F063'].isin(VALID_8PT)]
    numer = ita[ita['F063'].isin(MONTHLY_VALS)]
    if len(denom) > 0:
        print(f"  Include neg={include_neg}: {round(len(numer)/len(denom)*100)}% (n={len(denom)})")
# Check all F063 values including negatives
print("\nAll F063 values (including negatives):")
print(ita['F063'].value_counts().sort_index().head(20))

# === 6. India wave 2 ===
print("\n\n=== INDIA ===")
ind_w2 = wvs[(wvs['S003'] == 356) & (wvs['S002VS'] == 2)]
print(f"India W2 F028:")
print(ind_w2['F028'].value_counts().sort_index())
valid = ind_w2[ind_w2['F028'].isin(VALID_8PT)]
monthly = ind_w2[ind_w2['F028'].isin(MONTHLY_VALS)]
if len(valid) > 0:
    print(f"India W2: {round(len(monthly)/len(valid)*100)}% (n={len(valid)})")

# India wave 3
ind_w3 = wvs[(wvs['S003'] == 356) & (wvs['S002VS'] == 3)]
print(f"\nIndia W3 F028:")
print(ind_w3['F028'].value_counts().sort_index())
valid = ind_w3[ind_w3['F028'].isin(VALID_8PT)]
monthly = ind_w3[ind_w3['F028'].isin(MONTHLY_VALS)]
if len(valid) > 0:
    print(f"India W3: {round(len(monthly)/len(valid)*100)}% (n={len(valid)})")

# === 7. EVS: Check which countries have 1981 data ===
print("\n\n=== EVS year distribution by country ===")
for alpha in sorted(evs['COUNTRY_ALPHA'].unique()):
    sub = evs[evs['COUNTRY_ALPHA'] == alpha]
    years = sorted(sub['S020'].unique())
    print(f"  {alpha}: years={years}, n={len(sub)}")

# === 8. Check EVS for Russia/Latvia/Belarus ===
print("\n\n=== EVS ex-communist countries ===")
for alpha in ['BLR', 'BGR', 'HUN', 'LVA', 'POL', 'RUS', 'SVN']:
    sub = evs[evs['COUNTRY_ALPHA'] == alpha]
    if len(sub) == 0:
        print(f"  {alpha}: NOT in EVS")
        continue
    valid = sub[sub['F063'].isin(VALID_8PT)]
    monthly = sub[sub['F063'].isin(MONTHLY_VALS)]
    pct = round(len(monthly)/len(valid)*100) if len(valid) > 0 else 'N/A'
    print(f"  {alpha}: n={len(sub)}, valid={len(valid)}, attend%={pct}")

# === 9. WVS ex-communist wave 2 ===
print("\n\n=== WVS ex-communist wave 2 ===")
for s003, name in [(112, 'Belarus'), (100, 'Bulgaria'), (348, 'Hungary'),
                    (428, 'Latvia'), (616, 'Poland'), (643, 'Russia'), (705, 'Slovenia')]:
    sub = wvs[(wvs['S003'] == s003) & (wvs['S002VS'] == 2)]
    if len(sub) == 0:
        print(f"  {name} ({s003}): NOT in WVS wave 2")
        continue
    valid = sub[sub['F028'].isin(VALID_8PT)]
    monthly = sub[sub['F028'].isin(MONTHLY_VALS)]
    pct = round(len(monthly)/len(valid)*100) if len(valid) > 0 else 'N/A'
    print(f"  {name}: n={len(sub)}, valid={len(valid)}, attend%={pct}")
