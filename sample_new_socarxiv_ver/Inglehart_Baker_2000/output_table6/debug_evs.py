"""Debug EVS data for Table 6 discrepancies."""
import pandas as pd
import numpy as np

evs = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)

# Check F028 vs F063 for Hungary
hun = evs[evs['COUNTRY_ALPHA'] == 'HUN']
print('=== HUNGARY EVS ===')
print('F063 value counts:')
print(hun['F063'].value_counts().sort_index())
print()
print('F028 value counts:')
print(hun['F028'].value_counts().sort_index())
print()

# Check F028 vs F063 for Germany
deu = evs[evs['COUNTRY_ALPHA'] == 'DEU']
print('=== GERMANY EVS ===')
print('F063 value counts:')
print(deu['F063'].value_counts().sort_index())
print()
print('F028 value counts:')
print(deu['F028'].value_counts().sort_index())
print()

# Check Germany S001 for region split
print('Germany S001 values:', sorted(deu['S001'].unique()))
print()

# Check F028 vs F063 for Italy
ita = evs[evs['COUNTRY_ALPHA'] == 'ITA']
print('=== ITALY EVS ===')
print('F063 value counts:')
print(ita['F063'].value_counts().sort_index())
print()
print('F028 value counts:')
print(ita['F028'].value_counts().sort_index())
print()

# Check what S020 years are in EVS
print('S020 years in EVS:', sorted(evs['S020'].unique()))
print()

# Compute church attendance percentages using F028 vs F063 for key countries
MONTHLY_VALS = [1, 2, 3]
VALID_8PT = [1, 2, 3, 4, 5, 6, 7, 8]

for country in ['HUN', 'DEU', 'ITA']:
    c_data = evs[evs['COUNTRY_ALPHA'] == country]
    for col in ['F028', 'F063']:
        valid = c_data[c_data[col].isin(VALID_8PT)]
        monthly = c_data[c_data[col].isin(MONTHLY_VALS)]
        if len(valid) > 0:
            pct = round(len(monthly) / len(valid) * 100)
            print(f"{country} {col}: {pct}% (n_valid={len(valid)}, n_monthly={len(monthly)})")
        else:
            print(f"{country} {col}: No valid data")
    print()

# Check if there's a way to split East/West Germany in EVS
# S001 might be different codes for East vs West
print('=== Germany S001 detail ===')
for s001_val in sorted(deu['S001'].unique()):
    sub = deu[deu['S001'] == s001_val]
    valid_f = sub[sub['F063'].isin(VALID_8PT)]
    monthly_f = sub[sub['F063'].isin(MONTHLY_VALS)]
    if len(valid_f) > 0:
        pct = round(len(monthly_f) / len(valid_f) * 100)
    else:
        pct = 'N/A'
    print(f"  S001={s001_val}: n={len(sub)}, F063 attend%={pct}")

# WVS data - check wave info for problematic countries
print('\n\n=== WVS DATA ===')
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS', 'S003', 'S020', 'F028', 'S024', 'X048WVS'],
                   low_memory=False)

# Check Brazil wave 3
bra = wvs[(wvs['S003'] == 76) & (wvs['S002VS'] == 3)]
print('=== BRAZIL Wave 3 ===')
print('F028 value counts:')
print(bra['F028'].value_counts().sort_index())
bra_valid = bra[bra['F028'].isin(VALID_8PT)]
bra_monthly = bra[bra['F028'].isin(MONTHLY_VALS)]
if len(bra_valid) > 0:
    print(f"Brazil W3: {round(len(bra_monthly)/len(bra_valid)*100)}% (n={len(bra_valid)})")

# Try with broader valid values (maybe some coded differently)
print('\nBrazil W3 F028 ALL values:')
print(bra['F028'].value_counts().sort_index().to_string())

# Check S024 for Germany
deu_wvs = wvs[(wvs['S003'] == 276)]
print('\n=== GERMANY WVS ===')
print('S024 values in WVS Germany:')
print(deu_wvs['S024'].value_counts().sort_index())
print()

# Check S024 by wave
for w in [1, 2, 3]:
    deu_w = deu_wvs[deu_wvs['S002VS'] == w]
    if len(deu_w) > 0:
        print(f'Germany Wave {w}: S024 values = {sorted(deu_w["S024"].unique())}')
        for s024_val in sorted(deu_w['S024'].unique()):
            sub = deu_w[deu_w['S024'] == s024_val]
            valid = sub[sub['F028'].isin(VALID_8PT)]
            monthly = sub[sub['F028'].isin(MONTHLY_VALS)]
            if len(valid) > 0:
                pct = round(len(monthly) / len(valid) * 100)
            else:
                pct = 'N/A'
            print(f"  S024={s024_val}: n={len(sub)}, n_valid={len(valid)}, attend%={pct}")

# Check S Korea wave 2
kor = wvs[(wvs['S003'] == 410) & (wvs['S002VS'] == 2)]
print('\n=== S.KOREA Wave 2 ===')
print('F028 value counts:')
print(kor['F028'].value_counts().sort_index())
kor_valid = kor[kor['F028'].isin(VALID_8PT)]
kor_monthly = kor[kor['F028'].isin(MONTHLY_VALS)]
if len(kor_valid) > 0:
    print(f"S.Korea W2: {round(len(kor_monthly)/len(kor_valid)*100)}% (n={len(kor_valid)})")
