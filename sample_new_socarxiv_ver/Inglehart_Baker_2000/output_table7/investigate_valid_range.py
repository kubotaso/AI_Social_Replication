"""
Check if different valid range criteria affect the problematic cells.
"""
import pandas as pd
import numpy as np

wvs_path = "data/WVS_Time_Series_1981-2022_csv_v5_0.csv"

wvs = pd.read_csv(wvs_path, low_memory=False,
                   usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'S020', 'G006', 'S017'])

# Check various valid range criteria for problematic cells
def check_cell(country, wave, paper_val, df):
    if country == 'DEU':
        sub = df[(df['COUNTRY_ALPHA'] == country) & (df['S002VS'] == wave)]
        for g006_vals, label in [([2,3], 'EAST'), ([1,4], 'WEST')]:
            s = sub[sub['G006'].isin(g006_vals)]
            paper_v = 6 if label == 'EAST' else 16
            _check(s, f"DEU_{label} W{wave}", paper_v)
        return
    sub = df[(df['COUNTRY_ALPHA'] == country) & (df['S002VS'] == wave)]
    _check(sub, f"{country} W{wave}", paper_val)

def _check(sub, label, paper_val):
    print(f"\n{label}: N_total={len(sub)}, paper={paper_val}")
    for min_val, max_val in [(1, 10), (0, 10), (1, 9)]:
        valid = sub[sub['F063'].between(min_val, max_val)]
        if len(valid) == 0:
            continue
        pct = (valid['F063'] == 10).mean() * 100 if max_val == 10 else 0
        # For 0-10, check % of 10 out of 0-10 valid
        if min_val == 0:
            valid_10 = sub[sub['F063'].between(0, 10)]
            pct = (valid_10['F063'] == 10).mean() * 100
            label2 = f"valid range 0-10"
        elif min_val == 1:
            label2 = f"valid range 1-10"
        else:
            label2 = f"valid range 1-9"
        print(f"  {label2}: N={len(valid)}, %10={pct:.4f}% (round={round(pct)}, floor={int(pct)})")

    # Check the full distribution
    f063_vals = sub['F063'].value_counts().sort_index()
    print(f"  F063 distribution: {dict(f063_vals.items())}")

print("=== KEY PROBLEMATIC CELLS ===")
check_cell('IND', 2, 44, wvs)
check_cell('IND', 3, 56, wvs)
check_cell('DEU', 3, None, wvs)
check_cell('MEX', 3, 50, wvs)
check_cell('RUS', 3, 19, wvs)
check_cell('ZAF', 1, 50, wvs)
check_cell('ZAF', 2, 74, wvs)

# Check if any of these have -2 as a "valid near-10" code
print("\n=== CHECKING -2 CODE ===")
for country in ['IND', 'MEX', 'RUS', 'ZAF']:
    for wave in [1, 2, 3]:
        sub = wvs[(wvs['COUNTRY_ALPHA'] == country) & (wvs['S002VS'] == wave)]
        if len(sub) > 0:
            neg2 = len(sub[sub['F063'] == -2])
            if neg2 > 0:
                print(f"{country} W{wave}: F063=-2 count={neg2}")

# Check if RUS 1995 has any valid codes slightly different
print("\n=== RUSSIA 1995 DETAILED ===")
rus = wvs[(wvs['COUNTRY_ALPHA'] == 'RUS') & (wvs['S002VS'] == 3)]
print(f"Russia W3 F063 distribution:")
print(rus['F063'].value_counts().sort_index())
valid = rus[rus['F063'].between(1, 10)]
print(f"Valid N={len(valid)}, %10={(valid['F063']==10).mean()*100:.4f}%")

# DEU_EAST 1995 - what if we use different G006 values?
print("\n=== DEU_EAST 1995 - DIFFERENT G006 CODES ===")
deu_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'DEU') & (wvs['S002VS'] == 3)]
print(f"DEU W3 G006 distribution:")
print(deu_w3['G006'].value_counts().sort_index())
print(f"DEU W3 total: {len(deu_w3)}")

for g006_set, label in [
    ([2], 'G006=2 only'),
    ([3], 'G006=3 only'),
    ([2,3], 'G006=[2,3]'),
    ([2,3,-4], 'G006=[2,3,-4]'),
]:
    sub = deu_w3[deu_w3['G006'].isin(g006_set)]
    valid = sub[sub['F063'].between(1, 10)]
    if len(valid) > 0:
        pct = (valid['F063'] == 10).mean() * 100
        print(f"  {label}: N={len(valid)}, %10={pct:.4f}% (round={round(pct)}, floor={int(pct)})")
