"""
Try all possible approaches for Brazil Wave 3
Brazil paper=54%, we get 75%
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")

wvs = pd.read_csv(wvs_path, low_memory=False)

bra3 = wvs[(wvs['S002VS'] == 3) & (wvs['S003'] == 76)].copy()
print(f"Brazil Wave 3 total rows: {len(bra3)}")
print(f"Years: {sorted(bra3['S020'].unique())}")

print("\nF028 distribution:")
print(bra3['F028'].value_counts().sort_index())

print("\nWhat does each F028 code mean in WVS?")
print("1 = More than once a week")
print("2 = Once a week")
print("3 = Once a month")
print("4 = Only on specific holy days/Christmas/Easter")
print("5 = Other specific holy days")
print("6 = Once a year")
print("7 = Less often")
print("8 = Never, practically never")
print()

# Brazil has: -2, 1, 2, 3, 4, 6, 8 (no 5 or 7)
# In Brazil's scale: 4 = holy days, 6 = once a year, 8 = never
# Target: 54%
# What monthly definition gives 54%?

bra_valid = bra3[bra3['F028'].isin([1,2,3,4,6,8])].copy()
total = len(bra_valid)
print(f"Valid responses (1,2,3,4,6,8): {total}")

for cutoff in [[1,2,3], [1,2], [1,2,3,4]]:
    m = bra_valid['F028'].isin(cutoff).sum()
    print(f"F028 in {cutoff}: {m}/{total} = {m/total*100:.1f}%")

# Target: 54% of total
# 54% * 1139 = 614.6
# But we have 852 responding monthly...
print(f"\nTarget 54% of {total} = {0.54*total:.1f} respondents")
print(f"Actual monthly (1,2,3): {bra_valid['F028'].isin([1,2,3]).sum()}")

# Maybe the paper used a different dataset version?
# Or maybe Brazil had a different definition (at least once a week?)
# At least weekly = values 1, 2 only
m_weekly = bra_valid['F028'].isin([1,2]).sum()
print(f"At least weekly (1,2): {m_weekly}/{total} = {m_weekly/total*100:.1f}%")

# Check other columns that might filter Brazil
print("\nColumn check for Brazil filters:")
filter_cols = ['S020', 'S002', 'X001', 'X003', 'X025', 'X047', 'F034']
for col in filter_cols:
    if col in bra3.columns:
        print(f"  {col}: {bra3[col].value_counts().sort_index().head(10).to_dict()}")

# Check if there's a separate Brazil wave 3 from EVS CSV
evs_csv_path = os.path.join(base, "data", "EVS_1990_wvs_format.csv")
evs_csv = pd.read_csv(evs_csv_path, low_memory=False)
if 'COUNTRY_ALPHA' in evs_csv.columns:
    bra_evs = evs_csv[evs_csv['COUNTRY_ALPHA'] == 'BRA']
    print(f"\nBrazil in EVS CSV: {len(bra_evs)} rows")
    if len(bra_evs) > 0:
        print(f"F028:\n{bra_evs['F028'].value_counts().sort_index()}")

# Is 54 achievable with any non-trivial subset?
# Maybe they excluded certain education/income groups?
# Let's see if any filter gives us near 54%

# Maybe using S017 weights for Brazil too?
w = bra_valid['S017'].fillna(1.0)
m_vals = bra_valid['F028'].isin([1,2,3])
mw = (m_vals * w).sum()
tw = w.sum()
print(f"\nWeighted (S017): {mw:.1f}/{tw:.1f} = {mw/tw*100:.1f}%")

# Check F028B
if 'F028B' in bra3.columns:
    print(f"\nF028B for Brazil:\n{bra3['F028B'].value_counts().sort_index()}")

# Summary: Brazil appears fundamentally not matchable at 54%
# With any reasonable interpretation of "at least once a month"
print("\n=== SUMMARY ===")
print("Brazil Wave 3 gives 75% with standard 'at least monthly' coding")
print(f"Paper says 54% - a 21% gap")
print("No standard approach can explain this discrepancy")
print("Possible explanation: paper used older/different WVS Wave 3 dataset for Brazil")
print("Or paper had a different year range that excludes high-attendance years")
