"""
Extract EVS 1981 wave data from ZA4804 (EVS Longitudinal Data File 1981-2008).
"""
import pandas as pd
import numpy as np
import os

evs_long_path = '/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/OldFiles/Replication_Claude_IB/data/ZA4804_v3-1-0.dta/ZA4804_v3-1-0.dta'

# Read only the columns we need
print("Loading EVS Longitudinal data...")
evs = pd.read_stata(evs_long_path, convert_categoricals=False,
                     columns=['S002EVS', 's002vs', 'S003', 'S003A', 'F001', 'S017', 'S020'])
print(f"Shape: {evs.shape}")

# Check wave variable
print(f"\nS002EVS values: {sorted(evs['S002EVS'].unique())}")
print(f"s002vs values: {sorted(evs['s002vs'].unique())}")

# Wave 1 = 1981
w1 = evs[evs['S002EVS'] == 1]
print(f"\nEVS Wave 1 (1981): {len(w1)} rows")
print(f"S003 countries: {sorted(w1['S003'].unique())}")
print(f"Years: {sorted(w1['S020'].unique())}")

# Check F001 availability in Wave 1
print(f"\nF001 distribution in Wave 1:")
print(w1['F001'].value_counts().sort_index())
print(f"F001 valid (>0): {(w1['F001'] > 0).sum()}")

# Country mapping for S003
# Standard WVS country codes
country_map = {
    36: 'Australia', 56: 'Belgium', 124: 'Canada', 208: 'Denmark',
    246: 'Finland', 250: 'France', 276: 'Germany', 826: 'Great Britain',
    352: 'Iceland', 372: 'Ireland', 380: 'Italy', 392: 'Japan',
    410: 'South Korea', 484: 'Mexico', 528: 'Netherlands', 578: 'Norway',
    724: 'Spain', 752: 'Sweden', 840: 'United States',
    348: 'Hungary', 32: 'Argentina', 710: 'South Africa',
    909: 'Northern Ireland',
}

# Calculate % "Often" for each country in Wave 1
print(f"\n{'Country':<25} {'S003':>6} {'N_valid':>8} {'N_often':>8} {'Pct':>8} {'Rounded':>8}")
print("-" * 70)

for s003 in sorted(w1['S003'].unique()):
    sub = w1[w1['S003'] == s003]
    valid = sub[sub['F001'] > 0]['F001']
    if len(valid) > 0:
        pct = (valid == 1).mean() * 100
        name = country_map.get(int(s003), f'Unknown ({s003})')
        n_often = (valid == 1).sum()
        print(f"{name:<25} {int(s003):>6} {len(valid):>8} {n_often:>8} {pct:>8.2f} {round(pct):>8}")

# Also check S003A for split countries (e.g., West/East Germany, Northern Ireland)
print(f"\nS003A in Wave 1: {sorted(w1['S003A'].unique())}")

# Germany split in EVS Wave 1
# S003A might have West Germany and East Germany codes
ger = w1[w1['S003'] == 276]
if len(ger) > 0:
    print(f"\nGermany Wave 1: {len(ger)} rows")
    print(f"S003A values: {sorted(ger['S003A'].unique())}")
    # In 1981, only West Germany participated (East Germany was still communist)
    valid = ger[ger['F001'] > 0]['F001']
    if len(valid) > 0:
        pct = (valid == 1).mean() * 100
        print(f"  West Germany (all): {pct:.2f}% ({round(pct)}), n={len(valid)}")

# Northern Ireland
nir = w1[w1['S003'] == 909]
if len(nir) > 0:
    valid = nir[nir['F001'] > 0]['F001']
    pct = (valid == 1).mean() * 100
    print(f"  Northern Ireland: {pct:.2f}% ({round(pct)}), n={len(valid)}")

# Also check if GB includes Northern Ireland or if they're separate
gb = w1[w1['S003'] == 826]
if len(gb) > 0:
    valid = gb[gb['F001'] > 0]['F001']
    pct = (valid == 1).mean() * 100
    print(f"  Great Britain (S003=826): {pct:.2f}% ({round(pct)}), n={len(valid)}")
    # Check S003A
    print(f"  GB S003A values: {sorted(gb['S003A'].unique())}")

# Try weighted
print("\n\nWeighted results:")
print(f"{'Country':<25} {'Unweighted':>12} {'S017_weighted':>14}")
print("-" * 55)

for s003 in sorted(w1['S003'].unique()):
    sub = w1[w1['S003'] == s003]
    valid_mask = sub['F001'] > 0
    valid = sub[valid_mask]['F001']
    weights = sub[valid_mask]['S017'].fillna(1)
    if len(valid) > 0:
        pct_unw = (valid == 1).mean() * 100
        pct_w = ((valid == 1) * weights).sum() / weights.sum() * 100
        name = country_map.get(int(s003), f'Unknown ({s003})')
        print(f"{name:<25} {pct_unw:>10.2f}({round(pct_unw):>3}) {pct_w:>12.2f}({round(pct_w):>3})")
