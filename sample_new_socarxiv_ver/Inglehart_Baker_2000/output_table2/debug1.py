#!/usr/bin/env python3
"""Debug factor scores and correlations for Table 2."""
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_factor_analysis import compute_nation_level_factor_scores, COUNTRY_NAMES

# Get factor scores
scores, loadings, country_means = compute_nation_level_factor_scores(waves_wvs=[2, 3], include_evs=True)

# Check: Sweden should be high secular-rational (positive), Nigeria should be low (negative)
print("=== Factor Score Direction Check ===")
for c in ['SWE', 'JPN', 'DEU', 'NGA', 'IND', 'USA']:
    row = scores[scores['COUNTRY_ALPHA'] == c]
    if len(row) > 0:
        print(f"  {c}: trad_secrat={row['trad_secrat'].values[0]:.3f}")

# The paper's factor scores have:
# Secular-rational = positive (Japan, Sweden high)
# Traditional = negative (Nigeria, etc. low)
#
# The correlations in Table 2 are reported as POSITIVE for traditional items.
# This means the paper correlates each item with the TRADITIONAL end.
# If our trad_secrat is secular-rational = positive, then:
#   correlation with traditional = correlation with (-trad_secrat)
#
# But wait - let me check if the issue is that the factor analysis is not
# properly reproducing the original paper's factor structure.
#
# The key items for dimension 1 should be:
# A006 (God important), A042 (obedience), F120 (abortion), G006 (national pride), E018 (respect authority)
# These should all load heavily on the same factor.

print("\n=== Factor Loadings ===")
print(loadings.to_string(index=False))

# Check the A006 loading - it should be the strongest on trad/secrat
# In the paper, A006 loads .91 on dimension 1
print(f"\nA006 loading on trad_secrat: {loadings[loadings['item']=='A006']['trad_secrat'].values[0]:.3f}")
print("(Paper says .91)")

# Issue: A006 loading is only 0.074! That's very wrong.
# The problem might be in how A006 is recoded or scaled.
# Let me check the country means for A006
print("\n=== Country means for A006 (God important, 1-10) ===")
print("(After recoding: higher = more traditional)")
for c in ['SWE', 'JPN', 'NGA', 'IND', 'USA', 'BRA', 'CHN']:
    if c in country_means.index:
        print(f"  {c}: {country_means.loc[c, 'A006']:.2f}")

# The issue is clear: A006 has a very low loading (.074 vs .91 in paper)
# This suggests the factor analysis is not working correctly for this item.
# Let me check if A006 varies enough across countries
print(f"\nA006 stats: mean={country_means['A006'].mean():.2f}, std={country_means['A006'].std():.2f}")
print(f"A006 range: {country_means['A006'].min():.2f} to {country_means['A006'].max():.2f}")

# Let's also check correlations between country means of all factor items
print("\n=== Inter-item correlations at country level ===")
corr_matrix = country_means.corr()
print(corr_matrix.to_string())
