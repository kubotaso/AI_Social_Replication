#!/usr/bin/env python3
"""Find which 49 countries are in Table 5a."""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

wb = pd.read_csv(os.path.join(BASE_DIR, 'data/world_bank_indicators.csv'))
ind = wb[wb['indicator'] == 'SL.IND.EMPL.ZS']
wb_eco = set(ind['economy'].tolist())

# The 55 countries with PWT data
our_55 = ['ARG', 'AUS', 'AUT', 'BEL', 'BGD', 'BGR', 'BIH', 'BRA', 'CAN', 'CHE',
          'CHL', 'CHN', 'COL', 'CZE', 'DEU_E', 'DEU_W', 'DNK', 'DOM', 'ESP', 'FIN',
          'FRA', 'GBR', 'GHA', 'HRV', 'HUN', 'IND', 'IRL', 'ISL', 'ITA', 'JPN',
          'KOR', 'MEX', 'MKD', 'NGA', 'NLD', 'NOR', 'NZL', 'PAK', 'PER', 'PHL',
          'POL', 'PRI', 'PRT', 'ROU', 'RUS', 'SRB', 'SVK', 'SVN', 'SWE', 'TUR',
          'TWN', 'URY', 'USA', 'VEN', 'ZAF']

# Note DEU (unified) is in WB, but not DEU_E/DEU_W individually
# BIH, HRV, MKD, SRB, SVN are in WB (they were separate after 1991)
for c in our_55:
    in_wb = c in wb_eco
    if not in_wb:
        print(f'{c} NOT in WB industry data')

# Now try to run the actual regression with these 55 countries
# to see what we get versus the paper's 49
# The paper says N=49 for Models 1,3,4,5,6 and N=46 for Model 2

# The 55 - 49 = 6 missing. These are likely countries where either:
# a) the industry employment data is truly missing in the original 1980 ILO/WB data
# b) certain ex-communist/developing countries had no data in 1980

# From the previous attempts, the best N was 51 or 49 depending on exclusions
# Let's check: if we exclude BIH, MKD, SRB (former Yugoslavia that might lack independent 1980 data)
# + GHA, NGA (African), that's 5 more = 50... close

# Alternative: look at what's in the model 1 regression
# From the table (N=49), and we have GDP+Industrial, the missing 6 are:
# probably: BGD, PAK, GHA, NGA, BIH, MKD or SRB
# Actually: paper has 65 societies. 65-49=16 missing.
# We already have 10 without PWT: ARM, AZE, BLR, EST, GEO, LTU, LVA, MDA, NIR, UKR
# That's only 10. The paper says "Reduced Ns reflect missing data on independent variables"
# So 65-16=49 means 16 lack either GDP or industry data.

# Additional 6 that might lack GDP or industry data:
# IND/BGD/PAK might have PWT but the paper may use a different source
# GHA - has PWT 1980 data (976) but might be excluded for other reasons
# NGA - has PWT 1980 data too

# Let me try: could it be that "1980" industry data only exists for OECD-type countries?
# And the paper just uses ILO 1980 data directly (not WB 1991 approximation)?

# Print out the WB industry data for all years to see which have 1980 data
ind_all = ind[['economy'] + [c for c in ind.columns if c.startswith('YR')]]
# Find first year with data for each country
for _, row in ind_all.iterrows():
    eco = row['economy']
    first_yr = None
    for yr in range(1975, 2000):
        col = f'YR{yr}'
        if col in row and pd.notna(row[col]):
            first_yr = yr
            break
    if first_yr and first_yr <= 1985:
        print(f'{eco}: first year with data = {first_yr}, YR1980={row.get("YR1980")}')
    elif first_yr:
        print(f'{eco}: first year = {first_yr} (no 1980 data)')
