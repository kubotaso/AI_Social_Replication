#!/usr/bin/env python3
"""Check which countries are in WVS waves 2-3 and EVS."""
import pandas as pd
import os

# WVS countries in waves 2 and 3
df = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                  usecols=['S002VS', 'COUNTRY_ALPHA'], low_memory=False)

w2 = set(df[df['S002VS']==2]['COUNTRY_ALPHA'].unique())
w3 = set(df[df['S002VS']==3]['COUNTRY_ALPHA'].unique())
print(f"WVS Wave 2 (1990-91): {len(w2)} countries")
print(f"  {sorted(w2)}")
print(f"\nWVS Wave 3 (1995-98): {len(w3)} countries")
print(f"  {sorted(w3)}")

# EVS countries
evs = pd.read_csv('data/EVS_1990_wvs_format.csv', usecols=['COUNTRY_ALPHA'])
evs_c = set(evs['COUNTRY_ALPHA'].unique())
print(f"\nEVS 1990: {len(evs_c)} countries")
print(f"  {sorted(evs_c)}")

# Combined (latest per country)
all_c = w2 | w3 | evs_c
print(f"\nAll combined: {len(all_c)} countries")
print(f"  {sorted(all_c)}")

# Now check WB data availability
wb = pd.read_csv('data/world_bank_indicators.csv')
gdp_c = set(wb[(wb['indicator']=='NY.GNP.PCAP.PP.CD') & (wb['YR1995'].notna())]['economy'])
ind_c = set(wb[(wb['indicator']=='SL.IND.EMPL.ZS') & (wb['YR1991'].notna())]['economy'])
print(f"\nGNP 1995: {len(gdp_c)} countries")
print(f"Industry 1991: {len(ind_c)} countries")

# Intersection
have_wb = all_c & gdp_c & ind_c
no_wb = all_c - have_wb
print(f"\nCountries with both factor scores and WB data: {len(have_wb)}")
print(f"  {sorted(have_wb)}")
print(f"\nCountries WITHOUT WB data: {len(no_wb)}")
print(f"  {sorted(no_wb)}")

# Non-paper countries to exclude
non_paper = {'ALB', 'MNE', 'MLT', 'SLV', 'PAK'}  # PAK might be in paper
paper_with_wb = have_wb - non_paper
print(f"\nPaper countries with WB data: {len(paper_with_wb)}")
print(f"  {sorted(paper_with_wb)}")

# Check if DEU needs to be split
# In the paper: E. Germany + W. Germany = 2 observations
# In our data: DEU = 1 observation
# If paper splits, that's paper_with_wb + 1 (for the extra Germany)
print(f"\nIf Germany split: {len(paper_with_wb) + 1} observations")
