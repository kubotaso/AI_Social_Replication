#!/usr/bin/env python3
import pandas as pd
import numpy as np

# EVS data
evs = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)

# Check columns in EVS
print('EVS columns:', evs.columns.tolist())
deu_evs = evs[evs['COUNTRY_ALPHA']=='DEU']
if 'S024' in evs.columns:
    print('DEU EVS S024:', sorted(deu_evs['S024'].unique()))
else:
    print('S024 not in EVS')
if 'S003' in evs.columns:
    print('DEU EVS S003:', sorted(deu_evs['S003'].unique()))

# Check F034 coding in WVS codebook or by cross-referencing with F025
# F034 in WVS Time Series: 1=Catholic, 2=Protestant, 3=Other
# F025: 0=None, 1=Catholic, 2=Protestant, etc.
# Let's cross-check using WVS USA data
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F034','F025'],
                   low_memory=False)
wvs = wvs[wvs['S002VS'].isin([2,3])]

# Cross-tabulate F025 vs F034 for USA
usa = wvs[wvs['COUNTRY_ALPHA']=='USA']
usa_f025 = pd.to_numeric(usa['F025'], errors='coerce')
usa_f034 = pd.to_numeric(usa['F034'], errors='coerce')
print('\nUSA F025 vs F034 crosstab:')
ct = pd.crosstab(usa_f025[usa_f025>=0], usa_f034[usa_f034>=0])
print(ct)

# Same for Germany
deu = wvs[wvs['COUNTRY_ALPHA']=='DEU']
deu_f025 = pd.to_numeric(deu['F025'], errors='coerce')
deu_f034 = pd.to_numeric(deu['F034'], errors='coerce')
print('\nDEU F025 vs F034 crosstab:')
ct = pd.crosstab(deu_f025[deu_f025>=0], deu_f034[deu_f034>=0])
print(ct)

# Check India
ind = wvs[wvs['COUNTRY_ALPHA']=='IND']
ind_f025 = pd.to_numeric(ind['F025'], errors='coerce')
ind_f034 = pd.to_numeric(ind['F034'], errors='coerce')
print('\nIND F025 vs F034 crosstab:')
ct = pd.crosstab(ind_f025[ind_f025>=0], ind_f034[ind_f034>=0])
print(ct)

# Check Nigeria
nga = wvs[wvs['COUNTRY_ALPHA']=='NGA']
nga_f025 = pd.to_numeric(nga['F025'], errors='coerce')
nga_f034 = pd.to_numeric(nga['F034'], errors='coerce')
print('\nNGA F025 vs F034 crosstab:')
ct = pd.crosstab(nga_f025[nga_f025>=0], nga_f034[nga_f034>=0])
print(ct)

# For India: F025 6=Hindu, 5=Muslim
# Check
print('\nIND F025 coding:')
ind_f025_valid = ind_f025[ind_f025>=0]
print(ind_f025_valid.value_counts().sort_index())

# Nigeria F025 coding
print('\nNGA F025 coding:')
nga_f025_valid = nga_f025[nga_f025>=0]
print(nga_f025_valid.value_counts().sort_index())

# CHE (Switzerland)
che = wvs[wvs['COUNTRY_ALPHA']=='CHE']
che_f025 = pd.to_numeric(che['F025'], errors='coerce')
che_f034 = pd.to_numeric(che['F034'], errors='coerce')
print('\nCHE F025 vs F034 crosstab:')
ct = pd.crosstab(che_f025[che_f025>=0], che_f034[che_f034>=0])
print(ct)
