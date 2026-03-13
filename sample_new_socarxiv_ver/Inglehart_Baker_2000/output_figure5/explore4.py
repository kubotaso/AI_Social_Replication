#!/usr/bin/env python3
import pandas as pd
import numpy as np

# The paper says "West Germany" - need to check how to distinguish West/East Germany
# In WVS, S003=276 for Germany, S024=2763 for wave 3
# Check EVS for German sub-regions
evs = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)
deu_evs = evs[evs['COUNTRY_ALPHA']=='DEU']
print('DEU EVS rows:', len(deu_evs))
print('DEU EVS S020:', sorted(deu_evs['S020'].unique()))

# Check WVS for Germany - S024 2763 = wave 3 all Germany, but check if 276001/276002 exist
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','S003','S024','COUNTRY_ALPHA','S020'],
                   low_memory=False)

# Check all S024 codes for S003=276
deu_all = wvs[wvs['S003']==276]
print('\nAll DEU S024 codes:')
for wave in sorted(deu_all['S002VS'].unique()):
    wsub = deu_all[deu_all['S002VS']==wave]
    print(f"  Wave {wave}: S024={sorted(wsub['S024'].unique())}, S020={sorted(wsub['S020'].unique())}, n={len(wsub)}")

# Check if there's separate West Germany in EVS
# EVS may already be West Germany only (1990 survey)
# The EVS 1990 would have been done shortly after reunification
# Let's check the combined data approach
print('\nAll unique COUNTRY_ALPHA values for S003=276 in WVS:')
print(deu_all['COUNTRY_ALPHA'].unique())

# Also look for West Germany under a different code
wg = wvs[wvs['S024'].isin([276001, 276002])]
print('\nS024=276001 rows:', len(wg[wg['S024']==276001]))
print('S024=276002 rows:', len(wg[wg['S024']==276002]))

# Also check for S003=900 (West Germany in some codebooks)
wvs_900 = wvs[wvs['S003']==900]
print('\nS003=900 rows:', len(wvs_900))

# Check wave 2 for Germany
deu_w2 = wvs[(wvs['S003']==276) & (wvs['S002VS']==2)]
print('\nDEU wave 2:', len(deu_w2))
print('DEU wave 2 S024:', sorted(deu_w2['S024'].unique()) if len(deu_w2) > 0 else 'none')
