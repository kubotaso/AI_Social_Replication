import pandas as pd

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv', usecols=['COUNTRY_ALPHA','S002VS','S020'], low_memory=False)
pol = wvs[wvs['COUNTRY_ALPHA']=='POL']
print('Poland in WVS:')
print(pol.groupby('S002VS').agg({'S020': ['min','max','count']}).to_string())
print()
evs = pd.read_csv('data/EVS_1990_wvs_format.csv')
pol_evs = evs[evs['COUNTRY_ALPHA']=='POL']
print('Poland in EVS:')
print(pol_evs.shape[0], 'rows')
if len(pol_evs) > 0:
    print('Year:', pol_evs['S020'].min(), '-', pol_evs['S020'].max())
