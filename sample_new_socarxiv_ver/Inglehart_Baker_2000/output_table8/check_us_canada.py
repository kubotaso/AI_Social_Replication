import pandas as pd

# Check if US/Canada are in WVS Wave 1
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F001','S017'])
w1 = wvs[wvs['S002VS']==1]
for cc in ['USA','CAN']:
    sub = w1[w1['COUNTRY_ALPHA']==cc]
    valid = sub['F001'][sub['F001']>0]
    if len(valid) > 0:
        print(f'{cc} in WVS W1: n={len(valid)}, Often={100*(valid==1).mean():.2f}%')
    else:
        print(f'{cc} NOT in WVS W1 (or no F001)')

# Check WVS Wave 2 for Canada and US
w2 = wvs[wvs['S002VS']==2]
for cc in ['USA','CAN']:
    sub = w2[w2['COUNTRY_ALPHA']==cc]
    valid = sub['F001'][sub['F001']>0]
    if len(valid) > 0:
        print(f'{cc} in WVS W2: n={len(valid)}, Often={100*(valid==1).mean():.2f}%')
    else:
        print(f'{cc} NOT in WVS W2')

# The 1981 US data comes from EVS. Let's check if different weighting or
# a different N could explain the discrepancy
# Paper says US 1981 = 48%, EVS gives us 49.15% (uw) and 49.23% (wt)
# This is a consistent 1% overestimate

# Let's also re-check the EVS_1990_wvs_format.csv for US
evs90 = pd.read_csv('data/EVS_1990_wvs_format.csv')
us90 = evs90[evs90['COUNTRY_ALPHA']=='USA']
f = us90['F001']
valid = f[f>0]
print(f'\nUS EVS 1990: n={len(valid)}, Often={100*(valid==1).mean():.2f}%')
# This should be ~48%

# Canada 1981: paper says 38%, EVS longitudinal gives 36.91% (uw), 37.11% (wt)
# Both about 1% below. Could be a different data version.
# The paper may have used the original EVS 1981 release (ZA4438) rather than
# the longitudinal file (ZA4804) which was harmonized differently.
