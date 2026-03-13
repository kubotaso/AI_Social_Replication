import pandas as pd, numpy as np
from statsmodels.tsa.api import VAR
import warnings; warnings.filterwarnings('ignore')
df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'
fv = {'Industrial production': 'log_industrial_production', 'Capacity utilization': 'log_capacity_utilization', 'Employment': 'log_employment', 'Unemployment rate': 'unemp_male_2554', 'Housing starts': 'log_housing_starts', 'Personal income': 'log_personal_income_real', 'Retail sales': 'log_retail_sales_real', 'Consumption': 'log_consumption_real'}
cv = ['log_cpi', 'log_m1', 'log_m2', 'tbill_3m', 'treasury_10y', 'funds_rate']
ga = {'Industrial production': [36.6,3.1,15.4,8.7,8.0,0.8,27.4], 'Capacity utilization': [39.7,1.3,21.0,3.5,9.5,1.7,23.3], 'Employment': [38.9,7.0,10.5,0.6,9.8,2.7,30.6], 'Unemployment rate': [31.9,7.2,10.5,0.6,9.9,1.9,37.9], 'Housing starts': [28.8,1.4,3.9,1.8,38.6,14.3,11.2], 'Personal income': [48.2,4.3,20.8,0.1,6.9,3.3,16.3], 'Retail sales': [32.4,15.5,5.1,4.4,27.4,1.1,14.1], 'Consumption': [18.2,13.1,16.0,2.2,28.4,5.3,16.8]}
gb = {'Industrial production': [36.3,2.7,11.8,6.5,11.5,3.3,27.8], 'Capacity utilization': [39.9,2.4,12.4,4.5,10.8,5.6,24.3], 'Employment': [41.4,1.8,5.8,0.2,10.4,3.2,37.9], 'Unemployment rate': [44.9,1.3,4.9,1.3,11.6,2.2,33.8], 'Housing starts': [45.2,9.9,8.3,6.3,11.8,9.6,9.0], 'Personal income': [34.5,17.7,7.0,0.5,11.9,14.9,13.4], 'Retail sales': [49.2,6.0,9.9,2.7,16.7,4.1,11.4], 'Consumption': [18.9,21.1,13.2,3.3,11.7,16.4,15.5]}
print('Panel A best per variable:')
for vn, vc in fv.items():
    best = 0; best_cfg = ''
    for lags in [5, 6]:
        for ds in ['1959-01', '1959-07']:
            for t in ['c', 'ct']:
                vl = [vc] + cv
                sa = df.loc[ds:'1989-12', vl].dropna()
                ma = VAR(sa).fit(maxlags=lags, ic=None, trend=t)
                pa = ma.fevd(24).decomp[0, 23, :] * 100
                c = sum(1 for i in range(7) if abs(pa[i] - ga[vn][i]) <= 3)
                if c > best: best = c; best_cfg = f'lags={lags} ds={ds} t={t}'
    print(f'  {vn}: {best}/7 ({best_cfg})')
print()
print('Panel B best per variable:')
for vn, vc in fv.items():
    best = 0; best_cfg = ''
    for lags in [5, 6]:
        for ds in ['1959-01', '1959-07']:
            for t in ['c', 'ct']:
                vl = [vc] + cv
                sb = df.loc[ds:'1979-09', vl].dropna()
                mb = VAR(sb).fit(maxlags=lags, ic=None, trend=t)
                pb = mb.fevd(24).decomp[0, 23, :] * 100
                c = sum(1 for i in range(7) if abs(pb[i] - gb[vn][i]) <= 3)
                if c > best: best = c; best_cfg = f'lags={lags} ds={ds} t={t}'
    print(f'  {vn}: {best}/7 ({best_cfg})')
