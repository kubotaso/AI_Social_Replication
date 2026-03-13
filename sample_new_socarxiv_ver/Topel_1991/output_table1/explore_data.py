import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')
mask = (df['SEX'] == 1) & (df['AGE'] >= 20) & (df['AGE'] <= 60) & df['DWREAS'].isin([1, 2, 3]) & df['EMPSTAT'].isin([10, 12])
s = df[mask].copy()
print('After male/age/econ/employed:', len(s))
s2 = s[s['DWYEARS'] < 99]
print('After valid tenure:', len(s2))
print('DWYEARS range:', s2['DWYEARS'].min(), '-', s2['DWYEARS'].max())
valid_w = (s2['DWWEEKL'] > 0) & (s2['DWWEEKL'] < 9000) & (s2['DWWEEKC'] > 0) & (s2['DWWEEKC'] < 9000)
wg = s2[valid_w]
print('Valid wage sample:', len(wg))
print('DWLASTWRK in wage sample:')
print(wg['DWLASTWRK'].value_counts().sort_index().to_string())
vu = s2[s2['DWWKSUN'] < 999]
print('Valid unemp weeks sample:', len(vu))
bins = pd.cut(s2['DWYEARS'], bins=[-0.1, 5, 10, 20, 100], labels=['0-5','6-10','11-20','21+'])
print('Tenure bin distribution:')
print(bins.value_counts().sort_index().to_string())
