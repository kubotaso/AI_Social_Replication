import pandas as pd
df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
print('ffbond at 1970-01:', df.loc['1970-01-01', 'ffbond'])
print('funds - 10y:', df.loc['1970-01-01', 'funds_rate'] - df.loc['1970-01-01', 'treasury_10y'])
diff = df['funds_rate'] - df['treasury_10y'] - df['ffbond']
print('Max abs diff:', diff.abs().max())
print('Mean ffbond:', df.loc['1959-08':'1979-09', 'ffbond'].mean())
print('Mean funds:', df.loc['1959-08':'1979-09', 'funds_rate'].mean())
