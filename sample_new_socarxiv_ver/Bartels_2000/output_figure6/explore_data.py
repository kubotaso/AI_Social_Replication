import pandas as pd
df = pd.read_csv('anes_cumulative.csv', usecols=['VCF0004','VCF0301','VCF0707','VCF0902'], low_memory=False)
# Check unique values of VCF0902 for years 1970-1996
for yr in [1970, 1974, 1976, 1978, 1980]:
    sub = df[df['VCF0004'] == yr]
    print(f'Year {yr}: VCF0902 unique = {sorted(sub["VCF0902"].dropna().unique())}')
print()
# Check years available
cong_years = [1952,1956,1958,1960,1962,1964,1966,1968,1970,1972,1974,1976,1978,1980,1982,1984,1986,1988,1990,1992,1994,1996]
for yr in cong_years:
    sub = df[(df['VCF0004']==yr) & (df['VCF0707'].isin([1,2]))]
    n_902 = sub['VCF0902'].notna().sum()
    print(f'{yr}: voters={len(sub)}, with VCF0902={n_902}')
