import pandas as pd
df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')

# Compare cpaper_6m and cpaper_6m_long where both exist
sub = df.loc['1970-01':'1989-12', ['cpaper_6m', 'cpaper_6m_long', 'tbill_6m']].dropna()
print('N where both exist:', len(sub))
print('Correlation:', sub['cpaper_6m'].corr(sub['cpaper_6m_long']))
diff = sub['cpaper_6m'] - sub['cpaper_6m_long']
print('Mean diff (cpaper_6m - cpaper_6m_long):', diff.mean())
print('Max abs diff:', diff.abs().max())
print()

# cpbill comparison
cpbill_new = sub['cpaper_6m'] - sub['tbill_6m']
cpbill_old = sub['cpaper_6m_long'] - sub['tbill_6m']
print('Mean cpbill (cpaper_6m - tbill_6m):', cpbill_new.mean())
print('Mean cpbill (cpaper_6m_long - tbill_6m):', cpbill_old.mean())
print('Corr:', cpbill_new.corr(cpbill_old))

# Spot checks
for d in ['1980-01-01','1985-01-01','1989-06-01']:
    r = df.loc[d]
    print(f'{d}: cp6m={r["cpaper_6m"]:.2f}, cp6m_long={r["cpaper_6m_long"]:.2f}, diff={r["cpaper_6m"]-r["cpaper_6m_long"]:.2f}')

# Check what cpbill (original) is
print()
print('cpbill (original) at 1980-01:', df.loc['1980-01-01','cpbill'])
print('cpbill_long at 1980-01:', df.loc['1980-01-01','cpbill_long'])
print('cpbill = cpaper_6m - tbill_6m?', abs(df.loc['1980-01-01','cpbill'] - (df.loc['1980-01-01','cpaper_6m'] - df.loc['1980-01-01','tbill_6m'])) < 0.01)
