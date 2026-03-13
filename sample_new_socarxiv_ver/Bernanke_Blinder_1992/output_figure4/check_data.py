import pandas as pd
import numpy as np

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
mask = (df.index >= '1959-01-01') & (df.index <= '1978-12-31')
cols = ['bank_loans','bank_credit_total','bank_deposits_total','bank_deposits_check','bank_investments','bank_securities','cpi']
sub = df.loc[mask, cols]
print('Missing values:')
print(sub.isnull().sum())
print()
print('First 5:')
print(sub.head().to_string())
print()
print('Last 5:')
print(sub.tail().to_string())
print()

# Check if these are levels or changes
print('Are these levels or changes?')
print('bank_loans range:', sub['bank_loans'].min(), 'to', sub['bank_loans'].max())
print('bank_credit_total range:', sub['bank_credit_total'].min(), 'to', sub['bank_credit_total'].max())
print('bank_deposits_total range:', sub['bank_deposits_total'].min(), 'to', sub['bank_deposits_total'].max())
print('bank_deposits_check range:', sub['bank_deposits_check'].min(), 'to', sub['bank_deposits_check'].max())

# Check if bank_credit_total looks like changes (small numbers) or levels (large numbers)
print()
print('Checking if columns are levels or changes...')
for c in cols[:-1]:
    vals = sub[c].dropna()
    if len(vals) > 0:
        print(f'{c}: mean={vals.mean():.2f}, std={vals.std():.2f}, min={vals.min():.2f}, max={vals.max():.2f}')
