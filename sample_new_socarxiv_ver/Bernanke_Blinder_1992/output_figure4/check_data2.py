import pandas as pd
import numpy as np

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
mask = (df.index >= '1959-01-01') & (df.index <= '1978-12-31')

# Check bank_credit_total more carefully - is it changes or levels?
# Compare with bank_loans + bank_investments
sub = df.loc[mask]
print("bank_credit_total first 10:")
print(sub['bank_credit_total'].head(10))
print()

# If bank_credit_total is changes, we need to reconstruct levels
# Check log_bank_credit_total
print("log_bank_credit_total first 10:")
print(sub['log_bank_credit_total'].head(10))
print()

# Try computing total credit = loans + investments
computed_credit = sub['bank_loans'] + sub['bank_investments']
print("bank_loans + bank_investments first 10:")
print(computed_credit.head(10))
print()

# Securities should be total credit - loans = investments
print("bank_investments first 10:")
print(sub['bank_investments'].head(10))
print()

# Check when bank_deposits_total starts being available
deposits_avail = sub['bank_deposits_total'].dropna()
print(f"bank_deposits_total available from {deposits_avail.index[0]} to {deposits_avail.index[-1]}")
print(f"Number of observations: {len(deposits_avail)}")
print()

# Check log columns that are pre-computed
for c in ['log_bank_loans_real', 'log_bank_deposits_check_real', 'log_bank_investments_real']:
    vals = sub[c].dropna()
    if len(vals) > 0:
        print(f"{c}: available from {vals.index[0]} to {vals.index[-1]}, n={len(vals)}")
        print(f"  range: {vals.min():.4f} to {vals.max():.4f}")
    else:
        print(f"{c}: all missing")
print()

# Can we use bank_investments as a proxy for securities?
# Paper says securities = total loans & investments - total loans
# But we have bank_investments directly which is investment securities
# Let's check if bank_loans + bank_investments makes sense as total credit
print("Computed credit (loans + investments) vs bank_credit_total:")
print("loans + investments:", (sub['bank_loans'] + sub['bank_investments']).head(5).values)
print("bank_credit_total:", sub['bank_credit_total'].head(5).values)
print()
print("Ratio:", ((sub['bank_loans'] + sub['bank_investments']) / sub['bank_credit_total']).head(5).values)
print()

# Check bank_deposits_check as fallback for deposits
print("bank_deposits_check first and last 5:")
print(sub['bank_deposits_check'].head(5))
print(sub['bank_deposits_check'].tail(5))
