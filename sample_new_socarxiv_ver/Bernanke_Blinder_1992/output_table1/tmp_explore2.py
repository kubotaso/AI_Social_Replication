"""Check capacity utilization column."""
import pandas as pd
import numpy as np
df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')

print("capacity_utilization (raw):", df.loc['1970-01':'1970-03', 'capacity_utilization'].values)
print("log_capacity_utilization:", df.loc['1970-01':'1970-03', 'log_capacity_utilization'].values)
print("np.log(capacity_utilization):", np.log(df.loc['1970-01':'1970-03', 'capacity_utilization'].values))
print()

# So is log_capacity_utilization = log(capacity_utilization)?
raw = df['capacity_utilization'].dropna()
logged = df['log_capacity_utilization'].dropna()
check = np.log(raw)
print(f"Max diff between log(raw) and logged column: {(check - logged.loc[check.index]).abs().max():.10f}")

# What the best code uses: log_capacity_utilization
# The paper says "capacity utilization" is in levels, NOT logs
# So maybe we should use capacity_utilization (raw) instead of log_capacity_utilization
