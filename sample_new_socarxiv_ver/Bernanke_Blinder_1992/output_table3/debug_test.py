import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

dep = 'log_consumption_real'
rhs = ['log_cpi', 'log_m1', 'log_m2', 'cpbill_long', 'term', 'funds_rate']
nlags = 6

all_v = [dep] + rhs
sub = df[all_v].copy()

# Create lags
ld = {}
for v in all_v:
    for lag in range(1, nlags + 1):
        ld[f'{v}_L{lag}'] = sub[v].shift(lag)

ldf = pd.DataFrame(ld, index=sub.index)
full = pd.concat([sub[[dep]], ldf], axis=1)
full = full.loc['1961-07':'1989-12'].dropna()

y = full[dep]
xc = [c for c in full.columns if c != dep]
X = sm.add_constant(full[xc])
m = sm.OLS(y, X).fit()

print(f'N: {int(m.nobs)}')
print(f'R2: {m.rsquared:.6f}')

# Test each RHS variable
for test_var in rhs:
    lag_names = [f'{test_var}_L{i}' for i in range(1, nlags + 1)]
    R = np.zeros((nlags, len(m.params)))
    for i, ln in enumerate(lag_names):
        R[i, list(m.params.index).index(ln)] = 1.0
    f = m.f_test(R)
    print(f'{test_var}: F={float(f.fvalue):.4f}, p={float(f.pvalue):.4f}')

# Print cpbill_long lag coefficients for consumption
print('\nCPBILL lag coefficients:')
for i in range(1, nlags + 1):
    ln = f'cpbill_long_L{i}'
    print(f'  {ln}: coef={m.params[ln]:.8f}, t={m.tvalues[ln]:.4f}, p={m.pvalues[ln]:.4f}')

print('\n--- Checking consumption_real data ---')
# Check if consumption_real might have issues
cons = df.loc['1985-01':'1985-06', ['consumption_real', 'consumption_nominal', 'cpi']]
print(cons)
print()
print('log_consumption_real stats:')
lc = df.loc['1961-07':'1989-12', 'log_consumption_real']
print(f'  min: {lc.min():.4f}, max: {lc.max():.4f}, mean: {lc.mean():.4f}')
print(f'  first diff stats:')
d = lc.diff()
print(f'  mean: {d.mean():.6f}, std: {d.std():.6f}')
