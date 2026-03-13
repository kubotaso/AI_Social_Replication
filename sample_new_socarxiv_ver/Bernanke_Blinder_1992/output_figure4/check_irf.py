import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'
df = df.loc['1959-01':'1978-12'].copy()

cpi = df['cpi']
log_dep_real = np.log(df['bank_deposits_check']) - np.log(cpi)

df_var = pd.DataFrame({
    'funds_rate': df['funds_rate'],
    'unemp': df['unemp_male_2554'],
    'log_cpi': df['log_cpi'],
    'log_dep': log_dep_real
}, index=df.index).dropna()

r = VAR(df_var).fit(maxlags=6, trend='c')
irf = r.irf(24)

# Check what attributes are available
print("IRF attributes:", [a for a in dir(irf) if not a.startswith('_')])
print()

# Check irfs
print("irf.irfs shape:", irf.irfs.shape)
print("irf.irfs[0,:,0]:", irf.irfs[0, :, 0])
print()

# Check orth_irfs if exists
if hasattr(irf, 'orth_irfs'):
    print("irf.orth_irfs shape:", irf.orth_irfs.shape)
    print("irf.orth_irfs[0,:,0]:", irf.orth_irfs[0, :, 0])
else:
    print("No orth_irfs attribute")
print()

# Check orth_ma_rep
if hasattr(irf, 'orth_ma_rep'):
    print("Has orth_ma_rep method")
else:
    print("No orth_ma_rep")

# Manual orthogonalized computation
sigma = r.sigma_u.values
P = np.linalg.cholesky(sigma)
ma = r.ma_rep(24)

print("\nManual orthogonalized IRF (MA @ P):")
for h in [0, 1, 12, 24]:
    orth_h = ma[h] @ P
    print(f"  h={h}: response to shock 0: {orth_h[:, 0]}")

print()
print("Compare with irf.irfs:")
for h in [0, 1, 12, 24]:
    print(f"  h={h}: irf.irfs[h,:,0] = {irf.irfs[h, :, 0]}")

# Check if irfs is just the MA rep
print("\nAre irfs = MA rep?")
for h in [0, 1, 12]:
    print(f"  h={h}: ma[h,:,0] = {ma[h,:,0]}")
    print(f"  h={h}: irfs[h,:,0] = {irf.irfs[h,:,0]}")
    print(f"  equal: {np.allclose(ma[h,:,0], irf.irfs[h,:,0])}")
