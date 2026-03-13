import pandas as pd
import numpy as np

df = pd.read_csv('gss1993_clean.csv')
df['ethnic'] = pd.to_numeric(df['ethnic'], errors='coerce')

# Check ethnic distribution
print('Ethnic value counts:')
print(df['ethnic'].value_counts().sort_index().to_dict())

# Current Hispanic: ethnic in [17, 22, 25]
# 17=Mexico, 22=Puerto Rico, 25=other Spanish
# But maybe the paper includes more codes?
# Check: what codes are associated with Hispanic in GSS?
# Standard Hispanic codes in GSS ETHNIC variable:
# 17=Mexico, 22=Puerto Rico, 25=Other Spanish
# Some versions also use: 2=Austrian? No.
# Actually, let me check what % are Hispanic
hisp = df['ethnic'].isin([17, 22, 25])
print(f'\nHispanic (17,22,25): {hisp.sum()} ({hisp.mean()*100:.1f}%)')

# Paper footnote 4 says: "4 percent are Hispanic American"
# Total sample ~1606, so ~64 Hispanic
print(f'Expected ~64 Hispanic (4% of 1606)')

# Also check race==3 (other race)
df['race'] = pd.to_numeric(df['race'], errors='coerce')
print(f'\nOther race (race==3): {(df["race"]==3).sum()}')
print(f'Race distribution: {df["race"].value_counts().sort_index().to_dict()}')

# The paper says 11% are Black (race==2)
black_pct = (df['race']==2).mean() * 100
print(f'Black %: {black_pct:.1f}% (paper says 11%)')
