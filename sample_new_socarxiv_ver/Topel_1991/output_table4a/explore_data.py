import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')
print('Age range:', df['age'].min(), '-', df['age'].max())
print('Govt:', df['govt_worker'].value_counts().to_dict())
print('Sex:', df['sex'].value_counts().to_dict())
print('White:', df['white'].value_counts().to_dict())
print('T0:', (df['tenure_topel']==0).sum())
print('T1+:', (df['tenure_topel']>=1).sum())
print()
for yr in [1973, 1974, 1975, 1976, 1977]:
    vals = df.loc[df['year']==yr, 'education_clean']
    print(f'Year {yr}: min={vals.min():.0f}, max={vals.max():.0f}, mean={vals.mean():.1f}')
