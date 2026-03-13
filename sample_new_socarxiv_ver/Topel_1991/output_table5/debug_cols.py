import pandas as pd
df = pd.read_csv('data/psid_panel.csv')
print('All columns:', df.columns.tolist())
print()
print('Region cross-tab:')
for r in [1,2,3,4,5,6]:
    sub = df[df['region']==r]
    if len(sub) > 0:
        ne = sub['region_ne'].mean()
        nc = sub['region_nc'].mean()
        s = sub['region_south'].mean()
        w = sub['region_west'].mean()
        print(f'  region={r}: NE={ne:.2f}, NC={nc:.2f}, South={s:.2f}, West={w:.2f}, N={len(sub)}')
print()
# Check lives_in_smsa
print('lives_in_smsa:', df['lives_in_smsa'].value_counts(dropna=False).to_dict())
# Check if there are more detailed location variables
for c in df.columns:
    if any(k in c.lower() for k in ['smsa','metro','urban','state','division','census']):
        print(f'{c}: {df[c].nunique()} unique values')
