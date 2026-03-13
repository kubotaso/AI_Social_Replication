import pandas as pd
import numpy as np

df = pd.read_csv('gss1993_clean.csv')
items = ['racmost','busing','racdif1','racdif2','racdif3']
for item in items:
    df[item] = pd.to_numeric(df[item], errors='coerce')
    print(item, 'valid:', df[item].notna().sum())

all5_valid = df[items].notna().all(axis=1)
print('All 5 racism valid:', all5_valid.sum())

minority = ['rap','reggae','blues','jazz','gospel','latin']
remaining = ['musicals','oldies','classicl','bigband','newage','opera','blugrass','folk','moodeasy','conrock','hvymetal','country']
for g in minority + remaining:
    df[g] = pd.to_numeric(df[g], errors='coerce')

all_min_valid = df[minority].isin([1,2,3,4,5]).all(axis=1)
all_rem_valid = df[remaining].isin([1,2,3,4,5]).all(axis=1)
print('All 6 minority valid:', all_min_valid.sum())
print('All 12 remaining valid:', all_rem_valid.sum())

df['realinc'] = pd.to_numeric(df['realinc'], errors='coerce')
df['hompop'] = pd.to_numeric(df['hompop'], errors='coerce')
df['income_pc'] = df['realinc'] / df['hompop']
df['education'] = pd.to_numeric(df['educ'], errors='coerce')
df['occ_prestige'] = pd.to_numeric(df['prestg80'], errors='coerce')
df['age_var'] = pd.to_numeric(df['age'], errors='coerce')

other_vars = ['education', 'income_pc', 'occ_prestige', 'age_var']
other_valid = df[other_vars].notna().all(axis=1)
print('Other IVs valid:', other_valid.sum())

print('DV1 + 5racism + otherIVs:', (all_min_valid & all5_valid & other_valid).sum())
print('DV1 + otherIVs only:', (all_min_valid & other_valid).sum())
print('DV2 + 5racism + otherIVs:', (all_rem_valid & all5_valid & other_valid).sum())
print('DV2 + otherIVs only:', (all_rem_valid & other_valid).sum())

# Try treating racism NAs as 0
df['r1'] = np.where(df['racmost'].isna(), 0, (df['racmost'] == 1).astype(int))
df['r2'] = np.where(df['busing'].isna(), 0, (df['busing'] == 2).astype(int))
df['r3'] = np.where(df['racdif1'].isna(), 0, (df['racdif1'] == 2).astype(int))
df['r4'] = np.where(df['racdif2'].isna(), 0, (df['racdif2'] == 2).astype(int))
df['r5'] = np.where(df['racdif3'].isna(), 0, (df['racdif3'] == 1).astype(int))
df['racism_na0'] = df['r1'] + df['r2'] + df['r3'] + df['r4'] + df['r5']

mask1 = all_min_valid & other_valid
print('\nWith racism NAs as 0:')
print('DV1 N:', mask1.sum())
print('racism_na0 mean:', df.loc[mask1, 'racism_na0'].mean().round(2))
print('racism_na0 SD:', df.loc[mask1, 'racism_na0'].std(ddof=1).round(2))

mask2 = all_rem_valid & other_valid
print('DV2 N:', mask2.sum())

# Try using racfew instead of racmost
print('\nracfew values:', pd.to_numeric(df['racfew'], errors='coerce').dropna().value_counts().sort_index().to_dict())
print('racfew valid:', pd.to_numeric(df['racfew'], errors='coerce').notna().sum())

# Try different combination: racfew instead of racmost
df['racfew'] = pd.to_numeric(df['racfew'], errors='coerce')
items_alt = ['racfew','busing','racdif1','racdif2','racdif3']
all5_alt = df[items_alt].notna().all(axis=1)
print('\nAlt 5 (racfew instead of racmost) valid:', all5_alt.sum())
print('DV1 + alt5 + otherIVs:', (all_min_valid & all5_alt & other_valid).sum())

# Check: what if racmost question was only asked of a subset?
# GSS sometimes has ballot-specific questions
# Let's check racopen, rachaf, racseg, racmar too
for v in ['racopen','rachaf','racseg','racmar','racfew']:
    s = pd.to_numeric(df[v], errors='coerce')
    print(f'{v}: valid={s.notna().sum()}, vals={s.dropna().value_counts().sort_index().to_dict()}')
