import pandas as pd
import numpy as np

df = pd.read_csv('gss1993_clean.csv')

items = ['racmost','busing','racdif1','racdif2','racdif3']
for v in items:
    df[v] = pd.to_numeric(df[v], errors='coerce')

all_valid = df[items].notna().all(axis=1)

# The paper items are:
# (1) "Would you have any objection to sending your children to a school
#      where more than half of the children are Black?" -> racmost
#      Racist = YES = object. In GSS: 1=Would object
# (2) "do you favor or oppose the busing..." -> busing
#      Racist = OPPOSE. In GSS: 2=Oppose
# (3) "On average Blacks have worse jobs, income, and housing...
#      Do you think these differences are mainly due to discrimination?"
#      -> racdif1. Racist = NO (not due to discrimination). In GSS: 2=No
# (4) "...because most Blacks don't have the chance for education..."
#      -> racdif2. Racist = NO (not because lack education). In GSS: 2=No
# (5) "...because most Blacks just don't have the motivation or will power..."
#      -> racdif3. Racist = YES (lack motivation). In GSS: 1=Yes

# But the data shows racdif3 is NEGATIVELY correlated with the other racist items
# This suggests racdif3 coding in the dataset is REVERSED from GSS documentation
# Maybe in this dataset: racdif3 1=No, 2=Yes (opposite of standard GSS)

# OR: the paper uses a DIFFERENT interpretation
# The paper says items are "coded in the same direction"
# If we code all as: higher value = more racist:
# racmost: 1=object(racist) -> racist=1, not=2 -> REVERSE to get higher=racist: 3-x
# Wait no, for binary: racist direction for each:
# racmost: racist if 1 -> code as 1
# busing: racist if 2 -> code as 1 when value=2
# racdif1: racist if 2 -> code as 1 when value=2
# racdif2: racist if 2 -> code as 1 when value=2
# racdif3: racist if 1 -> code as 1 when value=1

# The negative correlations of racdif3 with others mean that in THIS data,
# people who answer 1 on racdif3 tend to NOT be racist on other items
# So either: (a) our racdif3 coding is wrong, or (b) racdif3 just
# doesn't correlate well with the other items

# The paper says alpha=0.54. With racdif3 flipped, alpha=0.42.
# Still not 0.54. But the SAMPLE matters - the paper might report
# alpha from a different sample

# Let me try: with racdif3 FLIPPED (2=racist)
# And also check if dropping racdif2 helps
# Paper says factor analysis suggested removal of ONE item
# Maybe that one item is racdif2?

# 4-item version: racmost + busing + racdif1 + racdif3(flipped)
df['c_racmost'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
df['c_busing'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
df['c_racdif1'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
df['c_racdif2'] = np.where(df['racdif2'].notna(), (df['racdif2'] == 2).astype(float), np.nan)
df['c_racdif3_f'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 2).astype(float), np.nan)

# Test: 5 items with racdif3 flipped
coded5 = ['c_racmost','c_busing','c_racdif1','c_racdif2','c_racdif3_f']
valid5 = df.loc[all_valid, coded5]
s5 = valid5.sum(axis=1)
k5 = 5
iv5 = valid5.var(ddof=1).sum()
tv5 = s5.var(ddof=1)
a5 = (k5/(k5-1)) * (1 - iv5/tv5)
print(f'5 items (racdif3 flipped): mean={s5.mean():.2f}, SD={s5.std(ddof=1):.2f}, alpha={a5:.2f}, N={all_valid.sum()}')

# Test: 4 items: racmost + busing + racdif1 + racdif3(flipped) - DROP racdif2
coded4a = ['c_racmost','c_busing','c_racdif1','c_racdif3_f']
items4a = ['racmost','busing','racdif1','racdif3']
all_valid_4a = df[items4a].notna().all(axis=1)
valid4a = df.loc[all_valid_4a, coded4a]
s4a = valid4a.sum(axis=1)
k4a = 4
iv4a = valid4a.var(ddof=1).sum()
tv4a = s4a.var(ddof=1)
a4a = (k4a/(k4a-1)) * (1 - iv4a/tv4a)
print(f'4 items (drop racdif2): mean={s4a.mean():.2f}, SD={s4a.std(ddof=1):.2f}, alpha={a4a:.2f}, N={all_valid_4a.sum()}')

# Test: 5 items with racdif3 flipped, in the DV1 model sample
minority = ['rap','reggae','blues','jazz','gospel','latin']
remaining = ['musicals','oldies','classicl','bigband','newage','opera','blugrass','folk','moodeasy','conrock','hvymetal','country']
for g in minority + remaining:
    df[g] = pd.to_numeric(df[g], errors='coerce')
df['realinc'] = pd.to_numeric(df['realinc'], errors='coerce')
df['hompop'] = pd.to_numeric(df['hompop'], errors='coerce')
df['income_pc'] = df['realinc'] / df['hompop']
df['education'] = pd.to_numeric(df['educ'], errors='coerce')
df['occ_prestige'] = pd.to_numeric(df['prestg80'], errors='coerce')
df['age_var'] = pd.to_numeric(df['age'], errors='coerce')
all_min_valid = df[minority].isin([1,2,3,4,5]).all(axis=1)
all_rem_valid = df[remaining].isin([1,2,3,4,5]).all(axis=1)
other_valid = df[['education','income_pc','occ_prestige','age_var']].notna().all(axis=1)

# With all 5 items required, racdif3 flipped
racism_5f = df['c_racmost'] + df['c_busing'] + df['c_racdif1'] + df['c_racdif2'] + df['c_racdif3_f']
racism_5f[~all_valid] = np.nan

mask1 = all_min_valid & all_valid & other_valid
mask2 = all_rem_valid & all_valid & other_valid
print(f'\n5 items flipped racdif3, strict: DV1 N={mask1.sum()}, DV2 N={mask2.sum()}')
print(f'  racism mean={racism_5f[mask1].mean():.2f}, SD={racism_5f[mask1].std(ddof=1):.2f}')

# With 4+ required, NAs as 0
n_valid_items = df[coded5].notna().sum(axis=1)
racism_5f_na0 = df[['c_racmost','c_busing','c_racdif1','c_racdif2','c_racdif3_f']].fillna(0).sum(axis=1)
racism_5f_na0[n_valid_items < 4] = np.nan

mask1b = all_min_valid & (n_valid_items >= 4) & other_valid
mask2b = all_rem_valid & (n_valid_items >= 4) & other_valid
print(f'\n5 items (racdif3 flipped), 4+ valid NAs=0: DV1 N={mask1b.sum()}, DV2 N={mask2b.sum()}')
print(f'  racism mean={racism_5f_na0[mask1b].mean():.2f}, SD={racism_5f_na0[mask1b].std(ddof=1):.2f}')

# Try: person-mean imputation with racdif3 flipped
coded5_cols = [df['c_racmost'], df['c_busing'], df['c_racdif1'], df['c_racdif2'], df['c_racdif3_f']]
coded5_df = pd.concat(coded5_cols, axis=1)
coded5_df.columns = coded5
pmean = coded5_df.mean(axis=1)
for c in coded5:
    coded5_df[c] = coded5_df[c].fillna(pmean)
racism_imp = coded5_df.sum(axis=1)
racism_imp[n_valid_items < 4] = np.nan

mask1c = all_min_valid & (n_valid_items >= 4) & other_valid
mask2c = all_rem_valid & (n_valid_items >= 4) & other_valid
print(f'\n5 items (racdif3 flipped), person-mean imputation 4+: DV1 N={mask1c.sum()}, DV2 N={mask2c.sum()}')
print(f'  racism mean={racism_imp[mask1c].mean():.2f}, SD={racism_imp[mask1c].std(ddof=1):.2f}')
