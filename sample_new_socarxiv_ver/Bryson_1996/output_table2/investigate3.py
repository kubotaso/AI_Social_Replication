import pandas as pd
import numpy as np

df = pd.read_csv('gss1993_clean.csv')
items = ['racmost','busing','racdif1','racdif2','racdif3']
for item in items:
    df[item] = pd.to_numeric(df[item], errors='coerce')

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
other_valid = df[['education', 'income_pc', 'occ_prestige', 'age_var']].notna().all(axis=1)

# Key insight: paper says mean=2.65, SD=1.56 for racism score
# With 5 binary items, this is unusual but possible
# Let me check: require all 5 valid and compute stats
all5 = df[items].notna().all(axis=1)

# Binary coding (racist=1, not=0)
df['r1'] = (df['racmost'] == 1).astype(float)
df['r2'] = (df['busing'] == 2).astype(float)
df['r3'] = (df['racdif1'] == 2).astype(float)
df['r4'] = (df['racdif2'] == 2).astype(float)
df['r5'] = (df['racdif3'] == 1).astype(float)
df.loc[df['racmost'].isna(), 'r1'] = np.nan
df.loc[df['busing'].isna(), 'r2'] = np.nan
df.loc[df['racdif1'].isna(), 'r3'] = np.nan
df.loc[df['racdif2'].isna(), 'r4'] = np.nan
df.loc[df['racdif3'].isna(), 'r5'] = np.nan

racism_binary = df[['r1','r2','r3','r4','r5']].sum(axis=1)
racism_binary[~all5] = np.nan

print('Binary coding, all 5 required:')
print(f'  mean={racism_binary.dropna().mean():.2f}, SD={racism_binary.dropna().std(ddof=1):.2f}, N={racism_binary.notna().sum()}')

# What about: don't require all 5 valid, just treat NA items as non-racist (0)
df['r1_na0'] = np.where(df['racmost'].isna(), 0, (df['racmost'] == 1).astype(int))
df['r2_na0'] = np.where(df['busing'].isna(), 0, (df['busing'] == 2).astype(int))
df['r3_na0'] = np.where(df['racdif1'].isna(), 0, (df['racdif1'] == 2).astype(int))
df['r4_na0'] = np.where(df['racdif2'].isna(), 0, (df['racdif2'] == 2).astype(int))
df['r5_na0'] = np.where(df['racdif3'].isna(), 0, (df['racdif3'] == 1).astype(int))
racism_na0 = df['r1_na0'] + df['r2_na0'] + df['r3_na0'] + df['r4_na0'] + df['r5_na0']

mask = all_min_valid & other_valid
print(f'\nNAs as 0, DV1 sample:')
print(f'  mean={racism_na0[mask].mean():.2f}, SD={racism_na0[mask].std(ddof=1):.2f}, N={mask.sum()}')

# Maybe the direction is flipped for some items? Let me try alternative coding
# What if racmost=2 is racist? (object to school)
# GSS RACMOST: 1=Object, 2=Not object
# Object to school with mostly black = racist, so 1 is correct

# What if busing=1 is racist?
# GSS BUSING: 1=Favor, 2=Oppose
# Oppose busing = racist, so 2 is correct

# What if items use original values, not binary?
# racmost: 1,2; racdif1-3: 1,2; busing: 1,2
# All are already binary (1 or 2), not Likert
# Sum of raw values would be 5-10, not 0-5

# Try: maybe the paper uses racopen + busing + racdif1-3 instead of racmost
# racopen: 1=can discriminate, 2=cannot, 3=neither
# racist direction for racopen: 1 (allow discrimination)
df['racopen'] = pd.to_numeric(df['racopen'], errors='coerce')
df['ro'] = (df['racopen'] == 1).astype(float)
df.loc[df['racopen'].isna(), 'ro'] = np.nan

items_v2 = [df['ro'], df['r2'], df['r3'], df['r4'], df['r5']]
racism_v2 = pd.concat(items_v2, axis=1)
racism_v2_sum = racism_v2.sum(axis=1)
racism_v2_sum[racism_v2.isna().any(axis=1)] = np.nan
print(f'\nracopen+busing+racdif1-3 (all valid):')
print(f'  mean={racism_v2_sum.dropna().mean():.2f}, SD={racism_v2_sum.dropna().std(ddof=1):.2f}, N={racism_v2_sum.notna().sum()}')

# Maybe the mean/SD in the paper is from the TABLE 1 sample (all 18 genres valid)?
all18 = df[minority + remaining].isin([1,2,3,4,5]).all(axis=1)
mask18 = all18 & all5 & other_valid
print(f'\nAll 18 genres valid + all 5 racism + other IVs:')
print(f'  N={mask18.sum()}')
print(f'  racism mean={racism_binary[mask18].mean():.2f}, SD={racism_binary[mask18].std(ddof=1):.2f}')

# Check: what is item mean for each item in the all5 sample?
for item_name, coded in [('racmost','r1'),('busing','r2'),('racdif1','r3'),('racdif2','r4'),('racdif3','r5')]:
    m = df.loc[all5, coded].mean()
    print(f'  {item_name} coded mean: {m:.3f}')

# Maybe racmost coding is flipped? Let me try racmost=2 as racist
df['r1_flip'] = (df['racmost'] == 2).astype(float)
df.loc[df['racmost'].isna(), 'r1_flip'] = np.nan
racism_flip = df['r1_flip'] + df['r2'] + df['r3'] + df['r4'] + df['r5']
racism_flip[~all5] = np.nan
print(f'\nFlipped racmost (2=racist):')
print(f'  mean={racism_flip.dropna().mean():.2f}, SD={racism_flip.dropna().std(ddof=1):.2f}')

# What about using racmar instead of racmost?
df['racmar'] = pd.to_numeric(df['racmar'], errors='coerce')
print(f'\nracmar valid: {df["racmar"].notna().sum()}, vals: {df["racmar"].dropna().value_counts().sort_index().to_dict()}')
# RACMAR: 1=Favor law against interracial marriage, 2=Against such a law
# racist = 1
df['rm'] = (df['racmar'] == 1).astype(float)
df.loc[df['racmar'].isna(), 'rm'] = np.nan

items_v3_cols = ['rm','r2','r3','r4','r5']
for c in items_v3_cols:
    pass
racism_v3 = df[['rm','r2','r3','r4','r5']].sum(axis=1)
all_v3 = df[['rm','r2','r3','r4','r5']].notna().all(axis=1)
racism_v3[~all_v3] = np.nan
print(f'racmar+busing+racdif1-3 (all valid):')
print(f'  mean={racism_v3.dropna().mean():.2f}, SD={racism_v3.dropna().std(ddof=1):.2f}, N={racism_v3.notna().sum()}')

# Check with racfew
df['racfew'] = pd.to_numeric(df['racfew'], errors='coerce')
# RACFEW: "few Blacks in neighborhood" 1=would favor, 2=would not
# racist = 1
df['rf'] = (df['racfew'] == 1).astype(float)
df.loc[df['racfew'].isna(), 'rf'] = np.nan

racism_v4 = df[['rf','r2','r3','r4','r5']].sum(axis=1)
all_v4 = df[['rf','r2','r3','r4','r5']].notna().all(axis=1)
racism_v4[~all_v4] = np.nan
print(f'\nracfew+busing+racdif1-3 (all valid):')
print(f'  mean={racism_v4.dropna().mean():.2f}, SD={racism_v4.dropna().std(ddof=1):.2f}, N={racism_v4.notna().sum()}')
