import pandas as pd
import numpy as np

df = pd.read_csv('gss1993_clean.csv')

race_vars = ['racmost','busing','racdif1','racdif2','racdif3','racdif4','racfew','rachaf']
for v in race_vars:
    df[v] = pd.to_numeric(df[v], errors='coerce')

# Items with small variance (p close to 0 or 1):
# racfew: p=0.044 -> REMOVE
# rachaf: p=0.852 -> borderline, maybe REMOVE
# racdif2: p=0.870 -> borderline, maybe REMOVE

# Try: racmost + busing + racdif1 + racdif3 + racdif4
# (removed racfew, rachaf, racdif2 for small variance)
items_test = ['racmost', 'busing', 'racdif1', 'racdif3', 'racdif4']

for v in items_test:
    df[v] = pd.to_numeric(df[v], errors='coerce')

all_valid = df[items_test].notna().all(axis=1)
print(f'racmost+busing+racdif1+racdif3+racdif4, all valid: {all_valid.sum()}')

# Code in racist direction
df['c_racmost'] = (df['racmost'] == 1).astype(float)
df['c_busing'] = (df['busing'] == 2).astype(float)
df['c_racdif1'] = (df['racdif1'] == 2).astype(float)
df['c_racdif3'] = (df['racdif3'] == 1).astype(float)
df['c_racdif4'] = (df['racdif4'] == 1).astype(float)
for v in items_test:
    df.loc[df[v].isna(), 'c_'+v] = np.nan

coded = ['c_racmost','c_busing','c_racdif1','c_racdif3','c_racdif4']
racism = df[coded].sum(axis=1)
racism[~all_valid] = np.nan

print(f'Mean: {racism.dropna().mean():.2f}, SD: {racism.dropna().std(ddof=1):.2f}')

# Cronbach's alpha
from numpy import var
valid_data = df.loc[all_valid, coded]
k = len(coded)
item_vars = valid_data.var(ddof=1)
total_var = valid_data.sum(axis=1).var(ddof=1)
alpha = (k / (k-1)) * (1 - item_vars.sum() / total_var)
print(f'Alpha: {alpha:.2f}')

# Check N with music and other IVs
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

mask1 = all_min_valid & all_valid & other_valid
mask2 = all_rem_valid & all_valid & other_valid
print(f'\nDV1 N={mask1.sum()}, DV2 N={mask2.sum()}')
print(f'DV1 racism mean={racism[mask1].mean():.2f}, SD={racism[mask1].std(ddof=1):.2f}')

# That probably won't work well. Let me try other combos.
# Paper says factor analysis suggested removal of ONE item from a longer list
# Starting from: racmost, busing, racdif1, racdif2, racdif3 (the 5 I originally used)
# Factor analysis removes one -> 4 items? But paper says 5 items...

# Unless the initial list had 6+ items, factor analysis removed 1, leaving 5
# Starting from 6 items: racmost, busing, racdif1, racdif2, racdif3, racdif4
# Remove items with small variance: racdif2 (p=0.870) or racdif4 could be removed
# Factor analysis removes 1 more

# Let me try: busing + racdif1 + racdif2 + racdif3 + racdif4 (drop racmost via factor analysis)
items_v2 = ['busing','racdif1','racdif2','racdif3','racdif4']
all_valid_v2 = df[items_v2].notna().all(axis=1)
print(f'\nbusing+racdif1-4: all valid: {all_valid_v2.sum()}')

df['c2_busing'] = (df['busing'] == 2).astype(float)
df['c2_racdif1'] = (df['racdif1'] == 2).astype(float)
df['c2_racdif2'] = (df['racdif2'] == 2).astype(float)
df['c2_racdif3'] = (df['racdif3'] == 1).astype(float)
df['c2_racdif4'] = (df['racdif4'] == 1).astype(float)
for v in items_v2:
    df.loc[df[v].isna(), 'c2_'+v] = np.nan

coded_v2 = ['c2_busing','c2_racdif1','c2_racdif2','c2_racdif3','c2_racdif4']
racism_v2 = df[coded_v2].sum(axis=1)
racism_v2[~all_valid_v2] = np.nan
print(f'Mean: {racism_v2.dropna().mean():.2f}, SD: {racism_v2.dropna().std(ddof=1):.2f}')

valid_data_v2 = df.loc[all_valid_v2, coded_v2]
k2 = len(coded_v2)
item_vars_v2 = valid_data_v2.var(ddof=1)
total_var_v2 = valid_data_v2.sum(axis=1).var(ddof=1)
alpha_v2 = (k2 / (k2-1)) * (1 - item_vars_v2.sum() / total_var_v2)
print(f'Alpha: {alpha_v2:.2f}')

mask1 = all_min_valid & all_valid_v2 & other_valid
mask2 = all_rem_valid & all_valid_v2 & other_valid
print(f'DV1 N={mask1.sum()}, DV2 N={mask2.sum()}')
print(f'DV1 racism mean={racism_v2[mask1].mean():.2f}, SD={racism_v2[mask1].std(ddof=1):.2f}')

# Try: racmost + busing + racdif1 + racdif3 + racdif4 (drop racdif2 for small var)
# Already done above. Let me also check alpha.
print('\n--- Summary of candidate 5-item scales ---')

# Let me try ALL possible 5-item subsets from: racmost, busing, racdif1, racdif2, racdif3, racdif4
from itertools import combinations
all_items = ['racmost','busing','racdif1','racdif2','racdif3','racdif4']
coding = {
    'racmost': lambda x: (x == 1).astype(float),
    'busing': lambda x: (x == 2).astype(float),
    'racdif1': lambda x: (x == 2).astype(float),
    'racdif2': lambda x: (x == 2).astype(float),
    'racdif3': lambda x: (x == 1).astype(float),
    'racdif4': lambda x: (x == 1).astype(float),
}

for combo in combinations(all_items, 5):
    combo = list(combo)
    valid_mask = df[combo].notna().all(axis=1)

    coded_cols = []
    for item in combo:
        col = coding[item](df[item])
        col[df[item].isna()] = np.nan
        coded_cols.append(col)

    coded_df = pd.concat(coded_cols, axis=1)
    coded_df.columns = combo
    racism_sum = coded_df.sum(axis=1)
    racism_sum[~valid_mask] = np.nan

    m = racism_sum.dropna().mean()
    s = racism_sum.dropna().std(ddof=1)

    # Alpha
    vd = coded_df.loc[valid_mask]
    k = 5
    iv = vd.var(ddof=1).sum()
    tv = vd.sum(axis=1).var(ddof=1)
    a = (k/(k-1)) * (1 - iv/tv) if tv > 0 else 0

    mask1 = all_min_valid & valid_mask & other_valid
    mask2 = all_rem_valid & valid_mask & other_valid

    print(f'{"+".join(combo):>45}: N1={mask1.sum()}, N2={mask2.sum()}, mean={m:.2f}, SD={s:.2f}, alpha={a:.2f}')
