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
all_rem_valid = df[remaining].isin([1,2,3,4,5]).all(axis=1)
other_vars = ['education', 'income_pc', 'occ_prestige', 'age_var']
other_valid = df[other_vars].notna().all(axis=1)

# Try requiring at least 3 or 4 racism items
for min_req in [1, 2, 3, 4, 5]:
    n_valid = df[items].notna().sum(axis=1) >= min_req
    mask1 = all_min_valid & n_valid & other_valid
    mask2 = all_rem_valid & n_valid & other_valid
    print(f'Require {min_req}+ racism items: DV1 N={mask1.sum()}, DV2 N={mask2.sum()}')

# Maybe the paper uses racopen instead of racmost
# racopen: "refuse to sell to black" - racist if open == 2 (disagree/not let)?
# Let's check coding: racopen has values 1, 2, 3
# GSS: RACOPEN "Suppose there is a community-wide vote on the general housing issue.
# Which law would you vote for?" 1=Homeowner can decide, 2=Cannot discriminate, 3=neither
# racist = 1 (homeowner decides = allows discrimination)
print('\nTrying racopen instead of racmost:')
df['racopen_num'] = pd.to_numeric(df['racopen'], errors='coerce')
items_v2 = ['racopen_num','busing','racdif1','racdif2','racdif3']
all5_v2 = df[items_v2].notna().all(axis=1)
mask1 = all_min_valid & all5_v2 & other_valid
mask2 = all_rem_valid & all5_v2 & other_valid
print(f'racopen version: DV1 N={mask1.sum()}, DV2 N={mask2.sum()}')

# Try racfew instead
df['racfew_num'] = pd.to_numeric(df['racfew'], errors='coerce')
items_v3 = ['racfew_num','busing','racdif1','racdif2','racdif3']
all5_v3 = df[items_v3].notna().all(axis=1)
mask1 = all_min_valid & all5_v3 & other_valid
mask2 = all_rem_valid & all5_v3 & other_valid
print(f'racfew version: DV1 N={mask1.sum()}, DV2 N={mask2.sum()}')

# Try: use busing, racdif1-4 (4 items)
df['racdif4'] = pd.to_numeric(df['racdif4'], errors='coerce')
print(f'\nracdif4 valid: {df["racdif4"].notna().sum()}, vals: {df["racdif4"].dropna().value_counts().sort_index().to_dict()}')

# Try racmost + racdif1-4 (no busing)
items_v4 = ['racmost','racdif1','racdif2','racdif3','racdif4']
all5_v4 = df[items_v4].notna().all(axis=1)
mask1 = all_min_valid & all5_v4 & other_valid
mask2 = all_rem_valid & all5_v4 & other_valid
print(f'racmost+racdif1-4: DV1 N={mask1.sum()}, DV2 N={mask2.sum()}')

# Try busing + racdif1-4 (no racmost)
items_v5 = ['busing','racdif1','racdif2','racdif3','racdif4']
all5_v5 = df[items_v5].notna().all(axis=1)
mask1 = all_min_valid & all5_v5 & other_valid
mask2 = all_rem_valid & all5_v5 & other_valid
print(f'busing+racdif1-4: DV1 N={mask1.sum()}, DV2 N={mask2.sum()}')

# Maybe the paper treats racism items NA as 0 (non-racist) for SOME items?
# Try: require racdif1-3 valid, treat racmost and busing NAs as 0
mask_racdif = df[['racdif1','racdif2','racdif3']].notna().all(axis=1)
mask1 = all_min_valid & mask_racdif & other_valid
mask2 = all_rem_valid & mask_racdif & other_valid
print(f'\nRequire racdif1-3 valid, ignore racmost/busing NAs: DV1 N={mask1.sum()}, DV2 N={mask2.sum()}')

# Try: require 4+ racism items valid (treat remaining NA as 0)
for threshold in [3, 4]:
    n_valid_items = df[items].notna().sum(axis=1)
    mask = n_valid_items >= threshold
    # Compute racism score treating NAs as 0 for those with >= threshold valid
    r1 = np.where(df['racmost'].isna(), 0, (df['racmost'] == 1).astype(int))
    r2 = np.where(df['busing'].isna(), 0, (df['busing'] == 2).astype(int))
    r3 = np.where(df['racdif1'].isna(), 0, (df['racdif1'] == 2).astype(int))
    r4 = np.where(df['racdif2'].isna(), 0, (df['racdif2'] == 2).astype(int))
    r5 = np.where(df['racdif3'].isna(), 0, (df['racdif3'] == 1).astype(int))
    racism = r1 + r2 + r3 + r4 + r5

    mask1 = all_min_valid & mask & other_valid
    mask2 = all_rem_valid & mask & other_valid
    m = df.loc[mask1, 'racism_score_temp'] if False else racism[mask1]
    print(f'\nRequire {threshold}+ racism items, NAs as 0: DV1 N={mask1.sum()}, DV2 N={mask2.sum()}')
    print(f'  racism mean={racism[mask1].mean():.2f}, SD={pd.Series(racism[mask1]).std(ddof=1):.2f}')

# Maybe income is NOT per capita but just realinc?
print('\n\nTrying different income variable:')
# Without income_pc, use realinc directly
for inc_var in ['realinc', 'income_pc']:
    if inc_var == 'income_pc':
        other2 = ['education', 'income_pc', 'occ_prestige', 'age_var']
    else:
        other2 = ['education', 'realinc', 'occ_prestige', 'age_var']
    ov = df[other2].notna().all(axis=1)
    mask1 = all_min_valid & all5_valid_mask & ov
    print(f'{inc_var}: DV1 N={mask1.sum()}')

# Fix: define all5_valid_mask
all5_valid_mask = df[items].notna().all(axis=1)
for inc_var in ['realinc']:
    other2 = ['education', 'realinc', 'occ_prestige', 'age_var']
    ov = df[other2].notna().all(axis=1)
    mask1 = all_min_valid & all5_valid_mask & ov
    print(f'With realinc (not pc): DV1 N={mask1.sum()}')

# Check: how many missing for each IV
print('\nMissing counts:')
print('education:', df['education'].isna().sum())
print('realinc:', df['realinc'].isna().sum())
print('income_pc:', df['income_pc'].isna().sum())
print('occ_prestige:', df['occ_prestige'].isna().sum())
print('age_var:', df['age_var'].isna().sum())

# Maybe income91 instead of realinc?
df['income91'] = pd.to_numeric(df['income91'], errors='coerce')
print('income91 valid:', df['income91'].notna().sum())
