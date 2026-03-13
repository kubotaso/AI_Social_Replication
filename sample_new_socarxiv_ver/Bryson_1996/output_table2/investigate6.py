import pandas as pd
import numpy as np

df = pd.read_csv('gss1993_clean.csv')

# All racial attitude variables in the dataset
race_vars = ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4',
             'racfew', 'rachaf', 'racmar', 'racopen', 'racseg']

for v in race_vars:
    df[v] = pd.to_numeric(df[v], errors='coerce')
    n_valid = df[v].notna().sum()
    vals = df[v].dropna().value_counts().sort_index().to_dict()
    print(f'{v}: N={n_valid}, vals={vals}')

# Check which items are asked of the SAME respondents
# Cross-tabulate validity
print('\nCo-validity matrix (count of respondents with BOTH items valid):')
for v1 in race_vars:
    row = []
    for v2 in race_vars:
        both = (df[v1].notna() & df[v2].notna()).sum()
        row.append(both)
    print(f'{v1:>10}: {row}')

# The paper says items must be asked of same set of respondents
# And they maximize valid responses
# busing has 990, racdif1-4 have ~1000+
# racmost only 824 - likely different ballot

# Check: which items overlap most with busing?
print('\nOverlap with busing:')
busing_valid = df['busing'].notna()
for v in race_vars:
    both = (busing_valid & df[v].notna()).sum()
    print(f'  busing & {v}: {both}')

# Check racopen overlap
print('\nOverlap with racopen:')
racopen_valid = df['racopen'].notna()
for v in race_vars:
    both = (racopen_valid & df[v].notna()).sum()
    print(f'  racopen & {v}: {both}')

# The 5 items with best overlap: busing, racdif1, racdif2, racdif3, + one more
# racdif4 has 1003, racfew 1073, racopen 1034, racmar 1037
# Let's check which set of 5 gives the highest N when all 5 are required

# Check variance of each binary-coded item
print('\nItem variances (binary coded, racist direction):')
# racmost: 1=racist
p = (df['racmost'] == 1).mean()
print(f'racmost: p={df["racmost"].dropna().eq(1).mean():.3f}')
# busing: 2=racist
print(f'busing: p={df["busing"].dropna().eq(2).mean():.3f}')
# racdif1: 2=racist
print(f'racdif1: p={df["racdif1"].dropna().eq(2).mean():.3f}')
# racdif2: 2=racist (NOT because lack education)
print(f'racdif2: p={df["racdif2"].dropna().eq(2).mean():.3f}')
# racdif3: 1=racist (because lack motivation)
print(f'racdif3: p={df["racdif3"].dropna().eq(1).mean():.3f}')
# racdif4: what is this? "inborn disability" - 1=Yes (racist), 2=No
print(f'racdif4: p={df["racdif4"].dropna().eq(1).mean():.3f}')
# racfew: 1=favor (racist), 2=not
print(f'racfew: p={df["racfew"].dropna().eq(1).mean():.3f}')
# racopen: 1=discriminate ok (racist)
print(f'racopen: p={df["racopen"].dropna().eq(1).mean():.3f}')
# racmar: 1=favor law against interracial marriage (racist)
print(f'racmar: p={df["racmar"].dropna().eq(1).mean():.3f}')
# racseg: 1-4 scale, higher = less segregation preference? Let's check
print(f'racseg vals: {df["racseg"].dropna().value_counts().sort_index().to_dict()}')
# rachaf: 1=in favor of preferences for blacks, 2=against
print(f'rachaf: p={df["rachaf"].dropna().eq(2).mean():.3f}')

# Paper says "remove questions with extremely small variances"
# racfew has p=0.044 (very low variance - almost everyone says no)
# This would be removed!

# Let's try different 5-item sets
print('\n\nTrying 5-item sets:')

# Set A: busing + racdif1-4 (all asked of ~1000 respondents)
items_A = ['busing','racdif1','racdif2','racdif3','racdif4']
all_A = df[items_A].notna().all(axis=1)
print(f'Set A (busing+racdif1-4): all valid N={all_A.sum()}')

# Check with music + other IVs
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

# Compute N for each candidate set
candidate_sets = {
    'busing+racdif1-4': ['busing','racdif1','racdif2','racdif3','racdif4'],
    'racopen+busing+racdif1-3': ['racopen','busing','racdif1','racdif2','racdif3'],
    'racmar+busing+racdif1-3': ['racmar','busing','racdif1','racdif2','racdif3'],
    'rachaf+busing+racdif1-3': ['rachaf','busing','racdif1','racdif2','racdif3'],
    'racmost+busing+racdif1-3': ['racmost','busing','racdif1','racdif2','racdif3'],
    'racopen+racdif1-4': ['racopen','racdif1','racdif2','racdif3','racdif4'],
    'busing+racopen+racdif1-3': ['busing','racopen','racdif1','racdif2','racdif3'],
    'racmost+racdif1-4': ['racmost','racdif1','racdif2','racdif3','racdif4'],
}

for name, items in candidate_sets.items():
    all_valid_mask = df[items].notna().all(axis=1)
    n1 = (all_min_valid & all_valid_mask & other_valid).sum()
    n2 = (all_rem_valid & all_valid_mask & other_valid).sum()

    # Compute racism score
    coded = []
    for item in items:
        if item == 'racmost':
            coded.append((df[item] == 1).astype(float))
        elif item == 'busing':
            coded.append((df[item] == 2).astype(float))
        elif item == 'racdif1':
            coded.append((df[item] == 2).astype(float))
        elif item == 'racdif2':
            coded.append((df[item] == 2).astype(float))
        elif item == 'racdif3':
            coded.append((df[item] == 1).astype(float))
        elif item == 'racdif4':
            coded.append((df[item] == 1).astype(float))
        elif item == 'racopen':
            coded.append((df[item] == 1).astype(float))
        elif item == 'racmar':
            coded.append((df[item] == 1).astype(float))
        elif item == 'rachaf':
            coded.append((df[item] == 2).astype(float))

    racism = sum(coded)
    mask1 = all_min_valid & all_valid_mask & other_valid
    if mask1.sum() > 0:
        m = racism[mask1].mean()
        s = racism[mask1].std(ddof=1)
    else:
        m, s = 0, 0
    print(f'{name:>30}: DV1 N={n1}, DV2 N={n2}, racism mean={m:.2f}, SD={s:.2f}')
