import pandas as pd
import numpy as np

df = pd.read_csv('gss1993_clean.csv')

# Build DVs
minority_genres = ['rap', 'reggae', 'blues', 'jazz', 'gospel', 'latin']
remaining_genres = ['musicals', 'oldies', 'classicl', 'bigband', 'newage', 'opera',
                    'blugrass', 'folk', 'moodeasy', 'conrock', 'hvymetal', 'country']
all_genres = minority_genres + remaining_genres

for g in all_genres:
    df[g] = pd.to_numeric(df[g], errors='coerce')

minority_valid = df[minority_genres].isin([1,2,3,4,5]).all(axis=1)
remaining_valid = df[remaining_genres].isin([1,2,3,4,5]).all(axis=1)

df['dv_minority'] = np.nan
df.loc[minority_valid, 'dv_minority'] = (df.loc[minority_valid, minority_genres] >= 4).sum(axis=1)
df['dv_remaining'] = np.nan
df.loc[remaining_valid, 'dv_remaining'] = (df.loc[remaining_valid, remaining_genres] >= 4).sum(axis=1)

# Build racism items
for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())
df['r_racdif3'] = (df['racdif3'] == 1).astype(float).where(df['racdif3'].notna())
df['r_racdif4'] = (df['racdif4'] == 1).astype(float).where(df['racdif4'].notna())

# Correlations of each item with DVs
items = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3', 'r_racdif4']
print("=== Individual item correlations with DVs ===")
for item in items:
    for dv in ['dv_minority', 'dv_remaining']:
        t = df.dropna(subset=[dv, item])
        c = t[dv].corr(t[item])
        print(f'{item} vs {dv}: r={c:.4f} (n={len(t)})')
    print()

# Build several scale versions
coded5 = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3']
df['racism_5item'] = df[coded5].sum(axis=1, min_count=5)

coded6 = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3', 'r_racdif4']
df['racism_6item'] = df[coded6].sum(axis=1, min_count=6)

# Person mean imputation for 6 items (min 4)
racism_imp = []
for idx in df.index:
    vals = [df.loc[idx, c] for c in coded6]
    n_valid = sum(1 for v in vals if not np.isnan(v))
    if n_valid >= 4:
        valid_vals = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_vals)
        racism_imp.append(sum(v if not np.isnan(v) else pm for v in vals))
    else:
        racism_imp.append(np.nan)
df['racism_6item_imp'] = racism_imp

# Person mean imputation for 5 items (min 4)
racism_imp5 = []
for idx in df.index:
    vals = [df.loc[idx, c] for c in coded5]
    n_valid = sum(1 for v in vals if not np.isnan(v))
    if n_valid >= 4:
        valid_vals = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_vals)
        racism_imp5.append(sum(v if not np.isnan(v) else pm for v in vals))
    else:
        racism_imp5.append(np.nan)
df['racism_5item_imp'] = racism_imp5

print("=== Scale correlations with DVs ===")
for scale in ['racism_5item', 'racism_6item', 'racism_6item_imp', 'racism_5item_imp']:
    for dv in ['dv_minority', 'dv_remaining']:
        t = df.dropna(subset=[dv, scale])
        c = t[dv].corr(t[scale])
        print(f'{scale} vs {dv}: r={c:.4f} (n={len(t)})')
    print()

# Stats for scales
print("=== Scale statistics ===")
for scale in ['racism_5item', 'racism_6item', 'racism_6item_imp', 'racism_5item_imp']:
    t = df[scale].dropna()
    print(f'{scale}: mean={t.mean():.3f}, SD={t.std(ddof=1):.3f}, n={len(t)}')

# Check region coding - GSS region values
print("\n=== Region variable ===")
print(df['region'].value_counts().sort_index())

# Check fund
print("\n=== Fund variable ===")
print(df['fund'].value_counts().sort_index())

# Check denom
print("\n=== Denom variable ===")
print(df['denom'].value_counts().sort_index().head(40))

# Check if Southern should include region 5,6,7 (South Atlantic, ESC, WSC)
print("\n=== Southern coding test ===")
for reg_vals in [[3], [5,6,7], [3,5,6,7]]:
    df['south_test'] = df['region'].isin(reg_vals).astype(int)
    print(f'Region {reg_vals}: n_south={df["south_test"].sum()}')

# Check income coding
print("\n=== Income per capita statistics ===")
df['realinc'] = pd.to_numeric(df['realinc'], errors='coerce')
df['hompop'] = pd.to_numeric(df['hompop'], errors='coerce')
df['income_pc'] = df['realinc'] / df['hompop']
print(f'realinc/hompop: mean={df["income_pc"].mean():.0f}, median={df["income_pc"].median():.0f}')
df['income91'] = pd.to_numeric(df['income91'], errors='coerce')
df['income91_pc'] = df['income91'] / df['hompop']
print(f'income91/hompop: mean={df["income91_pc"].mean():.2f}, median={df["income91_pc"].median():.2f}')

# Check DV distributions
print("\n=== DV distributions ===")
print(f'dv_minority: mean={df["dv_minority"].mean():.3f}, SD={df["dv_minority"].std(ddof=1):.3f}, n_valid={df["dv_minority"].notna().sum()}')
print(f'dv_remaining: mean={df["dv_remaining"].mean():.3f}, SD={df["dv_remaining"].std(ddof=1):.3f}, n_valid={df["dv_remaining"].notna().sum()}')

# Check if requiring all 18 genres valid changes things
all_valid = df[all_genres].isin([1,2,3,4,5]).all(axis=1)
df['dv_minority_all18'] = np.nan
df.loc[all_valid, 'dv_minority_all18'] = (df.loc[all_valid, minority_genres] >= 4).sum(axis=1)
df['dv_remaining_all18'] = np.nan
df.loc[all_valid, 'dv_remaining_all18'] = (df.loc[all_valid, remaining_genres] >= 4).sum(axis=1)
print(f'\nAll 18 valid: n={all_valid.sum()}')
print(f'dv_minority_all18: mean={df["dv_minority_all18"].mean():.3f}')
print(f'dv_remaining_all18: mean={df["dv_remaining_all18"].mean():.3f}')

# Try computing alpha for different scale combos
from itertools import combinations

print("\n=== Cronbach's alpha for various combos ===")
def cronbach_alpha(items_df):
    items_df = items_df.dropna()
    if len(items_df) < 10:
        return np.nan
    k = items_df.shape[1]
    var_sum = items_df.var(axis=0, ddof=1).sum()
    total_var = items_df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return 0
    return (k / (k - 1)) * (1 - var_sum / total_var)

all_items = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3', 'r_racdif4']
for combo_size in [5, 6]:
    for combo in combinations(all_items, combo_size):
        alpha = cronbach_alpha(df[list(combo)])
        print(f'{[c.replace("r_","") for c in combo]}: alpha={alpha:.3f}')
