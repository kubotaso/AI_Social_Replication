import pandas as pd
import numpy as np

df = pd.read_csv('gss1993_clean.csv')

# Check what's causing the N gap between 708 (all 5 racism valid) and 459 (full listwise)
items = ['racmost','busing','racdif1','racdif2','racdif3']
minority = ['rap','reggae','blues','jazz','gospel','latin']
remaining = ['musicals','oldies','classicl','bigband','newage','opera','blugrass','folk','moodeasy','conrock','hvymetal','country']

for col in items + minority + remaining + ['educ','realinc','hompop','prestg80','age']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

all5_racism = df[items].notna().all(axis=1)
all_min_valid = df[minority].isin([1,2,3,4,5]).all(axis=1)

df['income_pc'] = df['realinc'] / df['hompop']

# Check which IVs are causing the most missingness
print("Among those with all 5 racism items AND all 6 minority genres valid:")
base = all5_racism & all_min_valid
print(f"  Base N: {base.sum()}")
print(f"  + educ valid: {(base & df['educ'].notna()).sum()}")
print(f"  + realinc valid: {(base & df['realinc'].notna()).sum()}")
print(f"  + income_pc valid: {(base & df['income_pc'].notna()).sum()}")
print(f"  + prestg80 valid: {(base & df['prestg80'].notna()).sum()}")
print(f"  + age valid: {(base & df['age'].notna()).sum()}")
print(f"  All IVs: {(base & df['educ'].notna() & df['income_pc'].notna() & df['prestg80'].notna() & df['age'].notna()).sum()}")

# The gap: 708 (racism valid) -> need music + IVs
# If we DON'T use income per capita but just realinc:
print(f"\n  With realinc (not pc): {(base & df['educ'].notna() & df['realinc'].notna() & df['prestg80'].notna() & df['age'].notna()).sum()}")

# Check: maybe prestg80 is the bottleneck
print(f"\n  Without prestg80: {(base & df['educ'].notna() & df['income_pc'].notna() & df['age'].notna()).sum()}")

# Actually let me check: how many have racism valid + DV1 valid
print(f"\n  racism + DV1: {(all5_racism & all_min_valid).sum()}")

# Of those 510, how many have each IV?
sub = df[all5_racism & all_min_valid]
for v in ['educ','realinc','income_pc','prestg80','age']:
    if v == 'income_pc':
        print(f"  {v} valid: {sub['income_pc'].notna().sum()}")
    else:
        print(f"  {v} valid: {sub[v].notna().sum()}")

# I wonder if the paper doesn't use hompop at all - maybe just realinc
# or income91
print(f"\n  income91 valid (among racism+DV1): {pd.to_numeric(sub['income91'], errors='coerce').notna().sum()}")

# Try: all 5 racism, DV1, educ, realinc (not pc), prestg80, age
all_IVs_v2 = df['educ'].notna() & df['realinc'].notna() & df['prestg80'].notna() & df['age'].notna()
mask1 = all5_racism & all_min_valid & all_IVs_v2
print(f"\nWith realinc (not per capita): DV1 N={mask1.sum()}")

all_rem_valid = df[remaining].isin([1,2,3,4,5]).all(axis=1)
mask2 = all5_racism & all_rem_valid & all_IVs_v2
print(f"With realinc (not per capita): DV2 N={mask2.sum()}")

# Try: all 5 racism, DV1, educ, income91, prestg80, age
df['income91'] = pd.to_numeric(df['income91'], errors='coerce')
all_IVs_v3 = df['educ'].notna() & df['income91'].notna() & df['prestg80'].notna() & df['age'].notna()
mask1 = all5_racism & all_min_valid & all_IVs_v3
print(f"\nWith income91: DV1 N={mask1.sum()}")
mask2 = all5_racism & all_rem_valid & all_IVs_v3
print(f"With income91: DV2 N={mask2.sum()}")

# So realinc gives 459, same as income_pc (hompop doesn't add NAs)
# income91 gives same N

# What if prestg80 has a lot of NAs?
print(f"\nprestg80 NAs total: {df['prestg80'].isna().sum()}")
print(f"prestg80 valid total: {df['prestg80'].notna().sum()}")

# Among racism + DV1 valid, how many lack prestg80?
sub2 = df[all5_racism & all_min_valid]
print(f"prestg80 NAs among racism+DV1: {sub2['prestg80'].isna().sum()}")
print(f"realinc NAs among racism+DV1: {sub2['realinc'].isna().sum()}")

# The problem: 510 have racism + DV1 valid, but only 459 also have all IVs
# Paper has 644. So there's a 185-case gap we can't explain just by relaxing IV requirements

# Maybe: the paper doesn't use racmost at all! racmost has only 824 valid
# vs 990+ for the other items. Without racmost:
items_no_racmost = ['busing','racdif1','racdif2','racdif3']
all4_racism = df[items_no_racmost].notna().all(axis=1)

# 4-item racism scale
r1 = (df['busing'] == 2).astype(float)
r1[df['busing'].isna()] = np.nan
r2 = (df['racdif1'] == 2).astype(float)
r2[df['racdif1'].isna()] = np.nan
r3 = (df['racdif2'] == 2).astype(float)
r3[df['racdif2'].isna()] = np.nan
r4 = (df['racdif3'] == 1).astype(float)
r4[df['racdif3'].isna()] = np.nan
racism_4item = r1 + r2 + r3 + r4

mask1 = all_min_valid & all4_racism & all_IVs_v2
print(f"\n4-item (no racmost), all required: DV1 N={mask1.sum()}")
print(f"  racism mean={racism_4item[mask1].mean():.2f}, SD={racism_4item[mask1].std(ddof=1):.2f}")
mask2 = all_rem_valid & all4_racism & all_IVs_v2
print(f"  DV2 N={mask2.sum()}")

# Maybe the paper uses mean substitution for missing racism items?
# If an item is missing, replace with the mean of the available items for that person
# This would preserve the scale range 0-5 while keeping more cases

# Person-mean imputation for racism items
df['r_racmost'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
df['r_busing'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
df['r_racdif1'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
df['r_racdif2'] = np.where(df['racdif2'].notna(), (df['racdif2'] == 2).astype(float), np.nan)
df['r_racdif3'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 1).astype(float), np.nan)

coded = ['r_racmost','r_busing','r_racdif1','r_racdif2','r_racdif3']
person_mean = df[coded].mean(axis=1)  # mean of valid items
for c in coded:
    df[c + '_imp'] = df[c].fillna(person_mean)
coded_imp = [c + '_imp' for c in coded]
racism_imp = df[coded_imp].sum(axis=1)

# Require at least 4 items valid for imputation
n_valid = df[coded].notna().sum(axis=1)
racism_imp[n_valid < 4] = np.nan

mask1 = all_min_valid & (n_valid >= 4) & all_IVs_v2
print(f"\nMean-imputed (4+ valid): DV1 N={mask1.sum()}")
print(f"  racism mean={racism_imp[mask1].mean():.2f}, SD={racism_imp[mask1].std(ddof=1):.2f}")
mask2 = all_rem_valid & (n_valid >= 4) & all_IVs_v2
print(f"  DV2 N={mask2.sum()}")

# Require at least 3 items
mask1 = all_min_valid & (n_valid >= 3) & all_IVs_v2
print(f"\nMean-imputed (3+ valid): DV1 N={mask1.sum()}")
print(f"  racism mean={racism_imp[mask1].mean():.2f}, SD={racism_imp[mask1].std(ddof=1):.2f}")
mask2 = all_rem_valid & (n_valid >= 3) & all_IVs_v2
print(f"  DV2 N={mask2.sum()}")

# Scale to sum (multiply person mean by 5)
racism_scaled = person_mean * 5
racism_scaled[n_valid < 4] = np.nan
mask1 = all_min_valid & (n_valid >= 4) & all_IVs_v2
print(f"\nScaled mean*5 (4+ valid): DV1 N={mask1.sum()}")
print(f"  racism mean={racism_scaled[mask1].mean():.2f}, SD={racism_scaled[mask1].std(ddof=1):.2f}")
