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
other_valid = df[['education', 'income_pc', 'occ_prestige', 'age_var']].notna().all(axis=1)

# Binary coding with NAs as 0
df['r1'] = np.where(df['racmost'].isna(), 0, (df['racmost'] == 1).astype(int))
df['r2'] = np.where(df['busing'].isna(), 0, (df['busing'] == 2).astype(int))
df['r3'] = np.where(df['racdif1'].isna(), 0, (df['racdif1'] == 2).astype(int))
df['r4'] = np.where(df['racdif2'].isna(), 0, (df['racdif2'] == 2).astype(int))
df['r5'] = np.where(df['racdif3'].isna(), 0, (df['racdif3'] == 1).astype(int))
racism_na0 = df['r1'] + df['r2'] + df['r3'] + df['r4'] + df['r5']

n_valid_items = df[items].notna().sum(axis=1)

# Try various thresholds for minimum valid items, with NAs as 0
for threshold in [0, 1, 2, 3, 4, 5]:
    mask_thresh = n_valid_items >= threshold
    mask1 = all_min_valid & mask_thresh & other_valid
    mask2 = all_rem_valid & mask_thresh & other_valid
    m = racism_na0[mask1].mean()
    s = racism_na0[mask1].std(ddof=1)
    print(f'Threshold {threshold}+: DV1 N={mask1.sum()}, DV2 N={mask2.sum()}, '
          f'racism mean={m:.2f}, SD={s:.2f}')

# Key insight: paper N=644 for model1. Threshold 4+ gives 645. Very close!
# But mean=2.93, SD=0.99 vs paper 2.65, SD=1.56
# With threshold 0 (all cases): N=1001, mean=1.93, SD=1.59

# Hmm. Let me try: maybe the racism scale uses different item sets
# Paper says: (1) object to sending to school, (2) oppose busing,
# (3) diff not due to discrimination, (4) not because lack educ chance,
# (5) lack motivation/will
# These map to: racmost, busing, racdif1, racdif2, racdif3

# Wait - maybe racdif2 coding is wrong. Let me check:
# Paper says: "NOT because lack education chance" = racist
# racdif2: "On the average, blacks have worse jobs, income, and housing
# than white people. Do you think these differences are because most
# blacks don't have the chance for education that it takes to rise out of poverty?"
# 1=Yes, 2=No
# NOT because lack education = racist = value 2. That seems right.

# But paper mean 2.65 with 5 binary items means average item endorsement = 0.53
# With all 5 required: mean=3.04, avg endorsement=0.608
# The high endorsement for racdif2 (0.893) is pulling it up

# Let me try: maybe racdif2 coding is REVERSED
# What if racdif2=1 is racist (YES, it IS because lack education - wait that's not racist)
# Actually: "because most blacks don't have the chance for education"
# If you say YES (=1), you're saying it IS because lack of educational opportunity
# If you say NO (=2), you're saying it's NOT because lack of educational opportunity
# The paper says "NOT because lack education" = racist direction
# So value 2 = "No, it's not because lack of education" = racist
# That's what we have. racdif2=2 is correct for racist direction.

# Let me try a completely different approach: maybe the racism scale
# is NOT binary but uses the original response values
# racmost: 1=object (racist), 2=not object -> keep as 1 or 2
# If we use raw values: mean would be different

# Actually, let me re-read the paper more carefully about the racism scale
# The paper says: "a five-item racism scale that sums responses to items on
# racial attitudes (see Appendix Table A1)"
# "sums responses" might mean summing the RAW response values, not binary indicators!

# Let's try: recode so that higher = more racist, using original scale
# racmost: 1=Object (racist), 2=Not → recode: 1→1, 2→0
# busing: 1=Favor, 2=Oppose (racist) → recode: 1→0, 2→1
# racdif1: 1=Yes (due to discrimination, not racist), 2=No (racist) → 1→0, 2→1
# racdif2: 1=Yes (lack educ, not racist), 2=No (racist) → 1→0, 2→1
# racdif3: 1=Yes (lack motivation, racist), 2=No → 1→1, 2→0
# This is what we already have. Sum of binary = 0-5.

# Wait, what if the paper uses the UNrecoded values directly?
# racmost: 1(racist) + busing: 2(racist) + racdif1: 2(racist) + racdif2: 2(racist) + racdif3: 1(racist)
# Sum of raw values when racist: 1+2+2+2+1 = 8
# Sum of raw values when not racist: 2+1+1+1+2 = 7
# Range of raw sums: 5 to 10

# Let me try: just sum raw values after recoding to a consistent direction
# Recode all so higher = more racist:
# racmost: keep as is (1=racist, higher is less racist... wait 1<2)
# Actually 1=object (racist) is LOWER than 2=not object
# So we need to reverse some items
# Let's try: reverse racmost and racdif3 (where 1=racist)
# So all items: higher value = more racist
# racmost: reverse → 3 - racmost (1→2 racist, 2→1 not)
# busing: keep (2=racist)
# racdif1: keep (2=racist)
# racdif2: keep (2=racist)
# racdif3: reverse → 3 - racdif3 (1→2 racist, 2→1 not)
# Sum range: 5 to 10

# Hmm, that doesn't help. Let me just check descriptive stats
# with different approaches and find which matches mean=2.65, SD=1.56

# Method 1: binary, all 5 required
df['r1_strict'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
df['r2_strict'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
df['r3_strict'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
df['r4_strict'] = np.where(df['racdif2'].notna(), (df['racdif2'] == 2).astype(float), np.nan)
df['r5_strict'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 1).astype(float), np.nan)

coded = ['r1_strict','r2_strict','r3_strict','r4_strict','r5_strict']
all_valid = df[coded].notna().all(axis=1)
racism_strict = df.loc[all_valid, coded].sum(axis=1)
print(f'\nBinary, all 5 valid: mean={racism_strict.mean():.2f}, SD={racism_strict.std(ddof=1):.2f}')

# Paper says Table 1 (all 18 genres, standardized) uses the racism scale
# Table A1 in appendix: descriptive stats for all variables
# mean=2.65, SD=1.56 for racism scale
# This is the sample-wide stat (or for the Table 1 sample?)

# N for Table 1 was ~838. Let's check what racism looks like there
all18 = df[minority + remaining].isin([1,2,3,4,5]).all(axis=1)
# With NAs as 0, for Table 1 sample
mask_t1 = all18 & other_valid
print(f'\nTable 1 sample (all 18 valid + other IVs): N={mask_t1.sum()}')
print(f'  racism_na0 mean={racism_na0[mask_t1].mean():.2f}, SD={racism_na0[mask_t1].std(ddof=1):.2f}')
# With all 5 required
mask_t1_5 = all18 & all_valid & other_valid
print(f'  all 5 valid: N={mask_t1_5.sum()}, mean={racism_strict[mask_t1_5[all_valid]].mean():.2f}, SD={racism_strict[mask_t1_5[all_valid]].std(ddof=1):.2f}')

# I think the paper is reporting the Table 1 sample stats where N~838
# and racism NAs are treated as 0. mean=2.01 doesn't match 2.65 either.

# Let me try: maybe busing coding is flipped (1=oppose?)
# GSS: BUSING "In general, do you favor or oppose the busing of
# Black and white school children from one school district to another?"
# 1=Favor, 2=Oppose
# Oppose = racist direction = 2. That's correct.

# Let me try: use the raw values (1 or 2) and sum them up
# Without any binary recoding
raw_sum = df['racmost'] + df['busing'] + df['racdif1'] + df['racdif2'] + df['racdif3']
print(f'\nRaw sum (no recode): mean={raw_sum.dropna().mean():.2f}, SD={raw_sum.dropna().std(ddof=1):.2f}')
# Range would be 5-10

# Try: recode all items so racist = 1, non-racist = 0, then multiply by 2?
# No that makes no sense

# Let me check if income variable matters for the N
# Maybe use income91 instead of realinc/hompop
df['income91'] = pd.to_numeric(df['income91'], errors='coerce')
print(f'\nincome91 valid: {df["income91"].notna().sum()}')
print(f'realinc valid: {df["realinc"].notna().sum()}')

# With income91 instead
other_valid_v2 = df[['education', 'income91', 'occ_prestige', 'age_var']].notna().all(axis=1)
mask1 = all_min_valid & all_valid & other_valid_v2
print(f'DV1 + 5racism + income91: N={mask1.sum()}')

# With realinc (not per capita)
other_valid_v3 = df[['education', 'realinc', 'occ_prestige', 'age_var']].notna().all(axis=1)
mask1 = all_min_valid & all_valid & other_valid_v3
print(f'DV1 + 5racism + realinc: N={mask1.sum()}')

# Hmm: maybe income per capita doesn't add NAs beyond realinc
print(f'hompop valid: {df["hompop"].notna().sum()}')
print(f'income_pc valid: {df["income_pc"].notna().sum()}')

# The problem is purely the racism items. N=708 have all 5 valid
# but paper N=644 needs those 708 minus those missing on other vars
# 708 valid racism -> 459 when combined with DV1 and other IVs

# Wait: 708 is all 5 racism valid TOTAL, but only 459 also have DV1 valid + other IVs
# 1001 have DV1 + other IVs valid, 459 have DV1 + other IVs + all 5 racism
# So 1001-459=542 are losing from racism requirement
# To get 644, we need 644 from 1001, meaning 357 lost
# That's much less than 542 lost

# The only way to get N close to 644 is to be less strict about racism
# Require 4+ items: N=645, but stats don't match
# Unless... the paper reports stats from a DIFFERENT sample
# The paper's Table A1 descriptive stats might be from the Table 1 sample

# Let me try: require busing + racdif1 + racdif2 + racdif3 valid (4 items that have high validity)
# and treat racmost NA as 0
items_4 = ['busing','racdif1','racdif2','racdif3']
all4 = df[items_4].notna().all(axis=1)
# racism = racmost(NA->0) + busing + racdif1 + racdif2 + racdif3
racism_4req = df['r1'] + df[['r2_strict','r3_strict','r4_strict','r5_strict']].sum(axis=1)
# Hmm, let me be more careful
df['r1_na0'] = np.where(df['racmost'].isna(), 0, (df['racmost'] == 1).astype(int))
df['r2_v'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(int), np.nan)
df['r3_v'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(int), np.nan)
df['r4_v'] = np.where(df['racdif2'].notna(), (df['racdif2'] == 2).astype(int), np.nan)
df['r5_v'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 1).astype(int), np.nan)

racism_hybrid = df['r1_na0'] + df['r2_v'] + df['r3_v'] + df['r4_v'] + df['r5_v']
# This requires busing+racdif1-3 valid, racmost NA=0

mask1 = all_min_valid & all4 & other_valid
print(f'\nHybrid (4 items required, racmost NA=0):')
print(f'  DV1 N={mask1.sum()}')
print(f'  mean={racism_hybrid[mask1].mean():.2f}, SD={racism_hybrid[mask1].std(ddof=1):.2f}')

all_rem_valid = df[remaining].isin([1,2,3,4,5]).all(axis=1)
mask2 = all_rem_valid & all4 & other_valid
print(f'  DV2 N={mask2.sum()}')

# Try: 3 items required (racdif1 + racdif2 + racdif3), rest NA=0
items_3 = ['racdif1','racdif2','racdif3']
all3 = df[items_3].notna().all(axis=1)
df['r1_na0b'] = np.where(df['racmost'].isna(), 0, (df['racmost'] == 1).astype(int))
df['r2_na0b'] = np.where(df['busing'].isna(), 0, (df['busing'] == 2).astype(int))
racism_hybrid2 = df['r1_na0b'] + df['r2_na0b'] + df['r3_v'] + df['r4_v'] + df['r5_v']

mask1 = all_min_valid & all3 & other_valid
print(f'\nHybrid (racdif1-3 required, racmost+busing NA=0):')
print(f'  DV1 N={mask1.sum()}')
print(f'  mean={racism_hybrid2[mask1].mean():.2f}, SD={racism_hybrid2[mask1].std(ddof=1):.2f}')

mask2 = all_rem_valid & all3 & other_valid
print(f'  DV2 N={mask2.sum()}')
