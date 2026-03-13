import pandas as pd
import numpy as np

df = pd.read_csv('gss1993_clean.csv')

items = ['racmost','busing','racdif1','racdif2','racdif3']
for v in items:
    df[v] = pd.to_numeric(df[v], errors='coerce')

all_valid = df[items].notna().all(axis=1)

# Try: racdif3=2 as racist (flip direction)
# This means: "Do you think these differences are because most Blacks
# just don't have the motivation or will power?"
# 1=Yes, 2=No
# If 2=racist: "No, it's NOT because lack of motivation" = ???
# That doesn't make sense as racist...

# But let's check if it gives better alpha
df['c_racmost'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
df['c_busing'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
df['c_racdif1'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
df['c_racdif2'] = np.where(df['racdif2'].notna(), (df['racdif2'] == 2).astype(float), np.nan)
df['c_racdif3_flip'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 2).astype(float), np.nan)

coded_flip = ['c_racmost','c_busing','c_racdif1','c_racdif2','c_racdif3_flip']
valid_flip = df.loc[all_valid, coded_flip]
total_sum = valid_flip.sum(axis=1)

print('With racdif3 FLIPPED (2=racist):')
print(f'  mean={total_sum.mean():.2f}, SD={total_sum.std(ddof=1):.2f}')
k = 5
iv = valid_flip.var(ddof=1).sum()
tv = total_sum.var(ddof=1)
alpha = (k/(k-1)) * (1 - iv/tv)
print(f'  alpha={alpha:.2f}')
print('  Corr matrix:')
print(valid_flip.corr().round(3))

# Try: flip both racdif3 AND racdif2
# racdif2=1 as racist, racdif3=2 as racist
df['c_racdif2_flip'] = np.where(df['racdif2'].notna(), (df['racdif2'] == 1).astype(float), np.nan)
coded_flip2 = ['c_racmost','c_busing','c_racdif1','c_racdif2_flip','c_racdif3_flip']
valid_flip2 = df.loc[all_valid, coded_flip2]
total_sum2 = valid_flip2.sum(axis=1)
print('\nWith racdif2 AND racdif3 FLIPPED:')
print(f'  mean={total_sum2.mean():.2f}, SD={total_sum2.std(ddof=1):.2f}')
iv2 = valid_flip2.var(ddof=1).sum()
tv2 = total_sum2.var(ddof=1)
alpha2 = (k/(k-1)) * (1 - iv2/tv2)
print(f'  alpha={alpha2:.2f}')
print('  Corr matrix:')
print(valid_flip2.corr().round(3))

# The negative correlation of racdif3 with others is puzzling.
# Let me check the actual GSS coding more carefully
# In GSS: racdif3 might have DIFFERENT codes than I think
# Let me look at the actual values and see if there's a pattern

# Cross-tab racdif3 with racdif1 (should be positively correlated if both coded right)
ct = pd.crosstab(df['racdif1'], df['racdif3'], margins=True)
print('\nCross-tab racdif1 vs racdif3:')
print(ct)
# racdif1: 1=Yes(discrimination), 2=No(not discrimination, racist)
# racdif3: 1=Yes(lack motivation, racist), 2=No(not lack motivation)
# If both coded toward racist: racdif1=2 should go with racdif3=1
# But correlation is -0.288, meaning racdif1=2 goes with racdif3=2
# This means: people who deny discrimination also deny lack of motivation
# That's actually POSSIBLE - some people just disagree with all explanations

# Actually wait - the correlation structure makes more sense now:
# racmost (object to school) and busing (oppose busing) are positively correlated (0.148)
# racdif1 (not discrimination) is positive with racmost (0.209) and busing (0.244)
# racdif3 (lack motivation) is NEGATIVE with racdif1 (-0.288)
# This means: people who say "not due to discrimination" tend to say "not due to lack of motivation"
# They might instead say it's due to "inborn ability" (racdif4) or something else

# The KEY INSIGHT: racdif1=2 "not due to discrimination" is RACIST
# racdif3=1 "due to lack of motivation" is RACIST
# But they're negatively correlated because they represent DIFFERENT racist positions
# Someone who denies discrimination doesn't necessarily blame lack of motivation
# This explains the low alpha (0.01) - the items don't form a unidimensional scale

# The paper reports alpha=0.54 which is much higher
# Maybe the paper uses a DIFFERENT sample or processes the data differently
# Or maybe I have the wrong variables

# Let me check: maybe racdif3 in the GSS is actually a DIFFERENT question
# Let me look at what the original racdif questions are about

# Actually, let me try dropping racdif3 and adding racdif4 instead
# racdif4: "Do you think these differences are because most blacks have
# less in-born ability to learn?" 1=Yes(racist), 2=No
df['c_racdif4'] = np.where(pd.to_numeric(df['racdif4'], errors='coerce').notna(),
                           (pd.to_numeric(df['racdif4'], errors='coerce') == 1).astype(float), np.nan)

# 5 items: racmost, busing, racdif1, racdif2, racdif4 (drop racdif3)
coded_v3 = ['c_racmost','c_busing','c_racdif1','c_racdif2','c_racdif4']
df['racdif4'] = pd.to_numeric(df['racdif4'], errors='coerce')
items_v3 = ['racmost','busing','racdif1','racdif2','racdif4']
all_valid_v3 = df[items_v3].notna().all(axis=1)

valid_v3 = df.loc[all_valid_v3, coded_v3]
total_sum_v3 = valid_v3.sum(axis=1)
print(f'\nDrop racdif3, add racdif4:')
print(f'  mean={total_sum_v3.mean():.2f}, SD={total_sum_v3.std(ddof=1):.2f}')
iv3 = valid_v3.var(ddof=1).sum()
tv3 = total_sum_v3.var(ddof=1)
alpha3 = (k/(k-1)) * (1 - iv3/tv3)
print(f'  alpha={alpha3:.2f}')
print(f'  N all valid: {all_valid_v3.sum()}')
print('  Corr matrix:')
print(valid_v3.corr().round(3))

# racdif4 might not be mentioned in the paper though
# Paper lists: object to school, oppose busing, not due to discrimination,
# not because lack education, lack motivation
# These are clearly racmost, busing, racdif1, racdif2, racdif3

# The issue might be that the GSS has DIFFERENT response codes in 1993
# Maybe racdif3 values are reversed in this dataset

# Let me try one more thing: what if the ENTIRE scale uses raw response values
# not binary, and is coded so that items are in a consistent direction?
# racmost: 1=Object(racist), 2=Not object -> racist=1, not=2
# busing: 1=Favor, 2=Oppose(racist) -> not=1, racist=2
# racdif1: 1=Yes(discrim), 2=No(not discrim, racist) -> not=1, racist=2
# racdif2: 1=Yes(educ), 2=No(not educ, racist) -> not=1, racist=2
# racdif3: 1=Yes(motiv, racist), 2=No(not motiv) -> racist=1, not=2

# So racmost and racdif3 have racist=1, others have racist=2
# If we just SUM raw values:
# racist on all: 1+2+2+2+1 = 8
# not racist on all: 2+1+1+1+2 = 7
# Range 5-10, but effectively same as binary after rescaling

# The only explanation for alpha=0.54 is either:
# 1. Different items
# 2. Different coding of racdif3 (reversed)
# 3. Different data (maybe a different version of GSS 1993)

# Let me also check: perhaps the paper's reported alpha is from a LARGER sample
# (all valid on racism items) while the regression uses a smaller listwise sample
# The alpha of 0.54 might be correct for a different population
# But our computed alpha is 0.01 on 708 cases - that's definitely wrong

# Let me try: what if racdif3 coding in THIS dataset is reversed?
# Maybe in this dataset: racdif3=1 means "No" and racdif3=2 means "Yes"?
# That would explain the negative correlation
print('\n\nDebug: cross-tab racdif1 vs racdif3 for interpretation:')
print('racdif1: 1=Yes(because discrimination), 2=No(not discrimination)')
print('racdif3: 1=Yes(lack motivation), 2=No(not lack motivation)')
ct = pd.crosstab(df.loc[all_valid, 'racdif1'], df.loc[all_valid, 'racdif3'])
print(ct)
# If racdif1=2(not discrimination, racist) and racdif3=1(lack motivation, racist):
# these should be positively associated
# The cell [2,1] should be large
print('\nProportions:')
ct_norm = pd.crosstab(df.loc[all_valid, 'racdif1'], df.loc[all_valid, 'racdif3'], normalize='all')
print(ct_norm.round(3))
