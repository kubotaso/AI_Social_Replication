import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('gss1993_clean.csv')

minority_genres = ['rap', 'reggae', 'blues', 'jazz', 'gospel', 'latin']
remaining_genres = ['musicals', 'oldies', 'classicl', 'bigband', 'newage', 'opera',
                    'blugrass', 'folk', 'moodeasy', 'conrock', 'hvymetal', 'country']

for g in minority_genres + remaining_genres:
    df[g] = pd.to_numeric(df[g], errors='coerce')

# Check: what if "hispanic" is coded differently?
# The paper references "Hispanic" but GSS ethnic codes could vary
df['ethnic'] = pd.to_numeric(df['ethnic'], errors='coerce')
print("=== Ethnic variable distribution ===")
print(df['ethnic'].value_counts().sort_index())
print()

# Current: ethnic in [17, 22, 25]
# 17 = Mexico, 22 = Puerto Rico, 25 = ?
# GSS ETHNIC codes for Hispanic:
# 17 = Mexico, 22 = Puerto Rico, 25 = Other Spanish
# But also: 2 = Spanish, 15 = Central/South American
# Let's check broader Hispanic definitions
for hisp_codes in [[17, 22, 25], [2, 17, 22, 25], [2, 15, 17, 22, 25], [17, 22, 25, 38]]:
    n = df['ethnic'].isin(hisp_codes).sum()
    print(f'Hispanic codes {hisp_codes}: n={n}')

# Check Hispanic variable from GSS - maybe there's a dedicated variable
# Not in our data. Let's check other approaches

# Check: what if race=3 means something different?
df['race'] = pd.to_numeric(df['race'], errors='coerce')
print("\n=== Race variable ===")
print(df['race'].value_counts().sort_index())
print()

# What if "Other race" (race==3) partially overlaps with Hispanic?
# Let's check overlap
df['other_race'] = (df['race'] == 3).astype(int)
df['hispanic_v1'] = df['ethnic'].isin([17, 22, 25]).astype(int)
ct = pd.crosstab(df['other_race'], df['hispanic_v1'])
print("Crosstab other_race vs hispanic:")
print(ct)
print()

# What if Black respondents are excluded from Model 2 sample?
# No - the paper shows Black coefficient, so they're included

# What if there's something about how ethnic NA is handled?
print(f"ethnic NA count: {df['ethnic'].isna().sum()}")
print(f"ethnic == 97 (uncodeable): {(df['ethnic']==97).sum()}")

# Maybe the issue is that ethnic NaN should be treated as 0 (non-Hispanic)
# which is what we're doing. Let me verify.

# Key investigation: what makes the Black coefficient different?
# In our Model 2, Black=0.103-0.128. Paper says 0.042.
# This means Black people in our sample dislike more of the 12 remaining genres
# than the paper suggests. Maybe the racism scale should be absorbing more of this.

# Let me check: what's the bivariate relationship between Black and racism?
for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())
df['r_racdif3v2'] = (df['racdif3'] == 2).astype(float).where(df['racdif3'].notna())
df['r_racdif4'] = (df['racdif4'] == 1).astype(float).where(df['racdif4'].notna())

df['black'] = (df['race'] == 2).astype(int)
print("\n=== Black vs racism items ===")
for item in ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3v2', 'r_racdif4']:
    t = df.dropna(subset=[item])
    by_race = t.groupby('black')[item].mean()
    print(f'{item}: white={by_race.get(0, "NA"):.3f}, black={by_race.get(1, "NA"):.3f}')

# What if Bryson used a DIFFERENT set of racism items?
# The paper describes: "object to living near Blacks" + "oppose busing" + 3 RACDIF items
# But doesn't specify which 3 of the 4 RACDIF items

# Let me re-read: "racism scale composed of five dichotomous items"
# Items listed: (1) object to sending children to school where more than half are Black
# (2) oppose busing
# (3-5) whether racial differences in income/housing/employment are due to discrimination,
#        educational opportunities, and motivation/willpower

# So items (3)-(5) are: racdif1 (discrimination), racdif2 (education), racdif3 (motivation)
# NOT racdif4 (innate ability)

# But alpha=0.54 only matches with racdif4 instead of racdif2...
# Unless the paper's alpha is wrong or computed on a slightly different sample

# Let me check: what if we use the original 5 items but with a different
# missing data approach that gets N closer to 644?

# Build scale with all 5 original items, person-mean imputation min 3
for min_v in [3, 4, 5]:
    coded = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3v2']
    vals = []
    for idx in df.index:
        item_vals = [df.loc[idx, c] for c in coded]
        n_valid = sum(1 for v in item_vals if not np.isnan(v))
        if n_valid >= min_v:
            valid_v = [v for v in item_vals if not np.isnan(v)]
            pm = np.mean(valid_v)
            vals.append(sum(v if not np.isnan(v) else pm for v in item_vals))
        else:
            vals.append(np.nan)
    df['racism_test'] = vals
    t = df['racism_test'].dropna()
    print(f'\nOriginal 5 items (racdif3 flipped, racdif2 kept), min {min_v}: mean={t.mean():.3f}, SD={t.std(ddof=1):.3f}, n={len(t)}')

# Let me try: the 5 items from the paper literally (racmost, busing, racdif1, racdif2, racdif3)
# with racdif3 FLIPPED (2=racist to indicate lack of motivation is NOT the reason)
# Actually wait - racdif3: "do you think differences are because blacks lack motivation"
# 1 = yes (racist), 2 = no (not racist)
# So flipped = 2 is actually the NON-racist direction
# If we code racdif3==2 as racist, we're saying "not lacking motivation" is racist - that's wrong

# Let me reconsider: maybe racdif3 flipped means:
# Original GSS: 1=yes (differences due to lack of motivation), 2=no
# Racist direction: 1 (yes, blaming Black people)
# Anti-racist direction: 2 (no, not blaming Black people)
# So racdif3==1 as racist is correct

# But if we flip to racdif3==2 as "racist"... that alpha matches 0.54
# This suggests either the items are coded differently in the data
# or the paper made an error in describing the coding

# CRITICAL INSIGHT: Let me check the actual GSS codebook
# In GSS 1993, racdif3 is: "On the average (negroes/blacks/African-Americans)
# have worse jobs, income, and housing than white people.
# Do you think these differences are...
# Because most (negroes/blacks/African-Americans) just don't have the
# motivation or will power to pull themselves up out of poverty?"
# 1 = YES (this IS the racist direction - blaming Black people)
# 2 = NO

# So racdif3==1 IS the racist direction. But alpha only matches 0.54
# when we code racdif3==2 as racist. This is paradoxical.

# HYPOTHESIS: The data file might have racdif3 recoded/reversed from the original GSS
# If the data provider reversed the coding: 1=no, 2=yes
# Then racdif3==2 would be racist (yes, blame lack of motivation)

# Let's check: are there NEGATIVE correlations between racdif3 and other racism items
# that would suggest reversed coding?
print("\n=== Inter-item correlations (checking racdif3 coding) ===")
items = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3v2', 'r_racdif4']
item_df = df[items].dropna()
corr = item_df.corr()
print(corr.round(3))

# Now with racdif3 = 1 as racist
items2 = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2']
df['r_racdif3v1'] = (df['racdif3'] == 1).astype(float).where(df['racdif3'].notna())
items2b = items2 + ['r_racdif3v1', 'r_racdif4']
item_df2 = df[items2b].dropna()
print("\n=== With racdif3 = 1 as racist ===")
corr2 = item_df2.corr()
print(corr2.round(3))
