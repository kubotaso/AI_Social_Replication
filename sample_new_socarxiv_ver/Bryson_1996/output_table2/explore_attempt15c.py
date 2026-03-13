import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("gss1993_clean.csv")

# Check how many have all 5 racism items valid
for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

# The paper's 5 items: racmost, busing, racdif1, racdif2, racdif3
items_paper = ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3']
valid_all5 = df[items_paper].notna().all(axis=1)
print(f"All 5 paper items valid: {valid_all5.sum()}")

# Check with 1/2 coding only (excluding DK which might be coded differently)
for item in items_paper:
    vals = df[item].dropna().unique()
    print(f"  {item}: unique values = {sorted(vals)}, n_valid={df[item].notna().sum()}")

# What about items being in {1,2} only?
valid_12 = True
for item in items_paper:
    valid_12 = valid_12 & df[item].isin([1, 2])
print(f"All 5 items in {{1,2}}: {valid_12.sum()}")

# The paper says 912 valid cases. Let me check which combo gets 912
combos_to_check = [
    ('racmost+busing+racdif1+racdif2+racdif3', ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3']),
    ('racmost+busing+racdif1+racdif3+racdif4', ['racmost', 'busing', 'racdif1', 'racdif3', 'racdif4']),
    ('racmost+busing+racdif1+racdif2+racdif4', ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif4']),
]

for name, items in combos_to_check:
    valid = df[items].notna().all(axis=1)
    valid_12_only = True
    for item in items:
        valid_12_only = valid_12_only & df[item].isin([1, 2])
    print(f"{name}: all_notna={valid.sum()}, all_in_12={valid_12_only.sum()}")

# What if "912 valid cases" means having at least 1 valid racism item?
# Or at least 3?
for min_valid in [1, 2, 3, 4, 5]:
    n = df[items_paper].notna().sum(axis=1) >= min_valid
    print(f"At least {min_valid} of 5 valid: {n.sum()}")

# KEY INSIGHT: The paper says racdif3 is "because most Blacks just don't have the
# motivation or will power to pull themselves up out of poverty"
# racdif3 in GSS: 1=Yes, 2=No
# So racdif3==1 means "yes, lack motivation" which IS the racist direction
# But our best model uses racdif3==2 which means "No, NOT lack motivation"
# which would be the NON-racist direction!
# Wait - the paper says the item is coded "in the same direction" and the
# racist response is "yes" = 1.
# BUT: our instruction_summary says racdif3 racist if value == 1
# AND the best attempt (8) uses racdif3 == 2 (flipped!)

# Let me test: what if we use the CORRECT racdif3 coding (==1 racist)
# but with the paper's full set of items?
# The paper lists 5 items but says "factor analysis suggested removal of one item"
# Maybe the removed item IS racdif3, leaving only 4 items?

print("\n=== Testing 4-item scales (one item removed) ===")
all_items = ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3']
racist_vals = {'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif2': 2, 'racdif3': 1}

# Standard controls
minority_genres = ['rap', 'reggae', 'blues', 'jazz', 'gospel', 'latin']
remaining_genres = ['musicals', 'oldies', 'classicl', 'bigband', 'newage', 'opera',
                    'blugrass', 'folk', 'moodeasy', 'conrock', 'hvymetal', 'country']

for g in minority_genres + remaining_genres:
    df[g] = pd.to_numeric(df[g], errors='coerce')

minority_valid = df[minority_genres].isin([1,2,3,4,5]).all(axis=1)
df['dv_minority'] = np.nan
df.loc[minority_valid, 'dv_minority'] = (df.loc[minority_valid, minority_genres] >= 4).sum(axis=1)

remaining_valid = df[remaining_genres].isin([1,2,3,4,5]).all(axis=1)
df['dv_remaining'] = np.nan
df.loc[remaining_valid, 'dv_remaining'] = (df.loc[remaining_valid, remaining_genres] >= 4).sum(axis=1)

df['education'] = pd.to_numeric(df['educ'], errors='coerce')
df['realinc'] = pd.to_numeric(df['realinc'], errors='coerce')
df['hompop'] = pd.to_numeric(df['hompop'], errors='coerce')
df['income_pc'] = df['realinc'] / df['hompop']
df['occ_prestige'] = pd.to_numeric(df['prestg80'], errors='coerce')
df['female'] = (pd.to_numeric(df['sex'], errors='coerce') == 2).astype(int)
df['age_var'] = pd.to_numeric(df['age'], errors='coerce')
df['race'] = pd.to_numeric(df['race'], errors='coerce')
df['black'] = (df['race'] == 2).astype(int)
df['ethnic'] = pd.to_numeric(df['ethnic'], errors='coerce')
df['hispanic'] = (df['ethnic'].isin([17, 22, 25]) & (df['race'] != 3)).astype(int)
df['other_race'] = ((df['race'] == 3) & ~df['ethnic'].isin([17, 22, 25])).astype(int)
df['conservative_protestant'] = (pd.to_numeric(df['fund'], errors='coerce') == 1).astype(int)
df['no_religion'] = (pd.to_numeric(df['relig'], errors='coerce') == 4).astype(int)
df['southern'] = (pd.to_numeric(df['region'], errors='coerce') == 3).astype(int)

predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']

# Try removing each item one at a time
for remove_item in all_items:
    remaining_items = [i for i in all_items if i != remove_item]

    coded = []
    for item in remaining_items:
        col = f'r_{item}'
        df[col] = (df[item] == racist_vals[item]).astype(float).where(df[item].isin([1,2]))
        coded.append(col)

    # Simple sum, all 4 required
    df['racism_score'] = df[coded].sum(axis=1).where(df[coded].notna().all(axis=1))

    n_valid = df['racism_score'].notna().sum()
    mean_val = df['racism_score'].dropna().mean()
    sd_val = df['racism_score'].dropna().std(ddof=1)

    for dv_name, label in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
        vars_needed = [dv_name] + predictors
        model_df = df[vars_needed].dropna()
        N = len(model_df)
        if N < 100:
            continue
        y = model_df[dv_name].values
        X = model_df[predictors].values
        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        m = sm.OLS(y_z, sm.add_constant(X_z)).fit()

        racism_b = m.params[1]
        black_b = m.params[7]
        age_b = m.params[6]
        racism_p = m.pvalues[1]
        black_p = m.pvalues[7]
        age_p = m.pvalues[6]

        r2 = sm.OLS(y, sm.add_constant(X)).fit().rsquared

        sig_r = "***" if racism_p < 0.001 else "**" if racism_p < 0.01 else "*" if racism_p < 0.05 else ""
        sig_b = "***" if black_p < 0.001 else "**" if black_p < 0.01 else "*" if black_p < 0.05 else ""
        sig_a = "***" if age_p < 0.001 else "**" if age_p < 0.01 else "*" if age_p < 0.05 else ""

        print(f"  remove={remove_item:10s} {label} N={N:4d} n_racism={n_valid} "
              f"mean={mean_val:.2f} SD={sd_val:.2f} "
              f"Racism={racism_b:+.3f}{sig_r:3s} Black={black_b:+.3f}{sig_b:3s} "
              f"Age={age_b:+.3f}{sig_a:3s} R2={r2:.3f}")

# Also try: racdif3 coded as ==2 in the 4-item variant (without racdif3)
# this doesn't apply. But let's try the FULL 5-item with racdif3==1 (correct direction)
# and require all 5 valid
print("\n=== Full 5-item, racdif3==1 (correct), all 5 required ===")
coded = []
for item in all_items:
    col = f'r_{item}'
    df[col] = (df[item] == racist_vals[item]).astype(float).where(df[item].isin([1,2]))
    coded.append(col)

df['racism_score'] = df[coded].sum(axis=1).where(df[coded].notna().all(axis=1))
n_valid = df['racism_score'].notna().sum()
print(f"N with all 5 valid: {n_valid}")
print(f"Mean: {df['racism_score'].dropna().mean():.2f}, SD: {df['racism_score'].dropna().std(ddof=1):.2f}")

for dv_name, label in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    vars_needed = [dv_name] + predictors
    model_df = df[vars_needed].dropna()
    N = len(model_df)
    y = model_df[dv_name].values
    X = model_df[predictors].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    m = sm.OLS(y_z, sm.add_constant(X_z)).fit()
    r2 = sm.OLS(y, sm.add_constant(X)).fit().rsquared

    plabels = ['Racism', 'Educ', 'IncPC', 'Prest', 'Female', 'Age', 'Black', 'Hisp', 'Other', 'ConsProt', 'NoRel', 'South']
    print(f"\n  {label} N={N} R2={r2:.3f}")
    for i, pl in enumerate(plabels):
        sig = "***" if m.pvalues[i+1] < 0.001 else "**" if m.pvalues[i+1] < 0.01 else "*" if m.pvalues[i+1] < 0.05 else ""
        print(f"    {pl:10s} {m.params[i+1]:+.3f} {sig}")

# Also: what about racdif3==2 being "Not because lack motivation" = 1 for racist
# i.e. reverse the standard meaning? The GSS codebook says:
# racdif3: "On the average (negroes/blacks/African-Americans) have worse jobs, income,
# and housing than white people. Do you think these differences are...
# because most (negroes/blacks/African-Americans) just don't have the motivation
# or willpower to pull themselves up out of poverty?"
# 1 = Yes  2 = No
# So "Yes" (1) = agrees with racial stereotype = racist direction
# The paper says coding "in the same direction" and "motivation/will power"
# So racdif3==1 IS racist. Our best model uses ==2 which is WRONG!

# But wait -- with racdif3==1, M1 racism drops to 0.040 vs 0.130 target.
# With racdif3==2, M1 racism is 0.117. This is confusing.
# Let's check: maybe the paper actually used racdif3==2 as racist direction?
# That would mean "No, NOT because lack motivation" is considered racist.
# This seems backwards, but maybe racdif3 is reverse-coded in the GSS?

# Check racdif3 correlation with other racism items
print("\n=== Correlations of racdif3 with other items ===")
for rv3 in [1, 2]:
    df['r3_test'] = (df['racdif3'] == rv3).astype(float).where(df['racdif3'].notna())
    for other_item, other_rv in [('racmost', 1), ('busing', 2), ('racdif1', 2), ('racdif2', 2)]:
        df['other_test'] = (df[other_item] == other_rv).astype(float).where(df[other_item].notna())
        corr = df[['r3_test', 'other_test']].dropna().corr().iloc[0,1]
        print(f"  racdif3=={rv3} vs {other_item}=={other_rv}: r={corr:.3f}")
    print()
