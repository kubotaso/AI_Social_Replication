import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("gss1993_clean.csv")

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

# Standard controls
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

# Test overlapping Hispanic/Other race coding
# Paper: Hispanic (ethnic in 17,22,25), Other race (race==3)
# These overlap when race==3 AND ethnic in 17,22,25
df['hispanic_overlap'] = df['ethnic'].isin([17, 22, 25]).astype(int)
df['other_race_overlap'] = (df['race'] == 3).astype(int)

# Non-overlapping (current best)
df['hispanic_nonoverlap'] = (df['ethnic'].isin([17, 22, 25]) & (df['race'] != 3)).astype(int)
df['other_race_nonoverlap'] = ((df['race'] == 3) & ~df['ethnic'].isin([17, 22, 25])).astype(int)

df['conservative_protestant'] = (pd.to_numeric(df['fund'], errors='coerce') == 1).astype(int)
df['no_religion'] = (pd.to_numeric(df['relig'], errors='coerce') == 4).astype(int)
df['southern'] = (pd.to_numeric(df['region'], errors='coerce') == 3).astype(int)

# Best racism scale: old 5-item, racdif3==2, min 4
for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].isin([1,2]))
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].isin([1,2]))
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].isin([1,2]))
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].isin([1,2]))
df['r_racdif3'] = (df['racdif3'] == 2).astype(float).where(df['racdif3'].isin([1,2]))

coded = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3']
racism_vals = []
for idx in df.index:
    vals = [df.loc[idx, c] for c in coded]
    n_valid = sum(1 for v in vals if not np.isnan(v))
    if n_valid >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        racism_vals.append(sum(v if not np.isnan(v) else pm for v in vals))
    else:
        racism_vals.append(np.nan)
df['racism_score'] = racism_vals

# Test: overlapping vs non-overlapping Hispanic/Other race
for hisp_name, hisp_col, other_col in [
    ('overlap', 'hispanic_overlap', 'other_race_overlap'),
    ('nonoverlap', 'hispanic_nonoverlap', 'other_race_nonoverlap'),
]:
    predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
                  'female', 'age_var', 'black', hisp_col, other_col,
                  'conservative_protestant', 'no_religion', 'southern']

    for dv_name, label in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
        vars_needed = [dv_name] + predictors
        model_df = df[vars_needed].dropna()
        N = len(model_df)
        y = model_df[dv_name].values
        X = model_df[predictors].values
        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        m = sm.OLS(y_z, sm.add_constant(X_z)).fit()

        racism_b = m.params[1]
        black_b = m.params[7]
        hisp_b = m.params[8]
        other_b = m.params[9]
        age_b = m.params[6]

        racism_p = m.pvalues[1]
        black_p = m.pvalues[7]
        age_p = m.pvalues[6]

        sig_r = "***" if racism_p < 0.001 else "**" if racism_p < 0.01 else "*" if racism_p < 0.05 else ""
        sig_b = "***" if black_p < 0.001 else "**" if black_p < 0.01 else "*" if black_p < 0.05 else ""
        sig_a = "***" if age_p < 0.001 else "**" if age_p < 0.01 else "*" if age_p < 0.05 else ""

        r2 = sm.OLS(y, sm.add_constant(X)).fit().rsquared

        print(f"{hisp_name:12s} {label} N={N} Racism={racism_b:+.3f}{sig_r:3s} "
              f"Black={black_b:+.3f}{sig_b:3s} Hisp={hisp_b:+.3f} Other={other_b:+.3f} "
              f"Age={age_b:+.3f}{sig_a:3s} R2={r2:.3f}")

# What if we check additional racism items?
print("\n=== Additional racism variables ===")
for col in ['racfew', 'rachaf', 'racmar', 'racopen', 'racseg']:
    vals = pd.to_numeric(df[col], errors='coerce')
    valid = vals.dropna()
    print(f"  {col}: N={len(valid)}, unique={sorted(valid.unique())}")
    # Check correlation with racism score
    both = df[['racism_score']].copy()
    both['item'] = vals
    both = both.dropna()
    print(f"    Corr with racism_score: {both['racism_score'].corr(both['item']):.3f}")

# What if racfew, rachaf could replace racmost to increase N?
# racmost has only 824 valid vs 1073 for racfew
# racfew: Would you object to having half your children's schoolmates be Black?
# racmost: Object to school where more than half are Black
# These are similar questions!
print("\n=== racfew as replacement for racmost ===")
df['racfew'] = pd.to_numeric(df['racfew'], errors='coerce')
print(f"racfew values: {sorted(df['racfew'].dropna().unique())}")
print(f"racfew N: {df['racfew'].notna().sum()}")

# Try replacing racmost with racfew
df['r_racfew'] = (df['racfew'] == 1).astype(float).where(df['racfew'].isin([1,2]))
coded_alt = ['r_racfew', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3']
racism_vals_alt = []
for idx in df.index:
    vals = [df.loc[idx, c] for c in coded_alt]
    n_valid = sum(1 for v in vals if not np.isnan(v))
    if n_valid >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        racism_vals_alt.append(sum(v if not np.isnan(v) else pm for v in vals))
    else:
        racism_vals_alt.append(np.nan)
df['racism_score_alt'] = racism_vals_alt

n_valid = df['racism_score_alt'].notna().sum()
mean_v = df['racism_score_alt'].dropna().mean()
sd_v = df['racism_score_alt'].dropna().std(ddof=1)
print(f"racfew scale: N={n_valid} mean={mean_v:.2f} SD={sd_v:.2f}")

predictors_alt = ['racism_score_alt', 'education', 'income_pc', 'occ_prestige',
                  'female', 'age_var', 'black', 'hispanic_nonoverlap', 'other_race_nonoverlap',
                  'conservative_protestant', 'no_religion', 'southern']

for dv_name, label in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    vars_needed = [dv_name] + predictors_alt
    model_df = df[vars_needed].dropna()
    N = len(model_df)
    y = model_df[dv_name].values
    X = model_df[predictors_alt].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    m = sm.OLS(y_z, sm.add_constant(X_z)).fit()

    racism_b = m.params[1]
    black_b = m.params[7]

    racism_p = m.pvalues[1]
    black_p = m.pvalues[7]

    sig_r = "***" if racism_p < 0.001 else "**" if racism_p < 0.01 else "*" if racism_p < 0.05 else ""
    sig_b = "***" if black_p < 0.001 else "**" if black_p < 0.01 else "*" if black_p < 0.05 else ""

    r2 = sm.OLS(y, sm.add_constant(X)).fit().rsquared
    print(f"  racfew {label} N={N} Racism={racism_b:+.3f}{sig_r:3s} Black={black_b:+.3f}{sig_b:3s} R2={r2:.3f}")

# Try: what about using racopen instead of busing?
# racopen: "Would you vote for a Black president?" (different question)
# Actually let's check rachaf: "Would you object to having a family member marry a Black person?"
print("\n=== Other item substitutions ===")
df['rachaf'] = pd.to_numeric(df['rachaf'], errors='coerce')
df['racmar'] = pd.to_numeric(df['racmar'], errors='coerce')

# Check what values rachaf and racmar take
print(f"rachaf values: {sorted(df['rachaf'].dropna().unique())}, N={df['rachaf'].notna().sum()}")
print(f"racmar values: {sorted(df['racmar'].dropna().unique())}, N={df['racmar'].notna().sum()}")

# Try 6-item scale: racmost, busing, racdif1, racdif2, racdif3, racfew
# With one item removed
# Actually this is getting too speculative. Let me focus on what we know works.

# FINAL TEST: What if education should use 'degree' (categorical) instead of 'educ' (years)?
print("\n=== Testing degree vs educ ===")
df['degree'] = pd.to_numeric(df['degree'], errors='coerce')
print(f"degree values: {sorted(df['degree'].dropna().unique())}")
print(f"degree N: {df['degree'].notna().sum()}")

predictors_deg = ['racism_score', 'degree', 'income_pc', 'occ_prestige',
                  'female', 'age_var', 'black', 'hispanic_nonoverlap', 'other_race_nonoverlap',
                  'conservative_protestant', 'no_religion', 'southern']

for dv_name, label in [('dv_remaining', 'M2')]:
    vars_needed = [dv_name] + predictors_deg
    model_df = df[vars_needed].dropna()
    N = len(model_df)
    y = model_df[dv_name].values
    X = model_df[predictors_deg].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    m = sm.OLS(y_z, sm.add_constant(X_z)).fit()

    racism_b = m.params[1]
    educ_b = m.params[2]
    black_b = m.params[7]

    racism_p = m.pvalues[1]
    black_p = m.pvalues[7]

    sig_r = "***" if racism_p < 0.001 else "**" if racism_p < 0.01 else "*" if racism_p < 0.05 else ""
    sig_b = "***" if black_p < 0.001 else "**" if black_p < 0.01 else "*" if black_p < 0.05 else ""

    print(f"  degree {label} N={N} Racism={racism_b:+.3f}{sig_r:3s} Black={black_b:+.3f}{sig_b:3s} Educ={educ_b:+.3f}")
