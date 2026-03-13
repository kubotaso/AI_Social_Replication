import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('gss1993_clean.csv')

minority_genres = ['rap', 'reggae', 'blues', 'jazz', 'gospel', 'latin']
remaining_genres = ['musicals', 'oldies', 'classicl', 'bigband', 'newage', 'opera',
                    'blugrass', 'folk', 'moodeasy', 'conrock', 'hvymetal', 'country']

for g in minority_genres + remaining_genres:
    df[g] = pd.to_numeric(df[g], errors='coerce')

minority_valid = df[minority_genres].isin([1,2,3,4,5]).all(axis=1)
remaining_valid = df[remaining_genres].isin([1,2,3,4,5]).all(axis=1)
df['dv_minority'] = np.nan
df.loc[minority_valid, 'dv_minority'] = (df.loc[minority_valid, minority_genres] >= 4).sum(axis=1)
df['dv_remaining'] = np.nan
df.loc[remaining_valid, 'dv_remaining'] = (df.loc[remaining_valid, remaining_genres] >= 4).sum(axis=1)

for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())
df['r_racdif3'] = (df['racdif3'] == 2).astype(float).where(df['racdif3'].notna())
df['r_racdif4'] = (df['racdif4'] == 1).astype(float).where(df['racdif4'].notna())

df['education'] = pd.to_numeric(df['educ'], errors='coerce')
df['realinc'] = pd.to_numeric(df['realinc'], errors='coerce')
df['hompop'] = pd.to_numeric(df['hompop'], errors='coerce')
df['income_pc'] = df['realinc'] / df['hompop']
df['occ_prestige'] = pd.to_numeric(df['prestg80'], errors='coerce')
df['female'] = (pd.to_numeric(df['sex'], errors='coerce') == 2).astype(int)
df['age_var'] = pd.to_numeric(df['age'], errors='coerce')
df['black'] = (pd.to_numeric(df['race'], errors='coerce') == 2).astype(int)
df['ethnic'] = pd.to_numeric(df['ethnic'], errors='coerce')
df['hispanic'] = df['ethnic'].isin([17, 22, 25]).astype(int)
df['other_race'] = (pd.to_numeric(df['race'], errors='coerce') == 3).astype(int)
df['conservative_protestant'] = (pd.to_numeric(df['fund'], errors='coerce') == 1).astype(int)
df['no_religion'] = (pd.to_numeric(df['relig'], errors='coerce') == 4).astype(int)
df['southern'] = (pd.to_numeric(df['region'], errors='coerce') == 3).astype(int)

predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']

predictor_labels = ['Racism score', 'Education', 'Household income per cap',
                    'Occupational prestige', 'Female', 'Age', 'Black', 'Hispanic',
                    'Other race', 'Conservative Protestant', 'No religion', 'Southern']

def build_scale(df, coded_items, min_valid):
    vals_list = []
    for idx in df.index:
        vals = [df.loc[idx, c] for c in coded_items]
        n_v = sum(1 for v in vals if not np.isnan(v))
        if n_v >= min_valid:
            valid_v = [v for v in vals if not np.isnan(v)]
            pm = np.mean(valid_v)
            vals_list.append(sum(v if not np.isnan(v) else pm for v in vals))
        else:
            vals_list.append(np.nan)
    return pd.Series(vals_list, index=df.index)

def run_both_models(df, predictors, pred_labels):
    for dv, dv_name in [('dv_minority', 'Model 1'), ('dv_remaining', 'Model 2')]:
        vars_needed = [dv] + predictors
        model_df = df[vars_needed].dropna()
        N = len(model_df)

        y = model_df[dv].values
        X = model_df[predictors].values
        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        model = sm.OLS(y_z, sm.add_constant(X_z)).fit()

        X_const = sm.add_constant(X)
        model_raw = sm.OLS(y, X_const).fit()

        betas = dict(zip(pred_labels, model.params[1:]))
        pvals = dict(zip(pred_labels, model.pvalues[1:]))

        print(f"  {dv_name}: N={N}, R2={model_raw.rsquared:.3f}, adjR2={model_raw.rsquared_adj:.3f}")
        for label in pred_labels:
            b = betas[label]
            p = pvals[label]
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"    {label:<30} {b:>8.3f} {sig:>5}")
        print(f"    Constant: {model_raw.params[0]:.3f}")
        print(f"    Racism stats: mean={model_df['racism_score'].mean():.3f}, SD={model_df['racism_score'].std(ddof=1):.3f}")

# Test the correct 5-item scale (alpha=0.54)
print("=== Correct 5-item scale: racmost, busing, racdif1, racdif3(flipped), racdif4 ===")
coded5 = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif3', 'r_racdif4']

print("\nMin 4 valid, person-mean imputation:")
df['racism_score'] = build_scale(df, coded5, 4)
run_both_models(df, predictors, predictor_labels)

print("\nMin 3 valid, person-mean imputation:")
df['racism_score'] = build_scale(df, coded5, 3)
run_both_models(df, predictors, predictor_labels)

# What about the 5-item scale with NA as 0?
print("\n=== Correct 5-item, NA=0 approach, min 3 ===")
n_valid = pd.concat([df[c] for c in coded5], axis=1).notna().sum(axis=1)
df['racism_score'] = sum(df[c].fillna(0) for c in coded5)
df.loc[n_valid < 3, 'racism_score'] = np.nan
run_both_models(df, predictors, predictor_labels)

print("\n=== Correct 5-item, NA=0 approach, min 4 ===")
df['racism_score'] = sum(df[c].fillna(0) for c in coded5)
df.loc[n_valid < 4, 'racism_score'] = np.nan
run_both_models(df, predictors, predictor_labels)

# Try the 6-item scale but with racdif2 coded in the OTHER direction
# racdif2: "differences because lack education chance"
# Standard: 2 = not due to education = racist direction
# But what if we flip it: 1 = due to education = identifies structural racism = anti-racist?
# Then racdif2 == 1 -> racist = 1
print("\n=== 6-item with racdif2 FLIPPED (1=racist) ===")
df['r_racdif2_flip'] = (df['racdif2'] == 1).astype(float).where(df['racdif2'].notna())
coded6_flip = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2_flip', 'r_racdif3', 'r_racdif4']
df['racism_score'] = build_scale(df, coded6_flip, 4) * (5.0/6.0)
run_both_models(df, predictors, predictor_labels)

# Check alpha for this combo
items_df = df[coded6_flip].dropna()
k = items_df.shape[1]
var_sum = items_df.var(axis=0, ddof=1).sum()
total_var = items_df.sum(axis=1).var(ddof=1)
alpha = (k / (k - 1)) * (1 - var_sum / total_var)
print(f"Alpha (6-item, racdif2 flipped): {alpha:.3f}")

# Maybe what we really need to explore is: what if some people coded as having
# valid race responses actually have race=0 or some unusual value?
print(f"\n=== Race values ===")
print(df['race'].value_counts(dropna=False).sort_index())

# Maybe the paper uses DIFFERENT age coding
# e.g., age in decades, or age - mean
# These don't affect standardized coefficients though

# What about income per capita: maybe realinc needs different handling
# Maybe people with hompop=0 should be excluded
print(f"\nhompop=0: {(df['hompop'] == 0).sum()}")
print(f"hompop distribution:")
print(df['hompop'].value_counts().sort_index())
