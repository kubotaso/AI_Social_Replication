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

for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

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

plabels = ['Racism', 'Educ', 'IncPC', 'Prest', 'Female', 'Age', 'Black', 'Hisp', 'Other', 'ConsProt', 'NoRel', 'South']

# Test the "correct" 5-item scale (racdif4 instead of racdif2) with various thresholds
items = ['racmost', 'busing', 'racdif1', 'racdif3', 'racdif4']
rv = {'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif3': 2, 'racdif4': 1}

coded = []
for item in items:
    col = f'r_{item}'
    df[col] = (df[item] == rv[item]).astype(float).where(df[item].isin([1,2]))
    coded.append(col)

for min_valid in [3, 4, 5]:
    racism_vals = []
    for idx in df.index:
        vals = [df.loc[idx, c] for c in coded]
        n = sum(1 for v in vals if not np.isnan(v))
        if n >= min_valid:
            vv = [v for v in vals if not np.isnan(v)]
            if n == 5:
                racism_vals.append(sum(vv))
            else:
                pm = np.mean(vv)
                racism_vals.append(sum(v if not np.isnan(v) else pm for v in vals))
        else:
            racism_vals.append(np.nan)
    df['racism_score'] = racism_vals

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
        r2 = sm.OLS(y, sm.add_constant(X)).fit().rsquared
        adj_r2 = sm.OLS(y, sm.add_constant(X)).fit().rsquared_adj

        print(f"\nalt5 min{min_valid} {label} N={N} R2={r2:.3f} AdjR2={adj_r2:.3f}")
        for i, pl in enumerate(plabels):
            sig = "***" if m.pvalues[i+1] < 0.001 else "**" if m.pvalues[i+1] < 0.01 else "*" if m.pvalues[i+1] < 0.05 else ""
            true_m1 = {'Racism': 0.130, 'Educ': -0.175, 'IncPC': -0.037, 'Prest': -0.020,
                       'Female': -0.057, 'Age': 0.163, 'Black': -0.132, 'Hisp': -0.058,
                       'Other': -0.017, 'ConsProt': 0.063, 'NoRel': 0.057, 'South': 0.024}
            true_m2 = {'Racism': 0.080, 'Educ': -0.242, 'IncPC': -0.065, 'Prest': 0.005,
                       'Female': -0.070, 'Age': 0.126, 'Black': 0.042, 'Hisp': -0.029,
                       'Other': 0.047, 'ConsProt': 0.048, 'NoRel': 0.024, 'South': 0.069}
            true_val = true_m1[pl] if label == 'M1' else true_m2[pl]
            diff = abs(m.params[i+1] - true_val)
            marker = " <--" if diff > 0.05 else ""
            print(f"    {pl:10s} {m.params[i+1]:+.3f} {sig:3s} (true={true_val:+.3f} diff={diff:.3f}){marker}")

# Now the big question: what if we combine the "correct" 5-item scale for
# descriptive stats accuracy (mean, SD, alpha match) with the "old" 5-item
# scale's regression performance?
# The issue is that racdif4 vs racdif2 swap changes the racism coefficient
# in both models. Maybe the answer is that the paper used racdif2 not racdif4
# despite what the text suggests.

# Let me do a FULL scoring comparison of old vs new 5-item scales
print("\n\n===== FULL COMPARISON: old5 vs alt5 =====")
for scale_name, s_items, s_rv in [
    ('old5', ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3'],
     {'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif2': 2, 'racdif3': 2}),
    ('alt5', ['racmost', 'busing', 'racdif1', 'racdif3', 'racdif4'],
     {'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif3': 2, 'racdif4': 1}),
]:
    coded = []
    for item in s_items:
        col = f'r_{item}'
        df[col] = (df[item] == s_rv[item]).astype(float).where(df[item].isin([1,2]))
        coded.append(col)

    racism_vals = []
    for idx in df.index:
        vals = [df.loc[idx, c] for c in coded]
        n = sum(1 for v in vals if not np.isnan(v))
        if n >= 4:
            vv = [v for v in vals if not np.isnan(v)]
            pm = np.mean(vv)
            racism_vals.append(sum(v if not np.isnan(v) else pm for v in vals))
        else:
            racism_vals.append(np.nan)
    df['racism_score'] = racism_vals

    for dv_name, label in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
        vars_needed = [dv_name] + predictors
        model_df = df[vars_needed].dropna()
        N = len(model_df)
        y = model_df[dv_name].values
        X = model_df[predictors].values
        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        m = sm.OLS(y_z, sm.add_constant(X_z)).fit()
        raw_m = sm.OLS(y, sm.add_constant(X)).fit()

        print(f"\n{scale_name} min4 {label} N={N} R2={raw_m.rsquared:.3f} AdjR2={raw_m.rsquared_adj:.3f}")
        for i, pl in enumerate(plabels):
            sig = "***" if m.pvalues[i+1] < 0.001 else "**" if m.pvalues[i+1] < 0.01 else "*" if m.pvalues[i+1] < 0.05 else ""
            print(f"    {pl:10s} {m.params[i+1]:+.3f} {sig:3s}  p={m.pvalues[i+1]:.4f}")
