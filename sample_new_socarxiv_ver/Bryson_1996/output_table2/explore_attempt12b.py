import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('gss1993_clean.csv')

# Check for a direct hispanic variable
print("Columns with 'hisp':", [c for c in df.columns if 'hisp' in c.lower()])

# The GSS has a 'hispanic' variable from 1990+ surveys
hisp_var = pd.to_numeric(df.get('hispanic', pd.Series()), errors='coerce')
print(f"\nhispanic variable distribution:")
print(hisp_var.value_counts().sort_index())
# GSS hispanic: 1=Not Hispanic, 2-50=various Hispanic subgroups
# Actually let's check what values it has
print(f"\nhispanic variable unique values: {sorted(hisp_var.dropna().unique())}")

# Check ethnic codes more carefully
ethnic = pd.to_numeric(df['ethnic'], errors='coerce')
# GSS ethnic codes for Hispanic:
# 17=Mexico, 22=Puerto Rico, 25=Spain, 38=Other Spanish
# But also: 2=Austria? 3=Belgium? etc.
# Let me check if ethnic==38 is "Other Spanish"
print(f"\nethnic==38 count: {(ethnic==38).sum()}")
print(f"ethnic==17 count: {(ethnic==17).sum()}")
print(f"ethnic==22 count: {(ethnic==22).sum()}")
print(f"ethnic==25 count: {(ethnic==25).sum()}")

# Check what GSS hispanic variable says about these respondents
mask_17 = ethnic == 17
mask_22 = ethnic == 22
mask_25 = ethnic == 25
mask_38 = ethnic == 38

for code, name in [(17, 'Mexican'), (22, 'Puerto Rican'), (25, 'Spanish'), (38, 'Other')]:
    mask = ethnic == code
    print(f"\nethnic=={code} ({name}):")
    print(f"  Count: {mask.sum()}")
    if mask.sum() > 0:
        print(f"  Race: {pd.to_numeric(df.loc[mask, 'race'], errors='coerce').value_counts().to_dict()}")
        print(f"  Hispanic var: {hisp_var[mask].value_counts().to_dict()}")

# Try broader Hispanic definition with 38
print("\n\n=== Testing broader Hispanic with ethnic in [17,22,25,38] ===")
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
df['conservative_protestant'] = (pd.to_numeric(df['fund'], errors='coerce') == 1).astype(int)
df['no_religion'] = (pd.to_numeric(df['relig'], errors='coerce') == 4).astype(int)
df['southern'] = (pd.to_numeric(df['region'], errors='coerce') == 3).astype(int)

for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3']:
    df[item] = pd.to_numeric(df[item], errors='coerce')
df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())
df['r_racdif3'] = (df['racdif3'] == 2).astype(float).where(df['racdif3'].notna())
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

pred_labels = ['Racism score', 'Education', 'Household income per cap',
               'Occupational prestige', 'Female', 'Age', 'Black', 'Hispanic',
               'Other race', 'Conservative Protestant', 'No religion', 'Southern']

# Use the GSS hispanic variable directly (1=not hispanic, 2+=hispanic)
df['hisp_gss'] = (hisp_var >= 2).astype(int)
print(f"\nGSS hispanic variable: {df['hisp_gss'].sum()} Hispanic respondents")

for hisp_name, hisp_col, other_col in [
    ('ethnic in 17,22,25 non-overlap',
     (df['ethnic'].isin([17, 22, 25]) & (df['race'] != 3)).astype(int),
     ((df['race'] == 3) & ~df['ethnic'].isin([17, 22, 25])).astype(int)),
    ('ethnic in 17,22,25,38 non-overlap',
     (df['ethnic'].isin([17, 22, 25, 38]) & (df['race'] != 3)).astype(int),
     ((df['race'] == 3) & ~df['ethnic'].isin([17, 22, 25, 38])).astype(int)),
    ('GSS hispanic var non-overlap',
     ((hisp_var >= 2) & (df['race'] != 3)).astype(int),
     ((df['race'] == 3) & ~(hisp_var >= 2)).astype(int)),
    ('ethnic in 17,22,25 overlapping',
     df['ethnic'].isin([17, 22, 25]).astype(int),
     (df['race'] == 3).astype(int)),
]:
    df['hispanic'] = hisp_col
    df['other_race'] = other_col
    predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
                  'female', 'age_var', 'black', 'hispanic', 'other_race',
                  'conservative_protestant', 'no_religion', 'southern']

    print(f"\n  === {hisp_name}: Hispanic={df['hispanic'].sum()}, Other={df['other_race'].sum()} ===")
    for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
        model_df = df[[dv] + predictors].dropna()
        N = len(model_df)
        y = model_df[dv].values
        X = model_df[predictors].values
        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
        model_raw = sm.OLS(y, sm.add_constant(X)).fit()
        betas = dict(zip(pred_labels, model.params[1:]))
        pvals = dict(zip(pred_labels, model.pvalues[1:]))
        print(f"    {dv_name}: N={N}, R2={model_raw.rsquared:.3f}, AdjR2={model_raw.rsquared_adj:.3f}")
        for lab in ['Racism score', 'Black', 'Hispanic', 'Other race']:
            b = betas[lab]
            p = pvals[lab]
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"      {lab:<30} {b:7.3f} {sig:<4} (p={p:.4f})")
