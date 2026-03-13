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
df['dv1'] = np.nan
df.loc[minority_valid, 'dv1'] = (df.loc[minority_valid, minority_genres] >= 4).sum(axis=1)
df['dv2'] = np.nan
df.loc[remaining_valid, 'dv2'] = (df.loc[remaining_valid, remaining_genres] >= 4).sum(axis=1)

for v in ['racmost','busing','racdif1','racdif3','racdif4','educ','realinc','hompop','prestg80','age','sex','race','ethnic','fund','relig','region']:
    df[v] = pd.to_numeric(df[v], errors='coerce')

df['education'] = df['educ']
df['income_pc'] = df['realinc'] / df['hompop']
df['occ_prestige'] = df['prestg80']
df['female'] = (df['sex'] == 2).astype(int)
df['age_var'] = df['age']
df['black'] = (df['race'] == 2).astype(int)
df['hispanic'] = df['ethnic'].isin([17, 22, 25]).astype(int)
df['other_race'] = (df['race'] == 3).astype(int)
df['conservative_protestant'] = (df['fund'] == 1).astype(int)
df['no_religion'] = (df['relig'] == 4).astype(int)
df['southern'] = (df['region'] == 3).astype(int)

# Correct scale: racmost, busing, racdif1, racdif3(flip), racdif4
df['r1'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
df['r2'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
df['r3'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
df['r4'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 2).astype(float), np.nan)
df['r5'] = np.where(df['racdif4'].notna(), (df['racdif4'] == 1).astype(float), np.nan)

coded = ['r1','r2','r3','r4','r5']
n_valid = pd.concat([df[c] for c in coded], axis=1).notna().sum(axis=1)

# Key idea: racmost has ~824 valid, other 4 items have ~1000 valid
# The paper says "all items must be asked of the same set of respondents"
# GSS split ballot: some respondents got racmost, others didn't
# For those NOT asked racmost: they have NA for racmost but valid for other 4
# The paper might:
# 1. Only include respondents who were asked ALL items (all 5 valid) -> N~450
# 2. Include everyone who was asked AT LEAST the core items, treat racmost NA as 0
# 3. Something else

# Let's check: among those with busing valid (in the racism ballot group),
# how many have racmost valid?
busing_valid = df['busing'].notna()
racmost_in_busing = df.loc[busing_valid, 'racmost'].notna().sum()
print(f'Busing valid: {busing_valid.sum()}')
print(f'Racmost valid among busing valid: {racmost_in_busing}')
print(f'Racmost NA among busing valid: {busing_valid.sum() - racmost_in_busing}')

# So ~231 respondents have busing valid but racmost NA
# These were presumably asked busing but NOT racmost (different ballot subsection?)

# In the GSS, racmost might have a different skip pattern
# Let's see if racmost is conditioned on having children
# RACMOST: "Would you have any objection to sending your children..."
# This might only be asked of respondents WITH children

# Check: how many respondents in each group
print(f'\nRacmost + busing + racdif1 + racdif3 + racdif4 all valid: {(df[["racmost","busing","racdif1","racdif3","racdif4"]].notna().all(axis=1)).sum()}')
print(f'busing + racdif1 + racdif3 + racdif4 all valid: {(df[["busing","racdif1","racdif3","racdif4"]].notna().all(axis=1)).sum()}')

# If the paper treats racmost NA as 0 (non-racist) for those in the ballot group
# (i.e., those who have busing valid), that would include more cases
# Let's try: require busing valid + 3 of other 4 valid, racmost NA=0
other4 = ['busing','racdif1','racdif3','racdif4']
other4_valid = df[other4].notna().all(axis=1)

# For racism score: racmost NA->0 if in ballot group (busing valid)
racism_final = df['r1'].fillna(0) + df['r2'] + df['r3'] + df['r4'] + df['r5']
# Require busing + at least 3 of other 4 (racdif1, racdif3, racdif4) valid
n_other4 = sum(df[v].notna().astype(int) for v in other4)
racism_final[n_other4 < 4] = np.nan  # require all 4 of other items
# racmost NA is treated as 0

preds_base = ['education', 'income_pc', 'occ_prestige', 'female', 'age_var',
              'black', 'hispanic', 'other_race', 'conservative_protestant',
              'no_religion', 'southern']
df['racism'] = racism_final
preds = ['racism'] + preds_base

print('\n=== racmost NA=0 when other 4 all valid ===')
for dv, dv_label in [('dv1', 'Model 1'), ('dv2', 'Model 2')]:
    mdf = df[[dv] + preds].dropna()
    N = len(mdf)
    y = mdf[dv].values
    X = mdf[preds].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    X_zc = sm.add_constant(X_z)
    res = sm.OLS(y_z, X_zc).fit()
    X_c = sm.add_constant(X)
    res_raw = sm.OLS(y, X_c).fit()

    print(f'\n  {dv_label}: N={N}')
    print(f'    racism beta={res.params[1]:.3f}, p={res.pvalues[1]:.4f}')
    print(f'    R2={res_raw.rsquared:.3f}, AdjR2={res_raw.rsquared_adj:.3f}')
    print(f'    const={res_raw.params[0]:.3f}')
    print(f'    racism mean={mdf["racism"].mean():.2f}, SD={mdf["racism"].std(ddof=1):.2f}')

    # Print all betas
    pred_labels = ['Racism score', 'Education', 'Household income per cap',
                   'Occupational prestige', 'Female', 'Age', 'Black', 'Hispanic',
                   'Other race', 'Conservative Protestant', 'No religion', 'Southern']
    for i, (label, beta, pval) in enumerate(zip(pred_labels, res.params[1:], res.pvalues[1:])):
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f'    {label:<30} {beta:>8.3f} {sig}')
