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
df['denom'] = pd.to_numeric(df['denom'], errors='coerce')
df['fund'] = pd.to_numeric(df['fund'], errors='coerce')
df['no_religion'] = (pd.to_numeric(df['relig'], errors='coerce') == 4).astype(int)
df['southern'] = (pd.to_numeric(df['region'], errors='coerce') == 3).astype(int)

pred_labels = ['Racism score', 'Education', 'Household income per cap',
               'Occupational prestige', 'Female', 'Age', 'Black', 'Hispanic',
               'Other race', 'Conservative Protestant', 'No religion', 'Southern']

# Test combinations: scale type x conservative protestant coding
configs = []

# Scale options
def build_scale(df, coded_items, min_valid, rescale=None):
    vals = []
    for idx in df.index:
        v = [df.loc[idx, c] for c in coded_items]
        n_v = sum(1 for x in v if not np.isnan(x))
        if n_v >= min_valid:
            valid = [x for x in v if not np.isnan(x)]
            pm = np.mean(valid)
            s = sum(x if not np.isnan(x) else pm for x in v)
            if rescale:
                s = s * rescale / len(coded_items)
            vals.append(s)
        else:
            vals.append(np.nan)
    return pd.Series(vals, index=df.index)

scale_old5 = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3']
scale_6item = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3', 'r_racdif4']

for scale_name, coded, min_v, rescale in [
    ('old5_m4', scale_old5, 4, None),
    ('6item_m4_r5', scale_6item, 4, 5),
]:
    for cp_name, cp_col in [
        ('fund1', (df['fund'] == 1).astype(int)),
        ('denom1', (df['denom'] == 1).astype(int)),
    ]:
        df['racism_score'] = build_scale(df, coded, min_v, rescale)
        df['conservative_protestant'] = cp_col

        predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
                      'female', 'age_var', 'black', 'hispanic', 'other_race',
                      'conservative_protestant', 'no_religion', 'southern']

        for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
            model_df = df[[dv] + predictors].dropna()
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

            config = f"{scale_name}+{cp_name}"
            print(f"{config:20s} {dv_name}: N={N:3d}, rac={betas['Racism score']:6.3f}(p={pvals['Racism score']:.4f}), "
                  f"blk={betas['Black']:6.3f}(p={pvals['Black']:.4f}), "
                  f"cp={betas['Conservative Protestant']:6.3f}, "
                  f"age={betas['Age']:6.3f}(p={pvals['Age']:.4f}), "
                  f"hisp={betas['Hispanic']:6.3f}(p={pvals['Hispanic']:.4f}), "
                  f"R2={model_raw.rsquared:.3f}, adjR2={model_raw.rsquared_adj:.3f}")
