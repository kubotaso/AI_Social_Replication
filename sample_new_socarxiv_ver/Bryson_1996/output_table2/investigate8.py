import pandas as pd
import numpy as np

df = pd.read_csv('gss1993_clean.csv')

items = ['racmost','busing','racdif1','racdif2','racdif3']
for v in items:
    df[v] = pd.to_numeric(df[v], errors='coerce')

# Code in racist direction
df['c_racmost'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
df['c_busing'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
df['c_racdif1'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
df['c_racdif2'] = np.where(df['racdif2'].notna(), (df['racdif2'] == 2).astype(float), np.nan)
df['c_racdif3'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 1).astype(float), np.nan)

coded = ['c_racmost','c_busing','c_racdif1','c_racdif2','c_racdif3']
all_valid = df[items].notna().all(axis=1)
valid_df = df.loc[all_valid, coded]

print(f'N valid: {all_valid.sum()}')
print(f'Racism sum: mean={valid_df.sum(axis=1).mean():.2f}, SD={valid_df.sum(axis=1).std(ddof=1):.2f}')

# Cronbach's alpha
k = 5
item_vars = valid_df.var(ddof=1)
print('\nItem variances:')
for c in coded:
    print(f'  {c}: var={valid_df[c].var(ddof=1):.4f}, mean={valid_df[c].mean():.3f}')

total_var = valid_df.sum(axis=1).var(ddof=1)
print(f'\nSum of item variances: {item_vars.sum():.4f}')
print(f'Total score variance: {total_var:.4f}')

alpha = (k / (k-1)) * (1 - item_vars.sum() / total_var)
print(f'Alpha = ({k}/{k-1}) * (1 - {item_vars.sum():.4f}/{total_var:.4f}) = {alpha:.4f}')

# Check inter-item correlations
print('\nInter-item correlation matrix:')
corr = valid_df.corr()
print(corr.round(3))

# Something's wrong - alpha should be 0.54
# Let me check if maybe the coding direction is wrong for some items
# Paper: "coded in the same direction"
# Try flipping racdif2 (maybe racdif2=1 is racist?)
# racdif2: "because most Blacks don't have the chance for education"
# If yes (=1): saying it's because lack of education chance (structural) = NOT racist
# If no (=2): saying it's NOT because lack of education = IS racist
# So racdif2=2 is racist. That's what we have.

# But wait - what if the paper codes racdif2 DIFFERENTLY?
# "NOT because lack education chance" = not racist according to some interpretations?
# Actually the question is: "Do you think these differences are because most blacks
# don't have the chance for education that it takes to rise out of poverty?"
# Yes (1) = recognizing structural barriers = not racist
# No (2) = denying structural barriers = racist
# Our coding seems right

# Let me try: maybe the issue is racdif2 is coded OPPOSITE
# Try racdif2=1 as racist (YES because lack education = acknowledging inferiority?)
df['c_racdif2_flip'] = np.where(df['racdif2'].notna(), (df['racdif2'] == 1).astype(float), np.nan)
coded_flip = ['c_racmost','c_busing','c_racdif1','c_racdif2_flip','c_racdif3']
valid_flip = df.loc[all_valid, coded_flip]
total_var_flip = valid_flip.sum(axis=1).var(ddof=1)
item_vars_flip = valid_flip.var(ddof=1).sum()
alpha_flip = (k/(k-1)) * (1 - item_vars_flip / total_var_flip)
m_flip = valid_flip.sum(axis=1).mean()
s_flip = valid_flip.sum(axis=1).std(ddof=1)
print(f'\nWith racdif2 flipped: mean={m_flip:.2f}, SD={s_flip:.2f}, alpha={alpha_flip:.2f}')
print(f'Corr matrix:')
print(valid_flip.corr().round(3))

# Try flipping racdif1 too
df['c_racdif1_flip'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 1).astype(float), np.nan)
coded_flip2 = ['c_racmost','c_busing','c_racdif1_flip','c_racdif2','c_racdif3']
valid_flip2 = df.loc[all_valid, coded_flip2]
total_var_flip2 = valid_flip2.sum(axis=1).var(ddof=1)
item_vars_flip2 = valid_flip2.var(ddof=1).sum()
alpha_flip2 = (k/(k-1)) * (1 - item_vars_flip2 / total_var_flip2)
m2 = valid_flip2.sum(axis=1).mean()
s2 = valid_flip2.sum(axis=1).std(ddof=1)
print(f'\nWith racdif1 flipped: mean={m2:.2f}, SD={s2:.2f}, alpha={alpha_flip2:.2f}')

# Try: use raw values (1 or 2) instead of binary, recode so higher = more racist
# racmost: 1=racist -> recode to 2, 2=not -> recode to 1
# busing: 2=racist -> keep as is
# racdif1: 2=racist -> keep as is
# racdif2: 2=racist -> keep as is
# racdif3: 1=racist -> recode: 1->2, 2->1
df['raw_racmost'] = np.where(df['racmost'].notna(), 3 - df['racmost'], np.nan)
df['raw_busing'] = np.where(df['busing'].notna(), df['busing'].astype(float), np.nan)
df['raw_racdif1'] = np.where(df['racdif1'].notna(), df['racdif1'].astype(float), np.nan)
df['raw_racdif2'] = np.where(df['racdif2'].notna(), df['racdif2'].astype(float), np.nan)
df['raw_racdif3'] = np.where(df['racdif3'].notna(), 3 - df['racdif3'], np.nan)

raw_coded = ['raw_racmost','raw_busing','raw_racdif1','raw_racdif2','raw_racdif3']
valid_raw = df.loc[all_valid, raw_coded]
raw_sum = valid_raw.sum(axis=1)
print(f'\nRaw values (higher=racist): mean={raw_sum.mean():.2f}, SD={raw_sum.std(ddof=1):.2f}')
# Range would be 5-10
total_var_raw = raw_sum.var(ddof=1)
item_vars_raw = valid_raw.var(ddof=1).sum()
alpha_raw = (k/(k-1)) * (1 - item_vars_raw / total_var_raw)
print(f'Alpha: {alpha_raw:.2f}')
# This should give same alpha as binary since it's a linear transform
