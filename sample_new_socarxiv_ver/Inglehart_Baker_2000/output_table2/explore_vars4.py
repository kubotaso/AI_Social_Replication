#!/usr/bin/env python3
"""
Investigate A044/A045 meaning and connection to 'own preferences vs understanding others'.
Also: investigate the 'make parents proud' item more carefully.
"""
import pandas as pd
import csv
import numpy as np

DATA_PATH = 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'
EVS_PATH = 'data/EVS_1990_wvs_format.csv'

PAPER_COUNTRIES = [
    'ARG', 'ARM', 'AUS', 'AUT', 'AZE', 'BEL', 'BGD', 'BGR', 'BIH', 'BLR',
    'BRA', 'CAN', 'CHE', 'CHL', 'CHN', 'COL', 'CZE', 'DEU', 'DNK', 'DOM',
    'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GEO', 'HRV', 'HUN', 'IND', 'IRL',
    'ISL', 'ITA', 'JPN', 'KOR', 'LTU', 'LVA', 'MDA', 'MEX', 'MKD',
    'NGA', 'NIR', 'NLD', 'NOR', 'NZL', 'PAK', 'PER', 'PHL', 'POL', 'PRI',
    'PRT', 'ROU', 'RUS', 'SRB', 'SVK', 'SVN', 'SWE', 'TUR', 'TWN', 'UKR',
    'URY', 'USA', 'VEN', 'ZAF'
]

def varimax(Phi, gamma=1.0, q=100, tol=1e-8):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        Lambda = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma/p) * Lambda @ np.diag(np.sum(Lambda**2, axis=0)))
        )
        R = u @ vt
        d_new = np.sum(s)
        if d_new - d < tol:
            break
        d = d_new
    return Phi @ R, R

with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

needed = ['S002VS', 'COUNTRY_ALPHA', 'S020',
          'F063', 'A042', 'A029', 'A034', 'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165',
          'A044', 'A045', 'A049',
          'A057', 'A058', 'A059', 'A060', 'A061', 'A062', 'A063', 'A064', 'A065', 'A066',
          'A067', 'A068', 'A069', 'A070', 'A071']
available = [c for c in needed if c in header]
wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
wvs = wvs[wvs['S002VS'].isin([2, 3])]

for col in available:
    if col not in ['COUNTRY_ALPHA', 'S002VS', 'S020']:
        wvs[col] = pd.to_numeric(wvs[col], errors='coerce')
        wvs[col] = wvs[col].where(wvs[col] >= 0, np.nan)

for v in ['A042', 'A029', 'A034']:
    if v in wvs.columns:
        wvs.loc[wvs[v] == 2, v] = 0
wvs['GOD_IMP'] = wvs['F063']
if all(v in wvs.columns for v in ['A042', 'A034', 'A029']):
    wvs['AUTONOMY'] = wvs['A042'] + wvs['A034'] - wvs['A029']

evs = pd.read_csv(EVS_PATH)
for col in evs.columns:
    if col not in ['COUNTRY_ALPHA', 'S002VS', 'S020']:
        evs[col] = pd.to_numeric(evs[col], errors='coerce')
        evs[col] = evs[col].where(evs[col] >= 0, np.nan)
if 'A006' in evs.columns:
    evs['GOD_IMP'] = evs['A006']
    evs['A006'] = np.nan
for v in ['A042', 'A034', 'A029']:
    if v in evs.columns:
        evs.loc[evs[v] == 2, v] = 0
if all(v in evs.columns for v in ['A042', 'A034', 'A029']):
    evs['AUTONOMY'] = evs['A042'] + evs['A034'] - evs['A029']

combined = pd.concat([wvs, evs], ignore_index=True, sort=False)
combined = combined[combined['COUNTRY_ALPHA'].isin(PAPER_COUNTRIES)]

latest = combined.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
latest.columns = ['COUNTRY_ALPHA', 'ly']
df = combined.merge(latest, on='COUNTRY_ALPHA')
df = df[df['S020'] == df['ly']].drop('ly', axis=1)

# Factor scores
FACTOR_ITEMS = ['GOD_IMP', 'AUTONOMY', 'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
df2 = df.copy()
for col, flip in [('F120', 11), ('G006', 5), ('E018', 4), ('Y002', 4), ('F118', 11)]:
    if col in df2.columns:
        df2[col] = flip - df2[col]
country_means_fa = df2.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean().dropna(thresh=7)
for col in FACTOR_ITEMS:
    country_means_fa[col] = country_means_fa[col].fillna(country_means_fa[col].mean())
corr = country_means_fa.corr().values
eigenvalues, eigenvectors = np.linalg.eigh(corr)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]; eigenvectors = eigenvectors[:, idx]
loadings = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
loadings, R = varimax(loadings)
loadings_df = pd.DataFrame(loadings, index=FACTOR_ITEMS, columns=['F1', 'F2'])
trad_items = ['AUTONOMY', 'F120', 'G006', 'E018']
f1_t = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items)
f2_t = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items)
tc = 0 if f1_t > f2_t else 1
if np.mean([loadings_df.iloc[loadings_df.index.get_loc(i), tc] for i in trad_items]) < 0:
    loadings[:, tc] *= -1
Z = (country_means_fa - country_means_fa.mean()) / country_means_fa.std()
scores = pd.Series(Z.values @ loadings[:, tc], index=country_means_fa.index)
if 'SWE' in scores.index and scores['SWE'] > 0:
    scores = -scores

print("A044/A045 analysis:")
print("="*60)

# A044: Show country means
if 'A044' in df.columns:
    print("\nA044 - wave 3 only?")
    w2n = combined[combined['S002VS']==2]['A044'].notna().sum()
    w3n = combined[combined['S002VS']==3]['A044'].notna().sum()
    print(f"  Wave 2: {w2n}, Wave 3: {w3n}")
    print("A044 value counts:", df['A044'].value_counts().sort_index().to_dict())
    print("Country means (sorted):")
    cm = df.groupby('COUNTRY_ALPHA')['A044'].mean().dropna()
    merged = pd.DataFrame({'item': cm, 'factor': scores}).dropna()
    print(cm.sort_values(ascending=False).head(10))
    print(f"A044 r={merged['item'].corr(merged['factor']):.3f}, N={len(merged)}")

print("\nA045 - wave 3 only:")
if 'A045' in df.columns:
    print("A045 value counts:", df['A045'].value_counts().sort_index().to_dict())
    # In WVS Wave 3, A045 is:
    # "Obedience" (importance as a quality children should learn at home)
    # BUT values 1-4 = ? OR is it an Inglehart-style item?
    # A045 could be "When jobs are scarce, men should have more right to a job than women"
    # OR it could be "Expressing one's own preferences"
    print("Country means (sorted desc):")
    cm = df.groupby('COUNTRY_ALPHA')['A045'].mean().dropna()
    print(cm.sort_values(ascending=False).head(10))
    print(cm.sort_values(ascending=False).tail(10))
    merged = pd.DataFrame({'item': cm, 'factor': scores}).dropna()
    print(f"A045 mean r={merged['item'].corr(merged['factor']):.3f}")
    # If A045 mean is high = traditional, then higher = preferring own expression (1=important)
    # but country means: traditional countries have high A045 means
    # This could be important to check what 1 means vs 4

# Check A057-A071: these might be Postmaterialism items
print("\n=== A057-A071 exploration ===")
for var in ['A057', 'A058', 'A059', 'A060', 'A061', 'A062', 'A063', 'A064', 'A065']:
    if var in df.columns:
        w2n = combined[combined['S002VS']==2][var].notna().sum()
        w3n = combined[combined['S002VS']==3][var].notna().sum()
        cm = df.groupby('COUNTRY_ALPHA')[var].mean().dropna()
        merged = pd.DataFrame({'item': cm, 'factor': scores}).dropna()
        if len(merged) >= 10:
            print(f"  {var}: W2={w2n}, W3={w3n}, r={merged['item'].corr(merged['factor']):.3f}, N={len(merged)}, vals={df[var].value_counts().sort_index().to_dict()}")

# A049 = "For the greater good of society, it is sometimes necessary that individual rights are curtailed"
print("\n=== A049 ===")
if 'A049' in df.columns:
    w2n = combined[combined['S002VS']==2]['A049'].notna().sum()
    w3n = combined[combined['S002VS']==3]['A049'].notna().sum()
    print(f"  A049: W2={w2n}, W3={w3n}")
    print(f"  Values: {df['A049'].value_counts().sort_index().to_dict()}")
    cm = df.groupby('COUNTRY_ALPHA')['A049'].mean().dropna()
    merged = pd.DataFrame({'item': cm, 'factor': scores}).dropna()
    if len(merged) >= 5:
        print(f"  r={merged['item'].corr(merged['factor']):.3f}, N={len(merged)}")

# === OWN PREFERENCES: Check what A044 and A045 actually are ===
# In WVS Wave 3 (1995-1998), A044/A045 correspond to:
# A044: "Choosing one's own way of life" or something similar
# A045: could be "Expressing one's own preferences clearly is more important than understanding others"
print("\n=== A044 high vs low countries ===")
if 'A044' in df.columns:
    cm44 = df.groupby('COUNTRY_ALPHA')['A044'].mean().dropna()
    print("High A044 countries (should be secular/traditional for own preference item):")
    print(cm44.sort_values(ascending=False).head(5))
    print("Low A044 countries:")
    print(cm44.sort_values(ascending=False).tail(5))

print("\n=== A045 high vs low countries ===")
if 'A045' in df.columns:
    cm45 = df.groupby('COUNTRY_ALPHA')['A045'].mean().dropna()
    print("High A045 countries:")
    print(cm45.sort_values(ascending=False).head(5))
    print("Low A045 countries:")
    print(cm45.sort_values(ascending=False).tail(5))
