#!/usr/bin/env python3
"""
Investigate A045 and other high-correlation items.
Also investigate 'make parents proud' - could be:
- A003 (first choice) - goals in life
- A045 (something in the A-series)
- F063 was already confirmed as GOD_IMP

Also look at what E116 represents in WVS codebook context.
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

# Find ALL A0xx variables to understand what A045 might be
a_vars = sorted([h for h in header if h.startswith('A0')])
print("A0xx variables available:", a_vars)

# Also look at all possible WVS question series
print("\nSearching for parent-proud related variables...")
# In WVS, 'make parents proud' could be in goals in life (A series)
# Typical WVS Wave 2 goals items: A003, A006, A008...
a003_vars = [h for h in header if h.startswith('A003')]
print("A003 variants:", a003_vars)

# Load the data
needed = ['S002VS', 'COUNTRY_ALPHA', 'S020',
          'F063', 'A042', 'A029', 'A034', 'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165',
          'A045', 'A046', 'A047', 'A048', 'A049', 'A050', 'A051',
          'E116', 'E118',
          'A003']
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

# Compute factor scores
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
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
loadings = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
loadings, R = varimax(loadings)
loadings_df = pd.DataFrame(loadings, index=FACTOR_ITEMS, columns=['F1', 'F2'])

trad_items = ['AUTONOMY', 'F120', 'G006', 'E018']
f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items)
f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items)
tc = 0 if f1_trad > f2_trad else 1
if np.mean([loadings_df.iloc[loadings_df.index.get_loc(i), tc] for i in trad_items]) < 0:
    loadings[:, tc] *= -1

means = country_means_fa.mean()
stds = country_means_fa.std()
Z = (country_means_fa - means) / stds
scores_arr = Z.values @ loadings[:, tc]
scores = pd.Series(scores_arr, index=country_means_fa.index)
if 'SWE' in scores.index and scores['SWE'] > 0:
    scores = -scores

print(f"\nFactor scores: SWE={scores.get('SWE', 'N/A'):.3f}, NGA={scores.get('NGA', 'N/A'):.3f}")

# === A045 Investigation ===
print("\n=== A045 INVESTIGATION ===")
if 'A045' in df.columns:
    print("A045 value counts:")
    print(df['A045'].value_counts().sort_index())
    print("Values 1-3 usually: 1=Important, 2=Not important (or 1=Mentioned, 2=Not mentioned)")
    print("A045 waves available:")
    for wave in [2, 3]:
        w = combined[combined['S002VS'] == wave]
        print(f"  Wave {wave}: {w['A045'].notna().sum()} non-null observations")

    # Correlation
    country_means = df.groupby('COUNTRY_ALPHA')['A045'].mean().dropna()
    merged = pd.DataFrame({'item': country_means, 'factor': scores}).dropna()
    print(f"A045 mean r={merged['item'].corr(merged['factor']):.3f}, N={len(merged)}")

    # Binary (% = 1)
    df['A045_pct1'] = (df['A045'] == 1).astype(float).where(df['A045'].notna())
    pct1 = df.groupby('COUNTRY_ALPHA')['A045_pct1'].mean().dropna()
    merged2 = pd.DataFrame({'item': pct1, 'factor': scores}).dropna()
    print(f"A045 pct=1 r={merged2['item'].corr(merged2['factor']):.3f}, N={len(merged2)}")

    # Show top/bottom countries (A045 pct=1)
    print("Top 5 countries (A045 pct=1):")
    print(pct1.sort_values(ascending=False).head(10))
    print("Bottom 5 countries:")
    print(pct1.sort_values(ascending=False).tail(5))

# === A003 check (goals in life - make parents proud) ===
print("\n=== A003 INVESTIGATION ===")
if 'A003' in df.columns:
    print("A003 value counts (1=Most important, 5=Least important?):")
    print(df['A003'].value_counts().sort_index().head(10))
    print("Waves:")
    for wave in [2, 3]:
        w = combined[combined['S002VS'] == wave]
        nn = w['A003'].notna().sum()
        print(f"  Wave {wave}: {nn} non-null")
    country_means = df.groupby('COUNTRY_ALPHA')['A003'].mean().dropna()
    merged = pd.DataFrame({'item': country_means, 'factor': scores}).dropna()
    if len(merged) >= 5:
        print(f"A003 r={merged['item'].corr(merged['factor']):.3f}, N={len(merged)}")

# === Search for 'make parents proud' ===
print("\n=== SEARCHING FOR MAKE PARENTS PROUD ===")
# In WVS Wave 2, this was asked as part of goals in life
# Specifically: A003 = make parents proud (in some versions)
# But the item is "One of your main goals in life has been to make your parents proud of you"
# This is different from the importance items

# Check all A0xx variables for wave distribution
print("A0xx wave 2 vs wave 3 coverage:")
for var in a_vars[:30]:
    if var in combined.columns:
        w2_n = combined[combined['S002VS'] == 2][var].notna().sum()
        w3_n = combined[combined['S002VS'] == 3][var].notna().sum()
        if w2_n > 100 or w3_n > 100:
            print(f"  {var}: W2={w2_n}, W3={w3_n}")

# === E116 Investigation ===
print("\n=== E116 INVESTIGATION ===")
if 'E116' in df.columns:
    print("E116 value counts:")
    print(df['E116'].value_counts().sort_index())
    print("E116 is 'having a strong leader who does not have to bother with parliament and elections'")
    print("In WVS: 1=Very good, 2=Fairly good, 3=Fairly bad, 4=Very bad")
    print("BUT for military government, it's a separate item - E116 might be about army rule")

    # Show countries high on E116<=2 (favorable to?)
    df['E116_pct12'] = (df['E116'] <= 2).astype(float).where(df['E116'].notna())
    pct12 = df.groupby('COUNTRY_ALPHA')['E116_pct12'].mean()
    print("\nTop 10 countries for E116<=2 (favorable):")
    print(pct12.sort_values(ascending=False).head(10))
    print("Bottom 5:")
    print(pct12.sort_values(ascending=False).tail(5))

    # Confirm the variable meaning by checking which countries are high
    merged = pd.DataFrame({'item': pct12, 'factor': scores}).dropna()
    print(f"\nE116 pct<=2 r={merged['item'].corr(merged['factor']):.3f}, N={len(merged)}")

    # Also try WVS E114 specifically
    if 'E114' in df.columns:
        print("E114 value counts (army rule directly):")
        print(df['E114'].value_counts().sort_index())
        df['E114_pct12'] = (df['E114'] <= 2).astype(float).where(df['E114'].notna())
        pct14 = df.groupby('COUNTRY_ALPHA')['E114_pct12'].mean()
        merged14 = pd.DataFrame({'item': pct14, 'factor': scores}).dropna()
        print(f"E114 pct<=2 r={merged14['item'].corr(merged14['factor']):.3f}, N={len(merged14)}")
        print("Top 10 countries for E114<=2:")
        print(pct14.sort_values(ascending=False).head(10))
