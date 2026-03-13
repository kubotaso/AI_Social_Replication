#!/usr/bin/env python3
"""
Deeper exploration of key variables for Table 2 replication.
Focus on:
1. E116 as potential 'favorable army rule' item
2. G001/G002 content and correlation with trad/sec factor
3. A044/A045/A046/A047 - possible 'own preferences' items
4. C001/C002 - interpersonal items
5. Better understanding of G007_01 scale and coding
"""
import pandas as pd
import csv
import numpy as np
from scipy.stats import pearsonr

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

# Load data
with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

needed = ['S002VS', 'COUNTRY_ALPHA', 'S020',
          'F063', 'A042', 'A029', 'A034', 'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165',
          'E114', 'E115', 'E116', 'E117', 'E118',
          'G001', 'G002',
          'A044', 'A045', 'A046', 'A047', 'A048',
          'C001', 'C002', 'C004',
          'G007_01', 'G007_03', 'G007_04', 'G007_05',
          'A006']  # WVS A006 = religion importance
available = [c for c in needed if c in header]
wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
wvs = wvs[wvs['S002VS'].isin([2, 3])]
wvs['_src'] = 'wvs'
for col in available:
    if col not in ['COUNTRY_ALPHA', 'S002VS', 'S020']:
        wvs[col] = pd.to_numeric(wvs[col], errors='coerce')
        wvs[col] = wvs[col].where(wvs[col] >= 0, np.nan)

# Recode child qualities to 0/1
for v in ['A042', 'A029', 'A034']:
    if v in wvs.columns:
        wvs.loc[wvs[v] == 2, v] = 0

wvs['GOD_IMP'] = wvs['F063']
if all(v in wvs.columns for v in ['A042', 'A034', 'A029']):
    wvs['AUTONOMY'] = wvs['A042'] + wvs['A034'] - wvs['A029']

# Load EVS
import os
evs = pd.read_csv(EVS_PATH)
evs['_src'] = 'evs'
for col in evs.columns:
    if col not in ['COUNTRY_ALPHA', 'S002VS', 'S020', '_src']:
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

# Get latest wave per country
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

country_means = df2.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean().dropna(thresh=7)
for col in FACTOR_ITEMS:
    country_means[col] = country_means[col].fillna(country_means[col].mean())

corr = country_means.corr().values
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

means = country_means.mean()
stds = country_means.std()
Z = (country_means - means) / stds
scores_arr = Z.values @ loadings[:, tc]
scores = pd.Series(scores_arr, index=country_means.index)

if 'SWE' in scores.index and scores['SWE'] > 0:
    scores = -scores

print(f"Factor scores computed for {len(scores)} countries")
print(f"SWE: {scores.get('SWE', 'N/A'):.3f} (should be negative)")
print(f"NGA: {scores.get('NGA', 'N/A'):.3f} (should be positive)")

print("\n=== ARMY RULE VARIABLES ===")
for var in ['E114', 'E115', 'E116', 'E117', 'E118']:
    if var in df.columns:
        # Try different codings
        # E116: 1=Very good, 4=Very bad (democracy good/military bad context)
        # So % saying 1-2 = favorable to THIS regime type
        country_pct12 = df.groupby('COUNTRY_ALPHA').apply(
            lambda x: (x[var] <= 2).sum() / x[var].notna().sum() if x[var].notna().sum() > 0 else np.nan
        )
        country_mean = df.groupby('COUNTRY_ALPHA')[var].mean()

        merged_pct = pd.DataFrame({'item': country_pct12, 'factor': scores}).dropna()
        merged_mean = pd.DataFrame({'item': country_mean, 'factor': scores}).dropna()

        if len(merged_pct) >= 5:
            r_pct = merged_pct['item'].corr(merged_pct['factor'])
            r_mean = merged_mean['item'].corr(merged_mean['factor'])
            print(f"  {var}: pct<=2 r={r_pct:.3f} (N={len(merged_pct)}), mean r={r_mean:.3f} (N={len(merged_mean)})")
            print(f"    inv_mean r={(-merged_mean['item']).corr(merged_mean['factor']):.3f}")

print("\n=== G001/G002 CORRELATION ===")
for var in ['G001', 'G002']:
    if var in combined.columns:
        # Wave 2 only
        w2 = combined[combined['S002VS'] == 2]
        w2_means = w2.groupby('COUNTRY_ALPHA')[var].mean().dropna()
        merged = pd.DataFrame({'item': w2_means, 'factor': scores}).dropna()
        print(f"  {var} wave 2 only: N={len(merged)}, r={merged['item'].corr(merged['factor']):.3f}")

        # All waves
        all_means = combined.groupby('COUNTRY_ALPHA')[var].mean().dropna()
        merged2 = pd.DataFrame({'item': all_means, 'factor': scores}).dropna()
        print(f"  {var} all waves: N={len(merged2)}, r={merged2['item'].corr(merged2['factor']):.3f}")

        # Latest wave
        df_var = df.groupby('COUNTRY_ALPHA')[var].mean().dropna()
        merged3 = pd.DataFrame({'item': df_var, 'factor': scores}).dropna()
        print(f"  {var} latest wave: N={len(merged3)}, r={merged3['item'].corr(merged3['factor']):.3f}")

        # Pct=1
        w2['tmp_pct1'] = (w2[var] == 1).astype(float).where(w2[var].notna())
        pct1 = w2.groupby('COUNTRY_ALPHA')['tmp_pct1'].mean().dropna()
        merged4 = pd.DataFrame({'item': pct1, 'factor': scores}).dropna()
        print(f"  {var} pct=1 wave2: N={len(merged4)}, r={merged4['item'].corr(merged4['factor']):.3f}")

print("\n=== A044/A045/A046/A047/A048 ITEMS ===")
for var in ['A044', 'A045', 'A046', 'A047', 'A048']:
    if var in df.columns:
        # All waves combined
        all_means = combined.groupby('COUNTRY_ALPHA')[var].mean().dropna()
        merged = pd.DataFrame({'item': all_means, 'factor': scores}).dropna()
        if len(merged) >= 5:
            print(f"  {var}: r={merged['item'].corr(merged['factor']):.3f}, N={len(merged)}, vals={df[var].value_counts().head(3).to_dict()}")

print("\n=== C001/C002/C004 ITEMS ===")
for var in ['C001', 'C002', 'C004']:
    if var in df.columns:
        all_means = combined.groupby('COUNTRY_ALPHA')[var].mean().dropna()
        merged = pd.DataFrame({'item': all_means, 'factor': scores}).dropna()
        if len(merged) >= 5:
            print(f"  {var}: r={merged['item'].corr(merged['factor']):.3f}, N={len(merged)}, vals={df[var].value_counts().head(3).to_dict()}")

print("\n=== G007 PROTECTIONISM ITEMS ===")
for var in ['G007_01', 'G007_03', 'G007_04', 'G007_05']:
    if var in combined.columns:
        # Wave 2 only
        w2 = combined[combined['S002VS'] == 2]
        w2_var = w2.copy()
        w2_var[var] = pd.to_numeric(w2_var[var], errors='coerce').where(lambda x: x >= 0)
        w2_means = w2_var.groupby('COUNTRY_ALPHA')[var].mean().dropna()
        if len(w2_means) >= 5:
            merged = pd.DataFrame({'item': w2_means, 'factor': scores}).dropna()
            print(f"  {var} wave2: r={merged['item'].corr(merged['factor']):.3f}, N={len(merged)}")
            # Also inverted
            print(f"  {var} wave2 inv: r={(-merged['item']).corr(merged['factor']):.3f}")

print("\n=== STRICTER: G007_01 coding investigation ===")
if 'G007_01' in combined.columns:
    w2 = combined[combined['S002VS'] == 2].copy()
    w2['G007_01'] = pd.to_numeric(w2['G007_01'], errors='coerce').where(lambda x: x >= 0)
    print("G007_01 wave 2 value counts:")
    print(w2['G007_01'].value_counts().sort_index())
    print("(1=Strongly agree to 5=Strongly disagree? OR reversed?)")

    w2_means = w2.groupby('COUNTRY_ALPHA')['G007_01'].mean().dropna()
    print(f"\nCountry means (sorted desc):")
    print(w2_means.sort_values(ascending=False).head(10))
    print(w2_means.sort_values(ascending=False).tail(5))

    # Pct strongly agree (1) = stricter limits preferred
    w2['G007_pct1'] = (w2['G007_01'] == 1).astype(float).where(w2['G007_01'].notna())
    pct1 = w2.groupby('COUNTRY_ALPHA')['G007_pct1'].mean()
    merged = pd.DataFrame({'item': pct1, 'factor': scores}).dropna()
    print(f"\nG007_01 pct=1: r={merged['item'].corr(merged['factor']):.3f}, N={len(merged)}")

    # Inverted mean
    w2['G007_inv'] = 6 - w2['G007_01']
    inv_means = w2.groupby('COUNTRY_ALPHA')['G007_inv'].mean().dropna()
    merged2 = pd.DataFrame({'item': inv_means, 'factor': scores}).dropna()
    print(f"G007_01 inv mean: r={merged2['item'].corr(merged2['factor']):.3f}, N={len(merged2)}")
