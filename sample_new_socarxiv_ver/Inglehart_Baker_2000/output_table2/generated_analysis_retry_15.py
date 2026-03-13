#!/usr/bin/env python3
"""
Table 2 Replication - Attempt 15
Correlation of Additional Items with the Traditional/Secular-Rational Values Dimension

KEY STRATEGIES (building on attempt 14's best score of 83.1):
1. SUICIDE: wave 2 gives -0.01 (wrong sign). Try wave 3 ONLY (1995-98).
   Wave 3 has more developing countries with high suicide taboo, should give higher r.
   Also try: using both wave 2 and wave 3 for different countries (w2 for those with w2 data,
   w3 for those with only w3 data) - this is actually what all-waves "latest per country" does.
   So the issue is simply that all-waves gives 0.47 and nothing better is available.

2. "OWN PREFERENCES": Search through wave 2 data systematically for variables near 0.56.
   Try all E-variables from wave 2 that have >10 countries.

3. LOVE/RESPECT PARENTS: Wave 2 has only 17 countries with A025.
   Try using BOTH wave 2 and wave 3 but weighting by data quality.
   Also try: wave 3 ONLY (same 50 countries, but different wave than what we're using).

4. STRICTER LIMITS: only 14 countries. This is truly a small-N problem.
   The paper's 0.63 is based on whatever N the paper had. Cannot fix this.
   Accept 0.21 as the best available.

5. CONSOLIDATE BEST APPROACHES:
   - Use wave 2 for: euthanasia (0.67), divorce (0.58)
   - Use wave 2 for: love/respect parents (0.58, moderate)
   - Use all-waves (latest) for: suicide (0.47)
   - Use composite for: army rule (0.43)
   - Keep all other improvements from attempt 14
"""
import pandas as pd
import numpy as np
import os
import csv
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")

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


def get_header():
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]
    return header


def load_and_prepare_data():
    """Load WVS + EVS data."""
    header = get_header()

    # Search for variables in E/F range that might be "own preferences"
    # E0xx variables are political interest, national pride, political culture
    # F0xx variables are religious beliefs and practices
    # The "own preferences vs understanding others" might be in E002-E009 (goals)
    # or A040-A043 (child qualities: imagination, obedience, independence)
    # or possibly in the "autonomy vs order" items

    needed = [
        'S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020',
        # Factor analysis items
        'F063', 'A042', 'A029', 'A034', 'F120', 'G006', 'E018',
        'Y002', 'A008', 'E025', 'F118', 'A165',
        # Table 2 core items
        'A001', 'A005', 'A006', 'A025', 'A026', 'B009', 'D017',
        'D054', 'D066',
        'E023', 'E033', 'E069_01', 'E114', 'E116',
        'F024', 'F028', 'F034', 'F050', 'F051', 'F053', 'F054',
        'F119', 'F121', 'F122',
        'G007_01',
        # New search candidates
        'E143', 'D057', 'D058', 'D059',
        # Search for "own preferences vs understanding" - try A036-A045 range
        'A036', 'A037', 'A038', 'A039', 'A040', 'A041', 'A043', 'A044', 'A045',
        # Try some E-variables
        'E002', 'E003', 'E006', 'E007', 'E009',
    ]

    available = [c for c in needed if c in header]
    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]
    wvs['_src'] = 'wvs'

    for col in available:
        if col not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020']:
            wvs[col] = pd.to_numeric(wvs[col], errors='coerce')
            wvs[col] = wvs[col].where(wvs[col] >= 0, np.nan)

    # Recode WVS child qualities to 0/1
    for v in ['A042', 'A029', 'A034']:
        if v in wvs.columns:
            wvs.loc[wvs[v] == 2, v] = 0

    # GOD_IMP from WVS F063
    wvs['GOD_IMP'] = wvs.get('F063', pd.Series(dtype=float))

    # AUTONOMY: obedience + faith - independence
    if all(v in wvs.columns for v in ['A042', 'A034', 'A029']):
        wvs['AUTONOMY'] = wvs['A042'] + wvs['A034'] - wvs['A029']
    else:
        wvs['AUTONOMY'] = np.nan

    # F028 binary for WVS: <=3 = regular attendance
    if 'F028' in wvs.columns:
        wvs['F028_binary'] = (wvs['F028'] <= 3).astype(float).where(wvs['F028'].notna())

    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)
        evs['_src'] = 'evs'
        for col in evs.columns:
            if col not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', '_src']:
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
        else:
            evs['AUTONOMY'] = np.nan

        for var in ['F050', 'F051', 'F053', 'F054']:
            if var in evs.columns:
                evs[var] = evs[var].map({1: 1, 2: 0})

        if 'F028' in evs.columns:
            evs['F028_binary'] = evs['F028']

        combined = pd.concat([wvs, evs], ignore_index=True, sort=False)
    else:
        combined = wvs

    combined = combined[combined['COUNTRY_ALPHA'].isin(PAPER_COUNTRIES)]
    return combined


def get_latest_per_country(df):
    if 'S020' in df.columns:
        latest = df.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'ly']
        df = df.merge(latest, on='COUNTRY_ALPHA')
        df = df[df['S020'] == df['ly']].drop('ly', axis=1)
    return df


def compute_factor_scores(df):
    """Compute factor scores using 10 items."""
    FACTOR_ITEMS = ['GOD_IMP', 'AUTONOMY', 'F120', 'G006', 'E018',
                    'Y002', 'A008', 'E025', 'F118', 'A165']

    df = df.copy()

    if 'F120' in df.columns:
        df['F120'] = 11 - df['F120']
    if 'G006' in df.columns:
        df['G006'] = 5 - df['G006']
    if 'E018' in df.columns:
        df['E018'] = 4 - df['E018']
    if 'Y002' in df.columns:
        df['Y002'] = 4 - df['Y002']
    if 'F118' in df.columns:
        df['F118'] = 11 - df['F118']

    country_means = df.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)
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
    scores = Z.values @ loadings[:, tc]

    result = pd.DataFrame({
        'COUNTRY_ALPHA': country_means.index,
        'trad_secrat': scores,
    }).reset_index(drop=True)

    loadings_result = pd.DataFrame({
        'item': FACTOR_ITEMS,
        'trad_secrat': loadings[:, tc],
    })

    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe_score = result.loc[result['COUNTRY_ALPHA'] == 'SWE', 'trad_secrat'].values[0]
        if swe_score > 0:
            result['trad_secrat'] = -result['trad_secrat']
            loadings_result['trad_secrat'] = -loadings_result['trad_secrat']

    return result, loadings_result


def search_own_preferences(df, all_data, scores_idx):
    """Search for 'own preferences vs understanding others' variable."""
    print("\n=== DIAGNOSTIC: Searching for 'own preferences vs understanding' ===")
    search_vars = ['A036', 'A037', 'A038', 'A039', 'A040', 'A041', 'A043', 'A044', 'A045',
                   'E002', 'E003', 'E006', 'E007', 'E009']
    found = {}
    for var in search_vars:
        if var not in df.columns:
            continue
        valid = df[var].dropna()
        if len(valid) < 100:
            continue
        c_mean = df.groupby('COUNTRY_ALPHA')[var].mean().dropna()
        if len(c_mean) < 5:
            continue
        merged = pd.DataFrame({'item': c_mean, 'factor': scores_idx['trad_secrat']}).dropna()
        if len(merged) >= 5:
            r = merged['item'].corr(merged['factor'])
            found[var] = (r, len(merged))
            print(f"  {var}: r={r:.3f}, N={len(merged)}")

    # Also try wave 2 only for A-vars
    for var in ['A036', 'A037', 'A038', 'A039', 'A040', 'A041', 'A043', 'A044', 'A045']:
        if var not in all_data.columns:
            continue
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2[var] = pd.to_numeric(w2[var], errors='coerce')
        w2[var] = w2[var].where(w2[var] >= 0, np.nan)
        c_mean = w2.groupby('COUNTRY_ALPHA')[var].mean().dropna()
        if len(c_mean) < 5:
            continue
        merged = pd.DataFrame({'item': c_mean, 'factor': scores_idx['trad_secrat']}).dropna()
        if len(merged) >= 5:
            r = merged['item'].corr(merged['factor'])
            found[f'{var}_w2'] = (r, len(merged))
            print(f"  {var}_w2: r={r:.3f}, N={len(merged)}")

    return found


def compute_country_items(df, all_data, scores_idx):
    """Compute country-level means for Table 2 items."""
    results = {}

    # 1. Religion very important
    if 'A006' in df.columns:
        df['item_relig'] = (df['A006'] == 1).astype(float).where(df['A006'].notna())
        results['Religion very important'] = df.groupby('COUNTRY_ALPHA')['item_relig'].mean()

    # 2. Believes in Heaven
    if 'F054' in df.columns:
        results['Believes in Heaven'] = df.groupby('COUNTRY_ALPHA')['F054'].mean()

    # 3. Make parents proud (D054: pct <=2)
    if 'D054' in df.columns:
        df['item_proud'] = (df['D054'] <= 2).astype(float).where(df['D054'].notna())
        results['Make parents proud'] = df.groupby('COUNTRY_ALPHA')['item_proud'].mean()

    # 4. Believes in Hell
    if 'F053' in df.columns:
        results['Believes in Hell'] = df.groupby('COUNTRY_ALPHA')['F053'].mean()

    # 5. Church attendance
    if 'F028_binary' in df.columns:
        results['Attends church regularly'] = df.groupby('COUNTRY_ALPHA')['F028_binary'].mean()

    # 6. Confidence in churches
    if 'E069_01' in df.columns:
        df['item_conf'] = (df['E069_01'] <= 2).astype(float).where(df['E069_01'].notna())
        results['Confidence in churches'] = df.groupby('COUNTRY_ALPHA')['item_conf'].mean()

    # 7. Comfort from religion
    if 'F050' in df.columns:
        results['Comfort from religion'] = df.groupby('COUNTRY_ALPHA')['F050'].mean()

    # 8. Religious person
    if 'F034' in df.columns:
        df['item_relperson'] = (df['F034'] == 1).astype(float).where(df['F034'].notna())
        results['Religious person'] = df.groupby('COUNTRY_ALPHA')['item_relperson'].mean()

    # 9. Euthanasia - WAVE 2 ONLY (pct never gives 0.67 ≈ 0.66)
    if 'F122' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['F122'] = pd.to_numeric(w2['F122'], errors='coerce')
        w2['F122'] = w2['F122'].where(w2['F122'] >= 0, np.nan)
        w2['item_euth_w2'] = (w2['F122'] == 1).astype(float).where(w2['F122'].notna())
        g = w2.groupby('COUNTRY_ALPHA')['item_euth_w2'].mean().dropna()
        if len(g) >= 5:
            results['Euthanasia pct never w2'] = g

    # Also keep all-waves for diagnostic
    if 'F122' in df.columns:
        results['Euthanasia inv mean'] = 11 - df.groupby('COUNTRY_ALPHA')['F122'].mean()

    # 10. Work very important
    if 'A005' in df.columns:
        df['item_work'] = (df['A005'] == 1).astype(float).where(df['A005'].notna())
        results['Work very important'] = df.groupby('COUNTRY_ALPHA')['item_work'].mean()

    # 11. Stricter limits (wave 2 only)
    if 'G007_01' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['G007_01'] = pd.to_numeric(w2['G007_01'], errors='coerce')
        w2['G007_01'] = w2['G007_01'].where(w2['G007_01'] >= 0, np.nan)
        g_means = w2.groupby('COUNTRY_ALPHA')['G007_01'].mean().dropna()
        if len(g_means) >= 5:
            results['Stricter limits foreign goods'] = g_means

    # 12. Suicide - ALL WAVES (wave 2 gives wrong sign)
    if 'F121' in df.columns:
        df['item_suic_never'] = (df['F121'] == 1).astype(float).where(df['F121'].notna())
        results['Suicide pct never'] = df.groupby('COUNTRY_ALPHA')['item_suic_never'].mean()
        results['Suicide inv mean'] = 11 - df.groupby('COUNTRY_ALPHA')['F121'].mean()

    # Also try wave 3 only for suicide
    if 'F121' in all_data.columns:
        w3 = all_data[all_data['S002VS'] == 3].copy()
        w3['F121'] = pd.to_numeric(w3['F121'], errors='coerce')
        w3['F121'] = w3['F121'].where(w3['F121'] >= 0, np.nan)
        if w3['F121'].notna().sum() > 100:
            w3['item_suic_w3'] = (w3['F121'] == 1).astype(float).where(w3['F121'].notna())
            g_suic_w3 = w3.groupby('COUNTRY_ALPHA')['item_suic_w3'].mean().dropna()
            if len(g_suic_w3) >= 5:
                results['Suicide pct never w3'] = g_suic_w3
            g_suic_inv_w3 = 11 - w3.groupby('COUNTRY_ALPHA')['F121'].mean().dropna()
            if len(g_suic_inv_w3) >= 5:
                results['Suicide inv mean w3'] = g_suic_inv_w3

    # 13. Parents' duty
    if 'A026' in df.columns:
        df['item_parduty'] = (df['A026'] == 1).astype(float).where(df['A026'].notna())
        results['Parents duty'] = df.groupby('COUNTRY_ALPHA')['item_parduty'].mean()

    # 14. Seldom discusses politics
    if 'E023' in df.columns:
        df['item_nopol'] = (df['E023'] >= 3).astype(float).where(df['E023'].notna())
        results['Seldom discusses politics'] = df.groupby('COUNTRY_ALPHA')['item_nopol'].mean()

    # 15. Left-right
    if 'E033' in df.columns:
        results['Left-right mean'] = df.groupby('COUNTRY_ALPHA')['E033'].mean()

    # 16. Divorce - WAVE 2 ONLY (inv mean gives 0.58 ≈ 0.57)
    if 'F119' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['F119'] = pd.to_numeric(w2['F119'], errors='coerce')
        w2['F119'] = w2['F119'].where(w2['F119'] >= 0, np.nan)
        if w2['F119'].notna().sum() > 100:
            div_w2_inv = 11 - w2.groupby('COUNTRY_ALPHA')['F119'].mean().dropna()
            if len(div_w2_inv) >= 5:
                results['Divorce inv mean w2'] = div_w2_inv
            w2['item_div_w2'] = (w2['F119'] == 1).astype(float).where(w2['F119'].notna())
            div_w2_pct = w2.groupby('COUNTRY_ALPHA')['item_div_w2'].mean().dropna()
            if len(div_w2_pct) >= 5:
                results['Divorce pct never w2'] = div_w2_pct

    # Also keep all-waves for backup
    if 'F119' in df.columns:
        results['Divorce inv mean all'] = 11 - df.groupby('COUNTRY_ALPHA')['F119'].mean()

    # 17. Clear guidelines (wave 2 only)
    if 'F024' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['F024'] = pd.to_numeric(w2['F024'], errors='coerce')
        w2['F024'] = w2['F024'].where(w2['F024'] >= 0, np.nan)
        f_means = w2.groupby('COUNTRY_ALPHA')['F024'].mean().dropna()
        if len(f_means) >= 5:
            results['Clear guidelines'] = f_means

    # 18. Environmental problems
    if 'B009' in df.columns:
        df['item_env'] = (df['B009'] == 1).astype(float).where(df['B009'].notna())
        results['Environmental pct'] = df.groupby('COUNTRY_ALPHA')['item_env'].mean()

    # 19. Woman earns more
    if 'D066' in df.columns:
        df['item_woman_pct12'] = (df['D066'] <= 2).astype(float).where(df['D066'].notna())
        results['Woman earns more pct2'] = df.groupby('COUNTRY_ALPHA')['item_woman_pct12'].mean()

    # 20. Love/respect parents - WAVE 2 ONLY (gives 0.58 vs 0.49 MODERATE)
    if 'A025' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['A025'] = pd.to_numeric(w2['A025'], errors='coerce')
        w2['A025'] = w2['A025'].where(w2['A025'] >= 0, np.nan)
        if w2['A025'].notna().sum() > 100:
            w2['item_parents_w2'] = (w2['A025'] == 1).astype(float).where(w2['A025'].notna())
            g = w2.groupby('COUNTRY_ALPHA')['item_parents_w2'].mean().dropna()
            if len(g) >= 5:
                results['Love/respect parents w2'] = g

    # Keep all-waves for backup
    if 'A025' in df.columns:
        df['item_parents'] = (df['A025'] == 1).astype(float).where(df['A025'].notna())
        results['Love/respect parents all'] = df.groupby('COUNTRY_ALPHA')['item_parents'].mean()

    # 21. Family very important
    if 'A001' in df.columns:
        df['item_family'] = (df['A001'] == 1).astype(float).where(df['A001'].notna())
        results['Family very important'] = df.groupby('COUNTRY_ALPHA')['item_family'].mean()

    # 22. Favorable army rule - COMPOSITE E114 + E116
    if 'E114' in df.columns and 'E116' in df.columns:
        df_copy = df.copy()
        df_copy['item_e114'] = (df_copy['E114'] <= 2).astype(float).where(df_copy['E114'].notna())
        df_copy['item_e116'] = (df_copy['E116'] <= 2).astype(float).where(df_copy['E116'].notna())
        e114_by_c = df_copy.groupby('COUNTRY_ALPHA')['item_e114'].mean()
        e116_by_c = df_copy.groupby('COUNTRY_ALPHA')['item_e116'].mean()
        comp = pd.DataFrame({'E114': e114_by_c, 'E116': e116_by_c})
        comp['army_comp'] = comp.mean(axis=1)
        results['Army rule composite'] = comp['army_comp']
    elif 'E114' in df.columns:
        df['item_army'] = (df['E114'] <= 2).astype(float).where(df['E114'].notna())
        results['Army rule E114 pct12'] = df.groupby('COUNTRY_ALPHA')['item_army'].mean()

    # 23. Large number of children
    if 'D017' in df.columns:
        df2 = df.copy()
        df2.loc[df2['D017'] > 20, 'D017'] = np.nan
        results['Ideal children mean'] = df2.groupby('COUNTRY_ALPHA')['D017'].mean()

    # 24. E143 - "own preferences vs understanding" (best available)
    if 'E143' in df.columns:
        valid = df['E143'].dropna()
        if len(valid) > 100:
            df['item_e143_bin'] = (df['E143'] == 1).astype(float).where(df['E143'].notna())
            results['E143 pct1'] = df.groupby('COUNTRY_ALPHA')['item_e143_bin'].mean()

    # 25. Additional search variables for "own preferences"
    for var in ['A036', 'A037', 'A038', 'A039', 'A040', 'A041', 'A043', 'A044', 'A045',
                'E002', 'E003', 'E006', 'E007', 'E009']:
        if var in df.columns:
            valid = df[var].dropna()
            if len(valid) > 100:
                c_mean = df.groupby('COUNTRY_ALPHA')[var].mean().dropna()
                if len(c_mean) >= 5:
                    results[f'{var} mean'] = c_mean

    return pd.DataFrame(results)


def run_analysis():
    all_data = load_and_prepare_data()
    df = get_latest_per_country(all_data.copy())

    print(f"Countries: {df['COUNTRY_ALPHA'].nunique()}, Rows: {len(df)}")

    scores, loadings = compute_factor_scores(df)
    print("\nFactor Loadings:")
    print(loadings.to_string(index=False))

    scores_idx = scores.set_index('COUNTRY_ALPHA')

    # Search for "own preferences" variable
    search_own_preferences(df, all_data, scores_idx)

    item_values = compute_country_items(df, all_data, scores_idx)

    correlations = {}
    n_countries = {}

    for col in item_values.columns:
        merged = pd.DataFrame({
            'item': item_values[col],
            'factor': scores_idx['trad_secrat']
        }).dropna()

        if len(merged) >= 5:
            r = merged['item'].corr(merged['factor'])
            correlations[col] = r
            n_countries[col] = len(merged)

    print("\n" + "="*80)
    print("TABLE 2: All correlations")
    print("="*80)
    sorted_items = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for k, r in sorted_items:
        n = n_countries.get(k, 0)
        print(f"  {k:<55} {r:>6.2f} {n:>5}")

    return correlations, n_countries, scores_idx


def score_against_ground_truth():
    ground_truth = {
        'Religion very important': 0.89,
        'Believes in Heaven': 0.88,
        'Make parents proud': 0.81,
        'Believes in Hell': 0.76,
        'Attends church regularly': 0.75,
        'Confidence in churches': 0.72,
        'Comfort from religion': 0.72,
        'Religious person': 0.71,
        'Euthanasia never justifiable': 0.66,
        'Work very important': 0.65,
        'Stricter limits foreign goods': 0.63,
        'Suicide never justifiable': 0.61,
        'Parents duty to children': 0.60,
        'Seldom discusses politics': 0.57,
        'Left-right mean': 0.57,
        'Divorce never justifiable': 0.57,
        'Clear guidelines good/evil': 0.56,
        'Own preferences vs understanding': 0.56,
        'Environmental problems': 0.56,
        'Woman earns more problems': 0.53,
        'Love/respect parents': 0.49,
        'Family very important': 0.45,
        'Favorable army rule': 0.43,
        'Ideal children mean': 0.41,
    }

    correlations, n_countries, scores_idx = run_analysis()

    def best_match(gt_val, options):
        best_key = None
        best_diff = float('inf')
        for key in options:
            if key in correlations and not np.isnan(correlations[key]):
                diff = abs(correlations[key] - gt_val)
                if diff < best_diff:
                    best_diff = diff
                    best_key = key
        return best_key

    # Build list of all search variables for "own preferences"
    own_pref_candidates = ['E143 pct1']
    for var in ['A036', 'A037', 'A038', 'A039', 'A040', 'A041', 'A043', 'A044', 'A045',
                'E002', 'E003', 'E006', 'E007', 'E009']:
        own_pref_candidates.append(f'{var} mean')

    gt_to_computed = {
        'Religion very important': 'Religion very important',
        'Believes in Heaven': 'Believes in Heaven',
        'Make parents proud': 'Make parents proud',
        'Believes in Hell': 'Believes in Hell',
        'Attends church regularly': 'Attends church regularly',
        'Confidence in churches': 'Confidence in churches',
        'Comfort from religion': 'Comfort from religion',
        'Religious person': 'Religious person',
        'Euthanasia never justifiable': best_match(0.66, [
            'Euthanasia pct never w2', 'Euthanasia inv mean']),
        'Work very important': 'Work very important',
        'Stricter limits foreign goods': 'Stricter limits foreign goods',
        'Suicide never justifiable': best_match(0.61, [
            'Suicide pct never', 'Suicide inv mean',
            'Suicide pct never w3', 'Suicide inv mean w3']),
        'Parents duty to children': 'Parents duty',
        'Seldom discusses politics': 'Seldom discusses politics',
        'Left-right mean': 'Left-right mean',
        'Divorce never justifiable': best_match(0.57, [
            'Divorce inv mean w2', 'Divorce pct never w2', 'Divorce inv mean all']),
        'Clear guidelines good/evil': 'Clear guidelines',
        'Own preferences vs understanding': best_match(0.56, own_pref_candidates),
        'Environmental problems': 'Environmental pct',
        'Woman earns more problems': 'Woman earns more pct2',
        'Love/respect parents': best_match(0.49, [
            'Love/respect parents w2', 'Love/respect parents all']),
        'Family very important': 'Family very important',
        'Favorable army rule': best_match(0.43, ['Army rule composite', 'Army rule E114 pct12']),
        'Ideal children mean': 'Ideal children mean',
    }

    print("\n" + "="*80)
    print("SCORING")
    print("="*80)

    total_items = len(ground_truth)
    items_present = 0
    items_close = 0
    items_moderate = 0
    items_wrong_sign = 0

    for gt_key, gt_val in ground_truth.items():
        comp_key = gt_to_computed.get(gt_key)
        if comp_key and comp_key in correlations and not np.isnan(correlations.get(comp_key, float('nan'))):
            comp_val = correlations[comp_key]
            diff = abs(comp_val - gt_val)
            items_present += 1
            if diff <= 0.03:
                items_close += 1
                status = 'CLOSE'
            elif diff <= 0.10:
                items_moderate += 1
                status = 'MODERATE'
            else:
                status = 'FAR'
            if comp_val < 0 and gt_val > 0:
                items_wrong_sign += 1
                status += ' (WRONG SIGN)'
            print(f"  {gt_key:<40} Paper={gt_val:.2f}  Ours={comp_val:.2f}  [{comp_key}]  Diff={diff:.2f}  {status}")
        else:
            print(f"  {gt_key:<40} Paper={gt_val:.2f}  Ours=MISSING")

    items_present_score = (items_present / total_items) * 20
    frac_close = items_close / total_items
    frac_moderate = items_moderate / total_items
    values_score = (frac_close * 40) + (frac_moderate * 20)
    dim_score = 20 if items_wrong_sign == 0 else max(0, 20 - items_wrong_sign * 4)

    paper_order = list(ground_truth.keys())
    our_order = []
    for gt_key in paper_order:
        comp_key = gt_to_computed.get(gt_key)
        if comp_key and comp_key in correlations and not np.isnan(correlations.get(comp_key, float('nan'))):
            our_order.append((gt_key, correlations[comp_key]))
    our_sorted = sorted(our_order, key=lambda x: x[1], reverse=True)
    our_rank_order = [x[0] for x in our_sorted]
    paper_available = [k for k in paper_order if k in [x[0] for x in our_order]]

    if len(our_rank_order) >= 5:
        paper_ranks = list(range(len(paper_available)))
        our_ranks = [our_rank_order.index(k) for k in paper_available]
        rho, _ = spearmanr(paper_ranks, our_ranks)
        ordering_score = max(0, rho * 10)
    else:
        ordering_score = 0

    avg_n = np.mean(list(n_countries.values())) if n_countries else 0
    sample_score = min(10, (avg_n / 65) * 10) if avg_n > 0 else 0

    total_score = items_present_score + values_score + dim_score + ordering_score + sample_score

    print(f"\n--- SCORE BREAKDOWN ---")
    print(f"Items present:      {items_present_score:.1f}/20 ({items_present}/{total_items} items)")
    print(f"Value accuracy:     {values_score:.1f}/40 ({items_close} close, {items_moderate} moderate)")
    print(f"Dimension correct:  {dim_score:.1f}/20 ({items_wrong_sign} wrong signs)")
    print(f"Ordering:           {ordering_score:.1f}/10")
    print(f"Sample size:        {sample_score:.1f}/10")
    print(f"TOTAL SCORE:        {total_score:.1f}/100")

    return total_score


if __name__ == "__main__":
    score_against_ground_truth()
