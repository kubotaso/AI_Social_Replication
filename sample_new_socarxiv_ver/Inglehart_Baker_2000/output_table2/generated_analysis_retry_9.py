#!/usr/bin/env python3
"""
Table 2 Replication - Attempt 9
Correlation of Additional Items with the Traditional/Secular-Rational Values Dimension

Key fixes from attempt 7 (best score 72.3):
1. D054 = "Make parents proud" (1=Agree strongly to 4=Strongly disagree)
   -> pct agreeing (1+2) correlates with traditional dimension
2. D066 = "Problem if woman earns more than husband" (1=Strongly agree to 4=Strongly disagree)
   -> was wrongly using D054 for this; now use D066
3. E116 = "Having the army rule" (1=Very good to 4=Very bad)
   -> was using E114 (strong leader, not army rule); E116 has r~0.64 in exploration
4. Euthanasia/suicide/divorce: try pct=1 only (never justifiable) as primary metric
5. Stricter limits (G007_01): try pct agree approach
6. Seldom discusses politics: use E023 == 3 (never)
7. Environmental: check B008 vs B009 coding
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


def load_and_prepare_data():
    """Load WVS + EVS data with careful handling of variable codings."""
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = [
        'S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020',
        # Factor analysis items
        'F063', 'A042', 'A029', 'A034', 'F120', 'G006', 'E018',
        'Y002', 'A008', 'E025', 'F118', 'A165',
        # Table 2 additional items
        'A001', 'A005', 'A006', 'A025', 'A026', 'B008', 'B009', 'D017',
        'D054',     # "Make parents proud" (1=Agree strongly, 4=Strongly disagree)
        'D066',     # "Problem if woman earns more" (1=Strongly agree, 4=Strongly disagree)
        'D066_01',  # Alternative version (5 categories)
        'E023', 'E033', 'E069_01',
        'E114',     # "Strong leader" (NOT army rule)
        'E116',     # "Having the army rule" (1=Very good, 4=Very bad) - CORRECT
        'F024', 'F028', 'F034', 'F050', 'F051', 'F053', 'F054',
        'F119', 'F121', 'F122',
        'G007_01',
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

        # EVS A006 is God importance 1-10 -> GOD_IMP
        if 'A006' in evs.columns:
            evs['GOD_IMP'] = evs['A006']
            evs['A006'] = np.nan  # Don't use EVS A006 for religion importance

        # EVS child qualities: 1=mentioned, 2=not -> 1/0
        for v in ['A042', 'A034', 'A029']:
            if v in evs.columns:
                evs.loc[evs[v] == 2, v] = 0

        if all(v in evs.columns for v in ['A042', 'A034', 'A029']):
            evs['AUTONOMY'] = evs['A042'] + evs['A034'] - evs['A029']
        else:
            evs['AUTONOMY'] = np.nan

        # EVS binary beliefs: 1=Yes/2=No -> 1/0
        for var in ['F050', 'F051', 'F053', 'F054']:
            if var in evs.columns:
                evs[var] = evs[var].map({1: 1, 2: 0})

        # EVS F028 is binary attendance
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
    """Compute factor scores using 10 items with AUTONOMY and GOD_IMP."""
    FACTOR_ITEMS = ['GOD_IMP', 'AUTONOMY', 'F120', 'G006', 'E018',
                    'Y002', 'A008', 'E025', 'F118', 'A165']

    df = df.copy()

    # Recode: higher = more traditional/survival
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

    # PCA via correlation matrix
    corr = country_means.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    loadings = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    loadings, R = varimax(loadings)

    loadings_df = pd.DataFrame(loadings, index=FACTOR_ITEMS, columns=['F1', 'F2'])

    # Identify traditional dimension
    trad_items = ['AUTONOMY', 'F120', 'G006', 'E018']
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items)
    tc = 0 if f1_trad > f2_trad else 1

    # Orient so positive = traditional
    if np.mean([loadings_df.iloc[loadings_df.index.get_loc(i), tc] for i in trad_items]) < 0:
        loadings[:, tc] *= -1

    # Compute factor scores using standardized data * loadings
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

    # Verify orientation: SWE should be negative (secular), NGA positive (traditional)
    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe_score = result.loc[result['COUNTRY_ALPHA'] == 'SWE', 'trad_secrat'].values[0]
        if swe_score > 0:
            result['trad_secrat'] = -result['trad_secrat']
            loadings_result['trad_secrat'] = -loadings_result['trad_secrat']

    return result, loadings_result


def compute_country_items(df, all_data):
    """Compute country-level means for Table 2 items.
    Uses latest-wave-per-country data; wave-2-only items use that subset.
    All items coded so higher = more traditional.
    """
    results = {}

    # 1. Religion very important (A006: 1=Very important, 4=Not important)
    # WVS only - EVS A006 was reassigned to GOD_IMP and set to NaN
    if 'A006' in df.columns:
        df['item_relig'] = (df['A006'] == 1).astype(float).where(df['A006'].notna())
        results['Religion very important'] = df.groupby('COUNTRY_ALPHA')['item_relig'].mean()

    # 2. Believes in Heaven (F054: 0=No, 1=Yes)
    if 'F054' in df.columns:
        results['Believes in Heaven'] = df.groupby('COUNTRY_ALPHA')['F054'].mean()

    # 3. Believes in Hell (F053: 0=No, 1=Yes)
    if 'F053' in df.columns:
        results['Believes in Hell'] = df.groupby('COUNTRY_ALPHA')['F053'].mean()

    # 4. Make parents proud (D054: 1=Agree strongly, 2=Agree, 3=Disagree, 4=Strongly disagree)
    # Pct agreeing (1+2) = more traditional
    if 'D054' in df.columns:
        df['item_proud'] = (df['D054'] <= 2).astype(float).where(df['D054'].notna())
        results['Make parents proud'] = df.groupby('COUNTRY_ALPHA')['item_proud'].mean()

    # 5. Church attendance (F028_binary harmonized across WVS/EVS)
    if 'F028_binary' in df.columns:
        results['Attends church regularly'] = df.groupby('COUNTRY_ALPHA')['F028_binary'].mean()

    # 6. Confidence in churches (E069_01: 1=A great deal, 4=None at all)
    if 'E069_01' in df.columns:
        df['item_conf'] = (df['E069_01'] <= 2).astype(float).where(df['E069_01'].notna())
        results['Confidence in churches'] = df.groupby('COUNTRY_ALPHA')['item_conf'].mean()

    # 7. Comfort from religion (F050: 1=Yes, 2=No -> recoded 0/1 for EVS)
    if 'F050' in df.columns:
        results['Comfort from religion'] = df.groupby('COUNTRY_ALPHA')['F050'].mean()

    # 8. Religious person (F034: 1=Religious person, 2=Not religious, 3=Atheist)
    if 'F034' in df.columns:
        df['item_relperson'] = (df['F034'] == 1).astype(float).where(df['F034'].notna())
        results['Religious person'] = df.groupby('COUNTRY_ALPHA')['item_relperson'].mean()

    # 9. Euthanasia (F122: 1-10, 1=Never justified, 10=Always)
    if 'F122' in df.columns:
        df['item_euth_never'] = (df['F122'] == 1).astype(float).where(df['F122'].notna())
        results['Euthanasia pct never'] = df.groupby('COUNTRY_ALPHA')['item_euth_never'].mean()
        results['Euthanasia inv mean'] = 11 - df.groupby('COUNTRY_ALPHA')['F122'].mean()

    # 10. Work very important (A005: 1=Very important, 4=Not important)
    if 'A005' in df.columns:
        df['item_work'] = (df['A005'] == 1).astype(float).where(df['A005'].notna())
        results['Work very important'] = df.groupby('COUNTRY_ALPHA')['item_work'].mean()

    # 11. Stricter limits foreign goods (G007_01: 1=Strongly agree, 5=Strongly disagree)
    # Wave 2 only; 1=want stricter limits -> traditional
    if 'G007_01' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['G007_01'] = pd.to_numeric(w2['G007_01'], errors='coerce')
        w2['G007_01'] = w2['G007_01'].where(w2['G007_01'] >= 0, np.nan)
        w2['item_strict'] = (w2['G007_01'] <= 2).astype(float).where(w2['G007_01'].notna())
        strict_pct = w2.groupby('COUNTRY_ALPHA')['item_strict'].mean().dropna()
        strict_inv = (6 - w2.groupby('COUNTRY_ALPHA')['G007_01'].mean()).dropna()
        if len(strict_pct) >= 5:
            results['Stricter limits pct agree'] = strict_pct
            results['Stricter limits inv mean'] = strict_inv

    # 12. Suicide (F121: 1-10)
    if 'F121' in df.columns:
        df['item_suic_never'] = (df['F121'] == 1).astype(float).where(df['F121'].notna())
        results['Suicide pct never'] = df.groupby('COUNTRY_ALPHA')['item_suic_never'].mean()
        results['Suicide inv mean'] = 11 - df.groupby('COUNTRY_ALPHA')['F121'].mean()

    # 13. Parents' duty (A026: 1=Agree strongly, 3=Disagree)
    if 'A026' in df.columns:
        df['item_parduty'] = (df['A026'] == 1).astype(float).where(df['A026'].notna())
        results['Parents duty'] = df.groupby('COUNTRY_ALPHA')['item_parduty'].mean()

    # 14. Seldom discusses politics (E023: 1=Frequently, 2=Occasionally, 3=Never)
    # Traditional = less political discussion
    if 'E023' in df.columns:
        df['item_nopol'] = (df['E023'] == 3).astype(float).where(df['E023'].notna())
        results['Seldom discusses politics'] = df.groupby('COUNTRY_ALPHA')['item_nopol'].mean()
        # Also try >= 3 (never only, same as == 3 in 3-category)
        df['item_nopol_ge3'] = (df['E023'] >= 2).astype(float).where(df['E023'].notna())
        results['Seldom discusses politics ge2'] = df.groupby('COUNTRY_ALPHA')['item_nopol_ge3'].mean()

    # 15. Left-right (E033: 1=Left, 10=Right)
    if 'E033' in df.columns:
        results['Left-right mean'] = df.groupby('COUNTRY_ALPHA')['E033'].mean()

    # 16. Divorce (F119: 1-10)
    if 'F119' in df.columns:
        df['item_div_never'] = (df['F119'] == 1).astype(float).where(df['F119'].notna())
        results['Divorce pct never'] = df.groupby('COUNTRY_ALPHA')['item_div_never'].mean()
        results['Divorce inv mean'] = 11 - df.groupby('COUNTRY_ALPHA')['F119'].mean()

    # 17. Clear guidelines (F024: wave 2 only)
    # "There are absolutely clear guidelines about what is good and evil"
    if 'F024' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['F024'] = pd.to_numeric(w2['F024'], errors='coerce')
        w2['F024'] = w2['F024'].where(w2['F024'] >= 0, np.nan)
        w2['item_clear'] = (w2['F024'] <= 2).astype(float).where(w2['F024'].notna())
        clear_pct = w2.groupby('COUNTRY_ALPHA')['item_clear'].mean().dropna()
        if len(clear_pct) >= 5:
            results['Clear guidelines'] = clear_pct

    # 18. Environmental problems (B008 or B009)
    for bvar in ['B008', 'B009']:
        if bvar in df.columns:
            bmax = df[bvar].max()
            df[f'item_env_{bvar}'] = (df[bvar] == 1).astype(float).where(df[bvar].notna())
            results[f'Environmental pct {bvar}'] = df.groupby('COUNTRY_ALPHA')[f'item_env_{bvar}'].mean()
            results[f'Environmental inv mean {bvar}'] = (bmax + 1) - df.groupby('COUNTRY_ALPHA')[bvar].mean()

    # 19. Woman earns more (D066: 1=Strongly agree, 4=Strongly disagree that it's a problem)
    # D066: "If a woman earns more money than her husband, it's almost certain to cause problems"
    # 1=Strongly agree = more traditional; pct agreeing (1+2) = traditional
    if 'D066' in df.columns:
        df['item_woman_pct1'] = (df['D066'] == 1).astype(float).where(df['D066'].notna())
        results['Woman earns more pct1'] = df.groupby('COUNTRY_ALPHA')['item_woman_pct1'].mean()
        df['item_woman_pct12'] = (df['D066'] <= 2).astype(float).where(df['D066'].notna())
        results['Woman earns more pct12'] = df.groupby('COUNTRY_ALPHA')['item_woman_pct12'].mean()
        results['Woman earns more inv mean'] = 5 - df.groupby('COUNTRY_ALPHA')['D066'].mean()

    # Also try D066_01 (5-category version)
    if 'D066_01' in df.columns:
        df['item_woman66_01'] = (df['D066_01'] <= 2).astype(float).where(df['D066_01'].notna())
        results['Woman earns more D066_01 pct12'] = df.groupby('COUNTRY_ALPHA')['item_woman66_01'].mean()

    # 20. Love/respect parents (A025: 1=Always, 2=Depends, 3=No obligation)
    if 'A025' in df.columns:
        df['item_parents'] = (df['A025'] == 1).astype(float).where(df['A025'].notna())
        results['Love/respect parents'] = df.groupby('COUNTRY_ALPHA')['item_parents'].mean()

    # 21. Family very important (A001: 1=Very important, 4=Not at all)
    if 'A001' in df.columns:
        df['item_family'] = (df['A001'] == 1).astype(float).where(df['A001'].notna())
        results['Family very important'] = df.groupby('COUNTRY_ALPHA')['item_family'].mean()

    # 22. Favorable army rule (E116: 1=Very good, 4=Very bad) - CORRECTED from E114
    if 'E116' in df.columns:
        df['item_army'] = (df['E116'] <= 2).astype(float).where(df['E116'].notna())
        results['Favorable army rule'] = df.groupby('COUNTRY_ALPHA')['item_army'].mean()

    # Also compute E114 for comparison
    if 'E114' in df.columns:
        df['item_strongldr'] = (df['E114'] <= 2).astype(float).where(df['E114'].notna())
        results['Strong leader E114'] = df.groupby('COUNTRY_ALPHA')['item_strongldr'].mean()

    # 23. Large number of children (D017: ideal number)
    if 'D017' in df.columns:
        df2 = df.copy()
        df2.loc[df2['D017'] > 20, 'D017'] = np.nan
        results['Ideal children mean'] = df2.groupby('COUNTRY_ALPHA')['D017'].mean()

    return pd.DataFrame(results)


def run_analysis():
    all_data = load_and_prepare_data()
    df = get_latest_per_country(all_data.copy())

    print(f"Countries: {df['COUNTRY_ALPHA'].nunique()}, Rows: {len(df)}")

    scores, loadings = compute_factor_scores(df)
    print("\nFactor Loadings (Trad/Sec-Rat, positive=traditional):")
    print(loadings.to_string(index=False))
    print(f"Countries with scores: {len(scores)}")

    item_values = compute_country_items(df, all_data)
    print(f"Items computed: {len(item_values.columns)}")

    scores_idx = scores.set_index('COUNTRY_ALPHA')

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
    print("TABLE 2: Correlations with Traditional/Secular-Rational Dimension")
    print("="*80)
    print(f"\n{'Item':<55} {'r':>8} {'N':>5}")
    print("-"*70)
    sorted_items = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for k, r in sorted_items:
        n = n_countries.get(k, 0)
        print(f"{k:<55} {r:>8.2f} {n:>5}")

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
            if key in correlations:
                diff = abs(correlations[key] - gt_val)
                if diff < best_diff:
                    best_diff = diff
                    best_key = key
        return best_key

    gt_to_computed = {
        'Religion very important': 'Religion very important',
        'Believes in Heaven': 'Believes in Heaven',
        'Make parents proud': 'Make parents proud',
        'Believes in Hell': 'Believes in Hell',
        'Attends church regularly': 'Attends church regularly',
        'Confidence in churches': 'Confidence in churches',
        'Comfort from religion': 'Comfort from religion',
        'Religious person': 'Religious person',
        'Euthanasia never justifiable': best_match(0.66, ['Euthanasia pct never', 'Euthanasia inv mean']),
        'Work very important': 'Work very important',
        'Stricter limits foreign goods': best_match(0.63, ['Stricter limits pct agree', 'Stricter limits inv mean']),
        'Suicide never justifiable': best_match(0.61, ['Suicide pct never', 'Suicide inv mean']),
        'Parents duty to children': 'Parents duty',
        'Seldom discusses politics': best_match(0.57, ['Seldom discusses politics', 'Seldom discusses politics ge2']),
        'Left-right mean': 'Left-right mean',
        'Divorce never justifiable': best_match(0.57, ['Divorce pct never', 'Divorce inv mean']),
        'Clear guidelines good/evil': 'Clear guidelines',
        'Own preferences vs understanding': None,  # Still MISSING - not found in codebook
        'Environmental problems': best_match(0.56, [
            'Environmental pct B008', 'Environmental inv mean B008',
            'Environmental pct B009', 'Environmental inv mean B009'
        ]),
        'Woman earns more problems': best_match(0.53, [
            'Woman earns more pct1', 'Woman earns more pct12',
            'Woman earns more inv mean', 'Woman earns more D066_01 pct12'
        ]),
        'Love/respect parents': 'Love/respect parents',
        'Family very important': 'Family very important',
        'Favorable army rule': 'Favorable army rule',
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
        if comp_key and comp_key in correlations:
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
        if comp_key and comp_key in correlations:
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
    score = score_against_ground_truth()
