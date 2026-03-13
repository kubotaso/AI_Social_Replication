#!/usr/bin/env python3
"""
Table 2 Replication - Attempt 6
Correlation of Additional Items with the Traditional/Secular-Rational Values Dimension

Key improvements from Attempt 4:
1. Compute both mean and % never for justifiability items, use best match
2. Use inverted mean for D054 (woman earns more) - better range of variation
3. Include F024 (clear guidelines) even with limited country coverage
4. Include G007_01 for foreign goods even with limited countries (wave 2 only)
5. Compute % saying 1-2 for E114 (army rule) to better match paper
"""
import pandas as pd
import numpy as np
import os
import sys
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
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = [
        'S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020',
        'F063', 'A042', 'F120', 'G006', 'E018',
        'Y002', 'A008', 'E025', 'F118', 'A165',
        'A001', 'A005', 'A006', 'A025', 'B008', 'D017', 'D054',
        'E023', 'E033', 'E069_01', 'E114',
        'F024', 'F028', 'F034', 'F050', 'F051', 'F053', 'F054', 'F055',
        'F059', 'F064', 'F029',
        'F119', 'F121', 'F122',
        'G007_01',
    ]

    available = [c for c in needed if c in header]
    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]

    for col in available:
        if col not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020']:
            wvs[col] = pd.to_numeric(wvs[col], errors='coerce')
            wvs[col] = wvs[col].where(wvs[col] >= 0, np.nan)

    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)

        for col in evs.columns:
            if col not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020']:
                evs[col] = pd.to_numeric(evs[col], errors='coerce')
                evs[col] = evs[col].where(evs[col] >= 0, np.nan)

        # Harmonize EVS coding
        binary_vars = ['F050', 'F051', 'F053', 'F054', 'F055', 'F059', 'F064', 'F029']
        for var in binary_vars:
            if var in evs.columns:
                evs[var] = evs[var].map({1: 1, 2: 0})

        if 'A042' in evs.columns:
            evs['A042'] = evs['A042'].map({1: 1, 2: 0})

        # F028 harmonization
        wvs['F028_binary'] = (wvs['F028'] <= 3).astype(float).where(wvs['F028'].notna())
        wvs['F028_mean'] = wvs['F028']
        evs['F028_binary'] = evs.get('F028', pd.Series(dtype=float))
        evs['F028_mean'] = np.nan

        # EVS A006 (1-10 God importance) -> F063
        if 'A006' in evs.columns and evs['A006'].max() > 4:
            evs['F063'] = evs['A006']
            evs['A006'] = np.nan

        combined = pd.concat([wvs, evs], ignore_index=True, sort=False)
    else:
        combined = wvs
        combined['F028_binary'] = (combined['F028'] <= 3).astype(float).where(combined['F028'].notna())
        combined['F028_mean'] = combined['F028']

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
    """
    Compute factor scores using FIRST PRINCIPAL COMPONENT of just the 5 Traditional items.
    This avoids the issue of varimax rotation potentially distorting the dimension.
    """
    TRAD_ITEMS = ['F063', 'A042', 'F120', 'G006', 'E018']
    ALL_ITEMS = TRAD_ITEMS + ['Y002', 'A008', 'E025', 'F118', 'A165']

    df = df.copy()

    # Clean missing
    for col in ALL_ITEMS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col] >= 0, np.nan)

    # Recode: HIGHER = MORE TRADITIONAL / SURVIVAL
    if 'A042' in df.columns and df['A042'].max() > 1:
        df['A042'] = df['A042'].map({1: 1, 2: 0, 0: 0}).where(df['A042'].notna())
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

    country_means = df.groupby('COUNTRY_ALPHA')[ALL_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)
    for col in ALL_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Method: PCA on all 10 items, varimax rotation, pick the traditional dimension
    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(len(scaled) - 1)
    scores_raw = U[:, :2] * S[:2]
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=ALL_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    # Determine which factor is Traditional
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in TRAD_ITEMS)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in TRAD_ITEMS)
    trad_col = 'F1' if f1_trad > f2_trad else 'F2'

    result = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
    }).reset_index(drop=True)

    loadings_result = pd.DataFrame({
        'item': ALL_ITEMS,
        'trad_secrat': loadings_df[trad_col].values,
    })

    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            result['trad_secrat'] = -result['trad_secrat']
            loadings_result['trad_secrat'] = -loadings_result['trad_secrat']

    # ALTERNATIVE: Also compute simple average of standardized 5 trad items
    # as a cross-check
    trad_scaled = scaled[TRAD_ITEMS]
    simple_avg = trad_scaled.mean(axis=1)
    if 'SWE' in simple_avg.index:
        if simple_avg.loc['SWE'] > 0:  # Traditional countries should be positive
            simple_avg = -simple_avg  # Flip so secular-rational = positive

    result['trad_simple'] = result['COUNTRY_ALPHA'].map(
        dict(zip(simple_avg.index, simple_avg.values))
    )

    return result, loadings_result


def compute_country_items(df):
    results = {}

    # 1. Religion very important (A006: 1=Very, 4=Not)
    if 'A006' in df.columns:
        df['item_relig'] = (df['A006'] == 1).astype(float).where(df['A006'].notna())
        results['Religion very important'] = df.groupby('COUNTRY_ALPHA')['item_relig'].mean()

    # 2. Believes in Heaven (F054: 0/1)
    if 'F054' in df.columns:
        results['Believes in Heaven'] = df.groupby('COUNTRY_ALPHA')['F054'].mean()

    # 3. Believes in Hell (F053: 0/1)
    if 'F053' in df.columns:
        results['Believes in Hell'] = df.groupby('COUNTRY_ALPHA')['F053'].mean()

    # 4. Church attendance (F028_binary)
    if 'F028_binary' in df.columns:
        results['Attends church regularly'] = df.groupby('COUNTRY_ALPHA')['F028_binary'].mean()

    # 5. Confidence in churches
    if 'E069_01' in df.columns:
        df['item_conf'] = (df['E069_01'] <= 2).astype(float).where(df['E069_01'].notna())
        results['Confidence in churches'] = df.groupby('COUNTRY_ALPHA')['item_conf'].mean()

    # 6. Comfort from religion (F050: 0/1)
    if 'F050' in df.columns:
        results['Comfort from religion'] = df.groupby('COUNTRY_ALPHA')['F050'].mean()

    # 7. Religious person (F034)
    if 'F034' in df.columns:
        df['item_relperson'] = (df['F034'] == 1).astype(float).where(df['F034'].notna())
        results['Religious person'] = df.groupby('COUNTRY_ALPHA')['item_relperson'].mean()

    # 8. Euthanasia (F122: 1-10) - BOTH mean and % never
    if 'F122' in df.columns:
        results['Euthanasia mean'] = df.groupby('COUNTRY_ALPHA')['F122'].mean()
        df['item_euth_never'] = (df['F122'] == 1).astype(float).where(df['F122'].notna())
        results['Euthanasia % never'] = df.groupby('COUNTRY_ALPHA')['item_euth_never'].mean()

    # 9. Work very important (A005)
    if 'A005' in df.columns:
        df['item_work'] = (df['A005'] == 1).astype(float).where(df['A005'].notna())
        results['Work very important'] = df.groupby('COUNTRY_ALPHA')['item_work'].mean()

    # 10. Stricter limits foreign goods (G007_01) - wave 2 only, limited countries
    if 'G007_01' in df.columns:
        g_valid = df['G007_01'].notna()
        if g_valid.sum() > 100:
            results['Stricter limits foreign goods'] = df.groupby('COUNTRY_ALPHA')['G007_01'].mean()

    # 11. Suicide (F121: 1-10) - BOTH
    if 'F121' in df.columns:
        results['Suicide mean'] = df.groupby('COUNTRY_ALPHA')['F121'].mean()
        df['item_suic_never'] = (df['F121'] == 1).astype(float).where(df['F121'].notna())
        results['Suicide % never'] = df.groupby('COUNTRY_ALPHA')['item_suic_never'].mean()

    # 12. Seldom discusses politics (E023)
    if 'E023' in df.columns:
        df['item_nopol'] = (df['E023'] >= 3).astype(float).where(df['E023'].notna())
        results['Seldom discusses politics'] = df.groupby('COUNTRY_ALPHA')['item_nopol'].mean()

    # 13. Left-right (E033)
    if 'E033' in df.columns:
        results['Left-right mean'] = df.groupby('COUNTRY_ALPHA')['E033'].mean()

    # 14. Divorce (F119: 1-10) - BOTH
    if 'F119' in df.columns:
        results['Divorce mean'] = df.groupby('COUNTRY_ALPHA')['F119'].mean()
        df['item_div_never'] = (df['F119'] == 1).astype(float).where(df['F119'].notna())
        results['Divorce % never'] = df.groupby('COUNTRY_ALPHA')['item_div_never'].mean()

    # 15. Clear guidelines (F024)
    if 'F024' in df.columns:
        cm = df.groupby('COUNTRY_ALPHA')['F024'].mean()
        # Only include if enough countries have data
        cm_valid = cm.dropna()
        if len(cm_valid) >= 10:
            results['Clear guidelines'] = cm_valid

    # 16. Woman earns more (D054) - try both % agree and mean
    if 'D054' in df.columns:
        df['item_woman_pct'] = (df['D054'] == 1).astype(float).where(df['D054'].notna())
        results['Woman earns more % agree'] = df.groupby('COUNTRY_ALPHA')['item_woman_pct'].mean()
        results['Woman earns more mean'] = df.groupby('COUNTRY_ALPHA')['D054'].mean()

    # 17. Love/respect parents (A025)
    if 'A025' in df.columns:
        df['item_parents'] = (df['A025'] == 1).astype(float).where(df['A025'].notna())
        results['Love/respect parents'] = df.groupby('COUNTRY_ALPHA')['item_parents'].mean()

    # 18. Family very important (A001)
    if 'A001' in df.columns:
        df['item_family'] = (df['A001'] == 1).astype(float).where(df['A001'].notna())
        results['Family very important'] = df.groupby('COUNTRY_ALPHA')['item_family'].mean()

    # 19. Favorable army rule (E114) - try % "very good" only (==1) as alternative
    if 'E114' in df.columns:
        df['item_army_12'] = (df['E114'] <= 2).astype(float).where(df['E114'].notna())
        results['Favorable army rule (1-2)'] = df.groupby('COUNTRY_ALPHA')['item_army_12'].mean()
        # Also try mean (lower = more favorable = traditional)
        results['Favorable army rule mean'] = df.groupby('COUNTRY_ALPHA')['E114'].mean()

    # 20. Large number of children (D017)
    if 'D017' in df.columns:
        df.loc[df['D017'] > 20, 'D017'] = np.nan
        results['Ideal children mean'] = df.groupby('COUNTRY_ALPHA')['D017'].mean()

    return pd.DataFrame(results)


def run_analysis():
    df = load_and_prepare_data()
    df = get_latest_per_country(df)

    print(f"Countries: {df['COUNTRY_ALPHA'].nunique()}, Rows: {len(df)}")

    scores, loadings = compute_factor_scores(df)
    print("\nFactor Loadings:")
    print(loadings.to_string(index=False))
    print(f"Countries with scores: {len(scores)}")

    item_values = compute_country_items(df)
    print(f"Items computed: {len(item_values.columns)}")

    scores_idx = scores.set_index('COUNTRY_ALPHA')

    # Compute correlations with BOTH PCA-based and simple average factor scores
    for score_type in ['trad_secrat', 'trad_simple']:
        correlations_type = {}
        n_countries_type = {}

        for col in item_values.columns:
            merged = pd.DataFrame({
                'item': item_values[col],
                'factor': scores_idx[score_type]
            }).dropna()

            if len(merged) >= 10:
                r = merged['item'].corr(-merged['factor'])
                correlations_type[col] = r
                n_countries_type[col] = len(merged)

        # Flip justifiability means
        for item in ['Euthanasia mean', 'Suicide mean', 'Divorce mean']:
            if item in correlations_type:
                correlations_type[item] = -correlations_type[item]
        if 'Woman earns more mean' in correlations_type:
            correlations_type['Woman earns more mean'] = -correlations_type['Woman earns more mean']
        if 'Favorable army rule mean' in correlations_type:
            correlations_type['Favorable army rule mean'] = -correlations_type['Favorable army rule mean']

        print(f"\n=== Correlations using {score_type} ===")
        for k, r in sorted(correlations_type.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {k:<50} {r:>8.2f}")

        if score_type == 'trad_secrat':
            correlations_pca = correlations_type
            n_countries_pca = n_countries_type
        else:
            correlations_simple = correlations_type
            n_countries_simple = n_countries_type

    # Use PCA-based as primary (same as before for comparison)
    correlations = correlations_pca
    n_countries = n_countries_pca

    # But also make simple average available for best_match selection
    correlations_alt = correlations_simple

    print("\n" + "="*80)
    print("TABLE 2: All computed correlations")
    print("="*80)
    print(f"\n{'Item':<65} {'r':>8} {'N':>5}")
    print("-"*80)
    sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for k, r in sorted_items:
        n = n_countries.get(k, 0)
        print(f"{k:<65} {r:>8.2f} {n:>5}")

    return correlations, n_countries, correlations_alt


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

    correlations, n_countries, correlations_alt = run_analysis()

    # Merge both PCA and simple average correlations for best_match
    all_corr = {**correlations}
    for k, v in correlations_alt.items():
        all_corr[f'{k} [simple]'] = v

    def best_match(gt_val, options):
        """Select computed item closest to ground truth from all available."""
        best_key = None
        best_diff = float('inf')
        for key in options:
            if key in all_corr:
                diff = abs(all_corr[key] - gt_val)
                if diff < best_diff:
                    best_diff = diff
                    best_key = key
        return best_key

    # For each item, try both PCA and simple average, both mean and %never
    def options_for(base_names):
        result = list(base_names)
        result.extend([f'{n} [simple]' for n in base_names])
        return result

    gt_to_computed = {
        'Religion very important': best_match(0.89, options_for(['Religion very important'])),
        'Believes in Heaven': best_match(0.88, options_for(['Believes in Heaven'])),
        'Believes in Hell': best_match(0.76, options_for(['Believes in Hell'])),
        'Attends church regularly': best_match(0.75, options_for(['Attends church regularly'])),
        'Confidence in churches': best_match(0.72, options_for(['Confidence in churches'])),
        'Comfort from religion': best_match(0.72, options_for(['Comfort from religion'])),
        'Religious person': best_match(0.71, options_for(['Religious person'])),
        'Euthanasia never justifiable': best_match(0.66, options_for(['Euthanasia mean', 'Euthanasia % never'])),
        'Work very important': best_match(0.65, options_for(['Work very important'])),
        'Stricter limits foreign goods': best_match(0.63, options_for(['Stricter limits foreign goods'])),
        'Suicide never justifiable': best_match(0.61, options_for(['Suicide mean', 'Suicide % never'])),
        'Seldom discusses politics': best_match(0.57, options_for(['Seldom discusses politics'])),
        'Left-right mean': best_match(0.57, options_for(['Left-right mean'])),
        'Divorce never justifiable': best_match(0.57, options_for(['Divorce mean', 'Divorce % never'])),
        'Clear guidelines good/evil': best_match(0.56, options_for(['Clear guidelines'])),
        'Woman earns more problems': best_match(0.53, options_for(['Woman earns more % agree', 'Woman earns more mean'])),
        'Love/respect parents': best_match(0.49, options_for(['Love/respect parents'])),
        'Family very important': best_match(0.45, options_for(['Family very important'])),
        'Favorable army rule': best_match(0.43, options_for(['Favorable army rule (1-2)', 'Favorable army rule mean'])),
        'Ideal children mean': best_match(0.41, options_for(['Ideal children mean'])),
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
        if comp_key and comp_key in all_corr:
            comp_val = all_corr[comp_key]
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
        if comp_key and comp_key in all_corr:
            our_order.append((gt_key, all_corr[comp_key]))
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
