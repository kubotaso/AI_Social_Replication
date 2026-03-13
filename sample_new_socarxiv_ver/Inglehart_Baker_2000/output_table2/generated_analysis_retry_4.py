#!/usr/bin/env python3
"""
Table 2 Replication - Attempt 4
Correlation of Additional Items with the Traditional/Secular-Rational Values Dimension

Key fixes from Attempt 3:
1. Handle EVS F028 coding (0/1 binary vs WVS 1-8 frequency) -- DO NOT combine directly
2. For items where EVS coding differs, compute country means separately then merge
3. Exclude countries not in paper's likely 65 societies (MLT, MNE, SLV, ALB)
4. Use inverted mean for church attendance instead of binary threshold
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

# Paper's likely 65 societies (from analysis of Figure 1 and paper text)
# Excluding: MLT (Malta), MNE (Montenegro), SLV (El Salvador), ALB (Albania)
# These are not typical for 1990s WVS
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
    """Load WVS + EVS with careful handling of variable codings."""
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

    # For WVS: clean negative values
    for col in available:
        if col not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020']:
            wvs[col] = pd.to_numeric(wvs[col], errors='coerce')
            wvs[col] = wvs[col].where(wvs[col] >= 0, np.nan)

    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)

        # Clean EVS negatives
        for col in evs.columns:
            if col not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020']:
                evs[col] = pd.to_numeric(evs[col], errors='coerce')
                evs[col] = evs[col].where(evs[col] >= 0, np.nan)

        # HARMONIZE EVS -> WVS coding

        # Binary variables: EVS uses 1=Yes/2=No, WVS uses 0=No/1=Yes
        binary_vars = ['F050', 'F051', 'F053', 'F054', 'F055', 'F059', 'F064', 'F029']
        for var in binary_vars:
            if var in evs.columns:
                evs[var] = evs[var].map({1: 1, 2: 0})

        # A042: EVS 1=mentioned/2=not mentioned -> 0/1
        if 'A042' in evs.columns:
            evs['A042'] = evs['A042'].map({1: 1, 2: 0})

        # F028: EVS is 0/1 (0=no, 1=yes regular attendance)
        # WVS is 1-8 (1=weekly+, 8=never)
        # CANNOT combine directly. We need to handle this specially.
        # For EVS: F028=1 means "attends regularly", F028=0 means "doesn't attend regularly"
        # For WVS: compute % attending at least monthly (F028 <= 3)
        # Solution: For each country, compute the "% attending regularly" separately
        # EVS F028: already 0/1 (proportion = mean)
        # WVS F028: convert to 0/1 (attending <= 3)

        # Convert WVS F028 to binary for church attendance
        wvs['F028_binary'] = (wvs['F028'] <= 3).astype(float).where(wvs['F028'].notna())
        # Keep original F028 for potential mean-based measure
        wvs['F028_mean'] = wvs['F028']

        # EVS F028 is already binary
        evs['F028_binary'] = evs['F028']
        evs['F028_mean'] = np.nan  # Cannot compute mean on 0/1 EVS data

        # A006 in EVS is 1-10 (God importance), in WVS it's 1-4 (religion importance)
        # We need to handle these separately
        # For the "religion is very important" item: use A006 from WVS (1-4 scale)
        # For the EVS: A006 is 1-10 (God importance) - not the same question
        # Best: compute "religion very important" only from WVS data where A006 is 1-4
        # And for EVS countries, this item will be missing (which is OK)

        # F063 in WVS: 1-10 God importance
        # EVS doesn't have F063 (they have A006 for this)
        # For factor analysis: we can map EVS A006 (1-10) to F063
        if 'A006' in evs.columns and evs['A006'].max() > 4:
            evs['F063'] = evs['A006']  # EVS A006 (1-10) maps to WVS F063
            evs['A006'] = np.nan  # Don't use EVS A006 for the 1-4 "religion important" item

        combined = pd.concat([wvs, evs], ignore_index=True, sort=False)
    else:
        combined = wvs
        combined['F028_binary'] = (combined['F028'] <= 3).astype(float).where(combined['F028'].notna())
        combined['F028_mean'] = combined['F028']

    # Filter to paper's countries
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
    FACTOR_ITEMS = ['F063', 'A042', 'F120', 'G006', 'E018',
                    'Y002', 'A008', 'E025', 'F118', 'A165']

    df = df.copy()

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

    country_means = df.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)

    for col in FACTOR_ITEMS:
        country_means[col] = country_means[col].fillna(country_means[col].mean())

    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(len(scaled) - 1)
    scores_raw = U[:, :2] * S[:2]
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    trad_items = ['A042', 'F120', 'G006', 'E018']
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items)
    trad_col = 'F1' if f1_trad > f2_trad else 'F2'

    result = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
    }).reset_index(drop=True)

    loadings_result = pd.DataFrame({
        'item': FACTOR_ITEMS,
        'trad_secrat': loadings_df[trad_col].values,
    })

    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            result['trad_secrat'] = -result['trad_secrat']
            loadings_result['trad_secrat'] = -loadings_result['trad_secrat']

    return result, loadings_result


def compute_country_items(df):
    results = {}

    # 1. Religion very important (A006: 1=Very, 4=Not at all) - WVS only
    if 'A006' in df.columns:
        df['item_relig'] = (df['A006'] == 1).astype(float).where(df['A006'].notna())
        results['Religion very important'] = df.groupby('COUNTRY_ALPHA')['item_relig'].mean()

    # 2. Believes in Heaven (F054: 0/1)
    if 'F054' in df.columns:
        results['Believes in Heaven'] = df.groupby('COUNTRY_ALPHA')['F054'].mean()

    # 3. Believes in Hell (F053: 0/1)
    if 'F053' in df.columns:
        results['Believes in Hell'] = df.groupby('COUNTRY_ALPHA')['F053'].mean()

    # 4. Church attendance - USE F028_binary (harmonized)
    if 'F028_binary' in df.columns:
        results['Attends church regularly'] = df.groupby('COUNTRY_ALPHA')['F028_binary'].mean()

    # 5. Confidence in churches (E069_01: 1=Great deal...4=None)
    if 'E069_01' in df.columns:
        df['item_conf'] = (df['E069_01'] <= 2).astype(float).where(df['E069_01'].notna())
        results['Confidence in churches'] = df.groupby('COUNTRY_ALPHA')['item_conf'].mean()

    # 6. Comfort from religion (F050: 0/1 harmonized)
    if 'F050' in df.columns:
        results['Comfort from religion'] = df.groupby('COUNTRY_ALPHA')['F050'].mean()

    # 7. Religious person (F034: 1=Religious...3=Atheist)
    if 'F034' in df.columns:
        df['item_relperson'] = (df['F034'] == 1).astype(float).where(df['F034'].notna())
        results['Religious person'] = df.groupby('COUNTRY_ALPHA')['item_relperson'].mean()

    # 8. Euthanasia (F122: 1-10) - use mean
    if 'F122' in df.columns:
        results['Euthanasia mean'] = df.groupby('COUNTRY_ALPHA')['F122'].mean()

    # 9. Work very important (A005: 1=Very...4=Not)
    if 'A005' in df.columns:
        df['item_work'] = (df['A005'] == 1).astype(float).where(df['A005'].notna())
        results['Work very important'] = df.groupby('COUNTRY_ALPHA')['item_work'].mean()

    # 10. Stricter limits foreign goods (G007_01)
    if 'G007_01' in df.columns:
        valid = df['G007_01'].notna()
        if valid.sum() > 100:
            results['Stricter limits foreign goods'] = df.groupby('COUNTRY_ALPHA')['G007_01'].mean()

    # 11. Suicide (F121: 1-10)
    if 'F121' in df.columns:
        results['Suicide mean'] = df.groupby('COUNTRY_ALPHA')['F121'].mean()

    # 12. Seldom discusses politics (E023: 1-4)
    if 'E023' in df.columns:
        df['item_nopol'] = (df['E023'] >= 3).astype(float).where(df['E023'].notna())
        results['Seldom discusses politics'] = df.groupby('COUNTRY_ALPHA')['item_nopol'].mean()

    # 13. Left-right (E033: 1-10)
    if 'E033' in df.columns:
        results['Left-right mean'] = df.groupby('COUNTRY_ALPHA')['E033'].mean()

    # 14. Divorce (F119: 1-10)
    if 'F119' in df.columns:
        results['Divorce mean'] = df.groupby('COUNTRY_ALPHA')['F119'].mean()

    # 15. Clear guidelines (F024)
    if 'F024' in df.columns:
        results['Clear guidelines'] = df.groupby('COUNTRY_ALPHA')['F024'].mean()

    # 16. Woman earns more (D054)
    if 'D054' in df.columns:
        df['item_woman'] = (df['D054'] == 1).astype(float).where(df['D054'].notna())
        results['Woman earns more problems'] = df.groupby('COUNTRY_ALPHA')['item_woman'].mean()

    # 17. Love/respect parents (A025: 1=Always, 2=Depends)
    if 'A025' in df.columns:
        df['item_parents'] = (df['A025'] == 1).astype(float).where(df['A025'].notna())
        results['Love/respect parents'] = df.groupby('COUNTRY_ALPHA')['item_parents'].mean()

    # 18. Family very important (A001: 1=Very...4=Not)
    if 'A001' in df.columns:
        df['item_family'] = (df['A001'] == 1).astype(float).where(df['A001'].notna())
        results['Family very important'] = df.groupby('COUNTRY_ALPHA')['item_family'].mean()

    # 19. Favorable army rule (E114: 1=Very good...4=Very bad)
    if 'E114' in df.columns:
        df['item_army'] = (df['E114'] <= 2).astype(float).where(df['E114'].notna())
        results['Favorable army rule'] = df.groupby('COUNTRY_ALPHA')['item_army'].mean()

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

    correlations = {}
    n_countries = {}

    for col in item_values.columns:
        merged = pd.DataFrame({
            'item': item_values[col],
            'factor': scores_idx['trad_secrat']
        }).dropna()

        if len(merged) >= 10:
            r = merged['item'].corr(-merged['factor'])
            correlations[col] = r
            n_countries[col] = len(merged)

    # Flip justifiability means (lower mean = more traditional)
    for item in ['Euthanasia mean', 'Suicide mean', 'Divorce mean']:
        if item in correlations:
            correlations[item] = -correlations[item]

    print("\n" + "="*80)
    print("TABLE 2: Correlations with Traditional/Secular-Rational Dimension")
    print("="*80)
    print(f"\n{'Item':<65} {'r':>8} {'N':>5}")
    print("-"*80)

    sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for k, r in sorted_items:
        n = n_countries.get(k, 0)
        print(f"{k:<65} {r:>8.2f} {n:>5}")

    return correlations, n_countries


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

    correlations, n_countries = run_analysis()

    gt_to_computed = {
        'Religion very important': 'Religion very important',
        'Believes in Heaven': 'Believes in Heaven',
        'Believes in Hell': 'Believes in Hell',
        'Attends church regularly': 'Attends church regularly',
        'Confidence in churches': 'Confidence in churches',
        'Comfort from religion': 'Comfort from religion',
        'Religious person': 'Religious person',
        'Euthanasia never justifiable': 'Euthanasia mean',
        'Work very important': 'Work very important',
        'Stricter limits foreign goods': 'Stricter limits foreign goods',
        'Suicide never justifiable': 'Suicide mean',
        'Seldom discusses politics': 'Seldom discusses politics',
        'Left-right mean': 'Left-right mean',
        'Divorce never justifiable': 'Divorce mean',
        'Clear guidelines good/evil': 'Clear guidelines',
        'Woman earns more problems': 'Woman earns more problems',
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
            print(f"  {gt_key:<40} Paper={gt_val:.2f}  Ours={comp_val:.2f}  Diff={diff:.2f}  {status}")
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
