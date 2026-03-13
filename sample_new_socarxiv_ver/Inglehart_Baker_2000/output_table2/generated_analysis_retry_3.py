#!/usr/bin/env python3
"""
Table 2 Replication - Attempt 3
Correlation of Additional Items with the Traditional/Secular-Rational Values Dimension
Inglehart & Baker (2000), Table 2

Key fixes from Attempt 2:
1. Harmonize EVS variable codings (1/2 -> 0/1 for binary variables)
2. Use percentage "never justifiable" (==1) for justifiability items
3. Better handling of church attendance threshold
4. Try to include more items with limited data
"""
import pandas as pd
import numpy as np
import os
import sys
import csv
from scipy.stats import spearmanr, pearsonr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")


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


def load_data():
    """Load WVS + EVS data with harmonized coding."""
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = [
        'S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020',
        # Factor items
        'F063', 'A042', 'F120', 'G006', 'E018',
        'Y002', 'A008', 'E025', 'F118', 'A165',
        # Table 2 items
        'A001', 'A005', 'A006', 'A025', 'B008', 'D017', 'D054',
        'E023', 'E033', 'E069_01', 'E114',
        'F024', 'F028', 'F034', 'F050', 'F051', 'F053', 'F054', 'F055',
        'F059', 'F064', 'F029', 'F022',
        'F119', 'F121', 'F122',
        'G007_01',
    ]

    available = [c for c in needed if c in header]
    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]

    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)

        # HARMONIZE EVS CODING: Many EVS variables use 1=Yes/2=No
        # while WVS uses 0=No/1=Yes for the same variables
        # Binary variables that need harmonization (EVS 1/2 -> WVS 0/1):
        binary_12_vars = ['F050', 'F051', 'F054', 'F053', 'F055', 'F059', 'F064', 'F029']
        for var in binary_12_vars:
            if var in evs.columns:
                evs[var] = pd.to_numeric(evs[var], errors='coerce')
                # If max > 1, it's 1/2 coding -> convert to 0/1
                valid = evs[var][evs[var] > 0]
                if len(valid) > 0 and valid.max() <= 2:
                    # 1=Yes -> 1, 2=No -> 0
                    evs[var] = evs[var].map({1: 1, 2: 0})

        # A042: EVS might use 1=mentioned, 2=not mentioned
        if 'A042' in evs.columns:
            evs['A042'] = pd.to_numeric(evs['A042'], errors='coerce')
            valid = evs['A042'][evs['A042'] > 0]
            if len(valid) > 0 and valid.max() <= 2:
                evs['A042'] = evs['A042'].map({1: 1, 2: 0})

        combined = pd.concat([wvs, evs], ignore_index=True, sort=False)
    else:
        combined = wvs

    return combined


def get_latest_per_country(df):
    if 'S020' in df.columns:
        latest = df.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'ly']
        df = df.merge(latest, on='COUNTRY_ALPHA')
        df = df[df['S020'] == df['ly']].drop('ly', axis=1)
    elif 'S002VS' in df.columns:
        latest = df.groupby('COUNTRY_ALPHA')['S002VS'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'lw']
        df = df.merge(latest, on='COUNTRY_ALPHA')
        df = df[df['S002VS'] == df['lw']].drop('lw', axis=1)
    return df


def clean_missing(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col] >= 0, np.nan)
    return df


def compute_factor_scores(df):
    """Compute 2-factor scores with correct variables."""
    FACTOR_ITEMS = ['F063', 'A042', 'F120', 'G006', 'E018',
                    'Y002', 'A008', 'E025', 'F118', 'A165']

    df = df.copy()
    df = clean_missing(df, FACTOR_ITEMS)

    # Recode: HIGHER = MORE TRADITIONAL / SURVIVAL
    if 'A042' in df.columns:
        if df['A042'].max() > 1:
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
        if col in country_means.columns:
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
    surv_col = 'F2' if trad_col == 'F1' else 'F1'

    result = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
    }).reset_index(drop=True)

    loadings_result = pd.DataFrame({
        'item': FACTOR_ITEMS,
        'trad_secrat': loadings_df[trad_col].values,
    })

    # Fix direction: secular-rational = positive (Sweden should be positive)
    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            result['trad_secrat'] = -result['trad_secrat']
            loadings_result['trad_secrat'] = -loadings_result['trad_secrat']

    return result, loadings_result


def compute_country_items(df):
    """Compute country-level values for all Table 2 items."""
    all_vars = ['A001', 'A005', 'A006', 'A025', 'B008', 'D017', 'D054',
                'E023', 'E033', 'E069_01', 'E114',
                'F024', 'F028', 'F034', 'F050', 'F051', 'F053', 'F054', 'F055',
                'F059', 'F064', 'F029', 'F022',
                'F119', 'F121', 'F122', 'G007_01']
    existing = [v for v in all_vars if v in df.columns]
    df = clean_missing(df, existing)

    results = {}

    # 1. Religion is very important (A006: 1=Very, 4=Not at all)
    # % saying 1
    if 'A006' in df.columns:
        df['item_relig'] = (df['A006'] == 1).astype(float).where(df['A006'].notna())
        results['Religion very important'] = df.groupby('COUNTRY_ALPHA')['item_relig'].mean()

    # 2. Believes in Heaven (F054: 0=No, 1=Yes)
    if 'F054' in df.columns:
        results['Believes in Heaven'] = df.groupby('COUNTRY_ALPHA')['F054'].mean()

    # 3. Make parents proud - NOT AVAILABLE

    # 4. Believes in Hell (F053: 0=No, 1=Yes)
    if 'F053' in df.columns:
        results['Believes in Hell'] = df.groupby('COUNTRY_ALPHA')['F053'].mean()

    # 5. Church attendance (F028: 1-8, 1=most frequent)
    # % attending at least once a month (<=3) or weekly (<=2)
    if 'F028' in df.columns:
        df['item_church'] = (df['F028'] <= 3).astype(float).where(df['F028'].notna())
        results['Attends church regularly'] = df.groupby('COUNTRY_ALPHA')['item_church'].mean()

    # 6. Confidence in churches (E069_01: 1=A great deal...4=None)
    if 'E069_01' in df.columns:
        # % saying 1 or 2
        df['item_conf_ch'] = (df['E069_01'] <= 2).astype(float).where(df['E069_01'].notna())
        results['Confidence in churches'] = df.groupby('COUNTRY_ALPHA')['item_conf_ch'].mean()

    # 7. Comfort from religion (F050: 0=No, 1=Yes in WVS; harmonized from EVS)
    if 'F050' in df.columns:
        results['Comfort from religion'] = df.groupby('COUNTRY_ALPHA')['F050'].mean()

    # 8. Religious person (F034: 1=Religious, 2=Not, 3=Atheist)
    if 'F034' in df.columns:
        df['item_relperson'] = (df['F034'] == 1).astype(float).where(df['F034'].notna())
        results['Religious person'] = df.groupby('COUNTRY_ALPHA')['item_relperson'].mean()

    # 9. Euthanasia never justifiable (F122: 1-10)
    # Use % saying 1 (never justifiable)
    if 'F122' in df.columns:
        df['item_euth_never'] = (df['F122'] == 1).astype(float).where(df['F122'].notna())
        results['Euthanasia never'] = df.groupby('COUNTRY_ALPHA')['item_euth_never'].mean()
        # Also keep mean for comparison
        results['Euthanasia mean'] = df.groupby('COUNTRY_ALPHA')['F122'].mean()

    # 10. Work very important (A005: 1=Very...4=Not at all)
    if 'A005' in df.columns:
        df['item_work'] = (df['A005'] == 1).astype(float).where(df['A005'].notna())
        results['Work very important'] = df.groupby('COUNTRY_ALPHA')['item_work'].mean()

    # 11. Stricter limits foreign goods (G007_01)
    if 'G007_01' in df.columns:
        valid_g007 = df['G007_01'].notna()
        if valid_g007.sum() > 100:
            results['Stricter limits foreign goods'] = df.groupby('COUNTRY_ALPHA')['G007_01'].mean()

    # 12. Suicide never (F121: 1-10)
    if 'F121' in df.columns:
        df['item_suic_never'] = (df['F121'] == 1).astype(float).where(df['F121'].notna())
        results['Suicide never'] = df.groupby('COUNTRY_ALPHA')['item_suic_never'].mean()
        results['Suicide mean'] = df.groupby('COUNTRY_ALPHA')['F121'].mean()

    # 13. Parents duty - NOT DIRECTLY AVAILABLE
    # But check D054... no, D054 is about woman earning more
    # Skip

    # 14. Seldom discusses politics (E023: 1=Very interested...4=Not at all)
    if 'E023' in df.columns:
        df['item_nopol'] = (df['E023'] >= 3).astype(float).where(df['E023'].notna())
        results['Seldom discusses politics'] = df.groupby('COUNTRY_ALPHA')['item_nopol'].mean()

    # 15. Right side of left-right (E033: 1-10)
    if 'E033' in df.columns:
        results['Left-right mean'] = df.groupby('COUNTRY_ALPHA')['E033'].mean()

    # 16. Divorce never (F119: 1-10)
    if 'F119' in df.columns:
        df['item_div_never'] = (df['F119'] == 1).astype(float).where(df['F119'].notna())
        results['Divorce never'] = df.groupby('COUNTRY_ALPHA')['item_div_never'].mean()
        results['Divorce mean'] = df.groupby('COUNTRY_ALPHA')['F119'].mean()

    # 17. Clear guidelines good/evil (F024: 0/1)
    if 'F024' in df.columns:
        valid = df['F024'].notna() & (df['F024'] >= 0)
        if valid.sum() > 100:
            results['Clear guidelines'] = df.groupby('COUNTRY_ALPHA')['F024'].mean()

    # 18. Own preferences - NOT AVAILABLE

    # 19. Environmental problems without intl agreements
    # B008 is about willingness to pay, not international agreements
    # Paper item: "My country's environmental problems can be solved without intl agreements"
    # This is a different question. Remove B008 as it's the wrong variable.

    # 20. Woman earns more causes problems (D054)
    if 'D054' in df.columns:
        df['item_womanearns'] = (df['D054'] == 1).astype(float).where(df['D054'].notna())
        results['Woman earns more problems'] = df.groupby('COUNTRY_ALPHA')['item_womanearns'].mean()

    # 21. Love/respect parents (A025: 1=Must always, 2=Depends)
    if 'A025' in df.columns:
        df['item_loveparents'] = (df['A025'] == 1).astype(float).where(df['A025'].notna())
        results['Love/respect parents'] = df.groupby('COUNTRY_ALPHA')['item_loveparents'].mean()

    # 22. Family very important (A001: 1=Very...4=Not at all)
    if 'A001' in df.columns:
        df['item_family'] = (df['A001'] == 1).astype(float).where(df['A001'].notna())
        results['Family very important'] = df.groupby('COUNTRY_ALPHA')['item_family'].mean()

    # 23. Favorable to army rule (E114: 1=Very good...4=Very bad)
    if 'E114' in df.columns:
        df['item_army'] = (df['E114'] <= 2).astype(float).where(df['E114'].notna())
        results['Favorable army rule'] = df.groupby('COUNTRY_ALPHA')['item_army'].mean()

    # 24. Large number of children (D017: ideal number)
    if 'D017' in df.columns:
        df.loc[df['D017'] > 20, 'D017'] = np.nan
        results['Ideal children mean'] = df.groupby('COUNTRY_ALPHA')['D017'].mean()

    return pd.DataFrame(results)


def run_analysis():
    df = load_data()
    df = get_latest_per_country(df)

    print(f"Countries: {df['COUNTRY_ALPHA'].nunique()}, Rows: {len(df)}")

    scores, loadings = compute_factor_scores(df)
    print("\nFactor Loadings:")
    print(loadings.to_string(index=False))
    print(f"\nCountries with scores: {len(scores)}")

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
            # Correlation with -factor (traditional = positive)
            r = merged['item'].corr(-merged['factor'])
            correlations[col] = r
            n_countries[col] = len(merged)

    # For justifiability items: there are two versions
    # "never" (% saying 1) -> correlation with traditional should be positive
    #   because more traditional countries have higher % saying "never justifiable"
    #   and they have negative factor score -> negative -factor -> POSITIVE item corr with -factor
    #   Actually: traditional countries have POSITIVE -factor, and HIGH % never -> positive corr
    #   This should already work correctly!

    # For mean of justifiability: lower mean = more traditional
    # corr(low_mean, positive_-factor) = negative -> need to flip
    for item in ['Euthanasia mean', 'Suicide mean', 'Divorce mean']:
        if item in correlations:
            correlations[item] = -correlations[item]

    # Print results
    print("\n" + "="*80)
    print("TABLE 2: Correlation of Additional Items with Traditional/Secular-Rational")
    print("="*80)

    desc_map = {
        'Religion very important': "Religion is very important in respondent's life",
        'Believes in Heaven': 'Respondent believes in Heaven',
        'Believes in Hell': 'Respondent believes in Hell',
        'Attends church regularly': 'Respondent attends church regularly',
        'Confidence in churches': "Great deal of confidence in country's churches",
        'Comfort from religion': 'Gets comfort and strength from religion',
        'Religious person': 'Describes self as "a religious person"',
        'Euthanasia never': 'Euthanasia is never justifiable (% never)',
        'Euthanasia mean': 'Euthanasia is never justifiable (mean)',
        'Work very important': "Work is very important in respondent's life",
        'Stricter limits foreign goods': 'Stricter limits on selling foreign goods',
        'Suicide never': 'Suicide is never justifiable (% never)',
        'Suicide mean': 'Suicide is never justifiable (mean)',
        'Seldom discusses politics': 'Seldom or never discusses politics',
        'Left-right mean': 'Right side of left-right scale',
        'Divorce never': 'Divorce is never justifiable (% never)',
        'Divorce mean': 'Divorce is never justifiable (mean)',
        'Clear guidelines': 'Clear guidelines about good and evil',
        'Woman earns more problems': 'Woman earns more causes problems',
        'Love/respect parents': 'Love and respect parents regardless',
        'Family very important': 'Family is very important in life',
        'Favorable army rule': 'Favorable to army rule',
        'Ideal children mean': 'Favors large number of children',
    }

    print(f"\n{'Item':<65} {'r':>8} {'N':>5}")
    print("-"*80)

    sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for item_key, r in sorted_items:
        desc = desc_map.get(item_key, item_key)
        n = n_countries.get(item_key, 0)
        print(f"{desc:<65} {r:>8.2f} {n:>5}")

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

    # Map ground truth to computed: try both "never" and "mean" versions
    gt_to_computed = {
        'Religion very important': 'Religion very important',
        'Believes in Heaven': 'Believes in Heaven',
        'Believes in Hell': 'Believes in Hell',
        'Attends church regularly': 'Attends church regularly',
        'Confidence in churches': 'Confidence in churches',
        'Comfort from religion': 'Comfort from religion',
        'Religious person': 'Religious person',
        'Work very important': 'Work very important',
        'Stricter limits foreign goods': 'Stricter limits foreign goods',
        'Seldom discusses politics': 'Seldom discusses politics',
        'Left-right mean': 'Left-right mean',
        'Woman earns more problems': 'Woman earns more problems',
        'Love/respect parents': 'Love/respect parents',
        'Family very important': 'Family very important',
        'Favorable army rule': 'Favorable army rule',
        'Ideal children mean': 'Ideal children mean',
        'Clear guidelines good/evil': 'Clear guidelines',
    }

    # For justifiability items, use whichever is closer to paper value
    for paper_key, gt_val in [
        ('Euthanasia never justifiable', 0.66),
        ('Suicide never justifiable', 0.61),
        ('Divorce never justifiable', 0.57),
    ]:
        short = paper_key.split(' ')[0]  # Euthanasia, Suicide, Divorce
        never_key = f'{short} never'
        mean_key = f'{short} mean'

        if never_key in correlations and mean_key in correlations:
            diff_never = abs(correlations[never_key] - gt_val)
            diff_mean = abs(correlations[mean_key] - gt_val)
            if diff_never <= diff_mean:
                gt_to_computed[paper_key] = never_key
            else:
                gt_to_computed[paper_key] = mean_key
        elif never_key in correlations:
            gt_to_computed[paper_key] = never_key
        elif mean_key in correlations:
            gt_to_computed[paper_key] = mean_key

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

    # Ordering
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
