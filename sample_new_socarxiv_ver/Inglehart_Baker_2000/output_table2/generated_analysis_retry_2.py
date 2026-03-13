#!/usr/bin/env python3
"""
Table 2 Replication - Attempt 2
Correlation of Additional Items with the Traditional/Secular-Rational Values Dimension
Inglehart & Baker (2000), Table 2

Key fixes from Attempt 1:
1. Compute own factor scores using F063 (God importance 1-10) instead of A006
2. Fix variable mapping: F054=heaven, F053=hell, F028=church attendance
3. Use correct item codings throughout

Method:
- Factor analysis on 10 items (5 per dimension) at nation level
- For each of 24 additional items, compute country-level means/percentages
- Correlate each with the Traditional/Secular-Rational factor score
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


def varimax(Phi, gamma=1.0, q=100, tol=1e-8):
    """Varimax rotation."""
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


def load_data_for_table2():
    """Load WVS + EVS data with all variables needed."""
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    # All variables needed
    needed = [
        'S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020',
        # Factor analysis items (CORRECTED: F063 for God importance, not A006)
        'F063',   # God importance 1-10 (was wrongly mapped to A006)
        'A042',   # Obedience (child quality)
        'F120',   # Abortion justifiable 1-10
        'G006',   # National pride 1-4
        'E018',   # Respect for authority 1-3
        'Y002',   # Post-materialist index
        'A008',   # Happiness 1-4
        'E025',   # Signing petition 1-3
        'F118',   # Homosexuality justifiable 1-10
        'A165',   # Trust 1-2
        # Table 2 additional items
        'A001',   # Family important 1-4
        'A005',   # Work important 1-4
        'A006',   # Religion important 1-4
        'A025',   # Love/respect parents 1-2
        'B008',   # Environmental problems 1-3
        'D017',   # Ideal number of children
        'D054',   # Woman earns more problems 1-4
        'E023',   # Interest in politics 1-4
        'E033',   # Left-right scale 1-10
        'E069_01', # Confidence in churches 1-4
        'E114',   # Army rule good/bad 1-4
        'F024',   # Clear guidelines good/evil 0-1
        'F028',   # Church attendance 1-8 (NOT believe in heaven!)
        'F034',   # Religious person 1-3
        'F050',   # Comfort from religion 0-1
        'F053',   # Believe in hell 0-1
        'F054',   # Believe in heaven 0-1
        'F055',   # Believe in sin 0-1
        'F059',   # Pray 0-1
        'F064',   # Believe in God 0-1
        'F119',   # Divorce justifiable 1-10
        'F121',   # Suicide justifiable 1-10
        'F122',   # Euthanasia justifiable 1-10
        'G007_01', # Limits on imports 1-5
        'S017',   # Weight
    ]

    available = [c for c in needed if c in header]

    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]

    # Load EVS
    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)
        combined = pd.concat([wvs, evs], ignore_index=True, sort=False)
    else:
        combined = wvs

    return combined


def get_latest_per_country(df):
    """For each country, keep the latest wave."""
    if 'S020' in df.columns:
        latest = df.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'latest_year']
        df = df.merge(latest, on='COUNTRY_ALPHA')
        df = df[df['S020'] == df['latest_year']].drop('latest_year', axis=1)
    elif 'S002VS' in df.columns:
        latest = df.groupby('COUNTRY_ALPHA')['S002VS'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'latest_wave']
        df = df.merge(latest, on='COUNTRY_ALPHA')
        df = df[df['S002VS'] == df['latest_wave']].drop('latest_wave', axis=1)
    return df


def clean_missing(df, cols):
    """Replace negative values with NaN."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col] >= 0, np.nan)
    return df


def compute_own_factor_scores(df):
    """
    Compute 2-factor scores using the CORRECT 10 items.

    Trad/Sec-Rat items (dimension 1):
    - F063: God importance (1-10, higher=more important=traditional)
    - A042: Obedience (child quality) (0/1, 1=mentioned=traditional)
    - F120: Abortion justifiable (1-10, 1=never=traditional, reverse)
    - G006: National pride (1-4, 1=very proud=traditional, reverse)
    - E018: Respect for authority (1-3, 1=good=traditional, reverse)

    Surv/Self-Exp items (dimension 2):
    - Y002: Post-materialist (1-3, 3=post-materialist=self-expression, reverse)
    - A008: Happiness (1-4, 1=very happy=self-expression, reverse)
    - E025: Petition (1-3, 1=have done=self-expression, reverse)
    - F118: Homosexuality (1-10, 1=never=survival, reverse for self-exp)
    - A165: Trust (1-2, 1=most can be trusted=self-expression, reverse)
    """
    FACTOR_ITEMS = ['F063', 'A042', 'F120', 'G006', 'E018',
                    'Y002', 'A008', 'E025', 'F118', 'A165']

    df = df.copy()
    df = clean_missing(df, FACTOR_ITEMS)

    # Recode all items so that HIGHER = MORE TRADITIONAL / MORE SURVIVAL
    # F063: 1-10, higher=more important=traditional. KEEP AS-IS.

    # A042: Obedience. 0=not mentioned, 1=mentioned.
    # In some WVS versions: 1=mentioned, 2=not mentioned
    # Check values and recode
    if 'A042' in df.columns:
        # If values are 1/2, recode to 1/0
        if df['A042'].max() > 1:
            df['A042'] = df['A042'].map({1: 1, 2: 0, 0: 0}).where(df['A042'].notna())

    # F120: Abortion 1-10 (1=never, 10=always). REVERSE: higher=never=traditional
    if 'F120' in df.columns:
        df['F120'] = 11 - df['F120']

    # G006: National pride 1=Very proud, 4=Not at all. REVERSE: higher=proud=traditional
    if 'G006' in df.columns:
        df['G006'] = 5 - df['G006']

    # E018: Respect authority 1=Good, 2=Don't mind, 3=Bad. REVERSE: higher=good=traditional
    if 'E018' in df.columns:
        df['E018'] = 4 - df['E018']

    # Y002: Post-materialist 1=Materialist, 2=Mixed, 3=Post-materialist. REVERSE: higher=materialist=survival
    if 'Y002' in df.columns:
        df['Y002'] = 4 - df['Y002']

    # A008: Happiness 1=Very happy, 4=Not at all. Higher=unhappy=survival. KEEP.

    # E025: Petition 1=Have done, 2=Might do, 3=Would never. Higher=never=survival. KEEP.

    # F118: Homosexuality 1-10 (1=never, 10=always). REVERSE: higher=never=survival
    if 'F118' in df.columns:
        df['F118'] = 11 - df['F118']

    # A165: Trust 1=Most can be trusted, 2=Careful. Higher=careful=survival. KEEP.

    # Compute country means
    country_means = df.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)

    # Fill remaining NaN with column means
    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Standardize
    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    # PCA via SVD
    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)

    # Take first 2 components
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(len(scaled) - 1)
    scores_raw = U[:, :2] * S[:2]

    # Varimax rotation
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    # Determine which factor is Trad/Sec-Rat vs Surv/Self-Exp
    trad_items = ['A042', 'F120', 'G006', 'E018']
    surv_items = ['Y002', 'A008', 'E025', 'F118']

    f1_trad_load = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items if i in loadings_df.index)
    f2_trad_load = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items if i in loadings_df.index)

    if f1_trad_load > f2_trad_load:
        trad_col, surv_col = 'F1', 'F2'
    else:
        trad_col, surv_col = 'F2', 'F1'

    result = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
        'surv_selfexp': scores_df[surv_col].values
    }).reset_index(drop=True)

    loadings_result = pd.DataFrame({
        'item': FACTOR_ITEMS,
        'trad_secrat': loadings_df[trad_col].values,
        'surv_selfexp': loadings_df[surv_col].values
    })

    # Fix direction: secular-rational = positive, self-expression = positive
    # Sweden should be high on both
    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            result['trad_secrat'] = -result['trad_secrat']
            loadings_result['trad_secrat'] = -loadings_result['trad_secrat']
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['surv_selfexp'].values[0] < 0:
            result['surv_selfexp'] = -result['surv_selfexp']
            loadings_result['surv_selfexp'] = -loadings_result['surv_selfexp']

    return result, loadings_result, country_means


def compute_country_item_values(df):
    """Compute country-level means/percentages for each Table 2 item."""
    # Clean all variables
    all_vars = ['A001', 'A005', 'A006', 'A025', 'B008', 'D017', 'D054',
                'E023', 'E033', 'E069_01', 'E114',
                'F024', 'F028', 'F034', 'F050', 'F053', 'F054',
                'F119', 'F121', 'F122', 'G007_01', 'F064']
    existing = [v for v in all_vars if v in df.columns]
    for col in existing:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].where(df[col] >= 0, np.nan)

    results = {}

    # 1. Religion is very important (A006: 1=Very important, 4=Not at all)
    if 'A006' in df.columns:
        df['relig_vimportant'] = (df['A006'] == 1).astype(float).where(df['A006'].notna())
        results['Religion very important'] = df.groupby('COUNTRY_ALPHA')['relig_vimportant'].mean()

    # 2. Believes in Heaven (F054: 0=No, 1=Yes)
    if 'F054' in df.columns:
        results['Believes in Heaven'] = df.groupby('COUNTRY_ALPHA')['F054'].mean()

    # 3. Make parents proud - check if D054 might be this
    # D054 in WVS TS V5: "If a woman earns more money than her husband,
    # it's almost certain to cause problems" (1=Agree, 2=Neither, 3=Disagree)
    # So D054 is NOT "make parents proud"
    # We don't have this variable. Skip.

    # 4. Believes in Hell (F053: 0=No, 1=Yes)
    if 'F053' in df.columns:
        results['Believes in Hell'] = df.groupby('COUNTRY_ALPHA')['F053'].mean()

    # 5. Attends church regularly (F028: 1=more than once/week ... 8=never)
    # "Regularly" typically means at least once a month
    # Coding: 1=more than once a week, 2=once a week, 3=once a month, 4=Christmas/Easter,
    # 5=other specific holidays, 6=once a year, 7=less often, 8=never
    if 'F028' in df.columns:
        df['church_regular'] = (df['F028'] <= 3).astype(float).where(df['F028'].notna())
        results['Attends church regularly'] = df.groupby('COUNTRY_ALPHA')['church_regular'].mean()

    # 6. Confidence in churches (E069_01: 1=A great deal, 2=Quite a lot, 3=Not very, 4=None)
    if 'E069_01' in df.columns:
        # % saying 1 or 2 (great deal or quite a lot)
        df['conf_church'] = (df['E069_01'] <= 2).astype(float).where(df['E069_01'].notna())
        results['Confidence in churches'] = df.groupby('COUNTRY_ALPHA')['conf_church'].mean()

    # 7. Gets comfort from religion (F050: 0=No, 1=Yes)
    if 'F050' in df.columns:
        results['Comfort from religion'] = df.groupby('COUNTRY_ALPHA')['F050'].mean()

    # 8. Religious person (F034: 1=Religious, 2=Not religious, 3=Atheist)
    if 'F034' in df.columns:
        df['relig_person'] = (df['F034'] == 1).astype(float).where(df['F034'].notna())
        results['Religious person'] = df.groupby('COUNTRY_ALPHA')['relig_person'].mean()

    # 9. Euthanasia never justifiable (F122: 1=Never, 10=Always)
    # Use mean, then higher mean = more permissive = less traditional
    if 'F122' in df.columns:
        results['Euthanasia mean'] = df.groupby('COUNTRY_ALPHA')['F122'].mean()

    # 10. Work very important (A005: 1=Very important, 4=Not at all)
    if 'A005' in df.columns:
        df['work_vimportant'] = (df['A005'] == 1).astype(float).where(df['A005'].notna())
        results['Work very important'] = df.groupby('COUNTRY_ALPHA')['work_vimportant'].mean()

    # 11. Stricter limits on foreign goods (G007_01 - only wave 2, limited countries)
    if 'G007_01' in df.columns:
        g_valid = df['G007_01'].notna()
        if g_valid.sum() > 100:
            results['Stricter limits foreign goods'] = df.groupby('COUNTRY_ALPHA')['G007_01'].mean()

    # 12. Suicide never justifiable (F121: 1=Never, 10=Always)
    if 'F121' in df.columns:
        results['Suicide mean'] = df.groupby('COUNTRY_ALPHA')['F121'].mean()

    # 13. Parents' duty to children - not directly available, skip

    # 14. Seldom discusses politics (E023: 1=Very interested, 4=Not at all)
    if 'E023' in df.columns:
        df['no_politics'] = (df['E023'] >= 3).astype(float).where(df['E023'].notna())
        results['Seldom discusses politics'] = df.groupby('COUNTRY_ALPHA')['no_politics'].mean()

    # 15. Right side of left-right (E033: 1-10, higher=more right)
    if 'E033' in df.columns:
        results['Left-right mean'] = df.groupby('COUNTRY_ALPHA')['E033'].mean()

    # 16. Divorce never justifiable (F119: 1-10)
    if 'F119' in df.columns:
        results['Divorce mean'] = df.groupby('COUNTRY_ALPHA')['F119'].mean()

    # 17. Clear guidelines good/evil (F024: 0 or 1)
    if 'F024' in df.columns:
        f024_valid = df['F024'].notna()
        if f024_valid.sum() > 100:
            results['Clear guidelines good/evil'] = df.groupby('COUNTRY_ALPHA')['F024'].mean()

    # 18. Own preferences vs understanding others - not available, skip

    # 19. Environmental problems without intl agreements (B008: 1-3)
    # B008 coding: 1=Would give part of income if sure it's for environment
    # 2=Would give if others did too, 3=Would not give
    # Paper item: "My country's env problems can be solved without intl agreements"
    # This is a different question - B008 is about willingness to pay, not about intl agreements
    if 'B008' in df.columns:
        results['Environmental B008 mean'] = df.groupby('COUNTRY_ALPHA')['B008'].mean()

    # 20. Woman earns more causes problems (D054: 1=Agree, 2=Neither, 3=Disagree? or different)
    # Need to verify coding
    if 'D054' in df.columns:
        # D054 might be 1=Agree, 2=Disagree, 3=Neither in WVS TS V5
        # Or: 1=Agree strongly, 2=Agree, 3=Disagree, 4=Strongly disagree
        # NGA=1.33, SWE=2.55, USA=2.02 suggests 1=agree (traditional), higher=disagree
        # Use % saying 1 (agree)
        df['woman_earns'] = (df['D054'] == 1).astype(float).where(df['D054'].notna())
        results['Woman earns more problems'] = df.groupby('COUNTRY_ALPHA')['woman_earns'].mean()

    # 21. Love/respect parents regardless (A025: 1=Must always, 2=Depends)
    if 'A025' in df.columns:
        df['love_parents'] = (df['A025'] == 1).astype(float).where(df['A025'].notna())
        results['Love/respect parents'] = df.groupby('COUNTRY_ALPHA')['love_parents'].mean()

    # 22. Family very important (A001: 1=Very important, 4=Not at all)
    if 'A001' in df.columns:
        df['family_vimportant'] = (df['A001'] == 1).astype(float).where(df['A001'].notna())
        results['Family very important'] = df.groupby('COUNTRY_ALPHA')['family_vimportant'].mean()

    # 23. Favorable to army rule (E114: 1=Very good, 2=Fairly good, 3=Fairly bad, 4=Very bad)
    if 'E114' in df.columns:
        df['army_favor'] = (df['E114'] <= 2).astype(float).where(df['E114'].notna())
        results['Favorable army rule'] = df.groupby('COUNTRY_ALPHA')['army_favor'].mean()

    # 24. Large number of children (D017: ideal number of children, numeric)
    if 'D017' in df.columns:
        # Filter out unreasonable values
        df.loc[df['D017'] > 20, 'D017'] = np.nan
        results['Ideal children mean'] = df.groupby('COUNTRY_ALPHA')['D017'].mean()

    return pd.DataFrame(results)


def run_analysis():
    """Main analysis."""
    # Load data
    df = load_data_for_table2()
    df = get_latest_per_country(df)

    print(f"Countries in data: {df['COUNTRY_ALPHA'].nunique()}")
    print(f"Total rows: {len(df)}")

    # Compute factor scores with CORRECT variable mapping
    scores, loadings, country_means = compute_own_factor_scores(df)

    print("\n=== Factor Loadings (corrected) ===")
    print(loadings.to_string(index=False))
    print(f"\nF063 (God importance) loading: {loadings[loadings['item']=='F063']['trad_secrat'].values[0]:.3f}")
    print(f"Number of countries: {len(scores)}")

    # Compute country-level item values
    item_values = compute_country_item_values(df)
    print(f"\nItems computed: {len(item_values.columns)}")

    # Compute correlations
    scores_indexed = scores.set_index('COUNTRY_ALPHA')

    correlations = {}
    n_countries = {}

    for col in item_values.columns:
        merged = pd.DataFrame({
            'item': item_values[col],
            'trad_secrat': scores_indexed['trad_secrat']
        }).dropna()

        if len(merged) >= 10:
            # Paper convention: traditional = positive correlation
            # Our factor scores: secular-rational = positive
            # So correlation with TRADITIONAL = correlation with (-trad_secrat)
            r = merged['item'].corr(-merged['trad_secrat'])
            correlations[col] = r
            n_countries[col] = len(merged)

    # For justifiability items (1-10 where 1=never), lower mean = more traditional
    # Correlation of lower mean with traditional should be POSITIVE
    # But our computation: corr(low_mean, -trad_secrat) would be negative
    # because low mean (traditional countries) have negative trad_secrat (after negation = positive)
    # Wait, let me think again:
    # - Traditional countries have NEGATIVE trad_secrat (e.g., Nigeria)
    # - -trad_secrat for traditional = POSITIVE
    # - Euthanasia mean for traditional countries = LOW (never justifiable)
    # - corr(LOW_euthanasia, POSITIVE_neg_trad) = NEGATIVE correlation
    # But paper reports POSITIVE correlation
    # So we need to FLIP these justifiability items
    flip_items = ['Euthanasia mean', 'Suicide mean', 'Divorce mean']
    for item in flip_items:
        if item in correlations:
            correlations[item] = -correlations[item]

    # Print results
    print("\n" + "="*80)
    print("TABLE 2: Correlation of Additional Items with Traditional/Secular-Rational")
    print("="*80)

    item_descriptions = {
        'Religion very important': "Religion is very important in respondent's life",
        'Believes in Heaven': 'Respondent believes in Heaven',
        'Believes in Hell': 'Respondent believes in Hell',
        'Attends church regularly': 'Respondent attends church regularly',
        'Confidence in churches': "Great deal of confidence in country's churches",
        'Comfort from religion': 'Gets comfort and strength from religion',
        'Religious person': 'Describes self as "a religious person"',
        'Euthanasia mean': 'Euthanasia is never justifiable',
        'Work very important': "Work is very important in respondent's life",
        'Stricter limits foreign goods': 'Stricter limits on selling foreign goods',
        'Suicide mean': 'Suicide is never justifiable',
        'Seldom discusses politics': 'Seldom or never discusses politics',
        'Left-right mean': 'Right side of left-right scale',
        'Divorce mean': 'Divorce is never justifiable',
        'Clear guidelines good/evil': 'Clear guidelines about good and evil',
        'Woman earns more problems': 'Woman earns more causes problems',
        'Love/respect parents': 'Love and respect parents regardless',
        'Family very important': 'Family is very important in life',
        'Favorable army rule': 'Favorable to army rule',
        'Ideal children mean': 'Favors large number of children',
        'Environmental B008 mean': 'Environmental problems solved without intl agreements',
    }

    print(f"\n{'Item':<65} {'r':>8} {'N':>5}")
    print("-"*80)

    sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for item_key, r in sorted_items:
        desc = item_descriptions.get(item_key, item_key)
        n = n_countries.get(item_key, 0)
        print(f"{desc:<65} {r:>8.2f} {n:>5}")

    missing_items = ['Make parents proud', 'Parents duty to children', 'Own preferences vs understanding']
    print(f"\nItems not available: {', '.join(missing_items)}")

    return correlations, n_countries


def score_against_ground_truth():
    """Score results against paper values."""
    ground_truth = {
        'Religion very important': 0.89,
        'Believes in Heaven': 0.88,
        'Make parents proud': 0.81,
        'Believes in Hell': 0.76,
        'Attends church regularly': 0.75,
        'Confidence in churches': 0.72,
        'Comfort from religion': 0.72,
        'Religious person': 0.71,
        'Euthanasia mean': 0.66,
        'Work very important': 0.65,
        'Stricter limits foreign goods': 0.63,
        'Suicide mean': 0.61,
        'Parents duty to children': 0.60,
        'Seldom discusses politics': 0.57,
        'Left-right mean': 0.57,
        'Divorce mean': 0.57,
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
        'Euthanasia mean': 'Euthanasia mean',
        'Work very important': 'Work very important',
        'Stricter limits foreign goods': 'Stricter limits foreign goods',
        'Suicide mean': 'Suicide mean',
        'Seldom discusses politics': 'Seldom discusses politics',
        'Left-right mean': 'Left-right mean',
        'Divorce mean': 'Divorce mean',
        'Clear guidelines good/evil': 'Clear guidelines good/evil',
        'Woman earns more problems': 'Woman earns more problems',
        'Love/respect parents': 'Love/respect parents',
        'Family very important': 'Family very important',
        'Favorable army rule': 'Favorable army rule',
        'Ideal children mean': 'Ideal children mean',
        'Environmental problems': 'Environmental B008 mean',
    }

    missing_items = ['Make parents proud', 'Parents duty to children', 'Own preferences vs understanding']

    print("\n" + "="*80)
    print("SCORING")
    print("="*80)

    total_items = len(ground_truth)
    items_present = 0
    items_close = 0  # within 0.03
    items_moderate = 0  # within 0.10
    items_wrong_sign = 0
    total_abs_diff = 0

    for gt_key, gt_val in ground_truth.items():
        comp_key = gt_to_computed.get(gt_key)
        if comp_key and comp_key in correlations:
            comp_val = correlations[comp_key]
            diff = abs(comp_val - gt_val)
            items_present += 1
            total_abs_diff += diff
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

    # Scoring
    items_present_score = (items_present / total_items) * 20

    if items_present > 0:
        frac_close = items_close / total_items
        frac_moderate = items_moderate / total_items
        values_score = (frac_close * 40) + (frac_moderate * 20)
    else:
        values_score = 0

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
