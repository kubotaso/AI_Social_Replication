#!/usr/bin/env python3
"""
Table 2 Replication: Correlation of Additional Items with the Traditional/Secular-Rational Values Dimension
Inglehart & Baker (2000), Table 2

Method:
1. Compute nation-level factor scores (Trad/Sec-Rat) using shared_factor_analysis.py
2. For each of 24 additional items, compute country-level means/percentages
3. Correlate each item's country-level value with the Trad/Sec-Rat factor score
4. Report Pearson correlations

NOTE: The paper reports correlations where HIGHER values = more TRADITIONAL.
So for items like "Religion is very important" (% saying very important), a positive
correlation means that countries scoring high on traditional values also have high
percentages saying religion is very important.

The factor scores from shared_factor_analysis.py have secular-rational = positive direction.
So we need to FLIP the sign: correlate with NEGATIVE of trad_secrat (i.e., traditional = positive).
"""
import pandas as pd
import numpy as np
import os
import sys
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")

from shared_factor_analysis import (
    compute_nation_level_factor_scores,
    FACTOR_ITEMS,
    clean_missing,
    COUNTRY_NAMES
)


def load_all_needed_vars():
    """Load WVS + EVS data with all variables needed for Table 2."""
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    # All variables needed for Table 2 items
    needed = [
        'S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020',
        # Factor analysis items
        'A006', 'A042', 'F120', 'G006', 'E018',
        'Y002', 'A008', 'E025', 'F118', 'A165',
        # Table 2 additional items
        'A001',  # Family important
        'A003',  # Leisure important (used for checking)
        'A004',  # Religion important in life
        'A005',  # Work important in life
        'A025',  # Love/respect parents
        'A029',  # Good manners (child quality - make parents proud proxy?)
        'B008',  # Environmental problems
        'D017',  # Ideal number of children
        'D054',  # Woman earns more -> problems
        'E023',  # Interest in politics (discuss politics)
        'E033',  # Left-right self-placement
        'E069_01',  # Confidence in churches
        'E114',  # Army rule
        'F024',  # Clear guidelines good/evil
        'F028',  # Believe in heaven
        'F034',  # Religious person
        'F050',  # Comfort from religion
        'F051',  # Believe in hell
        'F063',  # Church attendance
        'F119',  # Divorce justifiable
        'F121',  # Suicide justifiable
        'F122',  # Euthanasia justifiable
        'G007_01',  # Protectionism / foreign goods
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
    """For each country, keep the latest available wave."""
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


def compute_country_item_values(df):
    """
    Compute country-level means/percentages for each Table 2 item.
    Returns a DataFrame with COUNTRY_ALPHA as index and item names as columns.
    """
    # Clean missing values (negative = missing in WVS)
    all_vars = ['A001', 'A003', 'A004', 'A005', 'A006', 'A008', 'A025', 'A029',
                'B008', 'D017', 'D054', 'E023', 'E033', 'E069_01', 'E114',
                'F024', 'F028', 'F034', 'F050', 'F051', 'F063', 'F119', 'F121', 'F122',
                'G007_01']
    existing = [v for v in all_vars if v in df.columns]
    for col in existing:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].where(df[col] >= 0, np.nan)

    results = {}

    # 1. Religion is very important (A004: 1=Very important, 2=Rather important, 3=Not very, 4=Not at all)
    if 'A004' in df.columns:
        df['relig_important'] = (df['A004'] == 1).astype(float).where(df['A004'].notna())
        results['Religion very important'] = df.groupby('COUNTRY_ALPHA')['relig_important'].mean()

    # 2. Believes in Heaven (F028: 1=Yes, 2=No)
    if 'F028' in df.columns:
        df['believe_heaven'] = (df['F028'] == 1).astype(float).where(df['F028'].notna())
        results['Believes in Heaven'] = df.groupby('COUNTRY_ALPHA')['believe_heaven'].mean()

    # 3. Make parents proud - use A029 (good manners, child quality) as proxy
    # A029 is actually "Good manners" in children qualities (0=not mentioned, 1=mentioned)
    # The paper item "One of respondent's main goals is to make parents proud"
    # This doesn't have a direct WVS variable. Skip for now.

    # 4. Believes in Hell (F051: 1=Yes, 2=No)
    if 'F051' in df.columns:
        df['believe_hell'] = (df['F051'] == 1).astype(float).where(df['F051'].notna())
        results['Believes in Hell'] = df.groupby('COUNTRY_ALPHA')['believe_hell'].mean()

    # 5. Attends church regularly (F063: 1=More than once a week ... 7=Never)
    # "Regularly" = at least once a month (values 1-4)
    if 'F063' in df.columns:
        df['church_regular'] = (df['F063'] <= 4).astype(float).where(df['F063'].notna())
        results['Attends church regularly'] = df.groupby('COUNTRY_ALPHA')['church_regular'].mean()

    # 6. Confidence in churches (E069_01: 1=A great deal, 2=Quite a lot, 3=Not very much, 4=None at all)
    if 'E069_01' in df.columns:
        df['conf_church'] = (df['E069_01'] <= 2).astype(float).where(df['E069_01'].notna())
        results['Confidence in churches'] = df.groupby('COUNTRY_ALPHA')['conf_church'].mean()

    # 7. Gets comfort from religion (F050: 1=Yes, 2=No)
    if 'F050' in df.columns:
        df['comfort_relig'] = (df['F050'] == 1).astype(float).where(df['F050'].notna())
        results['Comfort from religion'] = df.groupby('COUNTRY_ALPHA')['comfort_relig'].mean()

    # 8. Religious person (F034: 1=Religious, 2=Not religious, 3=Atheist)
    if 'F034' in df.columns:
        df['relig_person'] = (df['F034'] == 1).astype(float).where(df['F034'].notna())
        results['Religious person'] = df.groupby('COUNTRY_ALPHA')['relig_person'].mean()

    # 9. Euthanasia never justifiable (F122: 1-10, 1=Never, 10=Always)
    if 'F122' in df.columns:
        # Use mean (lower mean = more traditional)
        results['Euthanasia mean'] = df.groupby('COUNTRY_ALPHA')['F122'].mean()

    # 10. Work very important (A005: 1=Very important, 2=Rather, 3=Not very, 4=Not at all)
    if 'A005' in df.columns:
        df['work_important'] = (df['A005'] == 1).astype(float).where(df['A005'].notna())
        results['Work very important'] = df.groupby('COUNTRY_ALPHA')['work_important'].mean()

    # 11. Stricter limits on foreign goods (G007_01: 1-5 scale, only wave 2 data)
    if 'G007_01' in df.columns:
        results['Stricter limits foreign goods'] = df.groupby('COUNTRY_ALPHA')['G007_01'].mean()

    # 12. Suicide never justifiable (F121: 1-10)
    if 'F121' in df.columns:
        results['Suicide mean'] = df.groupby('COUNTRY_ALPHA')['F121'].mean()

    # 13. Parents' duty to children (A025: 1=Parents duty to do best, 2=Parents have own lives)
    # Actually in WVS: A025 = 1 = Must always love and respect parents, 2 = Depends
    # Wait - A025 is "Love/respect parents" not "Parents' duty to children"
    # Need to check if there's another variable for "Parents' duty"
    # Actually, the paper item might be D054 area or different coding of A025
    # Let me use A025 for "Love and respect parents" item separately

    # 14. Seldom discusses politics (E023: 1=Very interested, 2=Somewhat, 3=Not very, 4=Not at all)
    if 'E023' in df.columns:
        # % saying 3 or 4 (seldom/never discusses)
        df['no_politics'] = (df['E023'] >= 3).astype(float).where(df['E023'].notna())
        results['Seldom discusses politics'] = df.groupby('COUNTRY_ALPHA')['no_politics'].mean()

    # 15. Right side of left-right scale (E033: 1-10)
    if 'E033' in df.columns:
        results['Left-right mean'] = df.groupby('COUNTRY_ALPHA')['E033'].mean()

    # 16. Divorce never justifiable (F119: 1-10)
    if 'F119' in df.columns:
        results['Divorce mean'] = df.groupby('COUNTRY_ALPHA')['F119'].mean()

    # 17. Clear guidelines about good and evil (F024: 0 or 1?)
    if 'F024' in df.columns:
        # F024: 1=There are absolutely clear guidelines about good and evil
        # 2=There can never be absolutely clear guidelines
        # Use % saying 1
        fvalid = df['F024'].notna() & (df['F024'] > 0)
        if fvalid.sum() > 0:
            df['clear_guidelines'] = (df['F024'] == 1).astype(float).where(fvalid)
            results['Clear guidelines good/evil'] = df.groupby('COUNTRY_ALPHA')['clear_guidelines'].mean()

    # 18. Woman earns more causes problems (D054: 1=Agree, 2=Disagree, 3=Neither, 4=DK?)
    # Actually WVS D054: 1=Agree, 2=Neither, 3=Disagree
    if 'D054' in df.columns:
        df['woman_earns_prob'] = (df['D054'] == 1).astype(float).where(df['D054'].notna())
        results['Woman earns more problems'] = df.groupby('COUNTRY_ALPHA')['woman_earns_prob'].mean()

    # 19. Love/respect parents regardless (A025: 1=Must always love/respect, 2=Depends)
    if 'A025' in df.columns:
        df['love_parents'] = (df['A025'] == 1).astype(float).where(df['A025'].notna())
        results['Love/respect parents'] = df.groupby('COUNTRY_ALPHA')['love_parents'].mean()

    # 20. Family very important (A001: 1=Very important, 2=Rather, 3=Not very, 4=Not at all)
    if 'A001' in df.columns:
        df['family_important'] = (df['A001'] == 1).astype(float).where(df['A001'].notna())
        results['Family very important'] = df.groupby('COUNTRY_ALPHA')['family_important'].mean()

    # 21. Favorable to army rule (E114: 1=Very good, 2=Fairly good, 3=Fairly bad, 4=Very bad)
    if 'E114' in df.columns:
        df['army_rule'] = (df['E114'] <= 2).astype(float).where(df['E114'].notna())
        results['Favorable army rule'] = df.groupby('COUNTRY_ALPHA')['army_rule'].mean()

    # 22. Large number of children (D017: ideal number of children)
    if 'D017' in df.columns:
        results['Ideal children mean'] = df.groupby('COUNTRY_ALPHA')['D017'].mean()

    # 23. Environmental problems / international agreements (B008: 1-3)
    if 'B008' in df.columns:
        # B008: 1=Would give part of my income...
        # Actually B008 in WVS is about protection of environment
        # Need to check exact coding
        results['Environmental B008 mean'] = df.groupby('COUNTRY_ALPHA')['B008'].mean()

    # 24. Express own preferences vs understanding others
    # This might not have a direct WVS variable available

    return pd.DataFrame(results)


def run_analysis():
    """Main analysis for Table 2."""
    # Step 1: Get factor scores
    scores, loadings, country_means = compute_nation_level_factor_scores(
        waves_wvs=[2, 3], include_evs=True
    )

    print("Factor loadings:")
    print(loadings.to_string(index=False))
    print(f"\nNumber of countries with factor scores: {len(scores)}")

    # Step 2: Load full data with all Table 2 variables
    df = load_all_needed_vars()
    df = get_latest_per_country(df)
    print(f"Countries in data: {df['COUNTRY_ALPHA'].nunique()}")

    # Step 3: Compute country-level item values
    item_values = compute_country_item_values(df)
    print(f"\nCountry-level item values computed for {len(item_values)} countries")
    print(f"Items computed: {list(item_values.columns)}")

    # Step 4: Merge with factor scores and compute correlations
    # The factor scores have secular-rational = positive (from shared_factor_analysis)
    # The paper reports correlations with TRADITIONAL values (positive = traditional)
    # So we need to use NEGATIVE of trad_secrat (which is secular-rational positive)
    # OR equivalently, negate the correlation sign

    scores_indexed = scores.set_index('COUNTRY_ALPHA')

    correlations = {}
    n_countries = {}
    for col in item_values.columns:
        # Merge on country
        merged = pd.DataFrame({
            'item': item_values[col],
            'trad_secrat': scores_indexed['trad_secrat']
        }).dropna()

        if len(merged) >= 10:
            # Correlate with NEGATIVE of trad_secrat since the paper defines
            # traditional = positive but our scores have secular-rational = positive
            r = merged['item'].corr(-merged['trad_secrat'])
            correlations[col] = r
            n_countries[col] = len(merged)

    # For items where lower mean = more traditional (justifiability scales),
    # the correlation with -trad_secrat would be negative (lower mean correlated with
    # traditional values). But the paper reports positive correlations.
    # For justifiability scales (1-10 where 1=never, 10=always):
    # Traditional = never justifiable = low scores
    # So correlation of low justifiability with traditional should be positive
    # Let's flip the sign for these items
    flip_items = ['Euthanasia mean', 'Suicide mean', 'Divorce mean']
    for item in flip_items:
        if item in correlations:
            correlations[item] = -correlations[item]

    # For left-right scale: paper says "right side" correlates with traditional
    # Higher E033 = more right = traditional -> positive correlation with traditional
    # Our correlation already should be positive if right = traditional

    # For environmental B008: need to check direction
    # B008 might need flipping depending on coding

    # Print results
    print("\n" + "="*80)
    print("TABLE 2: Correlation of Additional Items with Traditional/Secular-Rational")
    print("="*80)
    print(f"\n{'Item':<65} {'r':>8} {'N':>5}")
    print("-"*80)

    # Map our item names to paper descriptions
    item_map = {
        'Religion very important': 'Religion is very important in respondent\'s life',
        'Believes in Heaven': 'Respondent believes in Heaven',
        'Believes in Hell': 'Respondent believes in Hell',
        'Attends church regularly': 'Respondent attends church regularly',
        'Confidence in churches': 'Confidence in country\'s churches',
        'Comfort from religion': 'Gets comfort and strength from religion',
        'Religious person': 'Describes self as "a religious person"',
        'Euthanasia mean': 'Euthanasia is never justifiable',
        'Work very important': 'Work is very important in respondent\'s life',
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
        'Environmental B008 mean': 'Environmental problems/intl agreements',
    }

    # Sort by absolute correlation (descending)
    sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    for item_key, r in sorted_items:
        desc = item_map.get(item_key, item_key)
        n = n_countries.get(item_key, 0)
        print(f"{desc:<65} {r:>8.2f} {n:>5}")

    # Items we couldn't compute
    missing_items = [
        'Make parents proud',
        'Parents duty to children at own expense',
        'Own preferences vs understanding others',
    ]
    print(f"\nItems not available in dataset: {', '.join(missing_items)}")

    return correlations, n_countries


def score_against_ground_truth():
    """Score results against paper values."""
    # Ground truth from paper (Table 2)
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

    # Run analysis
    correlations, n_countries = run_analysis()

    # Map ground truth items to our computed items
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

    # Items that are missing
    missing_items = ['Make parents proud', 'Parents duty to children', 'Own preferences vs understanding']

    print("\n" + "="*80)
    print("SCORING")
    print("="*80)

    total_items = len(ground_truth)  # 24
    items_present = 0
    items_close = 0  # within 0.03
    items_moderate = 0  # within 0.10
    items_wrong_sign = 0
    total_abs_diff = 0
    matched_items = 0

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

    # Scoring rubric for correlation tables:
    # All items present: 20 pts
    # Loading/correlation values within 0.03: 40 pts
    # Correct dimension assignment: 20 pts
    # Ordering: 10 pts
    # Sample size and variance explained: 10 pts

    # Items present score (20 pts)
    items_present_score = (items_present / total_items) * 20

    # Correlation values score (40 pts)
    if items_present > 0:
        frac_close = items_close / total_items
        frac_moderate = items_moderate / total_items
        values_score = (frac_close * 40) + (frac_moderate * 20)  # partial credit for moderate
    else:
        values_score = 0

    # Correct dimension (all positive correlations -> correct): 20 pts
    dim_score = 20 if items_wrong_sign == 0 else max(0, 20 - items_wrong_sign * 4)

    # Ordering score (10 pts) - check if ranking matches
    # Compare rank order of our computed items vs paper
    paper_order = list(ground_truth.keys())
    our_order = []
    for gt_key in paper_order:
        comp_key = gt_to_computed.get(gt_key)
        if comp_key and comp_key in correlations:
            our_order.append((gt_key, correlations[comp_key]))
    our_sorted = sorted(our_order, key=lambda x: x[1], reverse=True)
    our_rank_order = [x[0] for x in our_sorted]
    paper_available = [k for k in paper_order if k in [x[0] for x in our_order]]

    # Compute rank correlation
    if len(our_rank_order) >= 5:
        from scipy.stats import spearmanr
        paper_ranks = list(range(len(paper_available)))
        our_ranks = [our_rank_order.index(k) for k in paper_available]
        rho, _ = spearmanr(paper_ranks, our_ranks)
        ordering_score = max(0, rho * 10)
    else:
        ordering_score = 0

    # Sample size score (10 pts)
    # Paper says 65 societies - check if we have close
    n_societies = len(set(n_countries.values()))  # rough
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
