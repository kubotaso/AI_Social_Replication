#!/usr/bin/env python3
"""
Replication of Table 3 from Inglehart & Baker (2000) - Attempt 2.
Correlations of 31 additional items with the Survival/Self-Expression Values dimension.

Fixes from attempt 1:
- Load ALL needed columns directly (not relying on shared_factor_analysis load which may miss some)
- Fix Outgroup index, Technology emphasis, Recycling, Environmental meeting, Free choice,
  Criminal records, Heavy drinkers, Science items
- Fix Prostitution (F125) - check value coding
- Fix Imagination/Tolerance - constant input issues (check coding)
- Fix Child needs both parents, University for boy, Democracy items
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared_factor_analysis import (
    clean_missing, get_latest_per_country,
    compute_nation_level_factor_scores, FACTOR_ITEMS, varimax
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output_table3")
DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")

# Ground truth from Table 3
GROUND_TRUTH = {
    "Men make better political leaders than women": 0.86,
    "Dissatisfied with financial situation": 0.83,
    "Woman needs children to be fulfilled": 0.83,
    "Outgroup rejection index": 0.81,
    "Favors technology emphasis": 0.78,
    "Has not recycled": 0.76,
    "Has not attended environmental meeting/petition": 0.75,
    "Job motivation: income/safety over accomplishment": 0.74,
    "Favors state ownership": 0.74,
    "Child needs both parents": 0.73,
    "Health not very good": 0.73,
    "Must always love/respect parents": 0.71,
    "Men more right to job when scarce": 0.69,
    "Prostitution never justifiable": 0.69,
    "Government should ensure provision": 0.68,
    "Not much free choice in life": 0.67,
    "University more important for boy": 0.67,
    "Not less emphasis on money": 0.66,
    "Rejects criminal records as neighbors": 0.66,
    "Rejects heavy drinkers as neighbors": 0.64,
    "Hard work important to teach child": 0.65,
    "Imagination not important to teach child": 0.62,
    "Tolerance not important to teach child": 0.62,
    "Science will help humanity": 0.60,
    "Leisure not very important": 0.60,
    "Friends not very important": 0.56,
    "Strong leader good": 0.58,
    "Would not take part in boycott": 0.56,
    "Govt ownership should increase": 0.55,
    "Democracy not necessarily best": 0.45,
    "Opposes economic aid to poorer countries": 0.42,
}


def load_all_data():
    """Load WVS + EVS data with ALL needed columns for Table 3."""
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    # All columns needed for Table 3 plus factor analysis
    needed = [
        'S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'S024',
        # Factor analysis items
        'A006', 'A042', 'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165',
        # Table 3 additional items
        'D059',  # Men better leaders
        'C006',  # Financial satisfaction
        'D018',  # Woman needs children
        'A124_02', 'A124_06', 'A124_07',  # Outgroup: race, immigrants, AIDS
        'E019',  # Technology emphasis
        'B003',  # Recycled
        'B002',  # Environmental meeting
        'C001', 'C011',  # Job motivation items
        'E036',  # State ownership / govt ownership
        'D022',  # Child needs both parents
        'A009',  # Health
        'A025',  # Respect parents
        'D060',  # Men right to job
        'F125',  # Prostitution justifiable
        'E037',  # Government responsibility
        'A173',  # Free choice in life
        'D058',  # University for boy
        'E014',  # Less emphasis on money
        'A124_08',  # Criminal records neighbors
        'A124_09',  # Heavy drinkers neighbors
        'A030',  # Hard work child quality
        'A032',  # Imagination child quality
        'A035',  # Tolerance child quality
        'E015',  # Science helps
        'A003',  # Leisure importance
        'A002',  # Friends importance
        'E114',  # Strong leader
        'E026',  # Boycott
        'E117',  # Democracy best
        'S017', 'S018',  # Weights
    ]

    available = [c for c in needed if c in header]
    print(f"Loading {len(available)} of {len(needed)} needed columns...")
    missing_cols = [c for c in needed if c not in header]
    if missing_cols:
        print(f"  Missing columns: {missing_cols}")

    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    # Keep waves 2 and 3 (1990s surveys)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]
    print(f"  WVS waves 2-3: {len(wvs)} rows")

    # Load EVS data
    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)
        print(f"  EVS: {len(evs)} rows, {len(evs.columns)} columns")
        combined = pd.concat([wvs, evs], ignore_index=True, sort=False)
    else:
        combined = wvs

    print(f"  Combined: {len(combined)} rows")
    return combined


def run_analysis(data_source=None):
    """Compute Table 3: correlations of 31 items with Survival/Self-Expression dimension."""

    # Step 1: Get nation-level factor scores
    scores_df, loadings_df, country_means_fa = compute_nation_level_factor_scores()
    factor_scores = scores_df[['COUNTRY_ALPHA', 'surv_selfexp']].copy()
    # Flip to survival polarity (survival = positive, self-expression = negative)
    factor_scores['survival_score'] = -factor_scores['surv_selfexp']

    print(f"\nFactor scores computed for {len(factor_scores)} countries")
    print("Top 5 survival (most traditional/survival):")
    top = factor_scores.nlargest(5, 'survival_score')
    for _, r in top.iterrows():
        print(f"  {r['COUNTRY_ALPHA']}: {r['survival_score']:.3f}")
    print("Bottom 5 (most self-expression):")
    bot = factor_scores.nsmallest(5, 'survival_score')
    for _, r in bot.iterrows():
        print(f"  {r['COUNTRY_ALPHA']}: {r['survival_score']:.3f}")

    # Step 2: Load full data for computing country-level means
    df = load_all_data()
    df = get_latest_per_country(df)

    # Clean all value columns
    value_cols = [c for c in df.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'S024', 'S017', 'S018']]
    df = clean_missing(df, value_cols)

    print(f"\nAfter filtering latest per country: {len(df)} rows, {df['COUNTRY_ALPHA'].nunique()} countries")

    # Diagnostic: Check data availability for each item
    diag_cols = ['D059','C006','D018','A124_02','A124_06','A124_07','E019','B003','B002',
                 'C001','C011','E036','D022','A009','A025','D060','F125','E037','A173',
                 'D058','E014','A124_08','A124_09','A030','A032','A035','E015','A003',
                 'A002','E114','E026','E117']
    print("\nData availability:")
    for c in diag_cols:
        if c in df.columns:
            valid = df[c].notna().sum()
            countries = df[df[c].notna()]['COUNTRY_ALPHA'].nunique()
            # Check value range
            vals = df[c].dropna()
            print(f"  {c}: {valid} valid obs, {countries} countries, range [{vals.min():.0f}, {vals.max():.0f}], unique={len(vals.unique())}")
        else:
            print(f"  {c}: NOT IN DATA")

    # Compute country-level indicators
    results = {}

    # 1. Men make better political leaders than women (D059)
    if 'D059' in df.columns:
        temp = df[df['D059'].notna()].copy()
        temp['val'] = (temp['D059'] <= 2).astype(float)
        results["Men make better political leaders than women"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 2. Dissatisfied with financial situation (C006)
    if 'C006' in df.columns:
        temp = df[df['C006'].notna()].copy()
        temp['val'] = 11 - temp['C006']
        results["Dissatisfied with financial situation"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 3. Woman needs children to be fulfilled (D018)
    if 'D018' in df.columns:
        temp = df[df['D018'].notna()].copy()
        temp['val'] = (temp['D018'] == 1).astype(float)
        results["Woman needs children to be fulfilled"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 4. Outgroup rejection index
    outgroup_cols = ['A124_02', 'A124_06', 'A124_07']
    avail_out = [c for c in outgroup_cols if c in df.columns]
    if avail_out:
        temp = df.copy()
        for c in avail_out:
            temp[c + '_rej'] = (temp[c] == 1).astype(float).where(temp[c].notna())
        rej_cols = [c + '_rej' for c in avail_out]
        temp['outgroup_idx'] = temp[rej_cols].mean(axis=1)
        results["Outgroup rejection index"] = temp.groupby('COUNTRY_ALPHA')['outgroup_idx'].mean()

    # 5. Favors technology emphasis (E019)
    if 'E019' in df.columns:
        temp = df[df['E019'].notna()].copy()
        # E019: Should there be more emphasis on technology?
        # Typically: 1=More emphasis is good, 2=Too much emphasis already, 3=Don't know
        temp['val'] = (temp['E019'] == 1).astype(float)
        results["Favors technology emphasis"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 6. Has not recycled (B003)
    if 'B003' in df.columns:
        temp = df[df['B003'].notna()].copy()
        # B003: Have recycled? 1=Have done, 2=Might do, 3=Would never
        temp['val'] = (temp['B003'] != 1).astype(float)
        results["Has not recycled"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 7. Has not attended environmental meeting/petition (B002)
    if 'B002' in df.columns:
        temp = df[df['B002'].notna()].copy()
        temp['val'] = (temp['B002'] != 1).astype(float)
        results["Has not attended environmental meeting/petition"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 8. Job motivation: income/safety over accomplishment
    # C001: Qualities of a job - good pay (1=mentioned, 2=not mentioned)
    # C011: Qualities of a job - good chance to use initiative (1=mentioned, 2=not mentioned)
    # Or build index: % mentioning good pay or safe job vs interesting job
    if 'C001' in df.columns:
        temp = df[df['C001'].notna()].copy()
        temp['val'] = (temp['C001'] == 1).astype(float)
        results["Job motivation: income/safety over accomplishment"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 9. Favors state ownership (E036)
    if 'E036' in df.columns:
        temp = df[df['E036'].notna()].copy()
        results["Favors state ownership"] = temp.groupby('COUNTRY_ALPHA')['E036'].mean()

    # 10. Child needs both parents (D022)
    if 'D022' in df.columns:
        temp = df[df['D022'].notna()].copy()
        # D022: 1=Needs both parents, 2=Not necessarily
        # But check if this might have different coding in some waves
        temp['val'] = (temp['D022'] == 1).astype(float)
        results["Child needs both parents"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 11. Health not very good (A009)
    if 'A009' in df.columns:
        temp = df[df['A009'].notna()].copy()
        # A009: 1=Very good, 2=Good, 3=Fair, 4=Poor, 5=Very poor
        temp['val'] = (temp['A009'] != 1).astype(float)
        results["Health not very good"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 12. Must always love/respect parents (A025)
    if 'A025' in df.columns:
        temp = df[df['A025'].notna()].copy()
        temp['val'] = (temp['A025'] == 1).astype(float)
        results["Must always love/respect parents"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 13. Men more right to job when scarce (D060)
    if 'D060' in df.columns:
        temp = df[df['D060'].notna()].copy()
        temp['val'] = (temp['D060'] == 1).astype(float)
        results["Men more right to job when scarce"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 14. Prostitution never justifiable (F125)
    if 'F125' in df.columns:
        temp = df[df['F125'].notna()].copy()
        # F125: 1-10, 1=never, 10=always justifiable
        # % saying never (1) or low values
        # Use mean reversed so higher = less justifiable = survival
        temp['val'] = 11 - temp['F125']
        results["Prostitution never justifiable"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 15. Government should ensure provision (E037)
    if 'E037' in df.columns:
        temp = df[df['E037'].notna()].copy()
        results["Government should ensure provision"] = temp.groupby('COUNTRY_ALPHA')['E037'].mean()

    # 16. Not much free choice in life (A173)
    if 'A173' in df.columns:
        temp = df[df['A173'].notna()].copy()
        temp['val'] = 11 - temp['A173']
        results["Not much free choice in life"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 17. University more important for boy (D058)
    if 'D058' in df.columns:
        temp = df[df['D058'].notna()].copy()
        temp['val'] = (temp['D058'] == 1).astype(float)
        results["University more important for boy"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 18. Not less emphasis on money (E014)
    if 'E014' in df.columns:
        temp = df[df['E014'].notna()].copy()
        # E014: Less emphasis on money. 1=Good, 2=Don't mind, 3=Bad
        # "Not favor less emphasis" = % NOT saying Good
        temp['val'] = (temp['E014'] != 1).astype(float)
        results["Not less emphasis on money"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 19. Rejects criminal records as neighbors (A124_08)
    if 'A124_08' in df.columns:
        temp = df[df['A124_08'].notna()].copy()
        temp['val'] = (temp['A124_08'] == 1).astype(float)
        results["Rejects criminal records as neighbors"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 20. Rejects heavy drinkers as neighbors (A124_09)
    if 'A124_09' in df.columns:
        temp = df[df['A124_09'].notna()].copy()
        temp['val'] = (temp['A124_09'] == 1).astype(float)
        results["Rejects heavy drinkers as neighbors"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 21. Hard work important to teach child (A030)
    if 'A030' in df.columns:
        temp = df[df['A030'].notna()].copy()
        temp['val'] = (temp['A030'] == 1).astype(float)
        results["Hard work important to teach child"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 22. Imagination not important to teach child (A032)
    if 'A032' in df.columns:
        temp = df[df['A032'].notna()].copy()
        # A032: 1=mentioned (important), 2=not mentioned
        # NOT important = 2 (not mentioned)
        temp['val'] = (temp['A032'] == 2).astype(float)
        results["Imagination not important to teach child"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 23. Tolerance not important to teach child (A035)
    if 'A035' in df.columns:
        temp = df[df['A035'].notna()].copy()
        temp['val'] = (temp['A035'] == 2).astype(float)
        results["Tolerance not important to teach child"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 24. Science will help humanity (E015)
    if 'E015' in df.columns:
        temp = df[df['E015'].notna()].copy()
        # E015: Science and technology. Often: 1=Will help, 2=No effect, 3=Will harm
        # % saying will help
        temp['val'] = (temp['E015'] == 1).astype(float)
        results["Science will help humanity"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 25. Leisure not very important (A003)
    if 'A003' in df.columns:
        temp = df[df['A003'].notna()].copy()
        temp['val'] = (temp['A003'] >= 3).astype(float)
        results["Leisure not very important"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 26. Friends not very important (A002)
    if 'A002' in df.columns:
        temp = df[df['A002'].notna()].copy()
        temp['val'] = (temp['A002'] >= 3).astype(float)
        results["Friends not very important"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 27. Strong leader good (E114)
    if 'E114' in df.columns:
        temp = df[df['E114'].notna()].copy()
        temp['val'] = (temp['E114'] <= 2).astype(float)
        results["Strong leader good"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 28. Would not take part in boycott (E026)
    if 'E026' in df.columns:
        temp = df[df['E026'].notna()].copy()
        temp['val'] = (temp['E026'] == 3).astype(float)
        results["Would not take part in boycott"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 29. Govt ownership should increase (E036 high values)
    if 'E036' in df.columns:
        temp = df[df['E036'].notna()].copy()
        temp['val'] = (temp['E036'] >= 6).astype(float)
        results["Govt ownership should increase"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 30. Democracy not necessarily best (E117)
    if 'E117' in df.columns:
        temp = df[df['E117'].notna()].copy()
        # E117: "Democracy may have problems but is the best system of government"
        # 1=Agree strongly, 2=Agree, 3=Disagree, 4=Disagree strongly
        # "Not necessarily best" = disagreeing = 3 or 4
        temp['val'] = (temp['E117'] >= 3).astype(float)
        results["Democracy not necessarily best"] = temp.groupby('COUNTRY_ALPHA')['val'].mean()

    # 31. Opposes economic aid to poorer countries
    # G008 not available. Skip this item.

    # Step 3: Correlate each item with survival_score
    print("\n" + "=" * 80)
    print("TABLE 3: Correlation of Additional Items with Survival/Self-Expression Dimension")
    print("=" * 80)
    print(f"\nCountries with factor scores: {len(factor_scores)}")

    corr_results = []
    for item_name, country_values in results.items():
        cv = country_values.reset_index()
        cv.columns = ['COUNTRY_ALPHA', 'item_val']

        merged = pd.merge(
            factor_scores[['COUNTRY_ALPHA', 'survival_score']],
            cv,
            on='COUNTRY_ALPHA',
            how='inner'
        )
        merged = merged.dropna(subset=['survival_score', 'item_val'])

        # Check for constant input
        if merged['item_val'].std() < 1e-10:
            print(f"  WARNING: {item_name} has constant values across countries!")
            corr_results.append({'item': item_name, 'correlation': np.nan, 'n_countries': len(merged), 'p_value': np.nan})
            continue

        n = len(merged)
        if n >= 5:
            r, p = stats.pearsonr(merged['survival_score'], merged['item_val'])
            corr_results.append({'item': item_name, 'correlation': round(r, 2), 'n_countries': n, 'p_value': p})
        else:
            corr_results.append({'item': item_name, 'correlation': np.nan, 'n_countries': n, 'p_value': np.nan})

    corr_df = pd.DataFrame(corr_results)
    corr_df = corr_df.sort_values('correlation', ascending=False).reset_index(drop=True)

    print("\nSURVIVAL VALUES EMPHASIZE THE FOLLOWING:                              Correlation")
    print("-" * 85)
    for _, row in corr_df.iterrows():
        item = row['item']
        r = row['correlation']
        n = row['n_countries']
        if pd.notna(r):
            print(f"  {item:<65s} {r:+.2f}  (n={n})")
        else:
            print(f"  {item:<65s}   NA  (n={n})")
    print("-" * 85)
    print("(SELF-EXPRESSION VALUES EMPHASIZE THE OPPOSITE)")

    return corr_df


def score_against_ground_truth(corr_df):
    """Score the results against ground truth from Table 3."""
    if corr_df is None:
        return 0

    total_items = len(GROUND_TRUTH)
    items_present = 0
    sign_matches = 0

    print("\n\n=== SCORING ===")
    print(f"{'Item':<55s} {'Paper':>6s} {'Got':>6s} {'Diff':>6s} {'Match'}")
    print("-" * 90)

    for item_name, true_r in GROUND_TRUTH.items():
        match = corr_df[corr_df['item'] == item_name]
        if len(match) > 0:
            items_present += 1
            got_r = match.iloc[0]['correlation']
            if pd.notna(got_r):
                diff = abs(got_r - true_r)
                sign_ok = (got_r > 0) == (true_r > 0)
                if sign_ok:
                    sign_matches += 1
                if diff <= 0.05:
                    status = "CLOSE"
                elif diff <= 0.10:
                    status = "PARTIAL"
                else:
                    status = "MISS"
                print(f"  {item_name:<53s} {true_r:+.2f}  {got_r:+.2f}  {diff:.2f}  {status}")
            else:
                print(f"  {item_name:<53s} {true_r:+.2f}    NA     -  MISSING_DATA")
        else:
            print(f"  {item_name:<53s} {true_r:+.2f}    --     -  NOT_FOUND")

    print("-" * 90)

    # Items present (20 pts)
    presence_score = (items_present / total_items) * 20

    # Correlation values (40 pts)
    value_score = 0
    for item_name, true_r in GROUND_TRUTH.items():
        match = corr_df[corr_df['item'] == item_name]
        if len(match) > 0:
            got_r = match.iloc[0]['correlation']
            if pd.notna(got_r):
                diff = abs(got_r - true_r)
                if diff <= 0.03:
                    value_score += 40 / total_items
                elif diff <= 0.05:
                    value_score += (40 / total_items) * 0.8
                elif diff <= 0.10:
                    value_score += (40 / total_items) * 0.5
                elif diff <= 0.15:
                    value_score += (40 / total_items) * 0.3
                elif diff <= 0.20:
                    value_score += (40 / total_items) * 0.1

    # Sign/dimension (20 pts)
    sign_score = (sign_matches / total_items) * 20

    # Ordering (10 pts)
    paper_order = list(GROUND_TRUTH.keys())
    got_items = corr_df[corr_df['item'].isin(paper_order)].sort_values('correlation', ascending=False)['item'].tolist()
    order_score = 0
    if len(got_items) >= 5:
        paper_ranks = {item: i for i, item in enumerate(paper_order)}
        got_ranks = {item: i for i, item in enumerate(got_items)}
        common = set(paper_ranks.keys()) & set(got_ranks.keys())
        if common:
            pr = [paper_ranks[i] for i in common]
            gr = [got_ranks[i] for i in common]
            rho, _ = stats.spearmanr(pr, gr)
            order_score = max(0, rho) * 10

    # Sample size (10 pts)
    median_n = corr_df['n_countries'].median() if len(corr_df) > 0 else 0
    if 60 <= median_n <= 70:
        n_score = 10
    elif 50 <= median_n <= 80:
        n_score = 7
    elif 40 <= median_n <= 90:
        n_score = 5
    else:
        n_score = 2

    total = presence_score + value_score + sign_score + order_score + n_score
    total = min(100, max(0, total))

    print(f"\nScore Breakdown:")
    print(f"  Items present:     {presence_score:.1f}/20 ({items_present}/{total_items} items)")
    print(f"  Correlation values: {value_score:.1f}/40")
    print(f"  Sign matches:      {sign_score:.1f}/20 ({sign_matches}/{total_items})")
    print(f"  Ordering:          {order_score:.1f}/10")
    print(f"  Sample size:       {n_score:.1f}/10 (median n={median_n:.0f})")
    print(f"  TOTAL:             {total:.1f}/100")

    return total


if __name__ == "__main__":
    corr_df = run_analysis()
    score = score_against_ground_truth(corr_df)
