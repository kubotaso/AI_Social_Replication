#!/usr/bin/env python3
"""
Replication of Table 3 from Inglehart & Baker (2000).
Correlations of 31 additional items with the Survival/Self-Expression Values dimension.

Method:
1. Compute nation-level Survival/Self-Expression factor scores using shared_factor_analysis.py
2. For each additional item, compute country-level mean/percentage
3. Correlate each item's country-level value with the Survival/Self-Expression factor score (Pearson r)
4. Table shows items sorted by correlation magnitude, with survival items positively correlated
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared_factor_analysis import (
    load_combined_data, clean_missing, get_latest_per_country,
    compute_nation_level_factor_scores, FACTOR_ITEMS
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output_table3")

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


def run_analysis(data_source=None):
    """
    Compute Table 3: correlations of 31 items with Survival/Self-Expression dimension.
    """
    # Step 1: Get nation-level factor scores
    scores_df, loadings_df, country_means_fa = compute_nation_level_factor_scores()

    # The factor scores use convention: self-expression = positive
    # For Table 3, survival = positive correlation, so we need survival_score = -surv_selfexp
    # Actually the paper reports correlations where survival items are positive.
    # The factor dimension has self-expression as positive.
    # So items associated with survival should correlate negatively with self-expression scores.
    # But the paper shows them as positive.
    # => The paper uses survival polarity: survival = positive on the dimension.
    # => We flip: survival_score = -surv_selfexp

    factor_scores = scores_df[['COUNTRY_ALPHA', 'surv_selfexp']].copy()
    # Flip to survival polarity (survival = positive)
    factor_scores['survival_score'] = -factor_scores['surv_selfexp']

    # Step 2: Load full data for computing country-level means of additional items
    df = load_combined_data(waves_wvs=[2, 3], include_evs=True)
    df = get_latest_per_country(df)

    # List of all additional columns we need
    needed_cols = [
        'COUNTRY_ALPHA', 'S002VS', 'S020',
        'D059', 'C006', 'D018',
        'A124_02', 'A124_06', 'A124_07',  # Outgroup: different race, immigrants, AIDS
        'E019',  # Technology emphasis
        'B003',  # Recycled
        'B002',  # Environmental meeting
        'C001', 'C011',  # Job motivation items
        'E036',  # State ownership
        'D022',  # Child needs both parents
        'A009',  # Health
        'A025',  # Respect parents
        'D060',  # Men right to job
        'F125',  # Prostitution justifiable
        'E037',  # Government responsibility
        'A173',  # Free choice in life
        'D058',  # University for boy
        'E014',  # Less emphasis on money
        'A124_08',  # Criminal records as neighbors
        'A124_09',  # Heavy drinkers as neighbors
        'A030',  # Hard work (child quality)
        'A032',  # Imagination (child quality)
        'A035',  # Tolerance (child quality)
        'E015',  # Science helps
        'A003',  # Leisure importance
        'A002',  # Friends importance
        'E114',  # Strong leader
        'E026',  # Boycott
        'E117',  # Democracy best
    ]

    available_cols = [c for c in needed_cols if c in df.columns]
    df_sub = df[available_cols].copy()

    # Clean missing values (negatives -> NaN)
    value_cols = [c for c in available_cols if c not in ['COUNTRY_ALPHA', 'S002VS', 'S020']]
    df_sub = clean_missing(df_sub, value_cols)

    # Now compute country-level indicators for each item
    results = {}

    # 1. Men make better political leaders than women (D059)
    # D059: 1=Agree strongly, 2=Agree, 3=Disagree, 4=Disagree strongly
    # % agreeing (1 or 2 = agree)
    if 'D059' in df_sub.columns:
        temp = df_sub[df_sub['D059'].notna()].copy()
        temp['val'] = (temp['D059'] <= 2).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Men make better political leaders than women"] = cm

    # 2. Dissatisfied with financial situation (C006)
    # C006: 1-10, 1=dissatisfied, 10=satisfied
    # Higher dissatisfaction = survival. Use mean, reversed (so higher = more dissatisfied)
    if 'C006' in df_sub.columns:
        temp = df_sub[df_sub['C006'].notna()].copy()
        temp['val'] = 11 - temp['C006']  # reverse: higher = dissatisfied
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Dissatisfied with financial situation"] = cm

    # 3. Woman needs children to be fulfilled (D018)
    # D018: 1=Needs children, 2=Not necessary
    # % agreeing (value == 1)
    if 'D018' in df_sub.columns:
        temp = df_sub[df_sub['D018'].notna()].copy()
        temp['val'] = (temp['D018'] == 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Woman needs children to be fulfilled"] = cm

    # 4. Outgroup rejection index (A124_02 + A124_06 + A124_07)
    # These are "would not like as neighbors" items: 1=mentioned, 2=not mentioned
    # % mentioning (rejecting) = average of mentioning across 3 items
    outgroup_cols = ['A124_02', 'A124_06', 'A124_07']
    avail_out = [c for c in outgroup_cols if c in df_sub.columns]
    if avail_out:
        temp = df_sub.copy()
        for c in avail_out:
            temp[c + '_rej'] = (temp[c] == 1).astype(float).where(temp[c].notna())
        rej_cols = [c + '_rej' for c in avail_out]
        temp['outgroup_idx'] = temp[rej_cols].mean(axis=1)
        cm = temp.groupby('COUNTRY_ALPHA')['outgroup_idx'].mean()
        results["Outgroup rejection index"] = cm

    # 5. Favors technology emphasis (E019)
    # E019: Should we emphasize technology? Various codings.
    # Typically 1=More emphasis, 2=Less emphasis, 3=Same
    # % saying more emphasis
    if 'E019' in df_sub.columns:
        temp = df_sub[df_sub['E019'].notna()].copy()
        temp['val'] = (temp['E019'] == 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Favors technology emphasis"] = cm

    # 6. Has not recycled (B003)
    # B003: Recycling action, typically 1=Have done, 2=Might do, 3=Would never
    # % who have NOT done (value != 1)
    if 'B003' in df_sub.columns:
        temp = df_sub[df_sub['B003'].notna()].copy()
        temp['val'] = (temp['B003'] != 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Has not recycled"] = cm

    # 7. Has not attended environmental meeting/petition (B002)
    # B002: Environmental action, 1=Have done, 2=Might do, 3=Would never
    # % who have NOT done
    if 'B002' in df_sub.columns:
        temp = df_sub[df_sub['B002'].notna()].copy()
        temp['val'] = (temp['B002'] != 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Has not attended environmental meeting/petition"] = cm

    # 8. Job motivation: income/safety over accomplishment (C001 vs C011)
    # C001: First choice of job quality. In WVS: 1=Good pay, 2=Not too much pressure,
    #        3=Good job security, 4=Respected job, 5=Good hours
    # C011: Second choice
    # Index: proportion choosing pay or security (materialist job values) over accomplishment
    # Actually C001 might be "Most important" with multiple items.
    # Let's try: % choosing "good pay" or "safe job" as first/second priority
    if 'C001' in df_sub.columns:
        temp = df_sub[df_sub['C001'].notna()].copy()
        # C001: 1=good pay is important. In some codings: 1=mentioned, 2=not mentioned
        # For job motivation index, combine good pay (C001) and job security (C002?)
        # Actually WVS uses C001=good pay mentioned, C008=good job security mentioned,
        # C011=interesting job mentioned, etc.
        # Since we may not have all, use C001 (good pay) as proxy
        temp['val'] = (temp['C001'] == 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Job motivation: income/safety over accomplishment"] = cm

    # 9. Favors state ownership (E036)
    # E036: 1-10 scale (1=Private ownership, 10=Government ownership)
    # Higher = more state ownership = survival
    if 'E036' in df_sub.columns:
        temp = df_sub[df_sub['E036'].notna()].copy()
        cm = temp.groupby('COUNTRY_ALPHA')['E036'].mean()
        results["Favors state ownership"] = cm

    # 10. Child needs both parents (D022)
    # D022: 1=Agree, 2=Disagree (child needs home with both parents)
    # % agreeing
    if 'D022' in df_sub.columns:
        temp = df_sub[df_sub['D022'].notna()].copy()
        temp['val'] = (temp['D022'] == 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Child needs both parents"] = cm

    # 11. Health not very good (A009)
    # A009: 1=Very good, 2=Good, 3=Fair, 4=Poor, 5=Very poor
    # % not describing as "very good" (value != 1)
    if 'A009' in df_sub.columns:
        temp = df_sub[df_sub['A009'].notna()].copy()
        temp['val'] = (temp['A009'] != 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Health not very good"] = cm

    # 12. Must always love/respect parents (A025)
    # A025: 1=Must always love and respect, 2=Don't have to if behave badly
    # % saying must always (value == 1)
    if 'A025' in df_sub.columns:
        temp = df_sub[df_sub['A025'].notna()].copy()
        temp['val'] = (temp['A025'] == 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Must always love/respect parents"] = cm

    # 13. Men more right to job when scarce (D060)
    # D060: 1=Agree, 2=Neither, 3=Disagree
    # % agreeing (value == 1)
    if 'D060' in df_sub.columns:
        temp = df_sub[df_sub['D060'].notna()].copy()
        temp['val'] = (temp['D060'] == 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Men more right to job when scarce"] = cm

    # 14. Prostitution never justifiable (F125)
    # F125: 1-10, 1=never, 10=always
    # % saying 1 (never) or use mean reversed
    if 'F125' in df_sub.columns:
        temp = df_sub[df_sub['F125'].notna()].copy()
        temp['val'] = (temp['F125'] == 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Prostitution never justifiable"] = cm

    # 15. Government should ensure provision (E037)
    # E037: 1-10 (1=People should be more responsible, 10=Government should ensure)
    # Higher = government responsible = survival
    if 'E037' in df_sub.columns:
        temp = df_sub[df_sub['E037'].notna()].copy()
        cm = temp.groupby('COUNTRY_ALPHA')['E037'].mean()
        results["Government should ensure provision"] = cm

    # 16. Not much free choice in life (A173)
    # A173: 1-10, 1=No choice, 10=A great deal of choice
    # Reverse: higher = no choice = survival
    if 'A173' in df_sub.columns:
        temp = df_sub[df_sub['A173'].notna()].copy()
        temp['val'] = 11 - temp['A173']  # reverse
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Not much free choice in life"] = cm

    # 17. University more important for boy (D058)
    # D058: 1=Agree, 2=Disagree
    # % agreeing
    if 'D058' in df_sub.columns:
        temp = df_sub[df_sub['D058'].notna()].copy()
        temp['val'] = (temp['D058'] == 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["University more important for boy"] = cm

    # 18. Not less emphasis on money (E014)
    # E014: Should there be less emphasis on money? 1=Good, 2=Don't mind, 3=Bad
    # % NOT saying good (not wanting less emphasis) = survival
    if 'E014' in df_sub.columns:
        temp = df_sub[df_sub['E014'].notna()].copy()
        temp['val'] = (temp['E014'] != 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Not less emphasis on money"] = cm

    # 19. Rejects criminal records as neighbors (A124_08)
    # 1=mentioned (would not like), 2=not mentioned
    if 'A124_08' in df_sub.columns:
        temp = df_sub[df_sub['A124_08'].notna()].copy()
        temp['val'] = (temp['A124_08'] == 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Rejects criminal records as neighbors"] = cm

    # 20. Rejects heavy drinkers as neighbors (A124_09)
    if 'A124_09' in df_sub.columns:
        temp = df_sub[df_sub['A124_09'].notna()].copy()
        temp['val'] = (temp['A124_09'] == 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Rejects heavy drinkers as neighbors"] = cm

    # 21. Hard work important to teach child (A030)
    # A030: 1=mentioned, 2=not mentioned
    # % mentioning
    if 'A030' in df_sub.columns:
        temp = df_sub[df_sub['A030'].notna()].copy()
        temp['val'] = (temp['A030'] == 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Hard work important to teach child"] = cm

    # 22. Imagination not important to teach child (A032)
    # A032: 1=mentioned, 2=not mentioned
    # % NOT mentioning
    if 'A032' in df_sub.columns:
        temp = df_sub[df_sub['A032'].notna()].copy()
        temp['val'] = (temp['A032'] == 2).astype(float)  # not mentioned = survival
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Imagination not important to teach child"] = cm

    # 23. Tolerance not important to teach child (A035)
    # A035: 1=mentioned, 2=not mentioned
    # % NOT mentioning
    if 'A035' in df_sub.columns:
        temp = df_sub[df_sub['A035'].notna()].copy()
        temp['val'] = (temp['A035'] == 2).astype(float)  # not mentioned = survival
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Tolerance not important to teach child"] = cm

    # 24. Science will help humanity (E015)
    # E015: 1=Will help, 2=Don't know, 3=Will harm
    # % saying will help
    if 'E015' in df_sub.columns:
        temp = df_sub[df_sub['E015'].notna()].copy()
        temp['val'] = (temp['E015'] == 1).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Science will help humanity"] = cm

    # 25. Leisure not very important (A003)
    # A003: 1=Very important, 2=Rather important, 3=Not very important, 4=Not at all important
    # % saying 3 or 4 (not very/not at all)
    if 'A003' in df_sub.columns:
        temp = df_sub[df_sub['A003'].notna()].copy()
        temp['val'] = (temp['A003'] >= 3).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Leisure not very important"] = cm

    # 26. Friends not very important (A002)
    # A002: 1=Very important, 2=Rather important, 3=Not very important, 4=Not at all important
    # % saying 3 or 4
    if 'A002' in df_sub.columns:
        temp = df_sub[df_sub['A002'].notna()].copy()
        temp['val'] = (temp['A002'] >= 3).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Friends not very important"] = cm

    # 27. Strong leader good (E114)
    # E114: 1=Very good, 2=Fairly good, 3=Fairly bad, 4=Very bad
    # % saying very good or fairly good
    if 'E114' in df_sub.columns:
        temp = df_sub[df_sub['E114'].notna()].copy()
        temp['val'] = (temp['E114'] <= 2).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Strong leader good"] = cm

    # 28. Would not take part in boycott (E026)
    # E026: 1=Have done, 2=Might do, 3=Would never
    # % saying would never (value == 3)
    if 'E026' in df_sub.columns:
        temp = df_sub[df_sub['E026'].notna()].copy()
        temp['val'] = (temp['E026'] == 3).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Would not take part in boycott"] = cm

    # 29. Govt ownership should increase (E036)
    # Already used E036 mean above. This is a separate item.
    # The paper lists both "favorable to state ownership" (mean of E036) and
    # "government ownership should increase" (% saying high values on E036)
    # Use % above midpoint (>= 6)
    if 'E036' in df_sub.columns:
        temp = df_sub[df_sub['E036'].notna()].copy()
        temp['val'] = (temp['E036'] >= 6).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Govt ownership should increase"] = cm

    # 30. Democracy not necessarily best (E117)
    # E117: 1=Very good, 2=Fairly good, 3=Bad, 4=Very bad (as form of govt)
    # OR could be: 1=Agree strongly...4=Disagree strongly with "Democracy may have problems but best"
    # % disagreeing (3 or 4) = democracy not best
    if 'E117' in df_sub.columns:
        temp = df_sub[df_sub['E117'].notna()].copy()
        temp['val'] = (temp['E117'] >= 3).astype(float)
        cm = temp.groupby('COUNTRY_ALPHA')['val'].mean()
        results["Democracy not necessarily best"] = cm

    # 31. Opposes economic aid to poorer countries
    # No G008 in dataset. Try using E033 (income equality) or look for an aid variable.
    # The paper says "Respondent opposes sending economic aid to poorer countries."
    # Try E001 or E033 as proxies... Actually let's see if there's something in G variables.
    # We don't have G008, so we'll skip or try a proxy.
    # Let's skip this one for now.

    # Step 3: Correlate each item with survival_score
    print("=" * 80)
    print("TABLE 3: Correlation of Additional Items with Survival/Self-Expression Dimension")
    print("=" * 80)
    print(f"\nNumber of countries with factor scores: {len(factor_scores)}")
    print()

    corr_results = []
    for item_name, country_values in results.items():
        # Merge with factor scores
        merged = pd.merge(
            factor_scores[['COUNTRY_ALPHA', 'survival_score']],
            country_values.reset_index().rename(columns={country_values.name if hasattr(country_values, 'name') and country_values.name else 0: 'item_val', 'index': 'COUNTRY_ALPHA'}),
            on='COUNTRY_ALPHA',
            how='inner'
        )
        # Handle column naming
        if 'item_val' not in merged.columns:
            # Find the value column
            val_col = [c for c in merged.columns if c not in ['COUNTRY_ALPHA', 'survival_score']][0]
            merged = merged.rename(columns={val_col: 'item_val'})

        merged = merged.dropna(subset=['survival_score', 'item_val'])
        n = len(merged)
        if n >= 5:
            r, p = stats.pearsonr(merged['survival_score'], merged['item_val'])
            corr_results.append({
                'item': item_name,
                'correlation': round(r, 2),
                'n_countries': n,
                'p_value': p
            })
        else:
            corr_results.append({
                'item': item_name,
                'correlation': np.nan,
                'n_countries': n,
                'p_value': np.nan
            })

    # Sort by correlation (descending, as in paper)
    corr_df = pd.DataFrame(corr_results)
    corr_df = corr_df.sort_values('correlation', ascending=False).reset_index(drop=True)

    print("SURVIVAL VALUES EMPHASIZE THE FOLLOWING:                              Correlation")
    print("-" * 80)
    for _, row in corr_df.iterrows():
        item = row['item']
        r = row['correlation']
        n = row['n_countries']
        if pd.notna(r):
            print(f"  {item:<65s} {r:+.2f}  (n={n})")
        else:
            print(f"  {item:<65s}   NA  (n={n})")
    print("-" * 80)
    print("(SELF-EXPRESSION VALUES EMPHASIZE THE OPPOSITE)")

    return corr_df


def score_against_ground_truth(corr_df):
    """Score the results against ground truth from Table 3."""
    if corr_df is None:
        return 0

    total_items = len(GROUND_TRUTH)
    items_present = 0
    close_matches = 0  # within 0.05
    partial_matches = 0  # within 0.10
    sign_matches = 0
    order_score = 0

    print("\n\n=== SCORING ===")
    print(f"{'Item':<55s} {'Paper':>6s} {'Got':>6s} {'Diff':>6s} {'Match'}")
    print("-" * 90)

    for item_name, true_r in GROUND_TRUTH.items():
        # Find matching row in results
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
                    close_matches += 1
                    status = "CLOSE"
                elif diff <= 0.10:
                    partial_matches += 1
                    status = "PARTIAL"
                else:
                    status = "MISS"
                print(f"  {item_name:<53s} {true_r:+.2f}  {got_r:+.2f}  {diff:.2f}  {status}")
            else:
                print(f"  {item_name:<53s} {true_r:+.2f}    NA     -  MISSING_DATA")
        else:
            print(f"  {item_name:<53s} {true_r:+.2f}    --     -  NOT_FOUND")

    print("-" * 90)

    # Scoring rubric for correlation table:
    # Loading/correlation values: 40 pts (values match within 0.03)
    # All items present: 20 pts
    # Correct dimension assignment: 20 pts (sign matches)
    # Ordering: 10 pts
    # Sample size: 10 pts

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

    # Ordering (10 pts) - check if rank order matches
    paper_order = list(GROUND_TRUTH.keys())
    got_items = corr_df[corr_df['item'].isin(paper_order)].sort_values('correlation', ascending=False)['item'].tolist()
    # Compute rank correlation
    if len(got_items) >= 5:
        paper_ranks = {item: i for i, item in enumerate(paper_order)}
        got_ranks = {item: i for i, item in enumerate(got_items)}
        common = set(paper_ranks.keys()) & set(got_ranks.keys())
        if common:
            pr = [paper_ranks[i] for i in common]
            gr = [got_ranks[i] for i in common]
            rho, _ = stats.spearmanr(pr, gr)
            order_score = max(0, rho) * 10
    else:
        order_score = 0

    # Sample size (10 pts) - we expect ~65 countries
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
