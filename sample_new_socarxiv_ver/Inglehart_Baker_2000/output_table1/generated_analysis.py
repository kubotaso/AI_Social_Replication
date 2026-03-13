#!/usr/bin/env python3
"""
Table 1 Replication: Items Characterizing Two Dimensions of Cross-Cultural Variation
Inglehart & Baker (2000)

Factor analysis with varimax rotation on 10 items.
Both nation-level (N=65 societies) and individual-level (N~165,594) analyses.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path for shared module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import (
    load_combined_data, clean_missing, get_latest_per_country,
    varimax, FACTOR_ITEMS
)

# Ground truth from the paper
GROUND_TRUTH = {
    'nation_level': {
        'trad_secrat': {
            'A006': 0.91,   # God important
            'A042': 0.89,   # Autonomy index (obedience+relig faith vs independence)
            'F120': 0.82,   # Abortion never justifiable
            'G006': 0.82,   # National pride
            'E018': 0.72,   # Respect for authority
        },
        'surv_selfexp': {
            'Y002': 0.86,   # Materialist/Postmaterialist
            'A008': 0.81,   # Not very happy
            'E025': 0.80,   # Would not sign petition
            'F118': 0.78,   # Homosexuality never justifiable
            'A165': 0.56,   # Careful about trusting
        }
    },
    'individual_level': {
        'trad_secrat': {
            'A006': 0.70,
            'A042': 0.61,
            'F120': 0.61,
            'G006': 0.60,
            'E018': 0.51,
        },
        'surv_selfexp': {
            'Y002': 0.59,
            'A008': 0.58,
            'E025': 0.59,
            'F118': 0.54,
            'A165': 0.44,
        }
    },
    'variance_explained': {
        'nation_dim1': 44,  # % of cross-national variation
        'nation_dim2': 26,
        'indiv_dim1': 26,   # % of individual-level variation
        'indiv_dim2': 13,
    },
    'N_nation': 65,
    'N_individual': 165594,
    'N_individual_smallest': 146789,
}


def recode_items_for_factor(df):
    """
    Recode the 10 items so higher values = traditional (dim1) or survival (dim2).

    The paper states that the items load positively on their respective dimensions,
    so we need to recode so that:
    - Traditional items: higher = more traditional
    - Survival items: higher = more survival-oriented
    """
    df = df.copy()
    items = FACTOR_ITEMS
    df = clean_missing(df, items)

    # A006: God important 1-10. Higher = more important = traditional. Keep as-is.

    # A042: Obedience. 1=mentioned (important), 2=not mentioned.
    # Higher obedience = more traditional. Recode: 1->1, 2->0
    if 'A042' in df.columns:
        df['A042'] = df['A042'].map({1: 1, 2: 0, 0: 0}).where(df['A042'].notna())

    # F120: Abortion 1-10 (1=never, 10=always). Reverse: higher = never = traditional
    if 'F120' in df.columns:
        df['F120'] = 11 - df['F120']

    # G006: National pride 1=Very proud...4=Not at all. Reverse: higher = proud = traditional
    if 'G006' in df.columns:
        df['G006'] = 5 - df['G006']

    # E018: Respect authority 1=Good, 2=Don't mind, 3=Bad. Reverse: higher = good = traditional
    if 'E018' in df.columns:
        df['E018'] = 4 - df['E018']

    # Y002: Post-materialist 1=Materialist, 2=Mixed, 3=Postmaterialist.
    # Reverse: higher = materialist = survival
    if 'Y002' in df.columns:
        df['Y002'] = 4 - df['Y002']

    # A008: Happiness 1=Very happy...4=Not at all. Higher = unhappy = survival. Keep as-is.

    # E025: Petition 1=Have done, 2=Might do, 3=Would never. Higher = never = survival. Keep as-is.

    # F118: Homosexuality 1-10 (1=never, 10=always). Reverse: higher = never = survival
    if 'F118' in df.columns:
        df['F118'] = 11 - df['F118']

    # A165: Trust 1=Most can be trusted, 2=Careful. Higher = careful = survival. Keep as-is.

    return df


def run_nation_level_analysis(df):
    """
    Run factor analysis at the nation level.
    1. Compute country means for 10 items
    2. Standardize
    3. Run PCA with 2 factors
    4. Apply varimax rotation
    5. Report loadings
    """
    # Recode items
    df = recode_items_for_factor(df)

    # Compute country means
    country_means = df.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()

    # Drop countries with too many missing items
    country_means = country_means.dropna(thresh=7)

    # Fill remaining NaN with column means
    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    n_countries = len(country_means)

    # Standardize
    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    # Correlation matrix
    corr = scaled.T @ scaled / (n_countries - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr.values)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take first 2 factors
    # Loadings = eigenvectors * sqrt(eigenvalues)
    loadings_raw = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])

    # Varimax rotation
    loadings_rot, R = varimax(loadings_raw)

    # Variance explained by each factor (after rotation)
    var_explained = (loadings_rot ** 2).sum(axis=0)
    total_var = len(FACTOR_ITEMS)  # Total variance = number of variables (standardized)
    pct_var = var_explained / total_var * 100

    # Identify which factor is Traditional and which is Survival
    trad_items = ['A006', 'A042', 'F120', 'G006', 'E018']
    surv_items = ['Y002', 'A008', 'E025', 'F118', 'A165']

    item_idx = {item: i for i, item in enumerate(FACTOR_ITEMS)}

    f1_trad = sum(abs(loadings_rot[item_idx[it], 0]) for it in trad_items)
    f2_trad = sum(abs(loadings_rot[item_idx[it], 1]) for it in trad_items)

    if f1_trad > f2_trad:
        trad_col, surv_col = 0, 1
    else:
        trad_col, surv_col = 1, 0

    # Ensure positive loadings on their respective factors
    # (Paper reports positive loadings for all items on their primary factor)
    if np.mean([loadings_rot[item_idx[it], trad_col] for it in trad_items]) < 0:
        loadings_rot[:, trad_col] = -loadings_rot[:, trad_col]
    if np.mean([loadings_rot[item_idx[it], surv_col] for it in surv_items]) < 0:
        loadings_rot[:, surv_col] = -loadings_rot[:, surv_col]

    loadings_df = pd.DataFrame({
        'item': FACTOR_ITEMS,
        'trad_secrat': loadings_rot[:, trad_col],
        'surv_selfexp': loadings_rot[:, surv_col],
    })

    pct_var_trad = pct_var[trad_col]
    pct_var_surv = pct_var[surv_col]

    return loadings_df, n_countries, pct_var_trad, pct_var_surv, eigenvalues


def run_individual_level_analysis(df):
    """
    Run factor analysis at the individual level.
    Use the correlation matrix approach for efficiency with large N.
    """
    # Recode items
    df = recode_items_for_factor(df)

    # Get complete cases for the 10 items (or use pairwise for larger N)
    data = df[FACTOR_ITEMS].copy()

    # Count valid responses per item
    item_counts = data.count()

    # Total N with at least 1 non-missing item
    total_n = data.dropna(how='all').shape[0]
    smallest_n = item_counts.min()

    # Use pairwise correlation
    corr = data.corr()

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr.values)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take first 2 factors
    loadings_raw = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])

    # Varimax rotation
    loadings_rot, R = varimax(loadings_raw)

    # Variance explained
    var_explained = (loadings_rot ** 2).sum(axis=0)
    total_var = len(FACTOR_ITEMS)
    pct_var = var_explained / total_var * 100

    # Identify factors
    trad_items = ['A006', 'A042', 'F120', 'G006', 'E018']
    surv_items = ['Y002', 'A008', 'E025', 'F118', 'A165']

    item_idx = {item: i for i, item in enumerate(FACTOR_ITEMS)}

    f1_trad = sum(abs(loadings_rot[item_idx[it], 0]) for it in trad_items)
    f2_trad = sum(abs(loadings_rot[item_idx[it], 1]) for it in trad_items)

    if f1_trad > f2_trad:
        trad_col, surv_col = 0, 1
    else:
        trad_col, surv_col = 1, 0

    # Ensure positive loadings
    if np.mean([loadings_rot[item_idx[it], trad_col] for it in trad_items]) < 0:
        loadings_rot[:, trad_col] = -loadings_rot[:, trad_col]
    if np.mean([loadings_rot[item_idx[it], surv_col] for it in surv_items]) < 0:
        loadings_rot[:, surv_col] = -loadings_rot[:, surv_col]

    loadings_df = pd.DataFrame({
        'item': FACTOR_ITEMS,
        'trad_secrat': loadings_rot[:, trad_col],
        'surv_selfexp': loadings_rot[:, surv_col],
    })

    pct_var_trad = pct_var[trad_col]
    pct_var_surv = pct_var[surv_col]

    return loadings_df, total_n, smallest_n, pct_var_trad, pct_var_surv


def run_analysis(data_source=None):
    """Main analysis function."""
    # Load combined WVS + EVS data
    df = load_combined_data(waves_wvs=[2, 3], include_evs=True)

    # For nation-level: keep latest wave per country
    df_nation = get_latest_per_country(df)

    # Run nation-level analysis
    nation_loadings, n_countries, nation_var1, nation_var2, eigenvalues = run_nation_level_analysis(df_nation)

    # Run individual-level analysis
    indiv_loadings, total_n, smallest_n, indiv_var1, indiv_var2 = run_individual_level_analysis(df_nation)

    # Format results
    item_names = {
        'A006': 'God is very important in respondent\'s life',
        'A042': 'Important for child to learn obedience and religious faith\n  rather than independence and determination (Autonomy index)',
        'F120': 'Abortion is never justifiable',
        'G006': 'Respondent has strong sense of national pride',
        'E018': 'Respondent favors more respect for authority',
        'Y002': 'Respondent gives priority to economic and physical security\n  over self-expression and quality-of-life (4-item index)',
        'A008': 'Respondent describes self as not very happy',
        'E025': 'Respondent has not signed and would not sign a petition',
        'F118': 'Homosexuality is never justifiable',
        'A165': 'You have to be very careful about trusting people',
    }

    trad_items_order = ['A006', 'A042', 'F120', 'G006', 'E018']
    surv_items_order = ['Y002', 'A008', 'E025', 'F118', 'A165']

    results = []
    results.append("=" * 90)
    results.append("TABLE 1: Items Characterizing Two Dimensions of Cross-Cultural Variation")
    results.append("=" * 90)
    results.append("")
    results.append(f"Nation-level analysis: N = {n_countries} societies")
    results.append(f"Individual-level analysis: N = {total_n:,} (smallest N for any item: {smallest_n:,})")
    results.append("")

    results.append("DIMENSION 1: Traditional vs. Secular-Rational Values")
    results.append("(Traditional values emphasize the following)")
    results.append("")
    results.append(f"{'Item':<65} {'Nation-Level':>12} {'Individual-Level':>16}")
    results.append("-" * 93)

    for item in trad_items_order:
        nl = nation_loadings[nation_loadings['item'] == item]['trad_secrat'].values[0]
        il = indiv_loadings[indiv_loadings['item'] == item]['trad_secrat'].values[0]
        name = item_names[item]
        if '\n' in name:
            lines = name.split('\n')
            results.append(f"{lines[0]:<65} {nl:>12.2f} {il:>16.2f}")
            for line in lines[1:]:
                results.append(f"  {line}")
        else:
            results.append(f"{name:<65} {nl:>12.2f} {il:>16.2f}")

    results.append("")
    results.append("DIMENSION 2: Survival vs. Self-Expression Values")
    results.append("(Survival values emphasize the following)")
    results.append("")
    results.append(f"{'Item':<65} {'Nation-Level':>12} {'Individual-Level':>16}")
    results.append("-" * 93)

    for item in surv_items_order:
        nl = nation_loadings[nation_loadings['item'] == item]['surv_selfexp'].values[0]
        il = indiv_loadings[indiv_loadings['item'] == item]['surv_selfexp'].values[0]
        name = item_names[item]
        if '\n' in name:
            lines = name.split('\n')
            results.append(f"{lines[0]:<65} {nl:>12.2f} {il:>16.2f}")
            for line in lines[1:]:
                results.append(f"  {line}")
        else:
            results.append(f"{name:<65} {nl:>12.2f} {il:>16.2f}")

    results.append("")
    results.append("Variance Explained:")
    results.append(f"  Dimension 1 (Traditional/Secular-Rational): {nation_var1:.1f}% (nation-level), {indiv_var1:.1f}% (individual-level)")
    results.append(f"  Dimension 2 (Survival/Self-Expression):     {nation_var2:.1f}% (nation-level), {indiv_var2:.1f}% (individual-level)")
    results.append("")
    results.append(f"Eigenvalues: {', '.join(f'{e:.3f}' for e in eigenvalues[:5])}")

    output = "\n".join(results)
    print(output)

    return {
        'nation_loadings': nation_loadings,
        'indiv_loadings': indiv_loadings,
        'n_countries': n_countries,
        'total_n': total_n,
        'smallest_n': smallest_n,
        'nation_var1': nation_var1,
        'nation_var2': nation_var2,
        'indiv_var1': indiv_var1,
        'indiv_var2': indiv_var2,
        'output': output,
    }


def score_against_ground_truth():
    """Compute score against the paper's values."""
    result = run_analysis()

    nation_loadings = result['nation_loadings']
    indiv_loadings = result['indiv_loadings']

    # Scoring rubric for factor analysis:
    # All items present: 20 points
    # Loading values match within 0.03: 40 points
    # Correct dimension assignment: 20 points
    # Ordering: 10 points
    # Sample size and variance explained: 10 points

    score = 0
    details = []

    # 1. All items present (20 points)
    all_items_present = True
    for item in FACTOR_ITEMS:
        if item not in nation_loadings['item'].values:
            all_items_present = False
            details.append(f"MISSING item: {item}")
    if all_items_present:
        score += 20
        details.append("All 10 items present: +20")

    # 2. Loading values match within 0.03 (40 points)
    # 20 loadings total (10 nation + 10 individual), each worth 2 points
    loading_score = 0
    max_loading_score = 40
    points_per_loading = max_loading_score / 20

    for level, loadings_df in [('nation', nation_loadings), ('individual', indiv_loadings)]:
        gt_level = GROUND_TRUTH[f'{level}_level']

        for dim, items in [('trad_secrat', ['A006', 'A042', 'F120', 'G006', 'E018']),
                           ('surv_selfexp', ['Y002', 'A008', 'E025', 'F118', 'A165'])]:
            for item in items:
                gt_val = gt_level[dim][item]
                gen_row = loadings_df[loadings_df['item'] == item]
                if len(gen_row) > 0:
                    gen_val = gen_row[dim].values[0]
                    diff = abs(gt_val - gen_val)
                    if diff <= 0.03:
                        loading_score += points_per_loading
                        details.append(f"  {level} {item} {dim}: paper={gt_val:.2f}, gen={gen_val:.2f}, diff={diff:.3f} -> MATCH")
                    elif diff <= 0.06:
                        loading_score += points_per_loading * 0.5
                        details.append(f"  {level} {item} {dim}: paper={gt_val:.2f}, gen={gen_val:.2f}, diff={diff:.3f} -> PARTIAL")
                    else:
                        details.append(f"  {level} {item} {dim}: paper={gt_val:.2f}, gen={gen_val:.2f}, diff={diff:.3f} -> MISS")

    score += loading_score
    details.append(f"Loading values: +{loading_score:.1f}/{max_loading_score}")

    # 3. Correct dimension assignment (20 points)
    # Check that Traditional items load highest on dim1 and Survival items on dim2
    dim_score = 0

    for level, loadings_df in [('nation', nation_loadings), ('individual', indiv_loadings)]:
        correct = 0
        for item in ['A006', 'A042', 'F120', 'G006', 'E018']:
            row = loadings_df[loadings_df['item'] == item]
            if len(row) > 0:
                if abs(row['trad_secrat'].values[0]) > abs(row['surv_selfexp'].values[0]):
                    correct += 1
        for item in ['Y002', 'A008', 'E025', 'F118', 'A165']:
            row = loadings_df[loadings_df['item'] == item]
            if len(row) > 0:
                if abs(row['surv_selfexp'].values[0]) > abs(row['trad_secrat'].values[0]):
                    correct += 1
        dim_score += (correct / 10) * 10
        details.append(f"  {level} dimension assignment: {correct}/10 correct")

    score += dim_score
    details.append(f"Dimension assignment: +{dim_score:.1f}/20")

    # 4. Ordering (10 points)
    # Within each dimension, items should be ordered by loading magnitude (descending)
    ordering_score = 0

    for level, loadings_df in [('nation', nation_loadings), ('individual', indiv_loadings)]:
        # Check Traditional dimension ordering
        trad_vals = []
        for item in ['A006', 'A042', 'F120', 'G006', 'E018']:
            row = loadings_df[loadings_df['item'] == item]
            if len(row) > 0:
                trad_vals.append(row['trad_secrat'].values[0])

        # Check Survival dimension ordering
        surv_vals = []
        for item in ['Y002', 'A008', 'E025', 'F118', 'A165']:
            row = loadings_df[loadings_df['item'] == item]
            if len(row) > 0:
                surv_vals.append(row['surv_selfexp'].values[0])

        # Check if descending
        trad_correct = all(trad_vals[i] >= trad_vals[i+1] for i in range(len(trad_vals)-1))
        surv_correct = all(surv_vals[i] >= surv_vals[i+1] for i in range(len(surv_vals)-1))

        if trad_correct:
            ordering_score += 2.5
        if surv_correct:
            ordering_score += 2.5
        details.append(f"  {level} ordering: trad={'OK' if trad_correct else 'WRONG'}, surv={'OK' if surv_correct else 'WRONG'}")

    score += ordering_score
    details.append(f"Ordering: +{ordering_score:.1f}/10")

    # 5. Sample size and variance explained (10 points)
    size_score = 0

    # N countries (2 pts)
    n_countries = result['n_countries']
    n_diff = abs(n_countries - GROUND_TRUTH['N_nation']) / GROUND_TRUTH['N_nation']
    if n_diff <= 0.05:
        size_score += 2
        details.append(f"  N countries: {n_countries} (paper: {GROUND_TRUTH['N_nation']}) -> MATCH")
    elif n_diff <= 0.15:
        size_score += 1
        details.append(f"  N countries: {n_countries} (paper: {GROUND_TRUTH['N_nation']}) -> PARTIAL")
    else:
        details.append(f"  N countries: {n_countries} (paper: {GROUND_TRUTH['N_nation']}) -> MISS")

    # Variance explained (8 pts, 2 per value)
    var_checks = [
        ('nation_dim1', result['nation_var1'], GROUND_TRUTH['variance_explained']['nation_dim1']),
        ('nation_dim2', result['nation_var2'], GROUND_TRUTH['variance_explained']['nation_dim2']),
        ('indiv_dim1', result['indiv_var1'], GROUND_TRUTH['variance_explained']['indiv_dim1']),
        ('indiv_dim2', result['indiv_var2'], GROUND_TRUTH['variance_explained']['indiv_dim2']),
    ]

    for name, gen_val, gt_val in var_checks:
        diff = abs(gen_val - gt_val)
        if diff <= 3:
            size_score += 2
            details.append(f"  {name}: gen={gen_val:.1f}%, paper={gt_val}% -> MATCH")
        elif diff <= 6:
            size_score += 1
            details.append(f"  {name}: gen={gen_val:.1f}%, paper={gt_val}% -> PARTIAL")
        else:
            details.append(f"  {name}: gen={gen_val:.1f}%, paper={gt_val}% -> MISS")

    score += size_score
    details.append(f"Sample size and variance explained: +{size_score:.1f}/10")

    total_score = min(100, round(score))
    details.append(f"\nTOTAL SCORE: {total_score}/100")

    print("\n" + "=" * 60)
    print("SCORING AGAINST GROUND TRUTH")
    print("=" * 60)
    for d in details:
        print(d)

    return total_score, details


if __name__ == "__main__":
    score, details = score_against_ground_truth()
