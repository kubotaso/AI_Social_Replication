#!/usr/bin/env python3
"""
Table 1 Replication: Items Characterizing Two Dimensions of Cross-Cultural Variation
Inglehart & Baker (2000) - Attempt 2

KEY FIXES:
- Use F063 from WVS (not A006) for "God importance" 10-point scale
- Construct Autonomy index from A042+A034-A029-A032
- These are harmonized in shared_factor_analysis.py as GOD_IMP and AUTONOMY
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import (
    load_combined_data, clean_missing, get_latest_per_country,
    varimax, FACTOR_ITEMS
)

GROUND_TRUTH = {
    'nation_level': {
        'trad_secrat': {
            'GOD_IMP': 0.91, 'AUTONOMY': 0.89, 'F120': 0.82, 'G006': 0.82, 'E018': 0.72,
        },
        'surv_selfexp': {
            'Y002': 0.86, 'A008': 0.81, 'E025': 0.80, 'F118': 0.78, 'A165': 0.56,
        }
    },
    'individual_level': {
        'trad_secrat': {
            'GOD_IMP': 0.70, 'AUTONOMY': 0.61, 'F120': 0.61, 'G006': 0.60, 'E018': 0.51,
        },
        'surv_selfexp': {
            'Y002': 0.59, 'A008': 0.58, 'E025': 0.59, 'F118': 0.54, 'A165': 0.44,
        }
    },
    'variance_explained': {
        'nation_dim1': 44, 'nation_dim2': 26, 'indiv_dim1': 26, 'indiv_dim2': 13,
    },
    'N_nation': 65,
    'N_individual': 165594,
}


def recode_items_for_factor(df):
    """Recode items so higher = traditional (dim1) or survival (dim2)."""
    df = df.copy()
    df = clean_missing(df, FACTOR_ITEMS)
    # GOD_IMP: 1-10, higher = important = traditional. Keep.
    # AUTONOMY: already (obedience+faith) - (independence+determination). Keep.
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
    return df


def run_nation_level_analysis(df):
    """PCA + varimax at nation level on country means."""
    df = recode_items_for_factor(df)
    country_means = df.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)
    for col in FACTOR_ITEMS:
        country_means[col] = country_means[col].fillna(country_means[col].mean())

    n_countries = len(country_means)
    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds
    corr = scaled.T @ scaled / (n_countries - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(corr.values)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    loadings_raw = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    loadings_rot, R = varimax(loadings_raw)

    var_explained = (loadings_rot ** 2).sum(axis=0)
    pct_var = var_explained / len(FACTOR_ITEMS) * 100

    trad_items = ['GOD_IMP', 'AUTONOMY', 'F120', 'G006', 'E018']
    surv_items = ['Y002', 'A008', 'E025', 'F118', 'A165']
    item_idx = {item: i for i, item in enumerate(FACTOR_ITEMS)}

    f1_trad = sum(abs(loadings_rot[item_idx[it], 0]) for it in trad_items)
    f2_trad = sum(abs(loadings_rot[item_idx[it], 1]) for it in trad_items)

    trad_col = 0 if f1_trad > f2_trad else 1
    surv_col = 1 - trad_col

    if np.mean([loadings_rot[item_idx[it], trad_col] for it in trad_items]) < 0:
        loadings_rot[:, trad_col] = -loadings_rot[:, trad_col]
    if np.mean([loadings_rot[item_idx[it], surv_col] for it in surv_items]) < 0:
        loadings_rot[:, surv_col] = -loadings_rot[:, surv_col]

    loadings_df = pd.DataFrame({
        'item': FACTOR_ITEMS,
        'trad_secrat': loadings_rot[:, trad_col],
        'surv_selfexp': loadings_rot[:, surv_col],
    })
    return loadings_df, n_countries, pct_var[trad_col], pct_var[surv_col], eigenvalues


def run_individual_level_analysis(df):
    """PCA + varimax at individual level using pairwise correlations."""
    df = recode_items_for_factor(df)
    data = df[FACTOR_ITEMS].copy()
    item_counts = data.count()
    total_n = data.dropna(how='all').shape[0]
    smallest_n = item_counts.min()

    corr = data.corr()
    eigenvalues, eigenvectors = np.linalg.eigh(corr.values)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    loadings_raw = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    loadings_rot, R = varimax(loadings_raw)

    var_explained = (loadings_rot ** 2).sum(axis=0)
    pct_var = var_explained / len(FACTOR_ITEMS) * 100

    trad_items = ['GOD_IMP', 'AUTONOMY', 'F120', 'G006', 'E018']
    surv_items = ['Y002', 'A008', 'E025', 'F118', 'A165']
    item_idx = {item: i for i, item in enumerate(FACTOR_ITEMS)}

    f1_trad = sum(abs(loadings_rot[item_idx[it], 0]) for it in trad_items)
    f2_trad = sum(abs(loadings_rot[item_idx[it], 1]) for it in trad_items)

    trad_col = 0 if f1_trad > f2_trad else 1
    surv_col = 1 - trad_col

    if np.mean([loadings_rot[item_idx[it], trad_col] for it in trad_items]) < 0:
        loadings_rot[:, trad_col] = -loadings_rot[:, trad_col]
    if np.mean([loadings_rot[item_idx[it], surv_col] for it in surv_items]) < 0:
        loadings_rot[:, surv_col] = -loadings_rot[:, surv_col]

    loadings_df = pd.DataFrame({
        'item': FACTOR_ITEMS,
        'trad_secrat': loadings_rot[:, trad_col],
        'surv_selfexp': loadings_rot[:, surv_col],
    })
    return loadings_df, total_n, smallest_n, pct_var[trad_col], pct_var[surv_col]


def run_analysis(data_source=None):
    """Main analysis function."""
    df = load_combined_data(waves_wvs=[2, 3], include_evs=True)
    df_nation = get_latest_per_country(df)

    # Nation-level: use latest wave per country (one data point per society)
    nation_loadings, n_countries, nation_var1, nation_var2, eigenvalues = run_nation_level_analysis(df_nation)
    # Individual-level: use ALL respondents from waves 2+3 (paper N=165,594)
    indiv_loadings, total_n, smallest_n, indiv_var1, indiv_var2 = run_individual_level_analysis(df)

    item_names = {
        'GOD_IMP': "God is very important in respondent's life",
        'AUTONOMY': "Important for child to learn obedience and religious faith\n  rather than independence and determination (Autonomy index)",
        'F120': 'Abortion is never justifiable',
        'G006': 'Respondent has strong sense of national pride',
        'E018': 'Respondent favors more respect for authority',
        'Y002': "Respondent gives priority to economic and physical security\n  over self-expression and quality-of-life (4-item index)",
        'A008': 'Respondent describes self as not very happy',
        'E025': 'Respondent has not signed and would not sign a petition',
        'F118': 'Homosexuality is never justifiable',
        'A165': 'You have to be very careful about trusting people',
    }

    results = []
    results.append("=" * 90)
    results.append("TABLE 1: Items Characterizing Two Dimensions of Cross-Cultural Variation")
    results.append("=" * 90)
    results.append("")
    results.append(f"Nation-level analysis: N = {n_countries} societies")
    results.append(f"Individual-level analysis: N = {total_n:,} (smallest N for any item: {smallest_n:,})")
    results.append("")

    for dim_label, dim_col, items in [
        ("DIMENSION 1: Traditional vs. Secular-Rational Values", 'trad_secrat',
         ['GOD_IMP', 'AUTONOMY', 'F120', 'G006', 'E018']),
        ("DIMENSION 2: Survival vs. Self-Expression Values", 'surv_selfexp',
         ['Y002', 'A008', 'E025', 'F118', 'A165'])
    ]:
        results.append(dim_label)
        results.append("")
        results.append(f"{'Item':<65} {'Nation-Level':>12} {'Individual-Level':>16}")
        results.append("-" * 93)
        for item in items:
            nl = nation_loadings[nation_loadings['item'] == item][dim_col].values[0]
            il = indiv_loadings[indiv_loadings['item'] == item][dim_col].values[0]
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
    results.append(f"\nEigenvalues: {', '.join(f'{e:.3f}' for e in eigenvalues[:5])}")

    output = "\n".join(results)
    print(output)

    return {
        'nation_loadings': nation_loadings, 'indiv_loadings': indiv_loadings,
        'n_countries': n_countries, 'total_n': total_n, 'smallest_n': smallest_n,
        'nation_var1': nation_var1, 'nation_var2': nation_var2,
        'indiv_var1': indiv_var1, 'indiv_var2': indiv_var2, 'output': output,
    }


def score_against_ground_truth():
    """Compute score against the paper's values."""
    result = run_analysis()
    nation_loadings = result['nation_loadings']
    indiv_loadings = result['indiv_loadings']

    score = 0
    details = []

    # 1. All items present (20 points)
    all_present = all(item in nation_loadings['item'].values for item in FACTOR_ITEMS)
    if all_present:
        score += 20
        details.append("All 10 items present: +20")

    # 2. Loading values (40 points)
    loading_score = 0
    pts = 40 / 20
    for level, ldf in [('nation', nation_loadings), ('individual', indiv_loadings)]:
        gt = GROUND_TRUTH[f'{level}_level']
        for dim, items in [('trad_secrat', ['GOD_IMP', 'AUTONOMY', 'F120', 'G006', 'E018']),
                           ('surv_selfexp', ['Y002', 'A008', 'E025', 'F118', 'A165'])]:
            for item in items:
                gt_val = gt[dim][item]
                gen = ldf[ldf['item'] == item][dim].values[0]
                diff = abs(gt_val - gen)
                if diff <= 0.03:
                    loading_score += pts
                    details.append(f"  {level} {item} {dim}: paper={gt_val:.2f}, gen={gen:.2f}, diff={diff:.3f} -> MATCH")
                elif diff <= 0.06:
                    loading_score += pts * 0.5
                    details.append(f"  {level} {item} {dim}: paper={gt_val:.2f}, gen={gen:.2f}, diff={diff:.3f} -> PARTIAL")
                else:
                    details.append(f"  {level} {item} {dim}: paper={gt_val:.2f}, gen={gen:.2f}, diff={diff:.3f} -> MISS")
    score += loading_score
    details.append(f"Loading values: +{loading_score:.1f}/40")

    # 3. Dimension assignment (20 points)
    dim_score = 0
    for level, ldf in [('nation', nation_loadings), ('individual', indiv_loadings)]:
        correct = 0
        for item in ['GOD_IMP', 'AUTONOMY', 'F120', 'G006', 'E018']:
            r = ldf[ldf['item'] == item]
            if abs(r['trad_secrat'].values[0]) > abs(r['surv_selfexp'].values[0]):
                correct += 1
        for item in ['Y002', 'A008', 'E025', 'F118', 'A165']:
            r = ldf[ldf['item'] == item]
            if abs(r['surv_selfexp'].values[0]) > abs(r['trad_secrat'].values[0]):
                correct += 1
        dim_score += (correct / 10) * 10
        details.append(f"  {level} dimension assignment: {correct}/10 correct")
    score += dim_score
    details.append(f"Dimension assignment: +{dim_score:.1f}/20")

    # 4. Ordering (10 points)
    ordering_score = 0
    for level, ldf in [('nation', nation_loadings), ('individual', indiv_loadings)]:
        tv = [ldf[ldf['item'] == i]['trad_secrat'].values[0]
              for i in ['GOD_IMP', 'AUTONOMY', 'F120', 'G006', 'E018']]
        sv = [ldf[ldf['item'] == i]['surv_selfexp'].values[0]
              for i in ['Y002', 'A008', 'E025', 'F118', 'A165']]
        if all(tv[i] >= tv[i+1] for i in range(4)):
            ordering_score += 2.5
        if all(sv[i] >= sv[i+1] for i in range(4)):
            ordering_score += 2.5
        details.append(f"  {level}: trad={'OK' if all(tv[i]>=tv[i+1] for i in range(4)) else 'WRONG'}, "
                       f"surv={'OK' if all(sv[i]>=sv[i+1] for i in range(4)) else 'WRONG'}")
    score += ordering_score
    details.append(f"Ordering: +{ordering_score:.1f}/10")

    # 5. Sample size and variance explained (10 points)
    size_score = 0
    nc = result['n_countries']
    nd = abs(nc - GROUND_TRUTH['N_nation']) / GROUND_TRUTH['N_nation']
    if nd <= 0.05:
        size_score += 2
        details.append(f"  N countries: {nc} (paper: 65) -> MATCH")
    elif nd <= 0.15:
        size_score += 1
        details.append(f"  N countries: {nc} (paper: 65) -> PARTIAL")
    else:
        details.append(f"  N countries: {nc} (paper: 65) -> MISS")

    for name, gen_val, gt_val in [
        ('nation_dim1', result['nation_var1'], 44),
        ('nation_dim2', result['nation_var2'], 26),
        ('indiv_dim1', result['indiv_var1'], 26),
        ('indiv_dim2', result['indiv_var2'], 13),
    ]:
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
