#!/usr/bin/env python3
"""
Table 1 Replication: Items Characterizing Two Dimensions of Cross-Cultural Variation
Inglehart & Baker (2000) - Attempt 3

Key fixes from attempt 2:
1. Fix god_important variable:
   - For WVS data: F063 is the 1-10 God importance scale
   - For EVS data: A006 is the 1-10 God importance scale (F063 in EVS is something different)
   - Detect by checking if A006 max > 4 per country (EVS=1-10, WVS=1-4)
2. Use proper autonomy index
3. Tag EVS vs WVS source during loading
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

# The 10 items for the factor analysis
ITEMS_FOR_FACTOR = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
                    'Y002', 'A008', 'E025', 'F118', 'A165']

GROUND_TRUTH = {
    'nation_level': {
        'trad_secrat': {
            'god_important': 0.91, 'autonomy_idx': 0.89, 'F120': 0.82, 'G006': 0.82, 'E018': 0.72,
        },
        'surv_selfexp': {
            'Y002': 0.86, 'A008': 0.81, 'E025': 0.80, 'F118': 0.78, 'A165': 0.56,
        }
    },
    'individual_level': {
        'trad_secrat': {
            'god_important': 0.70, 'autonomy_idx': 0.61, 'F120': 0.61, 'G006': 0.60, 'E018': 0.51,
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
    'N_individual_smallest': 146789,
}


def prepare_data():
    """Load and prepare the combined WVS + EVS dataset."""
    import csv

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
    EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")

    # Load WVS
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020',
              'A006', 'A008', 'A029', 'A032', 'A034', 'A042',
              'A165', 'E018', 'E025', 'F063', 'F118', 'F120',
              'G006', 'Y002']
    available = [c for c in needed if c in header]
    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]
    wvs['_source'] = 'wvs'

    # Load EVS
    evs = pd.read_csv(EVS_PATH)
    evs['_source'] = 'evs'

    # Combine
    df = pd.concat([wvs, evs], ignore_index=True, sort=False)

    # Clean missing values (negative = missing)
    all_vars = ['A006', 'F063', 'A029', 'A032', 'A034', 'A042',
                'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
    df = clean_missing(df, all_vars)

    # Construct god_important:
    # WVS: F063 is God importance 1-10
    # EVS: A006 is God importance 1-10
    df['god_important'] = np.nan

    # For WVS rows: use F063
    wvs_mask = df['_source'] == 'wvs'
    df.loc[wvs_mask, 'god_important'] = df.loc[wvs_mask, 'F063']

    # For EVS rows: use A006
    evs_mask = df['_source'] == 'evs'
    df.loc[evs_mask, 'god_important'] = df.loc[evs_mask, 'A006']

    # Construct autonomy index: (obedience + religious_faith) - (independence + determination)
    # Normalize child quality variables to 0/1 where 1=mentioned
    for col in ['A029', 'A032', 'A034', 'A042']:
        if col in df.columns:
            # WVS: 0=not mentioned, 1=mentioned
            # EVS: might be 1=mentioned, 2=not mentioned
            # Convert 2 -> 0
            df.loc[df[col] == 2, col] = 0

    has_all_child = df[['A029', 'A032', 'A034', 'A042']].notna().all(axis=1)
    df['autonomy_idx'] = np.nan
    df.loc[has_all_child, 'autonomy_idx'] = (
        df.loc[has_all_child, 'A042'] + df.loc[has_all_child, 'A034']
        - df.loc[has_all_child, 'A029'] - df.loc[has_all_child, 'A032']
    )

    # Recode other items so higher = traditional/survival
    df['F120'] = 11 - df['F120']   # Abortion: higher = never justifiable
    df['G006'] = 5 - df['G006']    # Pride: higher = very proud
    df['E018'] = 4 - df['E018']    # Respect: higher = good thing
    df['Y002'] = 4 - df['Y002']    # PostMat: higher = materialist
    # A008: higher = unhappy (survival) - keep as-is
    # E025: higher = would never sign (survival) - keep as-is
    df['F118'] = 11 - df['F118']   # Homo: higher = never justifiable
    # A165: higher = be careful (survival) - keep as-is

    return df


def run_factor_analysis(data_matrix, items):
    """Run PCA + varimax rotation on a data matrix."""
    # Standardize
    means = data_matrix.mean()
    stds = data_matrix.std()
    scaled = (data_matrix - means) / stds

    # Use correlation matrix
    if isinstance(scaled, pd.DataFrame):
        corr = scaled.corr().values
    else:
        corr = np.corrcoef(scaled.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take first 2 factors
    loadings_raw = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])

    # Varimax rotation
    loadings_rot, R = varimax(loadings_raw)

    # Variance explained
    var_explained = (loadings_rot ** 2).sum(axis=0)
    total_var = len(items)
    pct_var = var_explained / total_var * 100

    # Identify which factor is Traditional vs Survival
    trad_items = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']
    surv_items = ['Y002', 'A008', 'E025', 'F118', 'A165']
    item_idx = {item: i for i, item in enumerate(items)}

    f1_trad = sum(abs(loadings_rot[item_idx[it], 0]) for it in trad_items if it in item_idx)
    f2_trad = sum(abs(loadings_rot[item_idx[it], 1]) for it in trad_items if it in item_idx)

    if f1_trad > f2_trad:
        trad_col, surv_col = 0, 1
    else:
        trad_col, surv_col = 1, 0

    # Ensure positive loadings
    trad_mean = np.mean([loadings_rot[item_idx[it], trad_col] for it in trad_items if it in item_idx])
    surv_mean = np.mean([loadings_rot[item_idx[it], surv_col] for it in surv_items if it in item_idx])

    if trad_mean < 0:
        loadings_rot[:, trad_col] = -loadings_rot[:, trad_col]
    if surv_mean < 0:
        loadings_rot[:, surv_col] = -loadings_rot[:, surv_col]

    loadings_df = pd.DataFrame({
        'item': items,
        'trad_secrat': loadings_rot[:, trad_col],
        'surv_selfexp': loadings_rot[:, surv_col],
    })

    return loadings_df, pct_var[trad_col], pct_var[surv_col], eigenvalues


def run_analysis(data_source=None):
    """Main analysis function."""
    df = prepare_data()

    # For nation-level: keep latest wave per country
    df_nation = get_latest_per_country(df)

    # NATION-LEVEL ANALYSIS
    country_means = df_nation.groupby('COUNTRY_ALPHA')[ITEMS_FOR_FACTOR].mean()
    country_means = country_means.dropna(thresh=7)
    for col in ITEMS_FOR_FACTOR:
        country_means[col] = country_means[col].fillna(country_means[col].mean())
    n_countries = len(country_means)

    nation_loadings, nation_var1, nation_var2, nation_eigenvalues = \
        run_factor_analysis(country_means, ITEMS_FOR_FACTOR)

    # INDIVIDUAL-LEVEL ANALYSIS - use ALL respondents
    data_indiv = df[ITEMS_FOR_FACTOR].copy()
    item_counts = data_indiv.count()
    total_n = data_indiv.dropna(how='all').shape[0]
    smallest_n = item_counts.min()

    # Use pairwise correlation for individual-level
    corr = data_indiv.corr()
    eigenvalues_i, eigenvectors_i = np.linalg.eigh(corr.values)
    idx = np.argsort(eigenvalues_i)[::-1]
    eigenvalues_i = eigenvalues_i[idx]
    eigenvectors_i = eigenvectors_i[:, idx]

    loadings_raw_i = eigenvectors_i[:, :2] * np.sqrt(eigenvalues_i[:2])
    loadings_rot_i, R_i = varimax(loadings_raw_i)

    var_explained_i = (loadings_rot_i ** 2).sum(axis=0)
    pct_var_i = var_explained_i / len(ITEMS_FOR_FACTOR) * 100

    trad_items = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']
    surv_items = ['Y002', 'A008', 'E025', 'F118', 'A165']
    item_idx = {item: i for i, item in enumerate(ITEMS_FOR_FACTOR)}

    f1_trad_i = sum(abs(loadings_rot_i[item_idx[it], 0]) for it in trad_items)
    f2_trad_i = sum(abs(loadings_rot_i[item_idx[it], 1]) for it in trad_items)

    if f1_trad_i > f2_trad_i:
        trad_col_i, surv_col_i = 0, 1
    else:
        trad_col_i, surv_col_i = 1, 0

    if np.mean([loadings_rot_i[item_idx[it], trad_col_i] for it in trad_items]) < 0:
        loadings_rot_i[:, trad_col_i] = -loadings_rot_i[:, trad_col_i]
    if np.mean([loadings_rot_i[item_idx[it], surv_col_i] for it in surv_items]) < 0:
        loadings_rot_i[:, surv_col_i] = -loadings_rot_i[:, surv_col_i]

    indiv_loadings = pd.DataFrame({
        'item': ITEMS_FOR_FACTOR,
        'trad_secrat': loadings_rot_i[:, trad_col_i],
        'surv_selfexp': loadings_rot_i[:, surv_col_i],
    })
    indiv_var1 = pct_var_i[trad_col_i]
    indiv_var2 = pct_var_i[surv_col_i]

    # Print verification
    print("Country means verification:")
    for c in ['NGA', 'SWE', 'JPN', 'USA', 'IRL', 'MLT']:
        if c in country_means.index:
            g = country_means.loc[c, 'god_important']
            a = country_means.loc[c, 'autonomy_idx']
            print(f"  {c}: god={g:.2f}, autonomy={a:.2f}")
    print()

    # Format results
    item_names = {
        'god_important': 'God is very important in respondent\'s life',
        'autonomy_idx': 'Important for child to learn obedience and religious faith\n  rather than independence and determination (Autonomy index)',
        'F120': 'Abortion is never justifiable',
        'G006': 'Respondent has strong sense of national pride',
        'E018': 'Respondent favors more respect for authority',
        'Y002': 'Respondent gives priority to economic and physical security\n  over self-expression and quality-of-life (4-item index)',
        'A008': 'Respondent describes self as not very happy',
        'E025': 'Respondent has not signed and would not sign a petition',
        'F118': 'Homosexuality is never justifiable',
        'A165': 'You have to be very careful about trusting people',
    }

    trad_order = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']
    surv_order = ['Y002', 'A008', 'E025', 'F118', 'A165']

    results = []
    results.append("=" * 90)
    results.append("TABLE 1: Items Characterizing Two Dimensions of Cross-Cultural Variation")
    results.append("=" * 90)
    results.append("")
    results.append(f"Nation-level analysis: N = {n_countries} societies")
    results.append(f"Individual-level analysis: N = {total_n:,} (smallest N for any item: {smallest_n:,})")
    results.append("")

    results.append("DIMENSION 1: Traditional vs. Secular-Rational Values")
    results.append(f"{'Item':<65} {'Nation':>8} {'Individual':>12}")
    results.append("-" * 85)
    for item in trad_order:
        nl = nation_loadings[nation_loadings['item'] == item]['trad_secrat'].values[0]
        il = indiv_loadings[indiv_loadings['item'] == item]['trad_secrat'].values[0]
        name = item_names[item]
        lines = name.split('\n')
        results.append(f"{lines[0]:<65} {nl:>8.2f} {il:>12.2f}")
        for line in lines[1:]:
            results.append(f"  {line}")

    results.append("")
    results.append("DIMENSION 2: Survival vs. Self-Expression Values")
    results.append(f"{'Item':<65} {'Nation':>8} {'Individual':>12}")
    results.append("-" * 85)
    for item in surv_order:
        nl = nation_loadings[nation_loadings['item'] == item]['surv_selfexp'].values[0]
        il = indiv_loadings[indiv_loadings['item'] == item]['surv_selfexp'].values[0]
        name = item_names[item]
        lines = name.split('\n')
        results.append(f"{lines[0]:<65} {nl:>8.2f} {il:>12.2f}")
        for line in lines[1:]:
            results.append(f"  {line}")

    results.append("")
    results.append(f"Variance Explained:")
    results.append(f"  Dim 1: {nation_var1:.1f}% (nation), {indiv_var1:.1f}% (individual)")
    results.append(f"  Dim 2: {nation_var2:.1f}% (nation), {indiv_var2:.1f}% (individual)")

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

    score = 0
    details = []

    # 1. All items present (20 pts)
    all_present = all(item in nation_loadings['item'].values for item in ITEMS_FOR_FACTOR)
    if all_present:
        score += 20
        details.append("All 10 items present: +20")

    # 2. Loading values (40 pts)
    loading_score = 0
    pts_per = 40 / 20
    for level, ldf in [('nation', nation_loadings), ('individual', indiv_loadings)]:
        gt = GROUND_TRUTH[f'{level}_level']
        for dim, items in [('trad_secrat', ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']),
                           ('surv_selfexp', ['Y002', 'A008', 'E025', 'F118', 'A165'])]:
            for item in items:
                gt_val = gt[dim][item]
                row = ldf[ldf['item'] == item]
                if len(row) > 0:
                    gen_val = row[dim].values[0]
                    diff = abs(gt_val - gen_val)
                    if diff <= 0.03:
                        loading_score += pts_per
                        tag = "MATCH"
                    elif diff <= 0.06:
                        loading_score += pts_per * 0.5
                        tag = "PARTIAL"
                    else:
                        tag = "MISS"
                    details.append(f"  {level} {item} {dim}: paper={gt_val:.2f}, gen={gen_val:.2f}, diff={diff:.3f} -> {tag}")
    score += loading_score
    details.append(f"Loading values: +{loading_score:.1f}/40")

    # 3. Dimension assignment (20 pts)
    dim_score = 0
    for level, ldf in [('nation', nation_loadings), ('individual', indiv_loadings)]:
        correct = 0
        for item in ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']:
            row = ldf[ldf['item'] == item]
            if len(row) > 0 and abs(row['trad_secrat'].values[0]) > abs(row['surv_selfexp'].values[0]):
                correct += 1
        for item in ['Y002', 'A008', 'E025', 'F118', 'A165']:
            row = ldf[ldf['item'] == item]
            if len(row) > 0 and abs(row['surv_selfexp'].values[0]) > abs(row['trad_secrat'].values[0]):
                correct += 1
        dim_score += (correct / 10) * 10
        details.append(f"  {level}: {correct}/10 correct")
    score += dim_score
    details.append(f"Dimension assignment: +{dim_score:.1f}/20")

    # 4. Ordering (10 pts)
    ord_score = 0
    for level, ldf in [('nation', nation_loadings), ('individual', indiv_loadings)]:
        trad_vals = [ldf[ldf['item']==it]['trad_secrat'].values[0] for it in ['god_important','autonomy_idx','F120','G006','E018']]
        surv_vals = [ldf[ldf['item']==it]['surv_selfexp'].values[0] for it in ['Y002','A008','E025','F118','A165']]
        t_ok = all(trad_vals[i] >= trad_vals[i+1] for i in range(4))
        s_ok = all(surv_vals[i] >= surv_vals[i+1] for i in range(4))
        if t_ok: ord_score += 2.5
        if s_ok: ord_score += 2.5
        details.append(f"  {level}: trad={'OK' if t_ok else 'WRONG'}, surv={'OK' if s_ok else 'WRONG'}")
    score += ord_score
    details.append(f"Ordering: +{ord_score:.1f}/10")

    # 5. Sample size and variance (10 pts)
    size_score = 0
    n = result['n_countries']
    if abs(n - 65) / 65 <= 0.05:
        size_score += 2
        details.append(f"  N={n} (paper=65) -> MATCH")
    elif abs(n - 65) / 65 <= 0.15:
        size_score += 1
        details.append(f"  N={n} (paper=65) -> PARTIAL")
    else:
        details.append(f"  N={n} (paper=65) -> MISS")

    for name, gen, gt in [
        ('nation_dim1', result['nation_var1'], 44),
        ('nation_dim2', result['nation_var2'], 26),
        ('indiv_dim1', result['indiv_var1'], 26),
        ('indiv_dim2', result['indiv_var2'], 13),
    ]:
        d = abs(gen - gt)
        if d <= 3:
            size_score += 2
            details.append(f"  {name}: {gen:.1f}% (paper={gt}%) -> MATCH")
        elif d <= 6:
            size_score += 1
            details.append(f"  {name}: {gen:.1f}% (paper={gt}%) -> PARTIAL")
        else:
            details.append(f"  {name}: {gen:.1f}% (paper={gt}%) -> MISS")
    score += size_score
    details.append(f"Sample size/variance: +{size_score:.1f}/10")

    total_score = min(100, round(score))
    details.append(f"\nTOTAL SCORE: {total_score}/100")
    print("\n" + "=" * 60)
    print("SCORING")
    print("=" * 60)
    for d in details:
        print(d)
    return total_score, details


if __name__ == "__main__":
    score, details = score_against_ground_truth()
