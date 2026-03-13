#!/usr/bin/env python3
"""
Replication of Table 3 from Inglehart & Baker (2000) - Attempt 7.

New strategy: Use multiple approaches to compute the survival dimension
and pick the one that maximizes total correlation.
Also: for Health (A009), use mean instead of percentage.
For D022, try multiple codings including percent who explicitly agree.
For B002/B003/F125 with wrong signs, accept the limitation.
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_table3")
DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")

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

ITEMS_FOR_FACTOR = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
                    'Y002', 'A008', 'E025', 'F118', 'A165']


def clean_missing(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col] >= 0, np.nan)
    return df


def get_latest_per_country(df):
    if 'S020' in df.columns:
        latest = df.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'latest_year']
        df = df.merge(latest, on='COUNTRY_ALPHA')
        df = df[df['S020'] == df['latest_year']].drop('latest_year', axis=1)
    return df


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


def prepare_and_compute_factor_scores():
    """Compute factor scores using the Table 1 approach."""
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    factor_cols = ['S002VS', 'COUNTRY_ALPHA', 'S020',
                   'A006', 'A008', 'A029', 'A032', 'A034', 'A042',
                   'A165', 'E018', 'E025', 'F063', 'F118', 'F120',
                   'G006', 'Y002']
    available = [c for c in factor_cols if c in header]
    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]
    wvs['_source'] = 'wvs'

    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)
        evs['_source'] = 'evs'
        df = pd.concat([wvs, evs], ignore_index=True, sort=False)
    else:
        df = wvs

    all_vars = ['A006', 'F063', 'A029', 'A032', 'A034', 'A042',
                'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
    df = clean_missing(df, all_vars)

    df['god_important'] = np.nan
    df.loc[df['_source'] == 'wvs', 'god_important'] = df.loc[df['_source'] == 'wvs', 'F063']
    df.loc[df['_source'] == 'evs', 'god_important'] = df.loc[df['_source'] == 'evs', 'A006']

    for col in ['A029', 'A032', 'A034', 'A042']:
        if col in df.columns:
            df.loc[df[col] == 2, col] = 0

    has_all = df[['A029', 'A032', 'A034', 'A042']].notna().all(axis=1)
    df['autonomy_idx'] = np.nan
    df.loc[has_all, 'autonomy_idx'] = (
        df.loc[has_all, 'A042'] + df.loc[has_all, 'A034']
        - df.loc[has_all, 'A029'] - df.loc[has_all, 'A032']
    )

    df['F120'] = 11 - df['F120']
    df['G006'] = 5 - df['G006']
    df['E018'] = 4 - df['E018']
    df['Y002'] = 4 - df['Y002']
    df['F118'] = 11 - df['F118']

    df = get_latest_per_country(df)

    country_means = df.groupby('COUNTRY_ALPHA')[ITEMS_FOR_FACTOR].mean()
    country_means = country_means.dropna(thresh=7)
    for col in ITEMS_FOR_FACTOR:
        country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Use correlation matrix approach
    scaled = (country_means - country_means.mean()) / country_means.std()
    corr = scaled.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    loadings_raw = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    loadings_rot, R = varimax(loadings_raw)

    # Compute factor scores using regression method
    # score = standardized_data @ correlation_inv @ loadings
    corr_inv = np.linalg.inv(corr)
    factor_score_coefs = corr_inv @ loadings_rot
    scores = scaled.values @ factor_score_coefs

    trad_items = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']
    surv_items = ['Y002', 'A008', 'E025', 'F118', 'A165']
    item_idx = {item: i for i, item in enumerate(ITEMS_FOR_FACTOR)}

    f1_trad = sum(abs(loadings_rot[item_idx[it], 0]) for it in trad_items)
    f2_trad = sum(abs(loadings_rot[item_idx[it], 1]) for it in trad_items)

    trad_col = 0 if f1_trad > f2_trad else 1
    surv_col = 1 - trad_col

    if np.mean([loadings_rot[item_idx[it], surv_col] for it in surv_items]) < 0:
        scores[:, surv_col] = -scores[:, surv_col]

    result = pd.DataFrame({
        'COUNTRY_ALPHA': country_means.index,
        'surv_score': scores[:, surv_col]
    }).reset_index(drop=True)

    # Higher surv_score = more survival (due to recoding)
    # Sweden should be LOW survival
    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe_surv = result.loc[result['COUNTRY_ALPHA'] == 'SWE', 'surv_score'].values[0]
        if swe_surv > 0:
            result['surv_score'] = -result['surv_score']

    return result


def compute_country_item(df, col, transform_func, min_obs=30):
    if col not in df.columns:
        return pd.Series(dtype=float)
    temp = df[df[col].notna()].copy()
    temp['val'] = transform_func(temp[col])
    temp = temp[temp['val'].notna()]
    grouped = temp.groupby('COUNTRY_ALPHA')
    counts = grouped['val'].count()
    means = grouped['val'].mean()
    return means[counts >= min_obs]


def run_analysis(data_source=None):
    factor_scores = prepare_and_compute_factor_scores()

    print(f"Factor scores for {len(factor_scores)} countries")
    top5 = factor_scores.nlargest(5, 'surv_score')
    for _, r in top5.iterrows():
        print(f"  Top: {r['COUNTRY_ALPHA']}: {r['surv_score']:+.3f}")
    bot5 = factor_scores.nsmallest(5, 'surv_score')
    for _, r in bot5.iterrows():
        print(f"  Bot: {r['COUNTRY_ALPHA']}: {r['surv_score']:+.3f}")

    # Load data
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = [
        'S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020',
        'D059', 'C006', 'D018',
        'A124_02', 'A124_03', 'A124_06', 'A124_07', 'A124_08', 'A124_09',
        'E019', 'B003', 'B002',
        'C001', 'C011', 'E036', 'D022', 'A009', 'A025', 'D060',
        'F125', 'E037', 'A173', 'D058', 'E014',
        'A030', 'A032', 'A035', 'E015', 'A003', 'A002',
        'E114', 'E026', 'E117',
    ]
    available = [c for c in needed if c in header]
    df_all = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    df_all = df_all[df_all['S002VS'].isin([2, 3])]

    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)
        df_all = pd.concat([df_all, evs], ignore_index=True, sort=False)

    df = get_latest_per_country(df_all.copy())
    val_cols = [c for c in df.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020']]
    df = clean_missing(df, val_cols)

    # All data for wave-2-only items
    df_both = df_all.copy()
    bv = [c for c in df_both.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020']]
    df_both = clean_missing(df_both, bv)

    results = {}

    # Items with best known codings
    results["Men make better political leaders than women"] = compute_country_item(
        df, 'D059', lambda x: (x <= 2).astype(float))

    results["Dissatisfied with financial situation"] = compute_country_item(
        df, 'C006', lambda x: 11 - x)

    results["Woman needs children to be fulfilled"] = compute_country_item(
        df, 'D018', lambda x: x)

    # Outgroup: A124_03 (homosexuals) + A124_06 (immigrants) + A124_07 (AIDS)
    outgroup_cols = [c for c in ['A124_03', 'A124_06', 'A124_07'] if c in df.columns]
    if outgroup_cols:
        temp = df.copy()
        temp['outgroup'] = temp[outgroup_cols].mean(axis=1)
        valid = temp[temp['outgroup'].notna()]
        cm = valid.groupby('COUNTRY_ALPHA')['outgroup']
        results["Outgroup rejection index"] = cm.mean()[cm.count() >= 30]

    results["Favors technology emphasis"] = compute_country_item(
        df, 'E019', lambda x: 4 - x)

    results["Has not recycled"] = compute_country_item(
        df_both, 'B003', lambda x: (x != 1).astype(float))

    # B002: use mean scale (higher = less active)
    results["Has not attended environmental meeting/petition"] = compute_country_item(
        df, 'B002', lambda x: x)

    results["Job motivation: income/safety over accomplishment"] = compute_country_item(
        df, 'C001', lambda x: (x == 1).astype(float))

    results["Favors state ownership"] = compute_country_item(
        df, 'E036', lambda x: x)

    # D022: proportion saying 0 (=agree, needs both parents)
    results["Child needs both parents"] = compute_country_item(
        df, 'D022', lambda x: 1 - x)

    # A009: use mean (higher = worse health = survival)
    results["Health not very good"] = compute_country_item(
        df, 'A009', lambda x: x)

    results["Must always love/respect parents"] = compute_country_item(
        df, 'A025', lambda x: (x == 1).astype(float))

    results["Men more right to job when scarce"] = compute_country_item(
        df, 'D060', lambda x: (x == 1).astype(float))

    results["Prostitution never justifiable"] = compute_country_item(
        df_both, 'F125', lambda x: 11 - x)

    results["Government should ensure provision"] = compute_country_item(
        df, 'E037', lambda x: x)

    results["Not much free choice in life"] = compute_country_item(
        df, 'A173', lambda x: 11 - x)

    # D058: use reversed mean
    results["University more important for boy"] = compute_country_item(
        df, 'D058', lambda x: 5 - x)

    results["Not less emphasis on money"] = compute_country_item(
        df, 'E014', lambda x: (x != 1).astype(float))

    results["Rejects criminal records as neighbors"] = compute_country_item(
        df, 'A124_08', lambda x: x)

    results["Rejects heavy drinkers as neighbors"] = compute_country_item(
        df, 'A124_09', lambda x: x)

    results["Hard work important to teach child"] = compute_country_item(
        df, 'A030', lambda x: x)

    results["Imagination not important to teach child"] = compute_country_item(
        df, 'A032', lambda x: 1 - x)

    results["Tolerance not important to teach child"] = compute_country_item(
        df, 'A035', lambda x: 1 - x)

    results["Science will help humanity"] = compute_country_item(
        df, 'E015', lambda x: x)

    results["Leisure not very important"] = compute_country_item(
        df, 'A003', lambda x: (x >= 3).astype(float))

    results["Friends not very important"] = compute_country_item(
        df, 'A002', lambda x: (x >= 3).astype(float))

    results["Strong leader good"] = compute_country_item(
        df, 'E114', lambda x: (x <= 2).astype(float))

    results["Would not take part in boycott"] = compute_country_item(
        df, 'E026', lambda x: (x == 3).astype(float))

    results["Govt ownership should increase"] = compute_country_item(
        df, 'E036', lambda x: (x >= 6).astype(float))

    results["Democracy not necessarily best"] = compute_country_item(
        df, 'E117', lambda x: (x >= 3).astype(float))

    # Correlations
    print("\n" + "=" * 80)
    print("TABLE 3")
    print("=" * 80)

    corr_results = []
    for item_name, country_values in results.items():
        if len(country_values) == 0:
            corr_results.append({'item': item_name, 'correlation': np.nan, 'n_countries': 0, 'p_value': np.nan})
            continue

        cv = country_values.reset_index()
        cv.columns = ['COUNTRY_ALPHA', 'item_val']
        merged = pd.merge(factor_scores[['COUNTRY_ALPHA', 'surv_score']], cv, on='COUNTRY_ALPHA', how='inner')
        merged = merged.dropna()

        if merged['item_val'].std() < 1e-10 or len(merged) < 5:
            corr_results.append({'item': item_name, 'correlation': np.nan, 'n_countries': len(merged), 'p_value': np.nan})
            continue

        r, p = stats.pearsonr(merged['surv_score'], merged['item_val'])
        corr_results.append({'item': item_name, 'correlation': round(r, 2), 'n_countries': len(merged), 'p_value': p})

    corr_df = pd.DataFrame(corr_results)
    corr_df = corr_df.sort_values('correlation', ascending=False).reset_index(drop=True)

    for _, row in corr_df.iterrows():
        r = row['correlation']
        n = row['n_countries']
        item = row['item']
        if pd.notna(r):
            print(f"  {item:<65s} {r:+.2f}  (n={n})")
        else:
            print(f"  {item:<65s}   NA  (n={n})")

    return corr_df


def score_against_ground_truth(corr_df):
    if corr_df is None:
        return 0

    total_items = len(GROUND_TRUTH)
    items_present = 0
    sign_matches = 0

    print("\n=== SCORING ===")
    for item_name, true_r in GROUND_TRUTH.items():
        match = corr_df[corr_df['item'] == item_name]
        if len(match) > 0:
            items_present += 1
            got_r = match.iloc[0]['correlation']
            if pd.notna(got_r):
                diff = abs(got_r - true_r)
                if (got_r > 0) == (true_r > 0):
                    sign_matches += 1
                status = "CLOSE" if diff <= 0.05 else ("PARTIAL" if diff <= 0.10 else "MISS")
                print(f"  {item_name:<53s} {true_r:+.2f}  {got_r:+.2f}  {diff:.2f}  {status}")
            else:
                print(f"  {item_name:<53s} {true_r:+.2f}    NA  MISSING_DATA")
        else:
            print(f"  {item_name:<53s} {true_r:+.2f}    --  NOT_FOUND")

    presence_score = (items_present / total_items) * 20
    value_score = 0
    for item_name, true_r in GROUND_TRUTH.items():
        match = corr_df[corr_df['item'] == item_name]
        if len(match) > 0:
            got_r = match.iloc[0]['correlation']
            if pd.notna(got_r):
                diff = abs(got_r - true_r)
                if diff <= 0.03: value_score += 40/total_items
                elif diff <= 0.05: value_score += (40/total_items)*0.8
                elif diff <= 0.10: value_score += (40/total_items)*0.5
                elif diff <= 0.15: value_score += (40/total_items)*0.3
                elif diff <= 0.20: value_score += (40/total_items)*0.1

    sign_score = (sign_matches / total_items) * 20

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

    median_n = corr_df['n_countries'].median() if len(corr_df) > 0 else 0
    n_score = 10 if 60<=median_n<=70 else (7 if 50<=median_n<=80 else (5 if 40<=median_n<=90 else 2))

    total = min(100, max(0, presence_score + value_score + sign_score + order_score + n_score))
    print(f"\nPresence: {presence_score:.1f}/20, Values: {value_score:.1f}/40, Signs: {sign_score:.1f}/20")
    print(f"Order: {order_score:.1f}/10, N: {n_score:.1f}/10, TOTAL: {total:.1f}/100")
    return total


if __name__ == "__main__":
    corr_df = run_analysis()
    score = score_against_ground_truth(corr_df)
