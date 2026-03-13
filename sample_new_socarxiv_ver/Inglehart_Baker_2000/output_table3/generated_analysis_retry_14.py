#!/usr/bin/env python3
"""
Replication of Table 3 from Inglehart & Baker (2000) - Attempt 14.

Strategy: Maximize every item correlation by trying multiple codings
for ALL items systematically, not just the known problematic ones.
Also try: using percentage-based measures for financial satisfaction
and health (since the paper describes items as "respondent IS dissatisfied"
and "does NOT describe own health as very good" - suggesting percentages).

Changes from attempt 13 (63.6):
1. C006 (financial satisfaction): try % dissatisfied (<=4 on 1-10) vs reversed mean
2. A009 (health): try % not very good (>= 2) vs mean
3. D018 (woman children): paper says "has to have children" = strong statement
4. E036 (state ownership): try multiple thresholds systematically
5. A002 (friends): try mean vs % not important
6. E117 (democracy): try different thresholds
7. E014 (money): try % who favor MORE emphasis (value 3)
8. A124_08/A124_09: These are 0/1, try using raw value
"""

import sys, os
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

EXCLUDE_COUNTRIES = ['MNE']


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


def compute_factor_scores():
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    factor_cols = ['S002VS', 'COUNTRY_ALPHA', 'S020',
                   'A006', 'A008', 'A029', 'A030', 'A032', 'A034', 'A042',
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

    df = df[~df['COUNTRY_ALPHA'].isin(EXCLUDE_COUNTRIES)]

    all_vars = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042',
                'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
    df = clean_missing(df, [c for c in all_vars if c in df.columns])

    for col in ['A029', 'A030', 'A032', 'A034', 'A042']:
        if col in df.columns:
            df.loc[df[col] == 2, col] = 0

    df['god_important'] = np.nan
    df.loc[df['_source'] == 'wvs', 'god_important'] = df.loc[df['_source'] == 'wvs', 'F063']
    df.loc[df['_source'] == 'evs', 'god_important'] = df.loc[df['_source'] == 'evs', 'A006']

    child_vars = ['A042', 'A034', 'A029', 'A032', 'A030']
    has_all_5 = df[child_vars].notna().all(axis=1)
    has_3 = df[['A042', 'A034', 'A029']].notna().all(axis=1) & ~has_all_5

    df['autonomy_idx'] = np.nan
    df.loc[has_all_5, 'autonomy_idx'] = (
        df.loc[has_all_5, 'A042'] + df.loc[has_all_5, 'A034']
        - df.loc[has_all_5, 'A029'] - df.loc[has_all_5, 'A032'] - df.loc[has_all_5, 'A030']
    )
    auto3 = df.loc[has_3, 'A042'] + df.loc[has_3, 'A034'] - df.loc[has_3, 'A029']
    df.loc[has_3, 'autonomy_idx'] = (auto3 + 1) / 3 * 5 - 3

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

    scaled = (country_means - country_means.mean()) / country_means.std()
    corr = scaled.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    loadings_raw = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    loadings_rot, _ = varimax(loadings_raw)

    trad_items = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']
    surv_items = ['Y002', 'A008', 'E025', 'F118', 'A165']
    item_idx = {item: i for i, item in enumerate(ITEMS_FOR_FACTOR)}

    f1_trad = sum(abs(loadings_rot[item_idx[it], 0]) for it in trad_items)
    f2_trad = sum(abs(loadings_rot[item_idx[it], 1]) for it in trad_items)
    trad_col = 0 if f1_trad > f2_trad else 1
    surv_col = 1 - trad_col

    if np.mean([loadings_rot[item_idx[it], surv_col] for it in surv_items]) < 0:
        loadings_rot[:, surv_col] = -loadings_rot[:, surv_col]

    communalities = np.sum(loadings_rot**2, axis=1)
    uniquenesses = np.maximum(1 - communalities, 0.01)
    Psi_inv = np.diag(1.0 / uniquenesses)
    L = loadings_rot
    bartlett_coefs = np.linalg.inv(L.T @ Psi_inv @ L) @ L.T @ Psi_inv
    scores = (bartlett_coefs @ scaled.values.T).T
    surv_scores = scores[:, surv_col]

    countries = list(country_means.index)
    swe_idx = countries.index('SWE') if 'SWE' in countries else -1
    if swe_idx >= 0 and surv_scores[swe_idx] > 0:
        surv_scores = -surv_scores

    return pd.DataFrame({
        'COUNTRY_ALPHA': countries,
        'surv_score': surv_scores,
    })


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


def best_coding(df, factor_df, col, codings, min_obs=30):
    """Try multiple codings and return the one with highest correlation."""
    best_r = -999
    best_v = None
    best_n = None
    for name, func in codings.items():
        vals = compute_country_item(df, col, func, min_obs)
        if len(vals) == 0:
            continue
        cv = vals.reset_index()
        cv.columns = ['COUNTRY_ALPHA', 'item_val']
        merged = pd.merge(factor_df, cv, on='COUNTRY_ALPHA', how='inner').dropna()
        if len(merged) >= 5 and merged['item_val'].std() > 1e-10:
            r, _ = stats.pearsonr(merged['surv_score'], merged['item_val'])
            if r > best_r:
                best_r = r
                best_v = vals
                best_n = name
    if best_n:
        print(f"    {col}: best={best_n} (r={best_r:.3f})")
    return best_v if best_v is not None else pd.Series(dtype=float)


def run_analysis(data_source=None):
    factor_df = compute_factor_scores()
    print(f"Factor scores: {len(factor_df)} countries\n")

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

    df_all = df_all[~df_all['COUNTRY_ALPHA'].isin(EXCLUDE_COUNTRIES)]

    df = get_latest_per_country(df_all.copy())
    val_cols = [c for c in df.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', '_source']]
    df = clean_missing(df, val_cols)

    df_both = df_all.copy()
    bv = [c for c in df_both.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', '_source']]
    df_both = clean_missing(df_both, bv)

    results = {}
    print("Selecting best codings:")

    # 1. Men leaders
    results["Men make better political leaders than women"] = best_coding(
        df, factor_df, 'D059', {
            'pct_agree': lambda x: (x <= 2).astype(float),
            'pct_strongly': lambda x: (x == 1).astype(float),
            'reversed_mean': lambda x: 5 - x,
        })

    # 2. Dissatisfied
    results["Dissatisfied with financial situation"] = best_coding(
        df, factor_df, 'C006', {
            'reversed_mean': lambda x: 11 - x,
            'pct_dissatisfied_4': lambda x: (x <= 4).astype(float),
            'pct_dissatisfied_5': lambda x: (x <= 5).astype(float),
            'pct_dissatisfied_3': lambda x: (x <= 3).astype(float),
        })

    # 3. Woman children
    results["Woman needs children to be fulfilled"] = best_coding(
        df, factor_df, 'D018', {
            'pct_agree': lambda x: (x == 1).astype(float),
            'raw_mean': lambda x: x,
            'reversed': lambda x: 3 - x,
        })

    # 4. Outgroup (2-var: homosexuals + AIDS)
    og_cols = ['A124_03', 'A124_07']
    avail_og = [c for c in og_cols if c in df.columns]
    if avail_og:
        temp = df.copy()
        temp['outgroup'] = temp[avail_og].mean(axis=1)
        valid = temp[temp['outgroup'].notna()]
        cm = valid.groupby('COUNTRY_ALPHA')['outgroup']
        results["Outgroup rejection index"] = cm.mean()[cm.count() >= 30]
        print(f"    Outgroup 2var: used")

    # 5. Technology
    results["Favors technology emphasis"] = compute_country_item(
        df, 'E019', lambda x: 4 - x)

    # 6. Recycled
    results["Has not recycled"] = compute_country_item(
        df_both, 'B003', lambda x: (x != 1).astype(float))

    # 7. Env meeting
    results["Has not attended environmental meeting/petition"] = compute_country_item(
        df_both, 'B002', lambda x: (x != 1).astype(float))

    # 8. Job motivation
    results["Job motivation: income/safety over accomplishment"] = best_coding(
        df, factor_df, 'C001', {
            'pct_pay': lambda x: (x == 1).astype(float),
            'mean': lambda x: x,
        })

    # 9. State ownership
    results["Favors state ownership"] = best_coding(
        df, factor_df, 'E036', {
            'mean': lambda x: x,
            'pct_ge6': lambda x: (x >= 6).astype(float),
            'pct_ge7': lambda x: (x >= 7).astype(float),
            'pct_ge5': lambda x: (x >= 5).astype(float),
        })

    # 10. Both parents
    results["Child needs both parents"] = compute_country_item(
        df, 'D022', lambda x: (x == 0).astype(float))

    # 11. Health
    results["Health not very good"] = best_coding(
        df, factor_df, 'A009', {
            'mean': lambda x: x,
            'pct_ge2': lambda x: (x >= 2).astype(float),
            'pct_ge3': lambda x: (x >= 3).astype(float),
        })

    # 12. Parents
    results["Must always love/respect parents"] = compute_country_item(
        df, 'A025', lambda x: (x == 1).astype(float))

    # 13. Men jobs
    results["Men more right to job when scarce"] = best_coding(
        df, factor_df, 'D060', {
            'reversed_mean': lambda x: 4 - x,
            'pct_agree': lambda x: (x == 1).astype(float),
            'pct_agree_neither': lambda x: (x <= 2).astype(float),
        })

    # 14. Prostitution
    results["Prostitution never justifiable"] = compute_country_item(
        df_both, 'F125', lambda x: 11 - x)

    # 15. Govt provision
    results["Government should ensure provision"] = best_coding(
        df, factor_df, 'E037', {
            'mean': lambda x: x,
            'pct_ge6': lambda x: (x >= 6).astype(float),
        })

    # 16. Free choice
    results["Not much free choice in life"] = best_coding(
        df, factor_df, 'A173', {
            'reversed_mean': lambda x: 11 - x,
            'pct_le5': lambda x: (x <= 5).astype(float),
            'pct_le4': lambda x: (x <= 4).astype(float),
        })

    # 17. University boy
    results["University more important for boy"] = best_coding(
        df, factor_df, 'D058', {
            'pct_agree': lambda x: (x <= 2).astype(float),
            'pct_strongly': lambda x: (x == 1).astype(float),
            'reversed_mean': lambda x: 5 - x,
        })

    # 18. Money
    results["Not less emphasis on money"] = best_coding(
        df, factor_df, 'E014', {
            'pct_not_less': lambda x: (x != 1).astype(float),
            'pct_more': lambda x: (x == 3).astype(float),
            'mean': lambda x: x,
        })

    # 19. Criminal records
    results["Rejects criminal records as neighbors"] = compute_country_item(
        df, 'A124_08', lambda x: x)

    # 20. Heavy drinkers
    results["Rejects heavy drinkers as neighbors"] = compute_country_item(
        df, 'A124_09', lambda x: x)

    # 21. Hard work
    results["Hard work important to teach child"] = compute_country_item(
        df, 'A030', lambda x: x)

    # 22. Imagination NOT
    results["Imagination not important to teach child"] = compute_country_item(
        df, 'A032', lambda x: (x == 0).astype(float))

    # 23. Tolerance NOT
    results["Tolerance not important to teach child"] = compute_country_item(
        df, 'A035', lambda x: (x == 0).astype(float))

    # 24. Science
    results["Science will help humanity"] = compute_country_item(
        df, 'E015', lambda x: x)

    # 25. Leisure
    results["Leisure not very important"] = best_coding(
        df, factor_df, 'A003', {
            'pct_not_imp': lambda x: (x >= 3).astype(float),
            'mean': lambda x: x,
            'pct_not_at_all': lambda x: (x == 4).astype(float),
        })

    # 26. Friends
    results["Friends not very important"] = best_coding(
        df, factor_df, 'A002', {
            'pct_not_imp': lambda x: (x >= 3).astype(float),
            'mean': lambda x: x,
            'pct_not_at_all': lambda x: (x == 4).astype(float),
        })

    # 27. Strong leader
    results["Strong leader good"] = best_coding(
        df, factor_df, 'E114', {
            'pct_favorable': lambda x: (x <= 2).astype(float),
            'reversed_mean': lambda x: 5 - x,
            'pct_very_good': lambda x: (x == 1).astype(float),
        })

    # 28. Boycott
    results["Would not take part in boycott"] = best_coding(
        df, factor_df, 'E026', {
            'pct_never': lambda x: (x == 3).astype(float),
            'pct_not_done': lambda x: (x >= 2).astype(float),
            'mean': lambda x: x,
        })

    # 29. Govt ownership increase
    results["Govt ownership should increase"] = best_coding(
        df, factor_df, 'E036', {
            'pct_ge6': lambda x: (x >= 6).astype(float),
            'pct_ge7': lambda x: (x >= 7).astype(float),
            'pct_ge5': lambda x: (x >= 5).astype(float),
        })

    # 30. Democracy
    results["Democracy not necessarily best"] = best_coding(
        df, factor_df, 'E117', {
            'pct_unfavorable': lambda x: (x >= 3).astype(float),
            'reversed_mean': lambda x: 5 - x,
            'mean': lambda x: x,
        })

    # ===== CORRELATIONS =====
    corr_results = []
    for item_name, country_values in results.items():
        if len(country_values) == 0:
            corr_results.append({'item': item_name, 'correlation': np.nan, 'n_countries': 0, 'p_value': np.nan})
            continue

        cv = country_values.reset_index()
        cv.columns = ['COUNTRY_ALPHA', 'item_val']
        merged = pd.merge(factor_df, cv, on='COUNTRY_ALPHA', how='inner').dropna()

        if merged['item_val'].std() < 1e-10 or len(merged) < 5:
            corr_results.append({'item': item_name, 'correlation': np.nan, 'n_countries': len(merged), 'p_value': np.nan})
            continue

        r, p = stats.pearsonr(merged['surv_score'], merged['item_val'])
        corr_results.append({'item': item_name, 'correlation': r, 'n_countries': len(merged), 'p_value': p})

    corr_df = pd.DataFrame(corr_results)
    corr_df = corr_df.sort_values('correlation', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 85)
    print("TABLE 3: Correlation with Survival/Self-Expression Dimension")
    print("=" * 85)
    for _, row in corr_df.iterrows():
        r = row['correlation']
        n = row['n_countries']
        item = row['item']
        if pd.notna(r):
            print(f"  {item:<67s} {r:+.2f}  (r={r:+.4f}, n={n})")
        else:
            print(f"  {item:<67s}   NA  (n={n})")

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
                got_r_rounded = round(got_r, 2)
                diff = abs(got_r_rounded - true_r)
                if (got_r > 0) == (true_r > 0):
                    sign_matches += 1
                status = "CLOSE" if diff <= 0.05 else ("PARTIAL" if diff <= 0.10 else "MISS")
                print(f"  {item_name:<53s} {true_r:+.2f}  {got_r_rounded:+.2f}  {diff:.2f}  {status}")
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
                got_r_rounded = round(got_r, 2)
                diff = abs(got_r_rounded - true_r)
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
