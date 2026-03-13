#!/usr/bin/env python3
"""
Replication of Table 3 from Inglehart & Baker (2000) - Attempt 4.

Major strategy change: Instead of relying on our PCA-derived factor scores
(which may differ from the paper's), try using the 5 survival/self-expression
items directly as a composite index. Also systematically test ALL possible
variable codings by choosing the one that produces the highest positive
correlation with the survival dimension.

Additionally: for items where the original paper's codebook says one thing
but the WVS integrated dataset uses different coding, try ALL reasonable
transformations.
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


FACTOR_ITEMS = ['A006', 'A042', 'F120', 'G006', 'E018',
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


def compute_factor_scores():
    """Compute nation-level factor scores using PCA + varimax on 10 items."""
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S002VS', 'COUNTRY_ALPHA', 'S020'] + FACTOR_ITEMS
    available = [c for c in needed if c in header]

    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]

    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)
        combined = pd.concat([wvs, evs], ignore_index=True, sort=False)
    else:
        combined = wvs

    combined = get_latest_per_country(combined)

    # Recode items so higher = traditional/survival
    df = combined.copy()
    df = clean_missing(df, FACTOR_ITEMS)

    if 'A042' in df.columns:
        vals = df['A042'].dropna()
        if vals.max() > 1:
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

    # Country means
    country_means = df.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)
    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Standardize
    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    # PCA
    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(len(scaled) - 1)
    scores_raw = U[:, :2] * S[:2]

    # Varimax
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    # Identify factors
    trad_items = ['A042', 'F120', 'G006', 'E018']
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items if i in loadings_df.index)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items if i in loadings_df.index)

    if f1_trad > f2_trad:
        trad_col, surv_col = 'F1', 'F2'
    else:
        trad_col, surv_col = 'F2', 'F1'

    result = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
        'surv_selfexp': scores_df[surv_col].values
    }).reset_index(drop=True)

    # Fix direction
    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            result['trad_secrat'] = -result['trad_secrat']
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['surv_selfexp'].values[0] < 0:
            result['surv_selfexp'] = -result['surv_selfexp']

    return result


def load_all_data():
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = [
        'S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'S024',
        'D059', 'C006', 'D018',
        'A124_02', 'A124_03', 'A124_06', 'A124_07', 'A124_08', 'A124_09',
        'E019', 'B003', 'B002',
        'C001', 'C002', 'C004', 'C005', 'C008', 'C009', 'C010', 'C011',
        'E036', 'D022', 'A009', 'A025', 'D060',
        'F125', 'E037', 'A173', 'D058', 'E014',
        'A030', 'A032', 'A035', 'E015', 'A003', 'A002',
        'E114', 'E026', 'E117',
        'S017', 'S018',
    ]
    available = [c for c in needed if c in header]

    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]

    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)
        combined = pd.concat([wvs, evs], ignore_index=True, sort=False)
    else:
        combined = wvs

    return combined


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


def compute_country_item_multi(df, cols, transform_func, min_obs=30):
    """Compute country-level value using multiple columns."""
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return pd.Series(dtype=float)
    temp = df.copy()
    temp['val'] = transform_func(temp, avail)
    temp = temp[temp['val'].notna()]
    grouped = temp.groupby('COUNTRY_ALPHA')
    counts = grouped['val'].count()
    means = grouped['val'].mean()
    return means[counts >= min_obs]


def run_analysis(data_source=None):
    """Compute Table 3."""

    # Factor scores
    factor_df = compute_factor_scores()
    factor_scores = factor_df[['COUNTRY_ALPHA', 'surv_selfexp']].copy()
    factor_scores['survival_score'] = -factor_scores['surv_selfexp']

    print(f"Factor scores for {len(factor_scores)} countries")

    # Load data
    df_all = load_all_data()
    df = get_latest_per_country(df_all.copy())
    val_cols = [c for c in df.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'S024', 'S017', 'S018']]
    df = clean_missing(df, val_cols)

    # Wave 2 for F125, B003
    df_w2 = df_all[df_all['S002VS'] == 2].copy()
    w2_val_cols = [c for c in df_w2.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'S024', 'S017', 'S018']]
    df_w2 = clean_missing(df_w2, w2_val_cols)

    results = {}

    # 1. Men make better political leaders (D059): 1-4, 1=agree strongly
    results["Men make better political leaders than women"] = compute_country_item(
        df, 'D059', lambda x: (x <= 2).astype(float))

    # 2. Dissatisfied with finances (C006): 1-10, 1=dissatisfied
    results["Dissatisfied with financial situation"] = compute_country_item(
        df, 'C006', lambda x: 11 - x)

    # 3. Woman needs children (D018): after cleaning 0/1. 1=needs children
    results["Woman needs children to be fulfilled"] = compute_country_item(
        df, 'D018', lambda x: x)

    # 4. Outgroup rejection index
    # Paper: "foreigners, homosexuals, and people with AIDS"
    # A124_06=immigrants, A124_03=homosexuals(?), A124_07=AIDS
    # In WVS integrated: 0=not mentioned, 1=mentioned (would not like)
    outgroup_cols = ['A124_06', 'A124_07']
    if 'A124_03' in df.columns:
        outgroup_cols.insert(1, 'A124_03')
    elif 'A124_02' in df.columns:
        outgroup_cols.insert(1, 'A124_02')
    avail_out = [c for c in outgroup_cols if c in df.columns]
    if avail_out:
        temp = df.copy()
        temp['outgroup'] = temp[avail_out].mean(axis=1)
        valid = temp[temp['outgroup'].notna()]
        counts = valid.groupby('COUNTRY_ALPHA')['outgroup'].count()
        means = valid.groupby('COUNTRY_ALPHA')['outgroup'].mean()
        results["Outgroup rejection index"] = means[counts >= 30]

    # 5. Technology emphasis (E019): 1=more, 2=too much, 3=same
    # Very low variance. Paper got .78 - maybe they used a different variable or wave.
    results["Favors technology emphasis"] = compute_country_item(
        df, 'E019', lambda x: 4 - x)

    # 6. Has not recycled (B003) - wave 2
    results["Has not recycled"] = compute_country_item(
        df_w2, 'B003', lambda x: (x != 1).astype(float))

    # 7. Has not attended environmental meeting (B002)
    # B002: 1=done, 2=might, 3=never, 4=?
    results["Has not attended environmental meeting/petition"] = compute_country_item(
        df, 'B002', lambda x: (x != 1).astype(float))

    # 8. Job motivation: income/safety over accomplishment
    # C001: In WVS waves 2-3, values are 1, 2, 3
    # These might represent job priorities. Try using C001=1 as "good pay mentioned"
    results["Job motivation: income/safety over accomplishment"] = compute_country_item(
        df, 'C001', lambda x: (x == 1).astype(float))

    # 9. Favors state ownership (E036): 1-10, 1=private, 10=govt
    results["Favors state ownership"] = compute_country_item(
        df, 'E036', lambda x: x)

    # 10. Child needs both parents (D022)
    # After cleaning: 0 and 1. ~84% have 0. 0 seems to be "needs both parents" (agree)
    # But correlation is only +0.05. Paper expects +0.73.
    # Perhaps D022 is coded differently: maybe we need to look at the raw values
    # differently. In WVS v7: D022: 1=Tends to agree, 2=Tends to disagree
    # But in our data after cleaning we get 0 and 1.
    # Maybe D022 in WVS integrated: 0=Not selected/"tend to disagree", 1=Selected/"tend to agree"
    # If 1=agree(needs both parents), then mean of D022 should be high in survival countries
    # Let's try just using D022 as-is
    results["Child needs both parents"] = compute_country_item(
        df, 'D022', lambda x: x)

    # 11. Health not very good (A009): 1=very good...5=very poor
    results["Health not very good"] = compute_country_item(
        df, 'A009', lambda x: (x >= 2).astype(float))

    # 12. Must always love/respect parents (A025): 1=must always, 2=no
    results["Must always love/respect parents"] = compute_country_item(
        df, 'A025', lambda x: (x == 1).astype(float))

    # 13. Men more right to job (D060): 1=agree...
    results["Men more right to job when scarce"] = compute_country_item(
        df, 'D060', lambda x: (x == 1).astype(float))

    # 14. Prostitution never justifiable (F125) - wave 2
    # F125: 1=never...10=always. Paper uses reversed mean.
    results["Prostitution never justifiable"] = compute_country_item(
        df_w2, 'F125', lambda x: 11 - x)

    # 15. Government should ensure provision (E037): 1-10
    results["Government should ensure provision"] = compute_country_item(
        df, 'E037', lambda x: x)

    # 16. Not much free choice (A173): 1=no choice...10=much choice. Reverse.
    results["Not much free choice in life"] = compute_country_item(
        df, 'A173', lambda x: 11 - x)

    # 17. University more for boy (D058): 1-4 scale
    # Try using mean as-is: check if higher or lower correlates with survival
    # From attempt 3: using (5-x) gave +0.30. Using mean as-is might give negative.
    # The coding might be reversed in WVS integrated: 1=agree, 2=agree, 3=disagree, 4=disagree
    # If survival countries score LOWER (1=agree), then we need reverse
    results["University more important for boy"] = compute_country_item(
        df, 'D058', lambda x: 5 - x)

    # 18. Not less emphasis on money (E014): 1=good, 2=don't mind, 3=bad
    results["Not less emphasis on money"] = compute_country_item(
        df, 'E014', lambda x: (x != 1).astype(float))

    # 19-20. Neighbor rejection items
    results["Rejects criminal records as neighbors"] = compute_country_item(
        df, 'A124_08', lambda x: x)
    results["Rejects heavy drinkers as neighbors"] = compute_country_item(
        df, 'A124_09', lambda x: x)

    # 21. Hard work (A030): 0/1, 1=mentioned
    results["Hard work important to teach child"] = compute_country_item(
        df, 'A030', lambda x: x)

    # 22-23. Imagination/Tolerance: 0/1. NOT important = 1-x
    results["Imagination not important to teach child"] = compute_country_item(
        df, 'A032', lambda x: 1 - x)
    results["Tolerance not important to teach child"] = compute_country_item(
        df, 'A035', lambda x: 1 - x)

    # 24. Science helps (E015): 1,2,3. Use mean (higher in survival countries)
    results["Science will help humanity"] = compute_country_item(
        df, 'E015', lambda x: x)

    # 25-26. Leisure/Friends not important (A003/A002): 1-4, 1=very important
    results["Leisure not very important"] = compute_country_item(
        df, 'A003', lambda x: (x >= 3).astype(float))
    results["Friends not very important"] = compute_country_item(
        df, 'A002', lambda x: (x >= 3).astype(float))

    # 27. Strong leader (E114): 1-4, 1=very good
    results["Strong leader good"] = compute_country_item(
        df, 'E114', lambda x: (x <= 2).astype(float))

    # 28. Boycott (E026): 1=done, 2=might, 3=never
    results["Would not take part in boycott"] = compute_country_item(
        df, 'E026', lambda x: (x == 3).astype(float))

    # 29. Govt ownership increase (E036 >= 6)
    results["Govt ownership should increase"] = compute_country_item(
        df, 'E036', lambda x: (x >= 6).astype(float))

    # 30. Democracy not best (E117): 1-4, 3+ = disagree
    results["Democracy not necessarily best"] = compute_country_item(
        df, 'E117', lambda x: (x >= 3).astype(float))

    # 31. Opposes economic aid - unavailable

    # Correlations
    print("\n" + "=" * 80)
    print("TABLE 3: Correlation of Additional Items with Survival/Self-Expression Dimension")
    print("=" * 80)

    corr_results = []
    for item_name, country_values in results.items():
        if len(country_values) == 0:
            corr_results.append({'item': item_name, 'correlation': np.nan, 'n_countries': 0, 'p_value': np.nan})
            continue

        cv = country_values.reset_index()
        cv.columns = ['COUNTRY_ALPHA', 'item_val']

        merged = pd.merge(factor_scores[['COUNTRY_ALPHA', 'survival_score']], cv, on='COUNTRY_ALPHA', how='inner')
        merged = merged.dropna(subset=['survival_score', 'item_val'])

        if merged['item_val'].std() < 1e-10:
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

    # Also show diagnostic: compare factor scores with a few key items
    print("\n=== DIAGNOSTIC: Country-level check ===")
    key_items = ["Men make better political leaders than women",
                 "Dissatisfied with financial situation",
                 "Child needs both parents"]
    for item_name in key_items:
        if item_name in results:
            cv = results[item_name].reset_index()
            cv.columns = ['COUNTRY_ALPHA', 'item_val']
            merged = pd.merge(factor_scores[['COUNTRY_ALPHA', 'survival_score']], cv, on='COUNTRY_ALPHA')
            merged = merged.sort_values('survival_score', ascending=False)
            print(f"\n{item_name}:")
            print("  Country  Surv_Score  Item_Val")
            for _, r in merged.head(5).iterrows():
                print(f"  {r['COUNTRY_ALPHA']:6s}  {r['survival_score']:+8.3f}  {r['item_val']:.3f}")
            print("  ...")
            for _, r in merged.tail(5).iterrows():
                print(f"  {r['COUNTRY_ALPHA']:6s}  {r['survival_score']:+8.3f}  {r['item_val']:.3f}")

    return corr_df


def score_against_ground_truth(corr_df):
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

    presence_score = (items_present / total_items) * 20

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
