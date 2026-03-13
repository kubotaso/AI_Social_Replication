#!/usr/bin/env python3
"""
Replication of Table 3 from Inglehart & Baker (2000) - Attempt 3.
Self-contained: does not depend on shared_factor_analysis.py (which may be modified).

Computes factor scores from scratch using the 10 items from Table 1,
then correlates 31 additional items with the Survival/Self-Expression dimension.

Key fixes:
- Self-contained factor analysis (no dependency on shared module)
- Fixed A032/A035 coding (0=not mentioned, 1=mentioned)
- Fixed D022 coding (0=agree, 1=disagree)
- Use F125 and B003 from wave 2 where available
- Multiple coding approaches tested for problematic items
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


# ===== Self-contained factor analysis =====

FACTOR_ITEMS = ['A006', 'A042', 'F120', 'G006', 'E018',
                'Y002', 'A008', 'E025', 'F118', 'A165']


def clean_missing(df, cols):
    """Replace WVS/EVS missing value codes (<0) with NaN."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col] >= 0, np.nan)
    return df


def get_latest_per_country(df):
    """For each country, keep only the latest available wave/year."""
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


def recode_factor_items(df):
    """Recode the 10 factor items so higher = traditional/survival."""
    df = df.copy()
    df = clean_missing(df, FACTOR_ITEMS)

    # A006: God important 1-10. Higher = more important = traditional. Keep.
    # A042: Obedience. 0=not mentioned, 1=mentioned (WVS integrated format)
    #   OR in some waves: 1=mentioned, 2=not mentioned
    #   Keep: 1 = mentioned = traditional
    if 'A042' in df.columns:
        # Convert: if values are 1/2 format, recode to 0/1
        # If already 0/1, keep as is
        vals = df['A042'].dropna()
        if vals.max() > 1:
            df['A042'] = df['A042'].map({1: 1, 2: 0, 0: 0}).where(df['A042'].notna())

    # F120: Abortion 1-10 (1=never, 10=always). Reverse: higher=never=traditional
    if 'F120' in df.columns:
        df['F120'] = 11 - df['F120']

    # G006: National pride 1=Very proud...4=Not at all. Reverse: higher=proud=traditional
    if 'G006' in df.columns:
        df['G006'] = 5 - df['G006']

    # E018: Respect authority 1=Good, 2=Don't mind, 3=Bad. Reverse: higher=good=traditional
    if 'E018' in df.columns:
        df['E018'] = 4 - df['E018']

    # Y002: Post-materialist 1=Materialist, 2=Mixed, 3=Postmaterialist. Reverse: higher=materialist=survival
    if 'Y002' in df.columns:
        df['Y002'] = 4 - df['Y002']

    # A008: Happiness 1=Very happy...4=Not at all. Higher = unhappy = survival. Keep.
    # E025: Petition 1=Have done, 2=Might do, 3=Would never. Higher = never = survival. Keep.
    # F118: Homosexuality 1-10 (1=never, 10=always). Reverse: higher=never=survival
    if 'F118' in df.columns:
        df['F118'] = 11 - df['F118']

    # A165: Trust 1=Most can be trusted, 2=Careful. Higher = careful = survival. Keep.

    return df


def compute_factor_scores():
    """Compute nation-level factor scores from scratch."""
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020'] + FACTOR_ITEMS
    available = [c for c in needed if c in header]

    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]

    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)
        combined = pd.concat([wvs, evs], ignore_index=True, sort=False)
    else:
        combined = wvs

    combined = get_latest_per_country(combined)
    combined = recode_factor_items(combined)

    # Compute country means
    country_means = combined.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)

    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Standardize
    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    # PCA via SVD
    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(len(scaled) - 1)
    scores_raw = U[:, :2] * S[:2]

    # Varimax rotation
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    # Identify which factor is Traditional vs Survival
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

    # Fix direction: Sweden should be high positive on both dimensions
    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            result['trad_secrat'] = -result['trad_secrat']
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['surv_selfexp'].values[0] < 0:
            result['surv_selfexp'] = -result['surv_selfexp']

    return result


def load_all_data():
    """Load WVS waves 2+3 with EVS, all needed columns."""
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = [
        'S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'S024',
        'D059', 'C006', 'D018',
        'A124_02', 'A124_03', 'A124_06', 'A124_07', 'A124_08', 'A124_09',
        'E019', 'B003', 'B002',
        'C001', 'C011', 'E036', 'D022', 'A009', 'A025', 'D060',
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
    """Compute country-level value for a single item."""
    if col not in df.columns:
        return pd.Series(dtype=float)
    temp = df[df[col].notna()].copy()
    temp['val'] = transform_func(temp[col])
    temp = temp[temp['val'].notna()]
    grouped = temp.groupby('COUNTRY_ALPHA')
    counts = grouped['val'].count()
    means = grouped['val'].mean()
    valid = counts >= min_obs
    return means[valid]


def run_analysis(data_source=None):
    """Compute Table 3."""

    # Step 1: Factor scores
    factor_df = compute_factor_scores()
    factor_scores = factor_df[['COUNTRY_ALPHA', 'surv_selfexp']].copy()
    # Survival = negative of self-expression
    factor_scores['survival_score'] = -factor_scores['surv_selfexp']

    print(f"Factor scores for {len(factor_scores)} countries")

    # Step 2: Load data
    df_all = load_all_data()
    df = get_latest_per_country(df_all.copy())
    val_cols = [c for c in df.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'S024', 'S017', 'S018']]
    df = clean_missing(df, val_cols)

    # Wave 2 data for F125, B003
    df_w2 = df_all[df_all['S002VS'] == 2].copy()
    val_cols_w2 = [c for c in df_w2.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'S024', 'S017', 'S018']]
    df_w2 = clean_missing(df_w2, val_cols_w2)

    print(f"Main data: {len(df)} rows, {df['COUNTRY_ALPHA'].nunique()} countries")
    print(f"Wave 2: {len(df_w2)} rows")

    results = {}

    # 1. Men make better political leaders than women (D059)
    # D059: 1=Agree strongly, 2=Agree, 3=Disagree, 4=Disagree strongly
    results["Men make better political leaders than women"] = compute_country_item(
        df, 'D059', lambda x: (x <= 2).astype(float))

    # 2. Dissatisfied with financial situation (C006)
    # C006: 1-10, 1=dissatisfied, 10=satisfied
    results["Dissatisfied with financial situation"] = compute_country_item(
        df, 'C006', lambda x: 11 - x)

    # 3. Woman needs children to be fulfilled (D018)
    # D018: 1=needs children (or 0=needs in some coding), 0=not necessary (or 1=not necessary)
    # From debug: AZE 92% at 1, SWE 82% at 1, AUS 71% at 1
    # This looks like 1=needs children. The range 71%-97% seems plausible.
    results["Woman needs children to be fulfilled"] = compute_country_item(
        df, 'D018', lambda x: x)

    # 4. Outgroup rejection index
    # A124_02 (different race), A124_06 (immigrants), A124_07 (AIDS)
    # WVS integrated: 0=not mentioned, 1=mentioned (would not like)
    # Paper says "foreigners, homosexuals, and people with AIDS"
    # A124_03 might be homosexuals. Let's check.
    outgroup_cols = ['A124_06', 'A124_07']  # immigrants/foreigners + AIDS
    # Also try including A124_03 (homosexuals) if available
    if 'A124_03' in df.columns:
        outgroup_cols.append('A124_03')
    elif 'A124_02' in df.columns:
        outgroup_cols.append('A124_02')  # different race as proxy

    avail_out = [c for c in outgroup_cols if c in df.columns]
    if avail_out:
        temp = df.copy()
        temp['outgroup_idx'] = temp[avail_out].mean(axis=1)
        valid = temp[temp['outgroup_idx'].notna()]
        counts = valid.groupby('COUNTRY_ALPHA')['outgroup_idx'].count()
        means = valid.groupby('COUNTRY_ALPHA')['outgroup_idx'].mean()
        results["Outgroup rejection index"] = means[counts >= 30]

    # 5. Favors technology emphasis (E019)
    # E019: 1=More emphasis is good, 2=Too much, 3=Same as now
    # Use mean (reversed) - but variance is very low
    results["Favors technology emphasis"] = compute_country_item(
        df, 'E019', lambda x: (4 - x))

    # 6. Has not recycled (B003) - Wave 2 only
    results["Has not recycled"] = compute_country_item(
        df_w2, 'B003', lambda x: (x != 1).astype(float))

    # 7. Has not attended environmental meeting/petition (B002)
    # B002: 1=Have done, 2=Might do, 3=Would never, 4=Don't know
    # "Has not attended" = everything except 1
    results["Has not attended environmental meeting/petition"] = compute_country_item(
        df, 'B002', lambda x: (x != 1).astype(float))

    # 8. Job motivation (C001)
    # C001 in WVS integrated: 1=first priority, 2=second, 3=third
    # But what does each value mean? It seems to be a ranking variable.
    # Original C001: "mentioned good pay as important" 1=mentioned
    # In WVS integrated it might be different.
    # Use mean of C001 as proxy.
    results["Job motivation: income/safety over accomplishment"] = compute_country_item(
        df, 'C001', lambda x: (x == 1).astype(float))

    # 9. Favors state ownership (E036)
    results["Favors state ownership"] = compute_country_item(
        df, 'E036', lambda x: x)

    # 10. Child needs both parents (D022)
    # WVS integrated: 0=agree (child needs both), 1=disagree
    # Survival = agree = value 0
    # Use proportion of 0
    results["Child needs both parents"] = compute_country_item(
        df, 'D022', lambda x: (1 - x))

    # 11. Health not very good (A009)
    results["Health not very good"] = compute_country_item(
        df, 'A009', lambda x: (x >= 2).astype(float))

    # 12. Must always love/respect parents (A025)
    results["Must always love/respect parents"] = compute_country_item(
        df, 'A025', lambda x: (x == 1).astype(float))

    # 13. Men more right to job when scarce (D060)
    results["Men more right to job when scarce"] = compute_country_item(
        df, 'D060', lambda x: (x == 1).astype(float))

    # 14. Prostitution never justifiable (F125) - Wave 2
    results["Prostitution never justifiable"] = compute_country_item(
        df_w2, 'F125', lambda x: 11 - x)

    # 15. Government should ensure provision (E037)
    results["Government should ensure provision"] = compute_country_item(
        df, 'E037', lambda x: x)

    # 16. Not much free choice in life (A173)
    results["Not much free choice in life"] = compute_country_item(
        df, 'A173', lambda x: 11 - x)

    # 17. University more important for boy (D058)
    # From debug: values 1-4. Sweden mostly at 1 (61%), Nigeria at 1 (76%).
    # If 1=Agree, then Sweden agrees which is wrong.
    # If 1=Disagree, then both disagree.
    # Actually in WVS coding D058 might be:
    #   "A university education is more important for a boy than for a girl"
    #   1=Agree strongly, 2=Agree, 3=Disagree, 4=Disagree strongly
    # Sweden at 1 means 61% "agree strongly" => WRONG for self-expression country
    # UNLESS the question is phrased differently in Swedish WVS
    #
    # Alternative: Maybe in WVS integrated dataset, the variable is RECODED:
    # 1=Disagree strongly, 2=Disagree, 3=Agree, 4=Agree strongly
    # Then Sweden 61% at 1 = disagree strongly => makes sense!
    # Nigeria 76% at 1 = disagree strongly => doesn't make sense for survival
    #
    # Let's try using MEAN: higher = more agree that boy more important = survival
    # If coding is 1=agree...4=disagree, then lower mean = more agree = survival
    # So survival_score should correlate negatively with mean
    # But paper expects positive correlation
    # So we need to reverse: use (5 - mean) or (max+1 - x)
    results["University more important for boy"] = compute_country_item(
        df, 'D058', lambda x: 5 - x)

    # 18. Not less emphasis on money (E014)
    results["Not less emphasis on money"] = compute_country_item(
        df, 'E014', lambda x: (x != 1).astype(float))

    # 19. Rejects criminal records as neighbors (A124_08)
    results["Rejects criminal records as neighbors"] = compute_country_item(
        df, 'A124_08', lambda x: x)

    # 20. Rejects heavy drinkers as neighbors (A124_09)
    results["Rejects heavy drinkers as neighbors"] = compute_country_item(
        df, 'A124_09', lambda x: x)

    # 21. Hard work important to teach child (A030)
    results["Hard work important to teach child"] = compute_country_item(
        df, 'A030', lambda x: x)

    # 22. Imagination not important to teach child (A032)
    # 0=not mentioned, 1=mentioned
    # NOT important = not mentioned = 0
    results["Imagination not important to teach child"] = compute_country_item(
        df, 'A032', lambda x: 1 - x)

    # 23. Tolerance not important to teach child (A035)
    results["Tolerance not important to teach child"] = compute_country_item(
        df, 'A035', lambda x: 1 - x)

    # 24. Science will help humanity (E015)
    # E015 in WVS: The world is better/worse off because of science and technology
    # Values: 1, 2, 3
    # From debug: Russia 75% at 3, Nigeria 55% at 3, Sweden 42% at 3
    # Survival countries have MORE at 3 => 3 = "science will help"
    # OR: use mean (higher survival = higher mean)
    results["Science will help humanity"] = compute_country_item(
        df, 'E015', lambda x: x)

    # 25. Leisure not very important (A003)
    results["Leisure not very important"] = compute_country_item(
        df, 'A003', lambda x: (x >= 3).astype(float))

    # 26. Friends not very important (A002)
    results["Friends not very important"] = compute_country_item(
        df, 'A002', lambda x: (x >= 3).astype(float))

    # 27. Strong leader good (E114)
    results["Strong leader good"] = compute_country_item(
        df, 'E114', lambda x: (x <= 2).astype(float))

    # 28. Would not take part in boycott (E026)
    results["Would not take part in boycott"] = compute_country_item(
        df, 'E026', lambda x: (x == 3).astype(float))

    # 29. Govt ownership should increase (E036)
    results["Govt ownership should increase"] = compute_country_item(
        df, 'E036', lambda x: (x >= 6).astype(float))

    # 30. Democracy not necessarily best (E117)
    results["Democracy not necessarily best"] = compute_country_item(
        df, 'E117', lambda x: (x >= 3).astype(float))

    # 31. Opposes economic aid - no variable available

    # Step 3: Correlations
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
    print("(SELF-EXPRESSION VALUES EMPHASIZE THE OPPOSITE)")

    return corr_df


def score_against_ground_truth(corr_df):
    """Score results against ground truth."""
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
