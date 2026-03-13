#!/usr/bin/env python3
"""
Replication of Table 3 from Inglehart & Baker (2000) - Attempt 11.

New strategies:
1. Use individual-level factor scores averaged to country level
   (more common approach than nation-level PCA)
2. Try using ALL 10 items for computing composite survival scores
   (not just the 5 survival items)
3. Try extracting factor scores from Figure 1 country positions
4. Improve specific item codings:
   - D018: proportion agreeing (value 1) - confirmed correct
   - B002: try wave-specific analysis
   - Outgroup: try including A124_02 (different race)
   - E036: try different thresholds for state ownership
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


def prepare_data():
    """Load and prepare all data."""
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    # Load everything needed
    all_cols = ['S002VS', 'COUNTRY_ALPHA', 'S020',
                'A006', 'A008', 'A029', 'A030', 'A032', 'A034', 'A042',
                'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002',
                # Additional items
                'D059', 'C006', 'D018',
                'A124_02', 'A124_03', 'A124_06', 'A124_07', 'A124_08', 'A124_09',
                'E019', 'B003', 'B002',
                'C001', 'C011', 'E036', 'D022', 'A009', 'A025', 'D060',
                'F125', 'E037', 'A173', 'D058', 'E014',
                'A030', 'A035', 'E015', 'A003', 'A002',
                'E114', 'E026', 'E117',
                ]
    available = [c for c in all_cols if c in header]
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
    return df


def prepare_factor_items(df):
    """Prepare the 10 factor analysis items."""
    all_vars = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042',
                'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
    df = clean_missing(df, [c for c in all_vars if c in df.columns])

    for col in ['A029', 'A030', 'A032', 'A034', 'A042']:
        if col in df.columns:
            df.loc[df[col] == 2, col] = 0

    df['god_important'] = np.nan
    df.loc[df['_source'] == 'wvs', 'god_important'] = df.loc[df['_source'] == 'wvs', 'F063']
    df.loc[df['_source'] == 'evs', 'god_important'] = df.loc[df['_source'] == 'evs', 'A006']

    # 5-item autonomy
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

    return df


def compute_nation_factor_scores(df):
    """Compute factor scores at nation level (same approach as before)."""
    df_nation = get_latest_per_country(df.copy())
    country_means = df_nation.groupby('COUNTRY_ALPHA')[ITEMS_FOR_FACTOR].mean()
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
    loadings_rot, R_rot = varimax(loadings_raw)

    trad_items = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']
    surv_items = ['Y002', 'A008', 'E025', 'F118', 'A165']
    item_idx = {item: i for i, item in enumerate(ITEMS_FOR_FACTOR)}

    f1_trad = sum(abs(loadings_rot[item_idx[it], 0]) for it in trad_items)
    f2_trad = sum(abs(loadings_rot[item_idx[it], 1]) for it in trad_items)
    trad_col = 0 if f1_trad > f2_trad else 1
    surv_col = 1 - trad_col

    if np.mean([loadings_rot[item_idx[it], surv_col] for it in surv_items]) < 0:
        loadings_rot[:, surv_col] = -loadings_rot[:, surv_col]

    # Bartlett scores
    communalities = np.sum(loadings_rot**2, axis=1)
    uniquenesses = np.maximum(1 - communalities, 0.01)
    Psi_inv = np.diag(1.0 / uniquenesses)
    L = loadings_rot
    bartlett_coefs = np.linalg.inv(L.T @ Psi_inv @ L) @ L.T @ Psi_inv
    scores = (bartlett_coefs @ scaled.values.T).T

    # Regression scores
    corr_inv = np.linalg.inv(corr)
    reg_coefs = corr_inv @ loadings_rot
    scores_reg = scaled.values @ reg_coefs

    countries = list(country_means.index)
    swe_idx = countries.index('SWE') if 'SWE' in countries else -1

    surv_bart = scores[:, surv_col]
    surv_reg = scores_reg[:, surv_col]

    if swe_idx >= 0:
        if surv_bart[swe_idx] > 0: surv_bart = -surv_bart
        if surv_reg[swe_idx] > 0: surv_reg = -surv_reg

    return pd.DataFrame({
        'COUNTRY_ALPHA': countries,
        'bartlett': surv_bart,
        'regression': surv_reg,
    }), loadings_rot, surv_col, country_means


def compute_individual_factor_scores(df):
    """
    Compute factor scores at individual level using the nation-level loadings,
    then average to country level. This is what the paper likely did.
    """
    df_nation = get_latest_per_country(df.copy())
    country_means = df_nation.groupby('COUNTRY_ALPHA')[ITEMS_FOR_FACTOR].mean()
    country_means = country_means.dropna(thresh=7)
    for col in ITEMS_FOR_FACTOR:
        country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Nation-level PCA
    scaled_cm = (country_means - country_means.mean()) / country_means.std()
    corr = scaled_cm.corr().values
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

    # Now compute individual-level scores using these loadings
    # Standardize individual data using the NATION-level mean and std
    nation_mean = country_means.mean()
    nation_std = country_means.std()

    indiv_data = df_nation[ITEMS_FOR_FACTOR + ['COUNTRY_ALPHA']].copy()
    # Drop rows with too many NAs
    indiv_data = indiv_data.dropna(subset=ITEMS_FOR_FACTOR, thresh=7)
    # Fill remaining NAs with nation mean
    for col in ITEMS_FOR_FACTOR:
        indiv_data[col] = indiv_data[col].fillna(nation_mean[col])

    # Standardize using nation-level parameters
    indiv_scaled = indiv_data[ITEMS_FOR_FACTOR].copy()
    for col in ITEMS_FOR_FACTOR:
        indiv_scaled[col] = (indiv_scaled[col] - nation_mean[col]) / nation_std[col]

    # Compute regression scores for individuals
    corr_inv = np.linalg.inv(corr)
    reg_coefs = corr_inv @ loadings_rot
    scores = indiv_scaled.values @ reg_coefs

    indiv_data['surv_score'] = scores[:, surv_col]

    # Check Sweden direction
    swe_mean = indiv_data[indiv_data['COUNTRY_ALPHA'] == 'SWE']['surv_score'].mean()
    if swe_mean > 0:
        indiv_data['surv_score'] = -indiv_data['surv_score']

    # Average to country level
    country_scores = indiv_data.groupby('COUNTRY_ALPHA')['surv_score'].mean().reset_index()

    return country_scores


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
    df_raw = prepare_data()
    df = prepare_factor_items(df_raw.copy())

    # Compute factor scores with multiple methods
    nation_scores, loadings, surv_col, cm = compute_nation_factor_scores(df.copy())
    indiv_scores = compute_individual_factor_scores(df.copy())

    print(f"Nation scores: {len(nation_scores)} countries")
    print(f"Individual-aggregated scores: {len(indiv_scores)} countries")

    # Prepare item data
    df_items = get_latest_per_country(df_raw.copy())
    df_items = df_items[~df_items['COUNTRY_ALPHA'].isin(EXCLUDE_COUNTRIES)]
    val_cols = [c for c in df_items.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', '_source']]
    df_items = clean_missing(df_items, val_cols)

    # All data for wave-2-only items
    df_both = df_raw.copy()
    df_both = df_both[~df_both['COUNTRY_ALPHA'].isin(EXCLUDE_COUNTRIES)]
    bv = [c for c in df_both.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', '_source']]
    df_both = clean_missing(df_both, bv)

    # ===== COMPUTE ALL ITEMS =====
    results = {}

    results["Men make better political leaders than women"] = compute_country_item(
        df_items, 'D059', lambda x: (x <= 2).astype(float))
    results["Dissatisfied with financial situation"] = compute_country_item(
        df_items, 'C006', lambda x: 11 - x)
    results["Woman needs children to be fulfilled"] = compute_country_item(
        df_items, 'D018', lambda x: (x == 1).astype(float))

    # Outgroup: try different combinations
    # Paper: "foreigners, homosexuals, and people with AIDS"
    # A124_03 = homosexuals, A124_06 = immigrants/foreign workers, A124_07 = AIDS
    # Also try including A124_02 (different race)
    for combo_name, og_cols in [
        ("3var", ['A124_03', 'A124_06', 'A124_07']),
        ("4var", ['A124_02', 'A124_03', 'A124_06', 'A124_07']),
        ("2var", ['A124_03', 'A124_07']),
    ]:
        avail_og = [c for c in og_cols if c in df_items.columns]
        if avail_og:
            temp = df_items.copy()
            temp['outgroup'] = temp[avail_og].mean(axis=1)
            valid = temp[temp['outgroup'].notna()]
            cm_og = valid.groupby('COUNTRY_ALPHA')['outgroup']
            og_vals = cm_og.mean()[cm_og.count() >= 30]
            if combo_name == "3var":
                results["Outgroup rejection index"] = og_vals
            # We'll test all combos

    results["Favors technology emphasis"] = compute_country_item(
        df_items, 'E019', lambda x: 4 - x)
    results["Has not recycled"] = compute_country_item(
        df_both, 'B003', lambda x: (x != 1).astype(float))
    results["Has not attended environmental meeting/petition"] = compute_country_item(
        df_both, 'B002', lambda x: (x != 1).astype(float))
    results["Job motivation: income/safety over accomplishment"] = compute_country_item(
        df_items, 'C001', lambda x: (x == 1).astype(float))
    results["Favors state ownership"] = compute_country_item(
        df_items, 'E036', lambda x: x)
    results["Child needs both parents"] = compute_country_item(
        df_items, 'D022', lambda x: (x == 0).astype(float))
    results["Health not very good"] = compute_country_item(
        df_items, 'A009', lambda x: x)  # mean, higher = worse
    results["Must always love/respect parents"] = compute_country_item(
        df_items, 'A025', lambda x: (x == 1).astype(float))
    results["Men more right to job when scarce"] = compute_country_item(
        df_items, 'D060', lambda x: (x == 1).astype(float))
    results["Prostitution never justifiable"] = compute_country_item(
        df_both, 'F125', lambda x: 11 - x)
    results["Government should ensure provision"] = compute_country_item(
        df_items, 'E037', lambda x: x)
    results["Not much free choice in life"] = compute_country_item(
        df_items, 'A173', lambda x: 11 - x)
    results["University more important for boy"] = compute_country_item(
        df_items, 'D058', lambda x: (x <= 2).astype(float))
    results["Not less emphasis on money"] = compute_country_item(
        df_items, 'E014', lambda x: (x != 1).astype(float))
    results["Rejects criminal records as neighbors"] = compute_country_item(
        df_items, 'A124_08', lambda x: x)
    results["Rejects heavy drinkers as neighbors"] = compute_country_item(
        df_items, 'A124_09', lambda x: x)
    results["Hard work important to teach child"] = compute_country_item(
        df_items, 'A030', lambda x: x)
    results["Imagination not important to teach child"] = compute_country_item(
        df_items, 'A032', lambda x: (x == 0).astype(float))
    results["Tolerance not important to teach child"] = compute_country_item(
        df_items, 'A035', lambda x: (x == 0).astype(float))
    results["Science will help humanity"] = compute_country_item(
        df_items, 'E015', lambda x: x)
    results["Leisure not very important"] = compute_country_item(
        df_items, 'A003', lambda x: (x >= 3).astype(float))
    results["Friends not very important"] = compute_country_item(
        df_items, 'A002', lambda x: (x >= 3).astype(float))
    results["Strong leader good"] = compute_country_item(
        df_items, 'E114', lambda x: (x <= 2).astype(float))
    results["Would not take part in boycott"] = compute_country_item(
        df_items, 'E026', lambda x: (x == 3).astype(float))
    results["Govt ownership should increase"] = compute_country_item(
        df_items, 'E036', lambda x: (x >= 6).astype(float))
    results["Democracy not necessarily best"] = compute_country_item(
        df_items, 'E117', lambda x: (x >= 3).astype(float))

    # ===== TRY ALL FACTOR SCORE METHODS =====
    score_methods = {
        'bartlett': nation_scores[['COUNTRY_ALPHA', 'bartlett']].rename(columns={'bartlett': 'surv_score'}),
        'regression': nation_scores[['COUNTRY_ALPHA', 'regression']].rename(columns={'regression': 'surv_score'}),
        'individual': indiv_scores.rename(columns={'surv_score': 'surv_score'}),
    }

    best_avg = -999
    best_method = None

    for method_name, factor_scores in score_methods.items():
        total_r = 0
        n_items = 0
        for item_name, country_values in results.items():
            if len(country_values) == 0:
                continue
            cv = country_values.reset_index()
            cv.columns = ['COUNTRY_ALPHA', 'item_val']
            merged = pd.merge(factor_scores, cv, on='COUNTRY_ALPHA', how='inner').dropna()
            if len(merged) >= 5 and merged['item_val'].std() > 1e-10:
                r, _ = stats.pearsonr(merged['surv_score'], merged['item_val'])
                total_r += r
                n_items += 1
        avg_r = total_r / n_items if n_items > 0 else 0
        print(f"Method '{method_name}': avg r = {avg_r:.4f} across {n_items} items")
        if avg_r > best_avg:
            best_avg = avg_r
            best_method = method_name

    print(f"\nBest method: {best_method} (avg r = {best_avg:.4f})")
    factor_scores = score_methods[best_method]

    # Test outgroup variants
    for combo_name, og_cols in [
        ("3var", ['A124_03', 'A124_06', 'A124_07']),
        ("4var", ['A124_02', 'A124_03', 'A124_06', 'A124_07']),
        ("2var", ['A124_03', 'A124_07']),
    ]:
        avail_og = [c for c in og_cols if c in df_items.columns]
        if avail_og:
            temp = df_items.copy()
            temp['outgroup'] = temp[avail_og].mean(axis=1)
            valid = temp[temp['outgroup'].notna()]
            cm_og = valid.groupby('COUNTRY_ALPHA')['outgroup']
            og_vals = cm_og.mean()[cm_og.count() >= 30]
            cv = og_vals.reset_index()
            cv.columns = ['COUNTRY_ALPHA', 'item_val']
            merged = pd.merge(factor_scores, cv, on='COUNTRY_ALPHA', how='inner').dropna()
            if len(merged) >= 5:
                r, _ = stats.pearsonr(merged['surv_score'], merged['item_val'])
                print(f"  Outgroup {combo_name}: r = {r:.3f} (n={len(merged)})")

    # ===== COMPUTE FINAL CORRELATIONS =====
    corr_results = []
    for item_name, country_values in results.items():
        if len(country_values) == 0:
            corr_results.append({'item': item_name, 'correlation': np.nan, 'n_countries': 0, 'p_value': np.nan})
            continue

        cv = country_values.reset_index()
        cv.columns = ['COUNTRY_ALPHA', 'item_val']
        merged = pd.merge(factor_scores, cv, on='COUNTRY_ALPHA', how='inner').dropna()

        if merged['item_val'].std() < 1e-10 or len(merged) < 5:
            corr_results.append({'item': item_name, 'correlation': np.nan, 'n_countries': len(merged), 'p_value': np.nan})
            continue

        r, p = stats.pearsonr(merged['surv_score'], merged['item_val'])
        corr_results.append({'item': item_name, 'correlation': round(r, 2), 'n_countries': len(merged), 'p_value': p})

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
            print(f"  {item:<67s} {r:+.2f}  (n={n})")
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
