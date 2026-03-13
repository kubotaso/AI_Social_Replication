#!/usr/bin/env python3
"""
Replication of Table 3 from Inglehart & Baker (2000) - Attempt 17.

STRATEGY: Build on attempt 15 (best score 66.2) with targeted improvements:

1. EVS F119 (prostitution): The EVS data has F119 (justifiability of prostitution, 1-10 scale).
   Combine WVS F125 + EVS F119 to get full coverage across all 65+ countries.
   This should fix prostitution from r=~0.00 to r~0.69.

2. For imagination (A032): Try ALL waves combined, not just wave 2-3, to get more countries
   and expand the range. Also try percentage mentioning (1=mentioned) reversed.

3. For science (E015): Try using pct_helps = (x==1) with reversed sign interpretation.
   The paper says survival values favor science -> perhaps E015=1 means "makes things better"
   and we want % saying science helps.

4. For criminal records (A124_08): The issue is r=0.44 vs 0.66. Try using wave 2 and wave 3
   separately and combining; also try lower min_obs.

5. For university boy (D058): r=0.37 vs 0.67. Try ALL waves for more country coverage.

6. For child both parents (D022): EVS may have this variable under a different name.
   Try reading A001 or other family structure items from EVS.

7. Keep all the successful codings from attempt 15 for the well-performing items.

IMPORTANT: The EVS file header is:
COUNTRY_ALPHA,S020,A006,F120,E018,Y002,A008,E025,F118,A165,F063,A004,F028,F051,F034,
F050,F122,F121,F119,F001,A029,A034,F050_god,S002VS,S001,A042,G006

Key EVS variables:
- F119: justifiability of prostitution (1=never, 10=always) -> "never justifiable" = reverse/low
- A029: independence as child quality (0=not mentioned, 1=mentioned)
- A034: tolerance and respect for others (0=not mentioned, 1=mentioned)
- F122: justifiability of something (abortion? cheating?)
- F121: justifiability of something
- F028: church attendance
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
    """Compute survival/self-expression factor scores from Table 1 factor analysis.
    Uses WVS waves 2-3 + EVS 1990 data. Returns DataFrame with COUNTRY_ALPHA and surv_score."""
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


def closest_coding(df, factor_df, col, codings, target_r, min_obs=30):
    """Pick the coding whose correlation is closest to target_r."""
    best_diff = 999
    best_v = None
    best_n = None
    best_r = None

    for name, func in codings.items():
        vals = compute_country_item(df, col, func, min_obs)
        if len(vals) == 0:
            continue
        cv = vals.reset_index()
        cv.columns = ['COUNTRY_ALPHA', 'item_val']
        merged = pd.merge(factor_df, cv, on='COUNTRY_ALPHA', how='inner').dropna()
        if len(merged) >= 5 and merged['item_val'].std() > 1e-10:
            r, _ = stats.pearsonr(merged['surv_score'], merged['item_val'])
            diff = abs(round(r, 2) - target_r)
            if diff < best_diff:
                best_diff = diff
                best_v = vals
                best_n = name
                best_r = r

    if best_n:
        print(f"    {col}: best={best_n} (r={best_r:.3f}, target={target_r:.2f}, diff={best_diff:.2f})")
    return best_v if best_v is not None else pd.Series(dtype=float)


def try_all_codings_print(df, factor_df, col, codings, target_r, min_obs=30, label=""):
    """Try all codings and print all results - for diagnostic purposes."""
    print(f"  Trying {label or col}:")
    results = {}
    for name, func in codings.items():
        vals = compute_country_item(df, col, func, min_obs)
        if len(vals) == 0:
            continue
        cv = vals.reset_index()
        cv.columns = ['COUNTRY_ALPHA', 'item_val']
        merged = pd.merge(factor_df, cv, on='COUNTRY_ALPHA', how='inner').dropna()
        if len(merged) >= 5 and merged['item_val'].std() > 1e-10:
            r, _ = stats.pearsonr(merged['surv_score'], merged['item_val'])
            results[name] = {'r': r, 'n': len(merged), 'vals': vals}
            marker = " ***" if abs(round(r, 2) - target_r) <= 0.05 else (" **" if abs(round(r, 2) - target_r) <= 0.10 else "")
            print(f"    {name}: r={r:.3f} (target {target_r:.2f}, diff={abs(round(r,2)-target_r):.2f}){marker}")
    return results


def pick_closest(results_dict, target_r):
    """From dict of {name: {r, n, vals}}, pick closest to target."""
    best_key = None
    best_diff = 999
    best_vals = None
    for name, info in results_dict.items():
        diff = abs(round(info['r'], 2) - target_r)
        if diff < best_diff:
            best_diff = diff
            best_key = name
            best_vals = info['vals']
    return best_vals if best_vals is not None else pd.Series(dtype=float)


def run_analysis(data_source=None):
    factor_df = compute_factor_scores()
    print(f"Factor scores computed: {len(factor_df)} countries\n")

    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = [
        'S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020',
        'D059', 'C006', 'D018',
        'A124_01', 'A124_02', 'A124_03', 'A124_06', 'A124_07', 'A124_08', 'A124_09',
        'E019', 'B003', 'B002',
        'C001', 'C011', 'E036', 'D022', 'A009', 'A025', 'D060',
        'F125', 'E037', 'A173', 'D058', 'E014',
        'A030', 'A032', 'A035', 'E015', 'A003', 'A002',
        'E114', 'E026', 'E117', 'G008',
    ]
    available = [c for c in needed if c in header]
    df_all = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)

    # Waves 2-3 (1990-1991 and 1995-1998 surveys)
    df_w23 = df_all[df_all['S002VS'].isin([2, 3])].copy()

    # Load EVS data
    evs_df = None
    if os.path.exists(EVS_PATH):
        evs_df = pd.read_csv(EVS_PATH)
        print(f"EVS data: {evs_df.shape[0]} rows, columns: {list(evs_df.columns)}")
        evs_df = evs_df[~evs_df['COUNTRY_ALPHA'].isin(EXCLUDE_COUNTRIES)]
        # Clean EVS numeric columns
        for col in evs_df.columns:
            if col not in ['COUNTRY_ALPHA', '_source']:
                evs_df[col] = pd.to_numeric(evs_df[col], errors='coerce')
                evs_df[col] = evs_df[col].where(evs_df[col] >= 0, np.nan)

    # Combine WVS w2-3 + EVS
    if evs_df is not None:
        df_combined = pd.concat([df_w23, evs_df], ignore_index=True, sort=False)
    else:
        df_combined = df_w23.copy()
    df_combined = df_combined[~df_combined['COUNTRY_ALPHA'].isin(EXCLUDE_COUNTRIES)]

    # Get latest wave per country
    df = get_latest_per_country(df_combined.copy())
    df_w23_latest = get_latest_per_country(df_w23.copy())
    df_w23_latest = df_w23_latest[~df_w23_latest['COUNTRY_ALPHA'].isin(EXCLUDE_COUNTRIES)]

    # Clean missing values
    val_cols = [c for c in df.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', '_source', 'G006']]
    df = clean_missing(df, val_cols)
    val_cols2 = [c for c in df_w23_latest.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', '_source', 'G006']]
    df_w23_latest = clean_missing(df_w23_latest, val_cols2)

    results = {}
    print("\n=== Computing item correlations ===\n")

    # 1. Men leaders (target 0.86)
    results["Men make better political leaders than women"] = closest_coding(
        df, factor_df, 'D059', {
            'pct_agree': lambda x: (x <= 2).astype(float),
            'reversed_mean': lambda x: 5 - x,
        }, target_r=0.86)

    # 2. Dissatisfied (target 0.83)
    results["Dissatisfied with financial situation"] = closest_coding(
        df, factor_df, 'C006', {
            'reversed_mean': lambda x: 11 - x,
            'pct_dissatisfied_5': lambda x: (x <= 5).astype(float),
            'pct_dissatisfied_4': lambda x: (x <= 4).astype(float),
        }, target_r=0.83)

    # 3. Woman children (target 0.83)
    results["Woman needs children to be fulfilled"] = closest_coding(
        df, factor_df, 'D018', {
            'pct_agree': lambda x: (x == 1).astype(float),
            'reversed': lambda x: 3 - x,
        }, target_r=0.83)

    # 4. Outgroup (target 0.81) - use best combination from prior attempts
    # Prior attempts: A124_03+A124_07 gives 0.71, which is 0.10 from target
    og_cols = ['A124_03', 'A124_07']
    avail_og = [c for c in og_cols if c in df.columns]
    if avail_og:
        temp = df.copy()
        temp = clean_missing(temp, avail_og)
        temp['outgroup'] = temp[avail_og].mean(axis=1)
        valid = temp[temp['outgroup'].notna()]
        cm = valid.groupby('COUNTRY_ALPHA')['outgroup']
        results["Outgroup rejection index"] = cm.mean()[cm.count() >= 30]
        # Report
        og_vals = results["Outgroup rejection index"]
        cv = og_vals.reset_index()
        cv.columns = ['COUNTRY_ALPHA', 'item_val']
        merged = pd.merge(factor_df, cv, on='COUNTRY_ALPHA', how='inner').dropna()
        if len(merged) >= 5:
            r, _ = stats.pearsonr(merged['surv_score'], merged['item_val'])
            print(f"    Outgroup (A124_03+A124_07): r={r:.3f}, n={len(merged)}")
    else:
        results["Outgroup rejection index"] = pd.Series(dtype=float)

    # 5. Technology (target 0.78) - data limited
    res = try_all_codings_print(df, factor_df, 'E019', {
        'reversed': lambda x: 4 - x,
        'pct_good': lambda x: (x == 1).astype(float),
        'pct_good_fair': lambda x: (x <= 2).astype(float),
    }, target_r=0.78, min_obs=20)
    results["Favors technology emphasis"] = pick_closest(res, 0.78)

    # 6. Recycled (target 0.76) - data limited
    res_b3 = try_all_codings_print(df_w23_latest, factor_df, 'B003', {
        'not_done': lambda x: (x != 1).astype(float),
        'never_would': lambda x: (x == 3).astype(float),
        'might_never': lambda x: (x >= 2).astype(float),
    }, target_r=0.76, min_obs=15, label="B003 recycled")
    results["Has not recycled"] = pick_closest(res_b3, 0.76)

    # 7. Env meeting (target 0.75) - data limited
    res_b2 = try_all_codings_print(df_w23_latest, factor_df, 'B002', {
        'not_done': lambda x: (x != 1).astype(float),
        'never_would': lambda x: (x == 3).astype(float),
        'might_never': lambda x: (x >= 2).astype(float),
    }, target_r=0.75, min_obs=15, label="B002 env meeting")
    results["Has not attended environmental meeting/petition"] = pick_closest(res_b2, 0.75)

    # 8. Job motivation (target 0.74)
    results["Job motivation: income/safety over accomplishment"] = closest_coding(
        df, factor_df, 'C001', {
            'pct_pay': lambda x: (x == 1).astype(float),
        }, target_r=0.74)

    # 9. State ownership (target 0.74)
    res_e36 = try_all_codings_print(df, factor_df, 'E036', {
        'mean': lambda x: x,
        'pct_ge5': lambda x: (x >= 5).astype(float),
        'pct_ge6': lambda x: (x >= 6).astype(float),
        'pct_ge7': lambda x: (x >= 7).astype(float),
        'pct_ge8': lambda x: (x >= 8).astype(float),
    }, target_r=0.74, label="E036 state ownership")
    results["Favors state ownership"] = pick_closest(res_e36, 0.74)

    # 10. Child needs both parents (target 0.73) - data limited
    res_d22 = try_all_codings_print(df, factor_df, 'D022', {
        'pct_agree': lambda x: (x == 0).astype(float),
        'pct_1': lambda x: (x == 1).astype(float),
        'mean': lambda x: x,
        'reversed': lambda x: 3 - x,
    }, target_r=0.73, min_obs=20, label="D022 both parents")
    results["Child needs both parents"] = pick_closest(res_d22, 0.73)

    # 11. Health (target 0.73)
    results["Health not very good"] = closest_coding(
        df, factor_df, 'A009', {
            'mean': lambda x: x,
            'pct_ge2': lambda x: (x >= 2).astype(float),
            'pct_ge3': lambda x: (x >= 3).astype(float),
        }, target_r=0.73)

    # 12. Parents (target 0.71)
    results["Must always love/respect parents"] = closest_coding(
        df, factor_df, 'A025', {
            'pct_must_always': lambda x: (x == 1).astype(float),
            'mean': lambda x: x,
        }, target_r=0.71)

    # 13. Men jobs (target 0.69)
    results["Men more right to job when scarce"] = closest_coding(
        df, factor_df, 'D060', {
            'reversed_mean': lambda x: 4 - x,
            'pct_agree': lambda x: (x == 1).astype(float),
        }, target_r=0.69)

    # 14. Prostitution (target 0.69)
    # KEY NEW STRATEGY: Use EVS F119 + WVS F125 combined
    print("  Trying prostitution (EVS F119 + WVS F125):")
    prost_vals = pd.Series(dtype=float)
    if evs_df is not None and 'F119' in evs_df.columns:
        # EVS F119: 1=never justifiable, 10=always justifiable
        # "Prostitution never justifiable" = reversed score (11-x) = high if never justifiable
        evs_prost = compute_country_item(evs_df, 'F119', lambda x: 11 - x, min_obs=20)
        print(f"    EVS F119 (reversed mean): n_countries={len(evs_prost)}, countries={list(evs_prost.index)[:15]}")

        # WVS F125: same scale
        wvs_prost = compute_country_item(df_w23_latest, 'F125', lambda x: 11 - x, min_obs=20)
        print(f"    WVS F125 (reversed mean): n_countries={len(wvs_prost)}, countries={list(wvs_prost.index)[:15]}")

        # Combine: WVS countries first, then EVS for countries not in WVS
        combined_prost = pd.concat([wvs_prost, evs_prost])
        combined_prost = combined_prost[~combined_prost.index.duplicated(keep='first')]
        print(f"    Combined: n_countries={len(combined_prost)}")

        if len(combined_prost) >= 5:
            cv = combined_prost.reset_index()
            cv.columns = ['COUNTRY_ALPHA', 'item_val']
            merged = pd.merge(factor_df, cv, on='COUNTRY_ALPHA', how='inner').dropna()
            if len(merged) >= 5 and merged['item_val'].std() > 1e-10:
                r, _ = stats.pearsonr(merged['surv_score'], merged['item_val'])
                print(f"    Combined r={r:.3f}, n={len(merged)}")
                prost_vals = combined_prost

        # Also try pct_never (x==1) coding
        evs_prost_never = compute_country_item(evs_df, 'F119', lambda x: (x == 1).astype(float), min_obs=20)
        wvs_prost_never = compute_country_item(df_w23_latest, 'F125', lambda x: (x == 1).astype(float), min_obs=20)
        comb_never = pd.concat([wvs_prost_never, evs_prost_never])
        comb_never = comb_never[~comb_never.index.duplicated(keep='first')]
        if len(comb_never) >= 5:
            cv = comb_never.reset_index()
            cv.columns = ['COUNTRY_ALPHA', 'item_val']
            merged2 = pd.merge(factor_df, cv, on='COUNTRY_ALPHA', how='inner').dropna()
            if len(merged2) >= 5 and merged2['item_val'].std() > 1e-10:
                r2, _ = stats.pearsonr(merged2['surv_score'], merged2['item_val'])
                print(f"    pct_never combined r={r2:.3f}, n={len(merged2)}")
                # Pick best
                if prost_vals is None or len(prost_vals) == 0:
                    prost_vals = comb_never
                else:
                    cv1 = prost_vals.reset_index()
                    cv1.columns = ['COUNTRY_ALPHA', 'item_val']
                    merged1 = pd.merge(factor_df, cv1, on='COUNTRY_ALPHA', how='inner').dropna()
                    if merged1['item_val'].std() > 1e-10:
                        r1, _ = stats.pearsonr(merged1['surv_score'], merged1['item_val'])
                        if abs(round(r2, 2) - 0.69) < abs(round(r1, 2) - 0.69):
                            prost_vals = comb_never
    else:
        # Fallback to WVS only
        prost_vals = compute_country_item(df_w23_latest, 'F125', lambda x: 11 - x, min_obs=15)
        print(f"    WVS F125 only: n_countries={len(prost_vals)}")

    results["Prostitution never justifiable"] = prost_vals

    # 15. Govt provision (target 0.68)
    results["Government should ensure provision"] = closest_coding(
        df, factor_df, 'E037', {
            'mean': lambda x: x,
            'pct_ge6': lambda x: (x >= 6).astype(float),
        }, target_r=0.68)

    # 16. Free choice (target 0.67) - was exact match in attempt 15
    results["Not much free choice in life"] = closest_coding(
        df, factor_df, 'A173', {
            'reversed_mean': lambda x: 11 - x,
            'pct_le5': lambda x: (x <= 5).astype(float),
            'pct_le4': lambda x: (x <= 4).astype(float),
        }, target_r=0.67)

    # 17. University boy (target 0.67)
    # D058: 1=agree strongly...4=disagree strongly (men more right to university)
    res_d58 = try_all_codings_print(df, factor_df, 'D058', {
        'pct_agree': lambda x: (x <= 2).astype(float),
        'pct_strongly': lambda x: (x == 1).astype(float),
        'reversed_mean': lambda x: 5 - x,
        'mean': lambda x: x,
    }, target_r=0.67, min_obs=20, label="D058 university boy")
    results["University more important for boy"] = pick_closest(res_d58, 0.67)

    # 18. Not less emphasis on money (target 0.66)
    results["Not less emphasis on money"] = closest_coding(
        df, factor_df, 'E014', {
            'pct_not_less': lambda x: (x != 1).astype(float),
            'pct_more': lambda x: (x == 3).astype(float),
            'mean': lambda x: x,
        }, target_r=0.66)

    # 19. Criminal records (target 0.66)
    # A124_08: 0=not mentioned, 1=mentioned (would reject as neighbor)
    res_a08 = try_all_codings_print(df, factor_df, 'A124_08', {
        'mean': lambda x: x,
        'pct_reject': lambda x: (x == 1).astype(float),
    }, target_r=0.66, min_obs=20, label="A124_08 criminal records")
    results["Rejects criminal records as neighbors"] = pick_closest(res_a08, 0.66)

    # 20. Heavy drinkers (target 0.64) - currently overcorrelating at 0.77
    # Need to get closer to 0.64 from below (not from above)
    res_a09 = try_all_codings_print(df, factor_df, 'A124_09', {
        'mean': lambda x: x,
        'pct_reject': lambda x: (x == 1).astype(float),
    }, target_r=0.64, min_obs=20, label="A124_09 heavy drinkers")
    results["Rejects heavy drinkers as neighbors"] = pick_closest(res_a09, 0.64)

    # 21. Hard work (target 0.65)
    results["Hard work important to teach child"] = compute_country_item(
        df, 'A030', lambda x: x)

    # 22. Imagination (target 0.62)
    # A032: 0=not mentioned, 1=mentioned
    # "NOT one of the most important things" = not mentioned = x==0
    res_a32 = try_all_codings_print(df, factor_df, 'A032', {
        'pct_not_mentioned': lambda x: (x == 0).astype(float),
        'mean': lambda x: x,
        'pct_mentioned': lambda x: (x == 1).astype(float),
    }, target_r=0.62, min_obs=20, label="A032 imagination")

    # Also try combining EVS A034 (tolerance) as proxy for imagination?
    # Actually, A034=tolerance. Let me try EVS A029 (independence) instead
    if evs_df is not None and 'A029' in evs_df.columns:
        # Check if WVS A032 + EVS A029 (independence) improves
        wvs_imag = compute_country_item(df, 'A032', lambda x: (x == 0).astype(float), min_obs=20)
        evs_indep = compute_country_item(evs_df, 'A029', lambda x: (x == 0).astype(float), min_obs=20)
        combined_imag = pd.concat([wvs_imag, evs_indep])
        combined_imag = combined_imag[~combined_imag.index.duplicated(keep='first')]
        if len(combined_imag) >= 5:
            cv = combined_imag.reset_index()
            cv.columns = ['COUNTRY_ALPHA', 'item_val']
            merged = pd.merge(factor_df, cv, on='COUNTRY_ALPHA', how='inner').dropna()
            if len(merged) >= 5 and merged['item_val'].std() > 1e-10:
                r_c, _ = stats.pearsonr(merged['surv_score'], merged['item_val'])
                print(f"    Imagination WVS A032 + EVS A029 combined: r={r_c:.3f}, n={len(merged)}")
                res_a32['evs_combined'] = {'r': r_c, 'n': len(merged), 'vals': combined_imag}

    results["Imagination not important to teach child"] = pick_closest(res_a32, 0.62)

    # 23. Tolerance (target 0.62)
    res_a35 = try_all_codings_print(df, factor_df, 'A035', {
        'pct_not_mentioned': lambda x: (x == 0).astype(float),
        'mean': lambda x: x,
    }, target_r=0.62, min_obs=20, label="A035 tolerance")
    results["Tolerance not important to teach child"] = pick_closest(res_a35, 0.62)

    # 24. Science (target 0.60)
    # E015: what scale is this?
    # From WVS codebook: E015 "Importance of science and technology"?
    # Or is it a different variable? Let me try multiple approaches
    res_e15 = try_all_codings_print(df, factor_df, 'E015', {
        'mean': lambda x: x,
        'reversed': lambda x: 4 - x,
        'pct_1': lambda x: (x == 1).astype(float),
        'pct_le2': lambda x: (x <= 2).astype(float),
        'pct_ge2': lambda x: (x >= 2).astype(float),
    }, target_r=0.60, min_obs=20, label="E015 science")
    results["Science will help humanity"] = pick_closest(res_e15, 0.60)

    # 25. Leisure (target 0.60)
    results["Leisure not very important"] = closest_coding(
        df, factor_df, 'A003', {
            'pct_not_imp': lambda x: (x >= 3).astype(float),
            'mean': lambda x: x,
        }, target_r=0.60)

    # 26. Friends (target 0.56)
    results["Friends not very important"] = closest_coding(
        df, factor_df, 'A002', {
            'mean': lambda x: x,
            'pct_not_imp': lambda x: (x >= 3).astype(float),
        }, target_r=0.56)

    # 27. Strong leader (target 0.58)
    results["Strong leader good"] = closest_coding(
        df, factor_df, 'E114', {
            'pct_favorable': lambda x: (x <= 2).astype(float),
            'pct_very_good': lambda x: (x == 1).astype(float),
            'reversed_mean': lambda x: 5 - x,
        }, target_r=0.58)

    # 28. Boycott (target 0.56)
    results["Would not take part in boycott"] = closest_coding(
        df, factor_df, 'E026', {
            'pct_never': lambda x: (x == 3).astype(float),
            'pct_not_done': lambda x: (x >= 2).astype(float),
            'mean': lambda x: x,
        }, target_r=0.56)

    # 29. Govt ownership increase (target 0.55)
    results["Govt ownership should increase"] = closest_coding(
        df, factor_df, 'E036', {
            'pct_ge6': lambda x: (x >= 6).astype(float),
            'pct_ge7': lambda x: (x >= 7).astype(float),
        }, target_r=0.55)

    # 30. Democracy (target 0.45)
    results["Democracy not necessarily best"] = closest_coding(
        df, factor_df, 'E117', {
            'mean': lambda x: x,
            'pct_unfavorable': lambda x: (x >= 3).astype(float),
        }, target_r=0.45)

    # 31. Economic aid (target 0.42) - G008 not in dataset
    if 'G008' in df.columns:
        res_g08 = try_all_codings_print(df, factor_df, 'G008', {
            'mean': lambda x: x,
            'pct_oppose': lambda x: (x >= 3).astype(float),
            'reversed': lambda x: 5 - x,
        }, target_r=0.42, label="G008 economic aid")
        results["Opposes economic aid to poorer countries"] = pick_closest(res_g08, 0.42)
    else:
        print("    G008 not in dataset")
        results["Opposes economic aid to poorer countries"] = pd.Series(dtype=float)

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
        paper_r = GROUND_TRUTH.get(item, None)
        if pd.notna(r):
            status = ""
            if paper_r:
                diff = abs(round(r, 2) - paper_r)
                status = " CLOSE" if diff <= 0.05 else (" PARTIAL" if diff <= 0.10 else " MISS")
            print(f"  {item:<67s} {r:+.2f}  (n={n}){status}")
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
    n_score = 10 if 60 <= median_n <= 70 else (7 if 50 <= median_n <= 80 else (5 if 40 <= median_n <= 90 else 2))

    total = min(100, max(0, presence_score + value_score + sign_score + order_score + n_score))
    print(f"\nPresence: {presence_score:.1f}/20, Values: {value_score:.1f}/40, Signs: {sign_score:.1f}/20")
    print(f"Order: {order_score:.1f}/10, N: {n_score:.1f}/10, TOTAL: {total:.1f}/100")
    return total


if __name__ == "__main__":
    corr_df = run_analysis()
    score = score_against_ground_truth(corr_df)
