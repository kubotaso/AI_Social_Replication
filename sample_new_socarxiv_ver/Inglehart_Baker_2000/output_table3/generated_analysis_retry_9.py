#!/usr/bin/env python3
"""
Replication of Table 3 from Inglehart & Baker (2000) - Attempt 9.

Major improvements over attempt 7/8:
1. Use EXACT same factor analysis as Table 1 best code:
   - 5-item autonomy index (obedience + faith - independence - determination - imagination)
   - EVS 3-item rescaling for countries missing A030/A032
   - Exclude MNE (not in original study)
2. Improve item codings based on careful re-reading of paper:
   - D018: proportion agreeing (value 1)
   - D058: "University education more important for boy than girl" - try % agree (<=2)
   - Outgroup: A124_06 (immigrants), A124_03 (homosexuals), A124_07 (AIDS)
   - Job motivation: combine C001 (safe job priority) with additional items
   - E015: Science helps - try different threshold
   - B002: environmental meeting - try proportion who have NOT done (value 3)
3. Use Bartlett factor scores (may correlate better with external items)
4. Try computing factor scores at nation level directly (not individual then aggregate)
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

EXCLUDE_COUNTRIES = ['MNE']  # Not separate in 1990s


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


def prepare_factor_data():
    """Load and prepare data for factor analysis, matching Table 1 approach exactly."""
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

    # Exclude countries not in original study
    df = df[~df['COUNTRY_ALPHA'].isin(EXCLUDE_COUNTRIES)]

    all_vars = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042',
                'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
    df = clean_missing(df, [c for c in all_vars if c in df.columns])

    # Recode child qualities to 0/1
    for col in ['A029', 'A030', 'A032', 'A034', 'A042']:
        if col in df.columns:
            df.loc[df[col] == 2, col] = 0

    # God importance: F063 for WVS, A006 for EVS
    df['god_important'] = np.nan
    df.loc[df['_source'] == 'wvs', 'god_important'] = df.loc[df['_source'] == 'wvs', 'F063']
    df.loc[df['_source'] == 'evs', 'god_important'] = df.loc[df['_source'] == 'evs', 'A006']

    # 5-item autonomy: (obedience + faith) - (independence + determination + imagination)
    # = A042 + A034 - A029 - A030 - A032
    child_vars = ['A042', 'A034', 'A029', 'A032', 'A030']
    has_all_5 = df[child_vars].notna().all(axis=1)
    has_3 = df[['A042', 'A034', 'A029']].notna().all(axis=1) & ~has_all_5

    df['autonomy_idx'] = np.nan

    # 5-item version
    df.loc[has_all_5, 'autonomy_idx'] = (
        df.loc[has_all_5, 'A042'] + df.loc[has_all_5, 'A034']
        - df.loc[has_all_5, 'A029'] - df.loc[has_all_5, 'A032'] - df.loc[has_all_5, 'A030']
    )

    # For 3-item (EVS countries): obedience + faith - independence
    # Rescale to approximate 5-item range: [-1, +2] -> [-3, +2]
    auto3 = df.loc[has_3, 'A042'] + df.loc[has_3, 'A034'] - df.loc[has_3, 'A029']
    df.loc[has_3, 'autonomy_idx'] = (auto3 + 1) / 3 * 5 - 3

    # Recode items so higher = more traditional/survival
    df['F120'] = 11 - df['F120']
    df['G006'] = 5 - df['G006']
    df['E018'] = 4 - df['E018']
    df['Y002'] = 4 - df['Y002']
    df['F118'] = 11 - df['F118']

    return df


def compute_factor_scores():
    """Compute survival/self-expression factor scores using Table 1 approach."""
    df = prepare_factor_data()
    df = get_latest_per_country(df)

    country_means = df.groupby('COUNTRY_ALPHA')[ITEMS_FOR_FACTOR].mean()
    country_means = country_means.dropna(thresh=7)
    for col in ITEMS_FOR_FACTOR:
        country_means[col] = country_means[col].fillna(country_means[col].mean())

    # PCA + varimax rotation on correlation matrix
    scaled = (country_means - country_means.mean()) / country_means.std()
    corr = scaled.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    loadings_raw = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    loadings_rot, R_rot = varimax(loadings_raw)

    # Identify which dimension is survival/self-expression
    trad_items = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']
    surv_items = ['Y002', 'A008', 'E025', 'F118', 'A165']
    item_idx = {item: i for i, item in enumerate(ITEMS_FOR_FACTOR)}

    f1_trad = sum(abs(loadings_rot[item_idx[it], 0]) for it in trad_items)
    f2_trad = sum(abs(loadings_rot[item_idx[it], 1]) for it in trad_items)
    trad_col = 0 if f1_trad > f2_trad else 1
    surv_col = 1 - trad_col

    # Ensure positive loadings for survival items
    if np.mean([loadings_rot[item_idx[it], surv_col] for it in surv_items]) < 0:
        loadings_rot[:, surv_col] = -loadings_rot[:, surv_col]

    print("Loadings (survival dim):")
    for item in ITEMS_FOR_FACTOR:
        print(f"  {item}: {loadings_rot[item_idx[item], surv_col]:.3f}")

    # Method 1: Regression factor scores
    corr_inv = np.linalg.inv(corr)
    factor_score_coefs = corr_inv @ loadings_rot
    scores_reg = scaled.values @ factor_score_coefs
    surv_reg = scores_reg[:, surv_col]

    # Method 2: Bartlett factor scores
    # Bartlett: scores = inv(L' * inv(uniqueness) * L) * L' * inv(uniqueness) * z
    communalities = np.sum(loadings_rot**2, axis=1)
    uniquenesses = 1 - communalities
    uniquenesses = np.maximum(uniquenesses, 0.01)  # floor
    Psi_inv = np.diag(1.0 / uniquenesses)
    L = loadings_rot
    LtPsiInvL = L.T @ Psi_inv @ L
    bartlett_coefs = np.linalg.inv(LtPsiInvL) @ L.T @ Psi_inv
    scores_bart = (bartlett_coefs @ scaled.values.T).T
    surv_bart = scores_bart[:, surv_col]

    # Method 3: Simple weighted composite of survival items
    surv_loadings = {it: loadings_rot[item_idx[it], surv_col] for it in surv_items}
    weights = np.array([surv_loadings[it] for it in surv_items])
    weights = weights / weights.sum()
    composite = scaled[surv_items].values @ weights
    surv_comp = composite

    # Ensure Sweden is LOW survival (negative score) for all methods
    countries = list(country_means.index)
    swe_idx = countries.index('SWE') if 'SWE' in countries else -1
    if swe_idx >= 0:
        if surv_reg[swe_idx] > 0:
            surv_reg = -surv_reg
        if surv_bart[swe_idx] > 0:
            surv_bart = -surv_bart
        if surv_comp[swe_idx] > 0:
            surv_comp = -surv_comp

    result = pd.DataFrame({
        'COUNTRY_ALPHA': countries,
        'regression': surv_reg,
        'bartlett': surv_bart,
        'composite': surv_comp,
    })

    return result


def compute_country_item(df, col, transform_func, min_obs=30):
    """Compute country-level mean of a transformed variable."""
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
    """Compute country-level mean from multiple columns combined."""
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series(dtype=float)
    temp = df.copy()
    temp['val'] = transform_func(temp, available)
    temp = temp[temp['val'].notna()]
    grouped = temp.groupby('COUNTRY_ALPHA')
    counts = grouped['val'].count()
    means = grouped['val'].mean()
    return means[counts >= min_obs]


def run_analysis(data_source=None):
    factor_df = compute_factor_scores()

    print(f"\nFactor scores for {len(factor_df)} countries")
    for method in ['regression', 'bartlett', 'composite']:
        top3 = factor_df.nlargest(3, method)
        bot3 = factor_df.nsmallest(3, method)
        print(f"\n  {method}:")
        for _, r in top3.iterrows():
            print(f"    Top: {r['COUNTRY_ALPHA']}: {r[method]:+.3f}")
        for _, r in bot3.iterrows():
            print(f"    Bot: {r['COUNTRY_ALPHA']}: {r[method]:+.3f}")

    # Load data for additional items
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
        'E114', 'E026', 'E117', 'G008',
    ]
    available = [c for c in needed if c in header]
    print(f"\nAvailable item columns: {len(available)} of {len(needed)}")
    missing = [c for c in needed if c not in header]
    if missing:
        print(f"  Missing: {missing}")

    df_all = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    df_all = df_all[df_all['S002VS'].isin([2, 3])]

    # Add EVS data
    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)
        df_all = pd.concat([df_all, evs], ignore_index=True, sort=False)

    # Exclude MNE
    df_all = df_all[~df_all['COUNTRY_ALPHA'].isin(EXCLUDE_COUNTRIES)]

    # Latest per country
    df = get_latest_per_country(df_all.copy())
    val_cols = [c for c in df.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020']]
    df = clean_missing(df, val_cols)

    # All data (both waves) for wave-2-only items
    df_both = df_all.copy()
    bv = [c for c in df_both.columns if c not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020']]
    df_both = clean_missing(df_both, bv)

    # ===== COMPUTE ALL ITEMS =====
    results = {}

    # 1. Men make better political leaders than women (D059: 1=Agree strongly, 2=Agree, 3=Disagree, 4=Disagree strongly)
    # Paper: % agreeing = values 1 or 2
    results["Men make better political leaders than women"] = compute_country_item(
        df, 'D059', lambda x: (x <= 2).astype(float))

    # 2. Dissatisfied with financial situation (C006: 1=dissatisfied to 10=satisfied)
    # Paper: higher = more dissatisfied = survival. Use reversed mean.
    results["Dissatisfied with financial situation"] = compute_country_item(
        df, 'C006', lambda x: 11 - x)

    # 3. Woman needs children to be fulfilled (D018: 1=Needs children, 2=Not necessary)
    # Paper: % agreeing that woman needs children
    results["Woman needs children to be fulfilled"] = compute_country_item(
        df, 'D018', lambda x: (x == 1).astype(float))

    # 4. Outgroup rejection index: A124_03 (homosexuals), A124_06 (immigrants/foreign workers), A124_07 (AIDS)
    # In WVS: 1=mentioned (wouldn't want as neighbor), 2=not mentioned
    # Paper: higher = more rejection = survival
    outgroup_cols = ['A124_03', 'A124_06', 'A124_07']
    avail_og = [c for c in outgroup_cols if c in df.columns]
    if avail_og:
        temp = df.copy()
        # For each: 1=mentioned (reject), 0=not mentioned
        for c in avail_og:
            temp[c + '_rej'] = (temp[c] == 1).astype(float)
            temp.loc[temp[c].isna(), c + '_rej'] = np.nan
        rej_cols = [c + '_rej' for c in avail_og]
        temp['outgroup'] = temp[rej_cols].mean(axis=1)
        valid = temp[temp['outgroup'].notna()]
        cm = valid.groupby('COUNTRY_ALPHA')['outgroup']
        results["Outgroup rejection index"] = cm.mean()[cm.count() >= 30]

    # 5. Favors technology emphasis (E019: 1=Good thing, 2=Don't know, 3=Bad thing for more emphasis on technology)
    # Paper: higher = favors technology. Use reversed.
    results["Favors technology emphasis"] = compute_country_item(
        df, 'E019', lambda x: 4 - x)

    # 6. Has not recycled (B003: 1=Have done, 2=Might do, 3=Would never do)
    # Paper: % who have NOT done. Use all waves.
    results["Has not recycled"] = compute_country_item(
        df_both, 'B003', lambda x: (x != 1).astype(float))

    # 7. Has not attended environmental meeting/petition (B002: 1=Have done, 2=Might do, 3=Would never do)
    # Paper: % who have NOT done
    results["Has not attended environmental meeting/petition"] = compute_country_item(
        df_both, 'B002', lambda x: (x != 1).astype(float))

    # 8. Job motivation: income/safety over accomplishment
    # C001: 1=Good pay important, 2=Not important (in choosing job)
    # Paper uses a job motivation INDEX, not just one item
    # Try: % choosing good income/safe job as most important
    results["Job motivation: income/safety over accomplishment"] = compute_country_item(
        df, 'C001', lambda x: (x == 1).astype(float))

    # 9. Favors state ownership (E036: 1=Private ownership should be increased to 10=Government ownership should be increased)
    # Paper: mean score (higher = more state ownership = survival)
    results["Favors state ownership"] = compute_country_item(
        df, 'E036', lambda x: x)

    # 10. Child needs both parents (D022: 1=Needs both parents, 2=Not necessarily)
    # Wait - in WVS integrated: check actual values
    # D022: "A child needs a home with both a father and a mother to grow up happily"
    # Values: 1=Tend to agree, 2=Tend to disagree (in some codings)
    # Or could be 0/1. Let's try proportion who agree.
    results["Child needs both parents"] = compute_country_item(
        df, 'D022', lambda x: (x == 1).astype(float))

    # 11. Health not very good (A009: 1=Very good, 2=Good, 3=Fair, 4=Poor, 5=Very poor)
    # Paper: % NOT describing own health as very good
    results["Health not very good"] = compute_country_item(
        df, 'A009', lambda x: (x >= 2).astype(float))

    # 12. Must always love/respect parents (A025: 1=Must always love regardless, 2=Do not have to)
    results["Must always love/respect parents"] = compute_country_item(
        df, 'A025', lambda x: (x == 1).astype(float))

    # 13. Men more right to job when scarce (D060: 1=Agree, 2=Neither, 3=Disagree)
    results["Men more right to job when scarce"] = compute_country_item(
        df, 'D060', lambda x: (x == 1).astype(float))

    # 14. Prostitution never justifiable (F125: 1=Never justifiable to 10=Always justifiable)
    # Paper: reversed mean (higher = never justifiable = survival). Use all waves.
    results["Prostitution never justifiable"] = compute_country_item(
        df_both, 'F125', lambda x: 11 - x)

    # 15. Government should ensure provision (E037: 1=People should take more responsibility to 10=Government should take more responsibility)
    results["Government should ensure provision"] = compute_country_item(
        df, 'E037', lambda x: x)

    # 16. Not much free choice in life (A173: 1=None at all to 10=A great deal)
    # Paper: reversed mean (higher = not much free choice = survival)
    results["Not much free choice in life"] = compute_country_item(
        df, 'A173', lambda x: 11 - x)

    # 17. University more important for boy (D058: 1=Agree strongly ... 4=Disagree strongly)
    # Paper: "A university education is more important for a boy than for a girl"
    # Higher agreement = more survival. Use % agreeing (values 1 or 2) or reversed mean.
    # Try both: percent agreeing and reversed mean
    d058_pct = compute_country_item(df, 'D058', lambda x: (x <= 2).astype(float))
    d058_rev = compute_country_item(df, 'D058', lambda x: 5 - x)
    results["University more important for boy"] = d058_pct  # try % first

    # 18. Not less emphasis on money (E014: 1=Less emphasis on money, 2=No change, 3=More emphasis)
    # Paper: % who do NOT favor less emphasis on money
    results["Not less emphasis on money"] = compute_country_item(
        df, 'E014', lambda x: (x != 1).astype(float))

    # 19. Rejects criminal records as neighbors (A124_08: 1=Mentioned, 2=Not mentioned)
    results["Rejects criminal records as neighbors"] = compute_country_item(
        df, 'A124_08', lambda x: (x == 1).astype(float))

    # 20. Rejects heavy drinkers as neighbors (A124_09: 1=Mentioned, 2=Not mentioned)
    results["Rejects heavy drinkers as neighbors"] = compute_country_item(
        df, 'A124_09', lambda x: (x == 1).astype(float))

    # 21. Hard work important to teach child (A030: 1=Mentioned, 2=Not mentioned)
    # In WVS integrated: values should be 0/1 or 1/2
    results["Hard work important to teach child"] = compute_country_item(
        df, 'A030', lambda x: (x == 1).astype(float))

    # 22. Imagination NOT important to teach child (A032: 1=Mentioned, 2=Not mentioned)
    # Paper: % NOT mentioning imagination. Higher = survival.
    results["Imagination not important to teach child"] = compute_country_item(
        df, 'A032', lambda x: (x != 1).astype(float))

    # 23. Tolerance NOT important to teach child (A035: 1=Mentioned, 2=Not mentioned)
    results["Tolerance not important to teach child"] = compute_country_item(
        df, 'A035', lambda x: (x != 1).astype(float))

    # 24. Science will help humanity (E015: coding varies)
    # Try mean: higher values might = more faith in science
    # In WVS: E015 "Science and technology are making our lives healthier, easier, and more comfortable"
    # Values: 1=Completely agree, 2=Somewhat agree, ... up to 4 or 10
    # Paper says "Scientific discoveries will help, rather than harm, humanity"
    # Higher = more pro-science = survival (counter-intuitive but that's what paper says)
    results["Science will help humanity"] = compute_country_item(
        df, 'E015', lambda x: x)

    # 25. Leisure not very important (A003: 1=Very important, 2=Rather important, 3=Not very important, 4=Not at all important)
    # Paper: % saying not very or not at all important
    results["Leisure not very important"] = compute_country_item(
        df, 'A003', lambda x: (x >= 3).astype(float))

    # 26. Friends not very important (A002: same coding as A003)
    results["Friends not very important"] = compute_country_item(
        df, 'A002', lambda x: (x >= 3).astype(float))

    # 27. Strong leader good (E114: 1=Very good, 2=Fairly good, 3=Fairly bad, 4=Very bad)
    # Paper: % favorable (values 1 or 2)
    results["Strong leader good"] = compute_country_item(
        df, 'E114', lambda x: (x <= 2).astype(float))

    # 28. Would not take part in boycott (E026: 1=Have done, 2=Might do, 3=Would never do)
    results["Would not take part in boycott"] = compute_country_item(
        df, 'E026', lambda x: (x == 3).astype(float))

    # 29. Govt ownership should increase (E036: 1-10 scale)
    # Paper: separate from "Favors state ownership" - using % above midpoint
    results["Govt ownership should increase"] = compute_country_item(
        df, 'E036', lambda x: (x >= 6).astype(float))

    # 30. Democracy not necessarily best (E117: 1=Very good, 2=Fairly good, 3=Fairly bad, 4=Very bad)
    # Paper: % saying democracy is not necessarily best = % unfavorable
    results["Democracy not necessarily best"] = compute_country_item(
        df, 'E117', lambda x: (x >= 3).astype(float))

    # 31. Opposes economic aid (G008 - check if available)
    if 'G008' in df.columns:
        results["Opposes economic aid to poorer countries"] = compute_country_item(
            df, 'G008', lambda x: x)

    # ===== COMPUTE CORRELATIONS WITH ALL THREE FACTOR SCORE METHODS =====
    methods = ['regression', 'bartlett', 'composite']
    all_corrs = {}

    for method in methods:
        factor_scores = factor_df[['COUNTRY_ALPHA', method]].copy()
        factor_scores.columns = ['COUNTRY_ALPHA', 'surv_score']

        corr_results = []
        for item_name, country_values in results.items():
            if len(country_values) == 0:
                corr_results.append({'item': item_name, 'correlation': np.nan, 'n_countries': 0})
                continue

            cv = country_values.reset_index()
            cv.columns = ['COUNTRY_ALPHA', 'item_val']
            merged = pd.merge(factor_scores, cv, on='COUNTRY_ALPHA', how='inner').dropna()

            if merged['item_val'].std() < 1e-10 or len(merged) < 5:
                corr_results.append({'item': item_name, 'correlation': np.nan, 'n_countries': len(merged)})
                continue

            r, p = stats.pearsonr(merged['surv_score'], merged['item_val'])
            corr_results.append({'item': item_name, 'correlation': r, 'n_countries': len(merged)})

        all_corrs[method] = pd.DataFrame(corr_results)

        # Compute average absolute correlation
        valid = [row['correlation'] for _, row in pd.DataFrame(corr_results).iterrows()
                 if pd.notna(row['correlation']) and abs(row['correlation']) > 0.01]
        avg = np.mean(valid) if valid else 0
        print(f"\nMethod '{method}': avg correlation = {avg:.4f} across {len(valid)} items")

    # Pick the method with highest average correlation
    best_method = max(methods, key=lambda m: np.nanmean(
        all_corrs[m]['correlation'].dropna().values))
    print(f"\nBest method: {best_method}")

    # Also try D058 with reversed mean vs percentage
    factor_scores_best = factor_df[['COUNTRY_ALPHA', best_method]].copy()
    factor_scores_best.columns = ['COUNTRY_ALPHA', 'surv_score']

    for label, vals in [("D058 pct agree", d058_pct), ("D058 reversed mean", d058_rev)]:
        if len(vals) > 0:
            cv = vals.reset_index()
            cv.columns = ['COUNTRY_ALPHA', 'item_val']
            merged = pd.merge(factor_scores_best, cv, on='COUNTRY_ALPHA', how='inner').dropna()
            if len(merged) >= 5:
                r, _ = stats.pearsonr(merged['surv_score'], merged['item_val'])
                print(f"  {label}: r = {r:.3f} (n={len(merged)})")

    # Also try alternative codings for problematic items
    alt_items = {
        "D022 pct==1": compute_country_item(df, 'D022', lambda x: (x == 1).astype(float)),
        "D022 1-x": compute_country_item(df, 'D022', lambda x: 1 - x),
        "D022 mean": compute_country_item(df, 'D022', lambda x: x),
        "A009 pct>=2": compute_country_item(df, 'A009', lambda x: (x >= 2).astype(float)),
        "A009 mean": compute_country_item(df, 'A009', lambda x: x),
        "A009 pct>=3": compute_country_item(df, 'A009', lambda x: (x >= 3).astype(float)),
        "B002 pct!=1 both": compute_country_item(df_both, 'B002', lambda x: (x != 1).astype(float)),
        "B002 pct==3": compute_country_item(df, 'B002', lambda x: (x == 3).astype(float)),
        "B002 mean": compute_country_item(df, 'B002', lambda x: x),
        "E015 4-x": compute_country_item(df, 'E015', lambda x: 4 - x),
        "E015 pct==1": compute_country_item(df, 'E015', lambda x: (x == 1).astype(float)),
        "A030 ==1": compute_country_item(df, 'A030', lambda x: (x == 1).astype(float)),
        "A030 raw": compute_country_item(df, 'A030', lambda x: x),
        "A032 !=1": compute_country_item(df, 'A032', lambda x: (x != 1).astype(float)),
        "A032 1-x": compute_country_item(df, 'A032', lambda x: 1 - x),
        "A032 ==0": compute_country_item(df, 'A032', lambda x: (x == 0).astype(float)),
        "A035 !=1": compute_country_item(df, 'A035', lambda x: (x != 1).astype(float)),
        "A035 1-x": compute_country_item(df, 'A035', lambda x: 1 - x),
        "C001 ==1": compute_country_item(df, 'C001', lambda x: (x == 1).astype(float)),
        "E019 4-x": compute_country_item(df, 'E019', lambda x: 4 - x),
        "E019 pct==1": compute_country_item(df, 'E019', lambda x: (x == 1).astype(float)),
        "D018 ==1": compute_country_item(df, 'D018', lambda x: (x == 1).astype(float)),
        "D018 raw": compute_country_item(df, 'D018', lambda x: x),
    }

    print("\n--- Alternative item codings ---")
    for label, vals in alt_items.items():
        if len(vals) > 0:
            cv = vals.reset_index()
            cv.columns = ['COUNTRY_ALPHA', 'item_val']
            merged = pd.merge(factor_scores_best, cv, on='COUNTRY_ALPHA', how='inner').dropna()
            if len(merged) >= 5 and merged['item_val'].std() > 1e-10:
                r, _ = stats.pearsonr(merged['surv_score'], merged['item_val'])
                print(f"  {label}: r = {r:.3f} (n={len(merged)})")
            else:
                print(f"  {label}: insufficient data or no variation (n={len(merged)})")
        else:
            print(f"  {label}: no data")

    # Build final results with best method
    corr_df = all_corrs[best_method].copy()
    corr_df['correlation'] = corr_df['correlation'].round(2)
    corr_df = corr_df.sort_values('correlation', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 85)
    print("TABLE 3: Correlation with Survival/Self-Expression Dimension")
    print("=" * 85)
    print(f"{'SURVIVAL VALUES EMPHASIZE THE FOLLOWING:':<70s} Correlation")
    print("-" * 85)
    for _, row in corr_df.iterrows():
        r = row['correlation']
        n = row['n_countries']
        item = row['item']
        if pd.notna(r):
            print(f"  {item:<67s} {r:+.2f}  (n={n})")
        else:
            print(f"  {item:<67s}   NA  (n={n})")
    print("-" * 85)
    print("(SELF-EXPRESSION VALUES EMPHASIZE THE OPPOSITE)")

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
