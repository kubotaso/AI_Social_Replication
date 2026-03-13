#!/usr/bin/env python3
"""
Table 1 Replication - Attempt 16

FUNDAMENTAL STRATEGY CHANGE: Address the autonomy loading problem.

Analysis of the problem:
- Best so far (attempt 15): autonomy nation=0.67 (paper=0.89), indiv=0.49 (paper=0.61)
- The autonomy index must have a much higher correlation with god_important,
  F120, G006, E018 at the country level to get 0.89 loading.

Root cause hypothesis:
The paper likely uses the 4-item autonomy index: obedience + faith - independence - determination
In the WVS longitudinal file, A032 (determination) is available in waves 2-3.
The 4-item version contrasts FOUR values (2 traditional vs 2 modern), giving
more reliable discrimination between societies.

Key change: Use proper 4-item index A042+A034-A029-A032 where all 4 available.
For EVS/missing cases, scale proportionally.

Additional change: The wave 2 countries span 1990-1994, wave 3 spans 1995-1998.
The paper uses the surveys collectively as a pool. We should include ALL observations
from waves 2-3 (not just the latest wave) when computing country means, similar to
how the paper pools all available observations for each country.

Also: Try using ALL observations (not just latest wave) for country means,
as the paper pools waves 2 and 3 surveys from all participating countries.
"""

import sys
import os
import csv
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from shared_factor_analysis import clean_missing, varimax

ITEMS = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
         'Y002', 'A008', 'E025', 'F118', 'A165']

EXCLUDE = ['MNE', 'MLT']

GROUND_TRUTH = {
    'nation_level': {
        'trad_secrat': {
            'god_important': 0.91, 'autonomy_idx': 0.89, 'F120': 0.82,
            'G006': 0.82, 'E018': 0.72,
        },
        'surv_selfexp': {
            'Y002': 0.86, 'A008': 0.81, 'E025': 0.80, 'F118': 0.78, 'A165': 0.56,
        }
    },
    'individual_level': {
        'trad_secrat': {
            'god_important': 0.70, 'autonomy_idx': 0.61, 'F120': 0.61,
            'G006': 0.60, 'E018': 0.51,
        },
        'surv_selfexp': {
            'Y002': 0.59, 'A008': 0.58, 'E025': 0.59, 'F118': 0.54, 'A165': 0.44,
        }
    },
    'variance_explained': {
        'nation_dim1': 44, 'nation_dim2': 26,
        'indiv_dim1': 26, 'indiv_dim2': 13
    },
    'N_nation': 65,
    'N_individual': 165594,
}


def load_raw_data():
    """Load raw WVS waves 2+3 and EVS 1990 data."""
    WVS_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
    EVS_PATH = os.path.join(BASE, "data/EVS_1990_wvs_format.csv")

    cols_needed = ['S002VS', 'COUNTRY_ALPHA', 'S020',
                   'A006', 'A008', 'A029', 'A030', 'A032', 'A034', 'A042',
                   'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002']

    with open(WVS_PATH, 'r') as f:
        header = [h.strip('"') for h in next(csv.reader(f))]
    avail = [c for c in cols_needed if c in header]

    wvs = pd.read_csv(WVS_PATH, usecols=avail, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]
    wvs['_src'] = 'wvs'

    evs = pd.read_csv(EVS_PATH)
    evs['_src'] = 'evs'

    df = pd.concat([wvs, evs], ignore_index=True, sort=False)
    df = df[~df['COUNTRY_ALPHA'].isin(EXCLUDE)]
    return df


def prepare_items(df):
    """
    Construct 10 factor items with careful variable coding.

    Key decisions:
    1. God importance: F063 for WVS (10-point), A006 for EVS 1990 (10-point)
    2. Autonomy: 4-item when A032 available: A042+A034-A029-A032
                 3-item fallback: A042+A034-A029 (then rescale to 4-item equivalent)
    3. Binary child quality items: recode 2->0 (not mentioned), keep 1 (mentioned)
    """
    all_raw = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042',
               'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
    df = clean_missing(df, [c for c in all_raw if c in df.columns])

    # Recode binary child quality items: 2=not mentioned -> 0
    for c in ['A029', 'A030', 'A032', 'A034', 'A042']:
        if c in df.columns:
            df.loc[df[c] == 2, c] = 0

    # GOD IMPORTANCE (higher = more important = traditional)
    df['god_important'] = np.nan
    if 'F063' in df.columns:
        df.loc[df['_src'] == 'wvs', 'god_important'] = df.loc[df['_src'] == 'wvs', 'F063']
    if 'A006' in df.columns:
        df.loc[df['_src'] == 'evs', 'god_important'] = df.loc[df['_src'] == 'evs', 'A006']

    # AUTONOMY INDEX (higher = more traditional = obedience/faith vs independence/determination)
    # 4-item: A042(obedience) + A034(faith) - A029(independence) - A032(determination)
    # Range: -2 to +2
    # 3-item: A042 + A034 - A029, range: -1 to +2, rescale to [-2, +2]
    child_4 = ['A042', 'A034', 'A029', 'A032']
    child_3 = ['A042', 'A034', 'A029']

    # Check availability
    has4_cols = all(c in df.columns for c in child_4)
    has3_cols = all(c in df.columns for c in child_3)

    df['autonomy_idx'] = np.nan
    if has4_cols:
        has4 = df[child_4].notna().all(axis=1)
        df.loc[has4, 'autonomy_idx'] = (
            df.loc[has4, 'A042'] + df.loc[has4, 'A034']
            - df.loc[has4, 'A029'] - df.loc[has4, 'A032']
        )

    if has3_cols:
        has3_only = df[child_3].notna().all(axis=1)
        if has4_cols:
            has3_only = has3_only & df['A032'].isna()
        auto3 = df.loc[has3_only, 'A042'] + df.loc[has3_only, 'A034'] - df.loc[has3_only, 'A029']
        # Rescale [-1,+2] to [-2,+2]: multiply by 4/3 and shift
        # 4-item range is [-2,+2] = span of 4
        # 3-item range is [-1,+2] = span of 3
        # Rescale: (auto3 - (-1)) / 3 * 4 + (-2) = (auto3+1)/3*4 - 2
        df.loc[has3_only, 'autonomy_idx'] = (auto3 + 1) / 3 * 4 - 2

    # RECODE: higher = traditional (dimension 1) or survival (dimension 2)
    df['F120'] = 11 - df['F120']   # abortion: 1=never->10, 10=always->1
    df['G006'] = 5 - df['G006']    # national pride: 1=very proud->4, 4=not proud->1
    df['E018'] = 4 - df['E018']    # authority: 1=good->3, 3=bad->1
    df['Y002'] = 4 - df['Y002']    # postmat: 1=mat->3, 3=postmat->1 (survival=high)
    df['F118'] = 11 - df['F118']   # homo: 1=never->10, 10=always->1

    return df


def pca_varimax(data_matrix, items):
    """Standard PCA + varimax rotation on a data matrix (N x p)."""
    # Use correlation matrix approach
    if isinstance(data_matrix, pd.DataFrame):
        corr = data_matrix.corr().values
        n = len(data_matrix)
    else:
        corr = np.corrcoef(data_matrix.T)
        n = data_matrix.shape[0]

    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    loadings = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    loadings, _ = varimax(loadings)

    var_exp = (loadings ** 2).sum(axis=0) / len(items) * 100

    trad_idx = [items.index(x) for x in ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']]
    surv_idx = [items.index(x) for x in ['Y002', 'A008', 'E025', 'F118', 'A165']]

    f1t = sum(abs(loadings[i, 0]) for i in trad_idx)
    f2t = sum(abs(loadings[i, 1]) for i in trad_idx)
    tc = 0 if f1t > f2t else 1
    sc = 1 - tc

    if np.mean([loadings[i, tc] for i in trad_idx]) < 0:
        loadings[:, tc] *= -1
    if np.mean([loadings[i, sc] for i in surv_idx]) < 0:
        loadings[:, sc] *= -1

    result = pd.DataFrame({
        'item': items,
        'trad_secrat': loadings[:, tc],
        'surv_selfexp': loadings[:, sc],
    })
    return result, var_exp[tc], var_exp[sc], eigenvalues


def run_analysis(data_source=None):
    """Run the full analysis."""
    df = load_raw_data()
    df = prepare_items(df)

    # -----------------------------------------------------------------------
    # NATION-LEVEL ANALYSIS
    # Use ALL observations from waves 2+3 per country (not just latest wave)
    # to maximize the number of respondents contributing to each country mean.
    # This is consistent with how Inglehart & Baker likely computed country means.
    # -----------------------------------------------------------------------
    cm = df.groupby('COUNTRY_ALPHA')[ITEMS].mean()
    cm = cm.dropna(thresh=7)  # require at least 7 of 10 items
    # Fill remaining missing with column mean
    for c in ITEMS:
        if c in cm.columns:
            cm[c] = cm[c].fillna(cm[c].mean())
    n_countries = len(cm)

    nation_load, nv1, nv2, neig = pca_varimax(cm, ITEMS)

    # -----------------------------------------------------------------------
    # INDIVIDUAL-LEVEL ANALYSIS
    # All respondents with valid data on at least some items
    # Use listwise deletion (complete cases) for the correlation matrix
    # -----------------------------------------------------------------------
    data_i = df[ITEMS].copy()
    total_n = data_i.dropna(how='all').shape[0]
    smallest_n = data_i.count().min()

    # PCA on complete cases
    complete = data_i.dropna()
    indiv_load, iv1, iv2, ieig = pca_varimax(complete.values, ITEMS)

    # PRINT RESULTS
    print(f"N countries: {n_countries}  (paper: 65)")
    print(f"N individual total: {total_n:,}")
    print(f"N individual complete: {len(complete):,}  (paper min: 146,789)")
    print()

    names = {
        'god_important': "God important",
        'autonomy_idx': "Autonomy index",
        'F120': 'Abortion never just.',
        'G006': 'National pride',
        'E018': 'Respect authority',
        'Y002': 'Materialist',
        'A008': 'Not happy',
        'E025': 'No petition',
        'F118': 'Homo never just.',
        'A165': 'Careful trust',
    }

    print(f"{'Item':<25} {'PaprN':>6} {'Nation':>7} {'DiffN':>6} | {'PaprI':>6} {'Indiv':>7} {'DiffI':>6}")
    print("-" * 68)
    for dim_col, its, gt_k in [('trad_secrat', ITEMS[:5], 'trad_secrat'),
                                ('surv_selfexp', ITEMS[5:], 'surv_selfexp')]:
        for it in its:
            nl = nation_load[nation_load['item'] == it][dim_col].values[0]
            il = indiv_load[indiv_load['item'] == it][dim_col].values[0]
            gt_n = GROUND_TRUTH['nation_level'][gt_k][it]
            gt_i = GROUND_TRUTH['individual_level'][gt_k][it]
            print(f"{names[it]:<25} {gt_n:>6.2f} {nl:>7.2f} {nl-gt_n:>+6.3f} | {gt_i:>6.2f} {il:>7.2f} {il-gt_i:>+6.3f}")
        print()

    print(f"Variance (nation):  Dim1={nv1:.1f}%  Dim2={nv2:.1f}%  (paper: 44/26%)")
    print(f"Variance (indiv):   Dim1={iv1:.1f}%  Dim2={iv2:.1f}%  (paper: 26/13%)")

    return {
        'nation_loadings': nation_load,
        'indiv_loadings': indiv_load,
        'n_countries': n_countries,
        'total_n': total_n,
        'smallest_n': int(smallest_n),
        'n_complete': len(complete),
        'nation_var1': nv1,
        'nation_var2': nv2,
        'indiv_var1': iv1,
        'indiv_var2': iv2,
    }


def score_against_ground_truth():
    """Compute automated score against paper's Table 1 values."""
    r = run_analysis()
    nl = r['nation_loadings']
    il = r['indiv_loadings']
    score = 0
    details = []

    # --- 1. Items present (20 pts) ---
    if all(i in nl['item'].values for i in ITEMS):
        score += 20
        details.append("Items present: +20/20")
    else:
        missing_items = [i for i in ITEMS if i not in nl['item'].values]
        details.append(f"Items MISSING: {missing_items} (+0/20)")

    # --- 2. Loading values (40 pts, 2 pts per loading) ---
    loading_score = 0
    match_c, partial_c, miss_c = 0, 0, 0
    pts_per = 2

    for level, ldf, key in [('nation', nl, 'nation_level'), ('indiv', il, 'individual_level')]:
        gt = GROUND_TRUTH[key]
        for dim_col, its, gt_k in [('trad_secrat', ITEMS[:5], 'trad_secrat'),
                                    ('surv_selfexp', ITEMS[5:], 'surv_selfexp')]:
            for it in its:
                gv = gt[gt_k][it]
                gen = ldf[ldf['item'] == it][dim_col].values[0]
                d = abs(gv - gen)
                if d <= 0.03:
                    loading_score += pts_per
                    match_c += 1
                    tag = "MATCH"
                elif d <= 0.06:
                    loading_score += pts_per * 0.5
                    partial_c += 1
                    tag = "PARTIAL"
                else:
                    miss_c += 1
                    tag = "MISS"
                details.append(
                    f"  {level:6s} {it:<14} paper={gv:.2f} gen={gen:.2f} d={d:.3f} {tag}"
                )

    score += loading_score
    details.append(
        f"Loadings: +{loading_score:.1f}/40 ({match_c} MATCH, {partial_c} PARTIAL, {miss_c} MISS)"
    )

    # --- 3. Correct dimension assignment (20 pts, 1 pt each) ---
    dim_score = 0
    for level, ldf in [('nation', nl), ('indiv', il)]:
        correct = 0
        for it in ITEMS[:5]:
            if (abs(ldf[ldf['item'] == it]['trad_secrat'].values[0]) >
                    abs(ldf[ldf['item'] == it]['surv_selfexp'].values[0])):
                correct += 1
        for it in ITEMS[5:]:
            if (abs(ldf[ldf['item'] == it]['surv_selfexp'].values[0]) >
                    abs(ldf[ldf['item'] == it]['trad_secrat'].values[0])):
                correct += 1
        dim_score += correct
        details.append(f"  Dimension {level}: {correct}/10")

    score += dim_score
    details.append(f"Dimensions: +{dim_score}/20")

    # --- 4. Ordering within each dimension (10 pts) ---
    ord_score = 0
    for level, ldf in [('nation', nl), ('indiv', il)]:
        tv = [ldf[ldf['item'] == i]['trad_secrat'].values[0] for i in ITEMS[:5]]
        sv = [ldf[ldf['item'] == i]['surv_selfexp'].values[0] for i in ITEMS[5:]]
        tok = all(tv[j] >= tv[j + 1] for j in range(4))
        sok = all(sv[j] >= sv[j + 1] for j in range(4))
        if tok:
            ord_score += 2.5
        if sok:
            ord_score += 2.5
        details.append(f"  Order {level}: trad={'OK' if tok else 'NO'} surv={'OK' if sok else 'NO'}")
        details.append(f"    trad: {[f'{v:.2f}' for v in tv]}")
        details.append(f"    surv: {[f'{v:.2f}' for v in sv]}")

    score += ord_score
    details.append(f"Ordering: +{ord_score:.1f}/10")

    # --- 5. Sample size and variance explained (10 pts) ---
    ss = 0
    n = r['n_countries']
    if abs(n - 65) / 65 <= 0.05:
        ss += 2
        details.append(f"  N countries={n} OK (+2)")
    elif abs(n - 65) / 65 <= 0.15:
        ss += 1
        details.append(f"  N countries={n} approx (+1)")
    else:
        details.append(f"  N countries={n} MISS (+0)")

    for nm, gn, gt in [
        ('nation_var1', r['nation_var1'], 44),
        ('nation_var2', r['nation_var2'], 26),
        ('indiv_var1', r['indiv_var1'], 26),
        ('indiv_var2', r['indiv_var2'], 13),
    ]:
        d = abs(gn - gt)
        if d <= 3:
            ss += 2
            details.append(f"  {nm}={gn:.1f}% OK (+2)")
        elif d <= 6:
            ss += 1
            details.append(f"  {nm}={gn:.1f}% approx (+1)")
        else:
            details.append(f"  {nm}={gn:.1f}% MISS (+0)")

    score += ss
    details.append(f"Size/Variance: +{ss}/10")

    total = min(100, round(score))
    print("\n" + "=" * 55)
    print("SCORING BREAKDOWN")
    print("=" * 55)
    for d in details:
        print(d)
    print(f"\nFINAL SCORE: {total}/100")
    return total, details


if __name__ == "__main__":
    score_against_ground_truth()
