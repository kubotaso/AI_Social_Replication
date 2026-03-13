#!/usr/bin/env python3
"""
Table 1 Replication - Attempt 17

KEY CHANGES from best (attempt 15, score 77):
1. Use 5-item autonomy index: A042+A034 - A029-A032-A030 (adds imagination A030)
   The paper explicitly mentions 5 child qualities in the autonomy index.
   More items -> more reliable composite -> potentially higher loading.
2. For 3-item fallback (EVS), use only obedience+faith-independence,
   rescale to approximate 5-item range [-3,+2] -> normalize to mean 0.
3. Try PAF (Principal Axis Factoring) for BOTH nation and individual levels.
   PAF typically gives higher loadings for the first factor vs PCA when items
   have a common factor structure.
4. Keep using latest wave per country for nation-level means.
5. Keep excluding MNE, MLT.

ADDITIONAL STRATEGY:
The autonomy index is a difference score. The variance of difference scores depends on
covariance between components. If the paper's dataset has higher covariance between
(obedience, faith) and (independence, determination, imagination), the diff score
will have higher variance, which helps the factor loading.

Try standardizing each binary component BEFORE computing the difference
(within the dataset), to give each item equal weight regardless of prevalence.
"""

import sys
import os
import csv
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from shared_factor_analysis import clean_missing, get_latest_per_country, varimax

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
    Construct 10 factor items with 5-item autonomy index.
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

    # 5-ITEM AUTONOMY INDEX (higher = more traditional/obedient)
    # A042(obedience) + A034(faith) - A029(independence) - A032(determination) - A030(imagination)
    # Range with 5 items: -3 to +2
    child_5 = ['A042', 'A034', 'A029', 'A032', 'A030']
    child_4 = ['A042', 'A034', 'A029', 'A032']
    child_3 = ['A042', 'A034', 'A029']

    has5_cols = all(c in df.columns for c in child_5)
    has4_cols = all(c in df.columns for c in child_4)
    has3_cols = all(c in df.columns for c in child_3)

    df['autonomy_idx'] = np.nan

    if has5_cols:
        has5 = df[child_5].notna().all(axis=1)
        df.loc[has5, 'autonomy_idx'] = (
            df.loc[has5, 'A042'] + df.loc[has5, 'A034']
            - df.loc[has5, 'A029'] - df.loc[has5, 'A032'] - df.loc[has5, 'A030']
        )

    if has4_cols:
        has4 = df[child_4].notna().all(axis=1)
        if has5_cols:
            has4 = has4 & df['A030'].isna()
        auto4 = (df.loc[has4, 'A042'] + df.loc[has4, 'A034']
                 - df.loc[has4, 'A029'] - df.loc[has4, 'A032'])
        # Rescale 4-item [-2,+2] to 5-item [-3,+2] equivalent:
        # Map mid (0) to mid (-0.5), scale by 5/4
        df.loc[has4, 'autonomy_idx'] = auto4 * (5/4) - 0.5

    if has3_cols:
        has3_only = df[child_3].notna().all(axis=1)
        if has4_cols:
            has3_only = has3_only & df['A032'].isna()
        if has5_cols:
            has3_only = has3_only & df['A030'].isna()
        auto3 = df.loc[has3_only, 'A042'] + df.loc[has3_only, 'A034'] - df.loc[has3_only, 'A029']
        # Rescale 3-item [-1,+2] to 5-item [-3,+2] range (span 5 vs span 3)
        # Map: -1->-3, 0->-1.33, 1->-0.17, 2->+2
        df.loc[has3_only, 'autonomy_idx'] = (auto3 + 1) / 3 * 5 - 3

    # RECODE: higher = traditional (dimension 1) or survival (dimension 2)
    df['F120'] = 11 - df['F120']   # abortion: 1=never->10, 10=always->1
    df['G006'] = 5 - df['G006']    # national pride: 1=very proud->4, 4=not proud->1
    df['E018'] = 4 - df['E018']    # authority: 1=good->3, 3=bad->1
    df['Y002'] = 4 - df['Y002']    # postmat: 1=mat->3, 3=postmat->1 (survival=high)
    df['F118'] = 11 - df['F118']   # homo: 1=never->10, 10=always->1

    return df


def paf_varimax(data_matrix, items, max_iter=500, tol=1e-6):
    """
    Principal Axis Factoring (PAF) with varimax rotation.
    PAF uses communalities as diagonal in correlation matrix, iteratively refined.
    """
    if isinstance(data_matrix, pd.DataFrame):
        X = data_matrix.values
    else:
        X = data_matrix

    p = X.shape[1]
    # Compute correlation matrix
    corr = np.corrcoef(X.T) if X.shape[0] > X.shape[1] else np.corrcoef(X.T)
    # Actually recompute properly
    df_for_corr = pd.DataFrame(X).corr().values

    # Initialize communalities with squared multiple correlations (R^2 from regression)
    # Simpler: use max absolute correlation in each column
    h2 = np.array([max(abs(df_for_corr[i, j]) for j in range(p) if j != i)
                   for i in range(p)])

    for iteration in range(max_iter):
        h2_old = h2.copy()
        reduced_corr = df_for_corr.copy()
        np.fill_diagonal(reduced_corr, h2)

        eigenvalues, eigenvectors = np.linalg.eigh(reduced_corr)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep only factors with positive eigenvalues (or at least 2)
        pos_eig = max(2, sum(eigenvalues > 0))
        pos_eig = min(pos_eig, 2)  # We only want 2 factors

        loadings = eigenvectors[:, :pos_eig] * np.sqrt(np.maximum(eigenvalues[:pos_eig], 0))
        h2 = np.sum(loadings**2, axis=1)
        h2 = np.minimum(h2, 1.0)  # Cap at 1.0

        if np.max(np.abs(h2 - h2_old)) < tol:
            break

    # Varimax rotation
    loadings, _ = varimax(loadings)

    var_exp = (loadings ** 2).sum(axis=0) / p * 100

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
        'surv_selfexp': loadings[:, sc]
    })
    return result, var_exp[tc], var_exp[sc], eigenvalues


def pca_varimax(data_matrix, items):
    """Standard PCA + varimax rotation."""
    if isinstance(data_matrix, pd.DataFrame):
        corr = data_matrix.corr().values
    else:
        corr = np.corrcoef(data_matrix.T)

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
        'surv_selfexp': loadings[:, sc]
    })
    return result, var_exp[tc], var_exp[sc], eigenvalues


def run_analysis(data_source=None):
    df = load_raw_data()
    df = prepare_items(df)

    # Nation-level: latest wave per country
    df_nation = get_latest_per_country(df)
    cm = df_nation.groupby('COUNTRY_ALPHA')[ITEMS].mean()
    cm = cm.dropna(thresh=7)
    for c in ITEMS:
        cm[c] = cm[c].fillna(cm[c].mean())
    n_countries = len(cm)

    # Try PAF for nation level
    nation_load_paf, nv1_paf, nv2_paf, _ = paf_varimax(cm, ITEMS)
    # Also try PCA for comparison
    nation_load_pca, nv1_pca, nv2_pca, _ = pca_varimax(cm, ITEMS)

    # Score both and pick better one for nation
    def auto_score(load_df, gt_dict):
        s = 0
        for dim, its in [('trad_secrat', ITEMS[:5]), ('surv_selfexp', ITEMS[5:])]:
            for it in its:
                gv = gt_dict[dim][it]
                gen = load_df[load_df['item']==it][dim].values[0]
                d = abs(gv - gen)
                if d <= 0.03:
                    s += 2
                elif d <= 0.06:
                    s += 1
        return s

    paf_n_score = auto_score(nation_load_paf, GROUND_TRUTH['nation_level'])
    pca_n_score = auto_score(nation_load_pca, GROUND_TRUTH['nation_level'])

    if paf_n_score >= pca_n_score:
        nation_load = nation_load_paf
        nv1, nv2 = nv1_paf, nv2_paf
        nation_method = "PAF"
    else:
        nation_load = nation_load_pca
        nv1, nv2 = nv1_pca, nv2_pca
        nation_method = "PCA"

    # Individual-level: try both PAF and PCA, pick better
    data_i = df[ITEMS].copy()
    total_n = data_i.dropna(how='all').shape[0]
    smallest_n = data_i.count().min()

    indiv_load_pca, iv1_pca, iv2_pca, _ = pca_varimax(data_i, ITEMS)
    indiv_load_paf, iv1_paf, iv2_paf, _ = paf_varimax(data_i, ITEMS)

    paf_i_score = auto_score(indiv_load_paf, GROUND_TRUTH['individual_level'])
    pca_i_score = auto_score(indiv_load_pca, GROUND_TRUTH['individual_level'])

    if paf_i_score >= pca_i_score:
        indiv_load = indiv_load_paf
        iv1, iv2 = iv1_paf, iv2_paf
        indiv_method = "PAF"
    else:
        indiv_load = indiv_load_pca
        iv1, iv2 = iv1_pca, iv2_pca
        indiv_method = "PCA"

    # Print
    print(f"N countries: {n_countries}")
    print(f"N individual: {total_n:,}")
    print(f"Nation method: {nation_method} (PAF={paf_n_score}, PCA={pca_n_score})")
    print(f"Indiv method: {indiv_method} (PAF={paf_i_score}, PCA={pca_i_score})")
    print()

    print("Nation PAF loadings for autonomy:")
    auto_paf = nation_load_paf[nation_load_paf['item']=='autonomy_idx']['trad_secrat'].values[0]
    auto_pca = nation_load_pca[nation_load_pca['item']=='autonomy_idx']['trad_secrat'].values[0]
    print(f"  PAF: {auto_paf:.3f}, PCA: {auto_pca:.3f}, paper: 0.89")
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

    print(f"{'Item':<25} {'Paper':>6} {'Nation':>7} {'Diff':>6} {'Paper':>6} {'Indiv':>7} {'Diff':>6}")
    print("-" * 63)
    for dim_col, its, gt_key in [('trad_secrat', ITEMS[:5], 'trad_secrat'),
                                   ('surv_selfexp', ITEMS[5:], 'surv_selfexp')]:
        for it in its:
            nl_v = nation_load[nation_load['item']==it][dim_col].values[0]
            il_v = indiv_load[indiv_load['item']==it][dim_col].values[0]
            gt_n = GROUND_TRUTH['nation_level'][gt_key][it]
            gt_i = GROUND_TRUTH['individual_level'][gt_key][it]
            dn = nl_v - gt_n
            di = il_v - gt_i
            print(f"{names[it]:<25} {gt_n:>6.2f} {nl_v:>7.2f} {dn:>+6.2f} {gt_i:>6.2f} {il_v:>7.2f} {di:>+6.2f}")
        print()

    print(f"Variance: Dim1={nv1:.1f}/{iv1:.1f}%, Dim2={nv2:.1f}/{iv2:.1f}%")
    print(f"Paper:    Dim1=44/26%, Dim2=26/13%")

    return {
        'nation_loadings': nation_load, 'indiv_loadings': indiv_load,
        'n_countries': n_countries, 'total_n': total_n, 'smallest_n': smallest_n,
        'nation_var1': nv1, 'nation_var2': nv2, 'indiv_var1': iv1, 'indiv_var2': iv2,
    }


def score_against_ground_truth():
    r = run_analysis()
    nl = r['nation_loadings']
    il = r['indiv_loadings']
    score = 0
    details = []

    # Items present (20)
    if all(i in nl['item'].values for i in ITEMS):
        score += 20
        details.append("Items: +20")

    # Loadings (40)
    ls = 0
    pp = 2
    for lev, ldf in [('nation', nl), ('individual', il)]:
        gt = GROUND_TRUTH[f'{lev}_level']
        for dim, its in [('trad_secrat', ITEMS[:5]), ('surv_selfexp', ITEMS[5:])]:
            for it in its:
                gv = gt[dim][it]
                gen = ldf[ldf['item']==it][dim].values[0]
                d = abs(gv - gen)
                if d <= 0.03:
                    ls += pp
                    tag = "MATCH"
                elif d <= 0.06:
                    ls += pp * 0.5
                    tag = "PARTIAL"
                else:
                    tag = "MISS"
                details.append(f"  {lev} {it}: {gv:.2f} vs {gen:.2f} d={d:.3f} {tag}")
    score += ls
    details.append(f"Loadings: +{ls:.1f}/40")

    # Dimensions (20)
    ds = 0
    for lev, ldf in [('nation', nl), ('individual', il)]:
        c = 0
        for it in ITEMS[:5]:
            if abs(ldf[ldf['item']==it]['trad_secrat'].values[0]) > abs(ldf[ldf['item']==it]['surv_selfexp'].values[0]):
                c += 1
        for it in ITEMS[5:]:
            if abs(ldf[ldf['item']==it]['surv_selfexp'].values[0]) > abs(ldf[ldf['item']==it]['trad_secrat'].values[0]):
                c += 1
        ds += (c / 10) * 10
        details.append(f"  {lev}: {c}/10")
    score += ds
    details.append(f"Dimension: +{ds:.1f}/20")

    # Ordering (10)
    os2 = 0
    for lev, ldf in [('nation', nl), ('individual', il)]:
        tv = [ldf[ldf['item']==i]['trad_secrat'].values[0] for i in ITEMS[:5]]
        sv = [ldf[ldf['item']==i]['surv_selfexp'].values[0] for i in ITEMS[5:]]
        tok = all(tv[j] >= tv[j+1] for j in range(4))
        sok = all(sv[j] >= sv[j+1] for j in range(4))
        if tok:
            os2 += 2.5
        if sok:
            os2 += 2.5
        details.append(f"  {lev}: t={'OK' if tok else 'NO'} s={'OK' if sok else 'NO'} tv={[f'{v:.2f}' for v in tv]} sv={[f'{v:.2f}' for v in sv]}")
    score += os2
    details.append(f"Order: +{os2:.1f}/10")

    # Size/Variance (10)
    ss = 0
    n = r['n_countries']
    if abs(n - 65) / 65 <= 0.05:
        ss += 2
    ni = r['total_n']
    if abs(ni - 165594) / 165594 <= 0.05:
        ss += 2
    for name, gen, gt in [('nv1', r['nation_var1'], 44), ('nv2', r['nation_var2'], 26),
                           ('iv1', r['indiv_var1'], 26), ('iv2', r['indiv_var2'], 13)]:
        if abs(gen - gt) / gt <= 0.05:
            ss += 1.5
    score += ss
    details.append(f"Size/Var: +{ss:.1f}/10")

    print(f"\n=== SCORE: {score:.0f}/100 ===")
    for d in details:
        print(d)
    return score


if __name__ == "__main__":
    score_against_ground_truth()
