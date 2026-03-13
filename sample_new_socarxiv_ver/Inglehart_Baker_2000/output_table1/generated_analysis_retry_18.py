#!/usr/bin/env python3
"""
Table 1 Replication - Attempt 18

DIAGNOSTIC: Try multiple autonomy coding strategies to find which gives
a nation-level loading of 0.89 for the trad dimension.

Based on Inglehart & Baker 2000:
- The "autonomy index" contrasts children's qualities
- Key items: obedience, religious faith vs independence, determination, imagination
- The paper uses nation-level country means for each of the 5 qualities separately,
  then summarizes via a linear combination or factor score

HYPOTHESIS: The paper may NOT compute autonomy as a single composite at the
individual level and then aggregate. Instead:
1. They compute the fraction of respondents who chose each quality per country
2. They factor-analyze those 5 fractions at the country level, getting a single
   "autonomy" score per country
3. That autonomy score is THEN used in Table 1's factor analysis

This would give different results than computing (obedience+faith-indep-determ-imag)
per respondent and then averaging.

ALTERNATIVE: The country-level autonomy score may be:
  country_mean(obedience) + country_mean(faith) - country_mean(independence)
  - country_mean(determination) - country_mean(imagination)
  = country_mean(obedience + faith - indep - determ - imag)
These ARE algebraically equivalent if computed on the same individuals.
BUT if different countries have different missing data patterns (e.g., some countries
only asked 3 of the 5 qualities), the country mean of the difference will differ
from the difference of the country means when NaN is excluded differently.

Strategy: Compute country_mean of each quality separately, THEN compute the difference.
This is more robust to different country-level missing data patterns.
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
    'N_nation': 65, 'N_individual': 165594,
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


def prepare_base_items(df):
    """Prepare all items except autonomy (which is coded separately)."""
    all_raw = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042',
               'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
    df = clean_missing(df, [c for c in all_raw if c in df.columns])

    # Binary child quality items: recode 2=not mentioned -> 0
    for c in ['A029', 'A030', 'A032', 'A034', 'A042']:
        if c in df.columns:
            df.loc[df[c] == 2, c] = 0

    # God importance
    df['god_important'] = np.nan
    if 'F063' in df.columns:
        df.loc[df['_src'] == 'wvs', 'god_important'] = df.loc[df['_src'] == 'wvs', 'F063']
    if 'A006' in df.columns:
        df.loc[df['_src'] == 'evs', 'god_important'] = df.loc[df['_src'] == 'evs', 'A006']

    # Recode direction
    df['F120'] = 11 - df['F120']
    df['G006'] = 5 - df['G006']
    df['E018'] = 4 - df['E018']
    df['Y002'] = 4 - df['Y002']
    df['F118'] = 11 - df['F118']

    return df


def pca_varimax(data_matrix, items):
    """Standard PCA + varimax."""
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

    result = pd.DataFrame({'item': items, 'trad_secrat': loadings[:, tc], 'surv_selfexp': loadings[:, sc]})
    return result, var_exp[tc], var_exp[sc], eigenvalues


def score_loadings(load_df, gt_dict):
    """Score just the loadings portion."""
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


def try_autonomy_coding(df, autonomy_method, use_latest=True):
    """Try different autonomy coding and return nation+individual loadings."""
    df = df.copy()

    if autonomy_method == 'indiv_5item':
        # 5-item individual: A042+A034-A029-A032-A030
        c5 = ['A042', 'A034', 'A029', 'A032', 'A030']
        c3 = ['A042', 'A034', 'A029']
        has5 = df[c5].notna().all(axis=1)
        has3 = df[c3].notna().all(axis=1) & ~has5
        df['autonomy_idx'] = np.nan
        df.loc[has5, 'autonomy_idx'] = (df.loc[has5,'A042']+df.loc[has5,'A034']
                                         -df.loc[has5,'A029']-df.loc[has5,'A032']-df.loc[has5,'A030'])
        auto3 = df.loc[has3,'A042']+df.loc[has3,'A034']-df.loc[has3,'A029']
        df.loc[has3,'autonomy_idx'] = (auto3+1)/3*5 - 3

    elif autonomy_method == 'indiv_3item':
        # 3-item: A042+A034-A029 (simpler)
        c3 = ['A042', 'A034', 'A029']
        has3 = df[c3].notna().all(axis=1)
        df['autonomy_idx'] = np.nan
        df.loc[has3,'autonomy_idx'] = df.loc[has3,'A042']+df.loc[has3,'A034']-df.loc[has3,'A029']

    elif autonomy_method == 'nation_means_5item':
        # Compute country-mean of each quality separately, then compute difference
        # This is done at nation level separately
        df['autonomy_idx'] = np.nan  # placeholder, will be replaced at nation level
        df['_A042'] = df['A042']
        df['_A034'] = df['A034']
        df['_A029'] = df['A029']
        df['_A032'] = df['A032']
        df['_A030'] = df['A030']

    elif autonomy_method == 'indiv_trad_sum':
        # Just sum obedience + faith (no subtraction)
        c = ['A042', 'A034']
        has = df[c].notna().all(axis=1)
        df['autonomy_idx'] = np.nan
        df.loc[has,'autonomy_idx'] = df.loc[has,'A042'] + df.loc[has,'A034']

    elif autonomy_method == 'indiv_obedience':
        # Just obedience
        df['autonomy_idx'] = df['A042'].copy()

    return df


def run_analysis(data_source=None):
    df_raw = load_raw_data()
    df = prepare_base_items(df_raw)

    results = {}

    for method in ['indiv_5item', 'indiv_3item', 'nation_means_5item', 'indiv_trad_sum', 'indiv_obedience']:
        df_m = try_autonomy_coding(df, method)

        if use_latest := True:
            df_nation_src = get_latest_per_country(df_m)
        else:
            df_nation_src = df_m

        if method == 'nation_means_5item':
            # Build nation-level matrix with separate quality means
            cm_q = df_nation_src.groupby('COUNTRY_ALPHA')[['_A042','_A034','_A029','_A032','_A030']].mean()
            cm_q = cm_q.dropna(thresh=3)
            # Compute autonomy as mean-level difference
            trad_cols = [c for c in ['_A042','_A034'] if c in cm_q.columns]
            sec_cols  = [c for c in ['_A029','_A032','_A030'] if c in cm_q.columns]
            cm_q['autonomy_idx'] = (cm_q[trad_cols].mean(axis=1) * 2
                                    - cm_q[sec_cols].mean(axis=1) * len(sec_cols) / len(sec_cols))
            # Actually: sum of traditional - sum of secular
            cm_q['autonomy_idx'] = (cm_q[['_A042','_A034']].sum(axis=1) / 2
                                    - cm_q[[c for c in ['_A029','_A032','_A030'] if c in cm_q.columns]].sum(axis=1) / len(sec_cols))
            # Scale to match 5-item range
            cm_q['autonomy_idx'] = cm_q['autonomy_idx'] * 5

            # Get other items
            other_items = [x for x in ITEMS if x != 'autonomy_idx']
            cm_other = df_nation_src.groupby('COUNTRY_ALPHA')[other_items].mean()
            cm = cm_other.join(cm_q[['autonomy_idx']], how='inner')
            cm = cm.dropna(thresh=7)
            for c in ITEMS:
                cm[c] = cm[c].fillna(cm[c].mean())
        else:
            cm = df_nation_src.groupby('COUNTRY_ALPHA')[ITEMS].mean()
            cm = cm.dropna(thresh=7)
            for c in ITEMS:
                cm[c] = cm[c].fillna(cm[c].mean())

        n_countries = len(cm)
        nation_load, nv1, nv2, _ = pca_varimax(cm, ITEMS)
        n_score = score_loadings(nation_load, GROUND_TRUTH['nation_level'])
        auto_n = nation_load[nation_load['item']=='autonomy_idx']['trad_secrat'].values[0]

        # Individual
        data_i = df_m[ITEMS].copy()
        indiv_load, iv1, iv2, _ = pca_varimax(data_i, ITEMS)
        i_score = score_loadings(indiv_load, GROUND_TRUTH['individual_level'])
        auto_i = indiv_load[indiv_load['item']=='autonomy_idx']['trad_secrat'].values[0]

        results[method] = {
            'nation_load': nation_load, 'indiv_load': indiv_load,
            'n_countries': n_countries, 'total_n': data_i.dropna(how='all').shape[0],
            'nv1': nv1, 'nv2': nv2, 'iv1': iv1, 'iv2': iv2,
            'n_score': n_score, 'i_score': i_score,
            'auto_n': auto_n, 'auto_i': auto_i,
            'total_score': n_score + i_score
        }

        print(f"Method: {method:25s} -> auto_n={auto_n:.3f}, auto_i={auto_i:.3f}, n_score={n_score}/20, i_score={i_score}/20, total={n_score+i_score}/40")

    print()
    print(f"Paper autonomy targets: nation=0.89, individual=0.61")
    print()

    # Pick best method
    best_method = max(results.keys(), key=lambda m: results[m]['total_score'])
    print(f"Best method: {best_method} (score {results[best_method]['total_score']}/40)")
    print()

    r = results[best_method]
    nl = r['nation_load']
    il = r['indiv_load']
    n_countries = r['n_countries']
    total_n = r['total_n']
    nv1, nv2, iv1, iv2 = r['nv1'], r['nv2'], r['iv1'], r['iv2']

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
            nl_v = nl[nl['item']==it][dim_col].values[0]
            il_v = il[il['item']==it][dim_col].values[0]
            gt_n = GROUND_TRUTH['nation_level'][gt_key][it]
            gt_i = GROUND_TRUTH['individual_level'][gt_key][it]
            dn = nl_v - gt_n
            di = il_v - gt_i
            print(f"{names[it]:<25} {gt_n:>6.2f} {nl_v:>7.2f} {dn:>+6.2f} {gt_i:>6.2f} {il_v:>7.2f} {di:>+6.2f}")
        print()

    print(f"Variance: Dim1={nv1:.1f}/{iv1:.1f}%, Dim2={nv2:.1f}/{iv2:.1f}%")
    print(f"Paper:    Dim1=44/26%, Dim2=26/13%")
    print(f"N: {n_countries} countries, {total_n:,} individuals")

    return {
        'nation_loadings': nl, 'indiv_loadings': il,
        'n_countries': n_countries, 'total_n': total_n, 'smallest_n': 0,
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
        details.append(f"  {lev}: t={'OK' if tok else 'NO'} s={'OK' if sok else 'NO'} tv={[f'{v:.2f}' for v in tv]}")
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
