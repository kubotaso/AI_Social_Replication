#!/usr/bin/env python3
"""
Table 1 Replication - Attempt 4

Key changes:
1. Autonomy index: Use obedience + religious_faith - independence (3 items, available for all countries)
   For countries WITH determination data (A032), use 4-item version
   For countries WITHOUT, use 3-item version rescaled to match 4-item range
2. Proper god_important: WVS=F063, EVS=A006
3. Individual level: all respondents from waves 2-3
4. Try to improve by better handling of missing data in autonomy
"""

import sys, os, csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import clean_missing, get_latest_per_country, varimax

ITEMS = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
         'Y002', 'A008', 'E025', 'F118', 'A165']

GROUND_TRUTH = {
    'nation_level': {
        'trad_secrat': {
            'god_important': 0.91, 'autonomy_idx': 0.89, 'F120': 0.82, 'G006': 0.82, 'E018': 0.72,
        },
        'surv_selfexp': {
            'Y002': 0.86, 'A008': 0.81, 'E025': 0.80, 'F118': 0.78, 'A165': 0.56,
        }
    },
    'individual_level': {
        'trad_secrat': {
            'god_important': 0.70, 'autonomy_idx': 0.61, 'F120': 0.61, 'G006': 0.60, 'E018': 0.51,
        },
        'surv_selfexp': {
            'Y002': 0.59, 'A008': 0.58, 'E025': 0.59, 'F118': 0.54, 'A165': 0.44,
        }
    },
    'variance_explained': {'nation_dim1': 44, 'nation_dim2': 26, 'indiv_dim1': 26, 'indiv_dim2': 13},
    'N_nation': 65, 'N_individual': 165594,
}


def load_data():
    """Load WVS + EVS data with source tagging."""
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WVS_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
    EVS_PATH = os.path.join(BASE, "data/EVS_1990_wvs_format.csv")

    cols_wvs = ['S002VS', 'COUNTRY_ALPHA', 'S020',
                'A006', 'A008', 'A029', 'A032', 'A034', 'A042',
                'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002']

    with open(WVS_PATH, 'r') as f:
        header = [h.strip('"') for h in next(csv.reader(f))]
    available = [c for c in cols_wvs if c in header]

    wvs = pd.read_csv(WVS_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]
    wvs['_src'] = 'wvs'

    evs = pd.read_csv(EVS_PATH)
    evs['_src'] = 'evs'

    df = pd.concat([wvs, evs], ignore_index=True, sort=False)
    return df


def prepare_items(df):
    """Construct and recode all 10 items."""
    # Clean missing
    all_vars = ['A006', 'F063', 'A029', 'A032', 'A034', 'A042',
                'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
    df = clean_missing(df, [c for c in all_vars if c in df.columns])

    # Normalize child quality items to 0/1
    for c in ['A029', 'A032', 'A034', 'A042']:
        if c in df.columns:
            df.loc[df[c] == 2, c] = 0

    # God importance
    df['god_important'] = np.nan
    wvs_mask = df['_src'] == 'wvs'
    evs_mask = df['_src'] == 'evs'
    df.loc[wvs_mask, 'god_important'] = df.loc[wvs_mask, 'F063']
    df.loc[evs_mask, 'god_important'] = df.loc[evs_mask, 'A006']

    # Autonomy index: try 4-item where possible, 3-item otherwise (rescaled)
    # 4-item: (obedience + faith) - (independence + determination): range -2 to +2
    # 3-item: (obedience + faith) - independence: range -1 to +2
    # To make comparable: rescale 3-item to match 4-item range
    df['autonomy_idx'] = np.nan

    has_4 = df[['A042', 'A034', 'A029', 'A032']].notna().all(axis=1)
    has_3 = df[['A042', 'A034', 'A029']].notna().all(axis=1) & ~has_4

    # 4-item version
    df.loc[has_4, 'autonomy_idx'] = (
        df.loc[has_4, 'A042'] + df.loc[has_4, 'A034']
        - df.loc[has_4, 'A029'] - df.loc[has_4, 'A032']
    )

    # 3-item version (rescaled to -2,+2 range)
    auto_3 = df.loc[has_3, 'A042'] + df.loc[has_3, 'A034'] - df.loc[has_3, 'A029']
    # 3-item range: -1 to +2, center=0.5. 4-item range: -2 to +2, center=0.
    # Linear rescale: map [-1,+2] to [-2,+2]
    # new = (old - (-1)) / (2 - (-1)) * (2 - (-2)) + (-2)
    # new = (old + 1) / 3 * 4 - 2
    df.loc[has_3, 'autonomy_idx'] = (auto_3 + 1) / 3 * 4 - 2

    # Recode remaining items
    df['F120'] = 11 - df['F120']    # higher = never justifiable
    df['G006'] = 5 - df['G006']     # higher = very proud
    df['E018'] = 4 - df['E018']     # higher = good thing
    df['Y002'] = 4 - df['Y002']     # higher = materialist
    df['F118'] = 11 - df['F118']    # higher = never justifiable
    # A008, E025, A165: keep as-is

    return df


def do_factor_analysis(data_matrix, items):
    """PCA + varimax on correlation matrix."""
    corr = data_matrix.corr().values

    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    loadings = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    loadings, _ = varimax(loadings)

    var_exp = (loadings ** 2).sum(axis=0) / len(items) * 100

    # Identify Traditional vs Survival
    trad_idx = [items.index(x) for x in ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']]
    surv_idx = [items.index(x) for x in ['Y002', 'A008', 'E025', 'F118', 'A165']]

    f1_trad = sum(abs(loadings[i, 0]) for i in trad_idx)
    f2_trad = sum(abs(loadings[i, 1]) for i in trad_idx)
    tc = 0 if f1_trad > f2_trad else 1
    sc = 1 - tc

    if np.mean([loadings[i, tc] for i in trad_idx]) < 0:
        loadings[:, tc] = -loadings[:, tc]
    if np.mean([loadings[i, sc] for i in surv_idx]) < 0:
        loadings[:, sc] = -loadings[:, sc]

    result = pd.DataFrame({
        'item': items,
        'trad_secrat': loadings[:, tc],
        'surv_selfexp': loadings[:, sc],
    })
    return result, var_exp[tc], var_exp[sc], eigenvalues


def run_analysis(data_source=None):
    """Main analysis."""
    df = load_data()
    df = prepare_items(df)

    # Nation-level: latest per country
    df_nation = get_latest_per_country(df)
    cm = df_nation.groupby('COUNTRY_ALPHA')[ITEMS].mean()
    cm = cm.dropna(thresh=7)
    for c in ITEMS:
        cm[c] = cm[c].fillna(cm[c].mean())
    n_countries = len(cm)

    nation_load, nv1, nv2, neig = do_factor_analysis(cm, ITEMS)

    # Individual-level: all respondents
    data_i = df[ITEMS].copy()
    total_n = data_i.dropna(how='all').shape[0]
    smallest_n = data_i.count().min()

    indiv_load, iv1, iv2, ieig = do_factor_analysis(data_i, ITEMS)

    # Print
    names = {
        'god_important': "God is very important in respondent's life",
        'autonomy_idx': "Autonomy index (obedience+faith vs independence+determination)",
        'F120': 'Abortion is never justifiable',
        'G006': 'Strong sense of national pride',
        'E018': 'Favors more respect for authority',
        'Y002': 'Priority to economic/physical security (Materialist)',
        'A008': 'Not very happy',
        'E025': 'Would not sign petition',
        'F118': 'Homosexuality never justifiable',
        'A165': 'Very careful about trusting people',
    }

    print("=" * 90)
    print("TABLE 1: Two Dimensions of Cross-Cultural Variation")
    print("=" * 90)
    print(f"\nNation-level: N = {n_countries}")
    print(f"Individual-level: N = {total_n:,} (smallest N: {smallest_n:,})")

    for dim_label, dim_col, items_list in [
        ("DIM 1: Traditional vs. Secular-Rational", 'trad_secrat',
         ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']),
        ("DIM 2: Survival vs. Self-Expression", 'surv_selfexp',
         ['Y002', 'A008', 'E025', 'F118', 'A165']),
    ]:
        print(f"\n{dim_label}")
        print(f"{'Item':<55} {'Nation':>8} {'Individual':>12}")
        print("-" * 75)
        for item in items_list:
            nl = nation_load[nation_load['item'] == item][dim_col].values[0]
            il = indiv_load[indiv_load['item'] == item][dim_col].values[0]
            print(f"{names[item]:<55} {nl:>8.2f} {il:>12.2f}")

    print(f"\nVariance explained:")
    print(f"  Dim 1: {nv1:.1f}% (nation), {iv1:.1f}% (individual)")
    print(f"  Dim 2: {nv2:.1f}% (nation), {iv2:.1f}% (individual)")

    return {
        'nation_loadings': nation_load, 'indiv_loadings': indiv_load,
        'n_countries': n_countries, 'total_n': total_n, 'smallest_n': smallest_n,
        'nation_var1': nv1, 'nation_var2': nv2, 'indiv_var1': iv1, 'indiv_var2': iv2,
    }


def score_against_ground_truth():
    """Score the results."""
    r = run_analysis()
    nl = r['nation_loadings']
    il = r['indiv_loadings']

    score = 0
    details = []

    # 1. All items (20 pts)
    if all(i in nl['item'].values for i in ITEMS):
        score += 20
        details.append("All items present: +20")

    # 2. Loadings (40 pts)
    ls = 0
    pp = 40 / 20
    for lev, ldf in [('nation', nl), ('indiv', il)]:
        gt = GROUND_TRUTH[f'{lev}{"" if lev=="nation" else "idual"}_level']
        for dim, its in [('trad_secrat', ITEMS[:5]), ('surv_selfexp', ITEMS[5:])]:
            for it in its:
                gv = gt[dim][it]
                gen = ldf[ldf['item'] == it][dim].values[0]
                d = abs(gv - gen)
                if d <= 0.03:
                    ls += pp; tag = "MATCH"
                elif d <= 0.06:
                    ls += pp * 0.5; tag = "PARTIAL"
                else:
                    tag = "MISS"
                details.append(f"  {lev} {it} {dim}: paper={gv:.2f}, gen={gen:.2f}, diff={d:.3f} -> {tag}")
    score += ls
    details.append(f"Loadings: +{ls:.1f}/40")

    # 3. Dimension assignment (20 pts)
    ds = 0
    for lev, ldf in [('nation', nl), ('indiv', il)]:
        c = 0
        for it in ITEMS[:5]:
            row = ldf[ldf['item'] == it]
            if abs(row['trad_secrat'].values[0]) > abs(row['surv_selfexp'].values[0]): c += 1
        for it in ITEMS[5:]:
            row = ldf[ldf['item'] == it]
            if abs(row['surv_selfexp'].values[0]) > abs(row['trad_secrat'].values[0]): c += 1
        ds += (c / 10) * 10
        details.append(f"  {lev}: {c}/10 correct")
    score += ds
    details.append(f"Dimension assignment: +{ds:.1f}/20")

    # 4. Ordering (10 pts)
    os2 = 0
    for lev, ldf in [('nation', nl), ('indiv', il)]:
        tv = [ldf[ldf['item'] == i]['trad_secrat'].values[0] for i in ITEMS[:5]]
        sv = [ldf[ldf['item'] == i]['surv_selfexp'].values[0] for i in ITEMS[5:]]
        tok = all(tv[j] >= tv[j+1] for j in range(4))
        sok = all(sv[j] >= sv[j+1] for j in range(4))
        if tok: os2 += 2.5
        if sok: os2 += 2.5
        details.append(f"  {lev}: trad={'OK' if tok else 'WRONG'} surv={'OK' if sok else 'WRONG'}")
    score += os2
    details.append(f"Ordering: +{os2:.1f}/10")

    # 5. Size & variance (10 pts)
    ss = 0
    n = r['n_countries']
    if abs(n - 65) / 65 <= 0.05:
        ss += 2; details.append(f"  N={n} -> MATCH")
    elif abs(n - 65) / 65 <= 0.15:
        ss += 1; details.append(f"  N={n} -> PARTIAL")
    else:
        details.append(f"  N={n} -> MISS")

    for nm, gn, gt in [('n_v1', r['nation_var1'], 44), ('n_v2', r['nation_var2'], 26),
                        ('i_v1', r['indiv_var1'], 26), ('i_v2', r['indiv_var2'], 13)]:
        d = abs(gn - gt)
        if d <= 3: ss += 2; details.append(f"  {nm}: {gn:.1f}% vs {gt}% -> MATCH")
        elif d <= 6: ss += 1; details.append(f"  {nm}: {gn:.1f}% vs {gt}% -> PARTIAL")
        else: details.append(f"  {nm}: {gn:.1f}% vs {gt}% -> MISS")
    score += ss
    details.append(f"Size/variance: +{ss:.1f}/10")

    total = min(100, round(score))
    details.append(f"\nTOTAL SCORE: {total}/100")
    print("\n" + "=" * 50)
    print("SCORING")
    print("=" * 50)
    for d in details:
        print(d)
    return total, details


if __name__ == "__main__":
    score_against_ground_truth()
