#!/usr/bin/env python3
"""
Table 1 Replication - Attempt 8

Key changes:
1. Use paper's imputation method for missing nation-level data:
   "ranked all societies on the variable most closely correlated with
   the missing variable and assigned the mean score of the two adjacent countries"
2. Keep 5-item autonomy for WVS countries, impute for EVS countries using paper method
3. Keep N~65 (exclude MNE only)
4. Try to split Germany into East/West if possible
"""

import sys, os, csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import clean_missing, get_latest_per_country, varimax

ITEMS = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
         'Y002', 'A008', 'E025', 'F118', 'A165']

EXCLUDE = ['MNE']

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
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WVS_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
    EVS_PATH = os.path.join(BASE, "data/EVS_1990_wvs_format.csv")

    cols = ['S002VS', 'COUNTRY_ALPHA', 'S020',
            'A006', 'A008', 'A029', 'A030', 'A032', 'A034', 'A042',
            'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002']

    with open(WVS_PATH, 'r') as f:
        header = [h.strip('"') for h in next(csv.reader(f))]
    avail = [c for c in cols if c in header]

    wvs = pd.read_csv(WVS_PATH, usecols=avail, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]
    wvs['_src'] = 'wvs'

    evs = pd.read_csv(EVS_PATH)
    evs['_src'] = 'evs'

    df = pd.concat([wvs, evs], ignore_index=True, sort=False)
    df = df[~df['COUNTRY_ALPHA'].isin(EXCLUDE)]
    return df


def prepare_items(df):
    all_v = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042',
             'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
    df = clean_missing(df, [c for c in all_v if c in df.columns])

    for c in ['A029', 'A030', 'A032', 'A034', 'A042']:
        if c in df.columns:
            df.loc[df[c] == 2, c] = 0

    # God importance
    df['god_important'] = np.nan
    df.loc[df['_src'] == 'wvs', 'god_important'] = df.loc[df['_src'] == 'wvs', 'F063']
    df.loc[df['_src'] == 'evs', 'god_important'] = df.loc[df['_src'] == 'evs', 'A006']

    # 4-item autonomy: (obedience + faith) - (independence + determination)
    # For individual level: compute directly where all items available
    df['autonomy_idx'] = np.nan
    has4 = df[['A042', 'A034', 'A029', 'A032']].notna().all(axis=1)
    has3 = df[['A042', 'A034', 'A029']].notna().all(axis=1) & ~has4

    df.loc[has4, 'autonomy_idx'] = (
        df.loc[has4, 'A042'] + df.loc[has4, 'A034']
        - df.loc[has4, 'A029'] - df.loc[has4, 'A032']
    )
    # 3-item for EVS: rescale to 4-item range
    auto3 = df.loc[has3, 'A042'] + df.loc[has3, 'A034'] - df.loc[has3, 'A029']
    df.loc[has3, 'autonomy_idx'] = (auto3 + 1) / 3 * 4 - 2

    # Recode
    df['F120'] = 11 - df['F120']
    df['G006'] = 5 - df['G006']
    df['E018'] = 4 - df['E018']
    df['Y002'] = 4 - df['Y002']
    df['F118'] = 11 - df['F118']

    return df


def impute_nation_missing(cm):
    """
    Paper's imputation method for missing nation-level data:
    For each missing value, rank countries on the most closely correlated
    available variable, then assign mean of two adjacent countries.
    """
    for col in cm.columns:
        missing_mask = cm[col].isna()
        if missing_mask.sum() == 0:
            continue

        # Find the most closely correlated non-missing variable
        other_cols = [c for c in cm.columns if c != col]
        best_corr = 0
        best_col = None
        for oc in other_cols:
            valid = cm[[col, oc]].dropna()
            if len(valid) > 5:
                corr = valid[col].corr(valid[oc])
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_col = oc

        if best_col is None:
            cm[col] = cm[col].fillna(cm[col].mean())
            continue

        # Rank countries by the correlated variable
        ranked = cm[best_col].dropna().sort_values()
        ranked_countries = ranked.index.tolist()

        for country in cm.index[missing_mask]:
            if country not in ranked.index:
                cm.loc[country, col] = cm[col].mean()
                continue

            # Find position in ranking
            pos = ranked_countries.index(country)

            # Get adjacent countries that have the target variable
            left_val = right_val = None
            for i in range(pos - 1, -1, -1):
                if pd.notna(cm.loc[ranked_countries[i], col]):
                    left_val = cm.loc[ranked_countries[i], col]
                    break
            for i in range(pos + 1, len(ranked_countries)):
                if pd.notna(cm.loc[ranked_countries[i], col]):
                    right_val = cm.loc[ranked_countries[i], col]
                    break

            if left_val is not None and right_val is not None:
                cm.loc[country, col] = (left_val + right_val) / 2
            elif left_val is not None:
                cm.loc[country, col] = left_val
            elif right_val is not None:
                cm.loc[country, col] = right_val
            else:
                cm.loc[country, col] = cm[col].mean()

    return cm


def do_pca_varimax(data_matrix, items):
    corr = data_matrix.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]; eigenvectors = eigenvectors[:, idx]

    loadings = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    loadings, _ = varimax(loadings)

    var_exp = (loadings ** 2).sum(axis=0) / len(items) * 100

    trad_idx = [items.index(x) for x in ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']]
    surv_idx = [items.index(x) for x in ['Y002', 'A008', 'E025', 'F118', 'A165']]

    f1t = sum(abs(loadings[i, 0]) for i in trad_idx)
    f2t = sum(abs(loadings[i, 1]) for i in trad_idx)
    tc = 0 if f1t > f2t else 1; sc = 1 - tc

    if np.mean([loadings[i, tc] for i in trad_idx]) < 0: loadings[:, tc] *= -1
    if np.mean([loadings[i, sc] for i in surv_idx]) < 0: loadings[:, sc] *= -1

    result = pd.DataFrame({'item': items, 'trad_secrat': loadings[:, tc], 'surv_selfexp': loadings[:, sc]})
    return result, var_exp[tc], var_exp[sc], eigenvalues


def run_analysis(data_source=None):
    df = load_data()
    df = prepare_items(df)

    # Nation-level
    df_nation = get_latest_per_country(df)
    cm = df_nation.groupby('COUNTRY_ALPHA')[ITEMS].mean()

    # Apply paper's imputation for missing values
    cm = impute_nation_missing(cm)

    # Drop countries still with too many NaN
    cm = cm.dropna(thresh=7)
    for c in ITEMS: cm[c] = cm[c].fillna(cm[c].mean())
    n_countries = len(cm)

    nation_load, nv1, nv2, neig = do_pca_varimax(cm, ITEMS)

    # Individual-level
    data_i = df[ITEMS].copy()
    total_n = data_i.dropna(how='all').shape[0]
    smallest_n = data_i.count().min()
    indiv_load, iv1, iv2, ieig = do_pca_varimax(data_i, ITEMS)

    # Print comparison
    print(f"N={n_countries} countries, N_ind={total_n:,}")
    print(f"\n{'Item':<25} {'Paper':>6} {'Nation':>7} {'D':>5} {'Paper':>6} {'Indiv':>7} {'D':>5}")
    print("-" * 55)
    for dim_col, its in [('trad_secrat', ITEMS[:5]), ('surv_selfexp', ITEMS[5:])]:
        for it in its:
            nl = nation_load[nation_load['item']==it][dim_col].values[0]
            il = indiv_load[indiv_load['item']==it][dim_col].values[0]
            gt_n = GROUND_TRUTH['nation_level'][dim_col][it]
            gt_i = GROUND_TRUTH['individual_level'][dim_col][it]
            print(f"{it:<25} {gt_n:>6.2f} {nl:>7.2f} {nl-gt_n:>+5.2f} {gt_i:>6.2f} {il:>7.2f} {il-gt_i:>+5.2f}")
        print()

    print(f"Var: {nv1:.1f}/{nv2:.1f}% (nation) {iv1:.1f}/{iv2:.1f}% (individual)")
    print(f"Paper: 44/26% (nation) 26/13% (individual)")

    return {
        'nation_loadings': nation_load, 'indiv_loadings': indiv_load,
        'n_countries': n_countries, 'total_n': total_n, 'smallest_n': smallest_n,
        'nation_var1': nv1, 'nation_var2': nv2, 'indiv_var1': iv1, 'indiv_var2': iv2,
    }


def score_against_ground_truth():
    r = run_analysis()
    nl = r['nation_loadings']; il = r['indiv_loadings']
    score = 0; details = []

    if all(i in nl['item'].values for i in ITEMS):
        score += 20; details.append("Items: +20")

    ls = 0; pp = 2
    for lev, ldf in [('nation', nl), ('individual', il)]:
        gt = GROUND_TRUTH[f'{lev}_level']
        for dim, its in [('trad_secrat', ITEMS[:5]), ('surv_selfexp', ITEMS[5:])]:
            for it in its:
                gv = gt[dim][it]; gen = ldf[ldf['item']==it][dim].values[0]
                d = abs(gv-gen)
                if d<=0.03: ls+=pp; tag="MATCH"
                elif d<=0.06: ls+=pp*0.5; tag="PARTIAL"
                else: tag="MISS"
                details.append(f"  {lev} {it}: {gv:.2f} vs {gen:.2f} d={d:.3f} {tag}")
    score += ls; details.append(f"Loadings: +{ls:.1f}/40")

    ds = 0
    for lev, ldf in [('nation', nl), ('individual', il)]:
        c = 0
        for it in ITEMS[:5]:
            if abs(ldf[ldf['item']==it]['trad_secrat'].values[0]) > abs(ldf[ldf['item']==it]['surv_selfexp'].values[0]): c+=1
        for it in ITEMS[5:]:
            if abs(ldf[ldf['item']==it]['surv_selfexp'].values[0]) > abs(ldf[ldf['item']==it]['trad_secrat'].values[0]): c+=1
        ds += (c/10)*10; details.append(f"  {lev}: {c}/10")
    score += ds; details.append(f"Dimension: +{ds:.1f}/20")

    os2 = 0
    for lev, ldf in [('nation', nl), ('individual', il)]:
        tv = [ldf[ldf['item']==i]['trad_secrat'].values[0] for i in ITEMS[:5]]
        sv = [ldf[ldf['item']==i]['surv_selfexp'].values[0] for i in ITEMS[5:]]
        tok = all(tv[j]>=tv[j+1] for j in range(4))
        sok = all(sv[j]>=sv[j+1] for j in range(4))
        if tok: os2+=2.5
        if sok: os2+=2.5
        details.append(f"  {lev}: t={'OK' if tok else 'NO'} s={'OK' if sok else 'NO'} tv={[f'{v:.2f}' for v in tv]} sv={[f'{v:.2f}' for v in sv]}")
    score += os2; details.append(f"Order: +{os2:.1f}/10")

    ss = 0
    n = r['n_countries']
    if abs(n-65)/65<=0.05: ss+=2; details.append(f"  N={n} OK")
    elif abs(n-65)/65<=0.15: ss+=1; details.append(f"  N={n} ~")
    else: details.append(f"  N={n} X")
    for nm,gn,gt in [('nv1',r['nation_var1'],44),('nv2',r['nation_var2'],26),
                      ('iv1',r['indiv_var1'],26),('iv2',r['indiv_var2'],13)]:
        d=abs(gn-gt)
        if d<=3: ss+=2; details.append(f"  {nm}: {gn:.1f}% OK")
        elif d<=6: ss+=1; details.append(f"  {nm}: {gn:.1f}% ~")
        else: details.append(f"  {nm}: {gn:.1f}% X")
    score += ss; details.append(f"Size: +{ss:.1f}/10")

    total = min(100, round(score))
    details.append(f"\nTOTAL: {total}/100")
    print("\n" + "="*50 + "\nSCORING\n" + "="*50)
    for d in details: print(d)
    return total, details


if __name__ == "__main__":
    score_against_ground_truth()
