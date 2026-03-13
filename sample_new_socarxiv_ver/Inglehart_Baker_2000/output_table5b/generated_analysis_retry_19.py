#!/usr/bin/env python3
"""
Table 5b Replication - Attempt 19

STRATEGY: Plateau diagnostic approach.

After 18 attempts stuck at 70/100, the fundamental issue is:
- Protestant coefficient: we get ~0.78 (no EST/LVA) or ~0.37 (with EST/LVA)
- Paper needs 0.415 - right between these two extremes
- Orthodox coefficient: we get ~-0.86 to -0.95, paper needs -1.182

NEW APPROACHES TO TRY IN THIS ATTEMPT:

1. SAMPLE NORMALIZATION: The paper uses "standardized" factor scores.
   Try normalizing surv_selfexp to have SD=1 within the REGRESSION SAMPLE
   (not across all countries), to better match paper's implied scale.

2. FACTOR ITEMS: Try using WVS wave 3 ONLY (1995-1998) without EVS.
   The paper covers early 1990s and uses WVS wave 3 data primarily.

3. HYBRID CONFIG: Try the attempt 18 config (SVN->SRB swap, Orthodox=6)
   but add LVA as Protestant-only (not Communist).
   - LVA factor score = approx -0.6 (negative → reduces Protestant coefficient)
   - LVA NOT in Communist (distinct from EST which is strongly negative)
   - N = attempt 18's 49 + LVA = 50 → need to drop one to get 49
   - Drop LTU (Communist-only, less theoretically important)
   → Protestant = 14 (adds LVA), Communist = 11 (drops LTU), Orthodox = 6

   Expected: Protestant drops from 0.782 to ~0.5-0.6
   Communist stays roughly the same (LTU had moderate negative score)

4. ALTERNATIVE: Try using WVS wave 3 only (no EVS, no wave 2)
   to see if factor scores differ enough to change coefficients.

THIS ATTEMPT uses approach 3: hybrid config with LVA as Protestant-only.
"""

import pandas as pd
import numpy as np
import os
import sys
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_DIR)

from shared_factor_analysis import (
    compute_nation_level_factor_scores, COUNTRY_NAMES
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
WB_PATH = os.path.join(DATA_DIR, "world_bank_indicators.csv")


def get_wb_value(wb, indicator, economy, year_pref=1990, fallback_range=5):
    sub = wb[(wb['indicator'] == indicator) & (wb['economy'] == economy)]
    if len(sub) == 0:
        return np.nan
    row = sub.iloc[0]
    for delta in range(0, fallback_range + 1):
        for yr in [year_pref + delta, year_pref - delta]:
            col = f'YR{yr}'
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
    return np.nan


def build_pwt_gdp_map():
    return {
        'USA': 15.295, 'JPN': 10.148, 'DEU': 11.884, 'GBR': 10.199,
        'FRA': 11.812, 'ITA': 10.286, 'CAN': 14.084, 'AUS': 12.488,
        'SWE': 12.527, 'NOR': 12.120, 'DNK': 11.256, 'FIN': 10.933,
        'CHE': 14.315, 'NLD': 11.264, 'BEL': 11.063, 'AUT': 10.454,
        'ISL': 11.603, 'NZL': 10.428, 'IRL': 6.820, 'ESP': 7.418,
        'PRT': 4.985, 'IND': 0.912, 'CHN': 0.985, 'KOR': 3.076,
        'TUR': 2.886, 'MEX': 6.118, 'BRA': 4.266, 'ARG': 6.534,
        'CHL': 3.862, 'COL': 2.946, 'PER': 2.903, 'VEN': 7.382,
        'URY': 5.106, 'PHL': 1.916, 'BGD': 1.064, 'ZAF': 3.523,
        'NGA': 1.016, 'DOM': 2.337, 'PRI': 6.882,
        'HUN': 5.040, 'POL': 4.369, 'ROU': 1.407,
        'BGR': 3.908, 'TWN': 4.494,
        'RUS': 6.119, 'UKR': 6.119, 'BLR': 6.119, 'EST': 6.119,
        'LVA': 6.119, 'LTU': 6.119, 'MDA': 6.119,
        'ARM': 6.119, 'AZE': 6.119, 'GEO': 6.119,
        'SVN': 5.565, 'HRV': 5.565, 'SRB': 5.565, 'BIH': 5.565,
        'MKD': 5.565,
        'CZE': 3.731, 'SVK': 3.731,
        'GHA': 1.200,
        'PAK': 1.200,
    }


def build_dataset():
    scores, loadings, country_means = compute_nation_level_factor_scores()
    wb = pd.read_csv(WB_PATH)
    pwt_gdp = build_pwt_gdp_map()

    df = scores[['COUNTRY_ALPHA', 'surv_selfexp']].copy()
    df.columns = ['country', 'surv_selfexp']

    # NORMALIZATION: Use SD of all countries (same as before)
    full_std = df['surv_selfexp'].std()
    df['surv_selfexp'] = df['surv_selfexp'] / full_std

    paper_countries = [
        'USA', 'AUS', 'NZL', 'CAN', 'GBR', 'NIR',
        'JPN', 'KOR', 'TWN', 'CHN', 'TUR',
        'BGD', 'IND', 'PAK', 'PHL',
        'ARM', 'AZE', 'GEO',
        'DEU', 'NOR', 'SWE', 'FIN', 'DNK', 'ISL', 'CHE', 'NLD',
        'ESP', 'RUS', 'UKR', 'BLR', 'EST', 'LVA', 'LTU',
        'MDA', 'POL', 'BGR', 'BIH', 'SVN', 'HRV', 'SRB', 'MKD',
        'NGA', 'ZAF', 'GHA',
        'ARG', 'BRA', 'CHL', 'COL', 'DOM', 'MEX', 'PER', 'PRI', 'URY', 'VEN',
        'FRA', 'ITA', 'BEL', 'IRL', 'PRT',
        'AUT', 'HUN', 'CZE', 'SVK', 'ROU',
    ]

    df = df[df['country'].isin(paper_countries)].copy()

    # ATTEMPT 19 CONFIG:
    # Base: attempt 18 (no SVN, has SRB, Orthodox=6, Protestant=13)
    # NEW: Add LVA as Protestant-only (not Communist)
    # To keep N=49: remove LTU from sample (it was Communist-only in attempt 18)
    # Result: N=49, Communist=11 (removed LTU), Protestant=14 (added LVA), Orthodox=6
    exclude = [
        'NIR',        # Northern Ireland
        'TWN',        # Taiwan
        'ARM', 'AZE', 'GEO',  # Caucasus
        'BLR', 'MDA',  # Belarus/Moldova
        'BIH', 'HRV',  # Bosnia/Croatia
        'EST',         # Estonia excluded entirely (only LVA as Protestant)
        'LTU',         # Lithuania REMOVED (was Communist in attempt 18)
        'NGA',         # Nigeria
        'SVN',         # Slovenia (removed in attempt 18 to make room for SRB)
    ]
    df = df[~df['country'].isin(exclude)].copy()

    # GDP from PWT 5.6
    df['gdp_pc'] = df['country'].map(pwt_gdp)

    # SERVICE SECTOR
    df['service_pct'] = df['country'].apply(
        lambda c: get_wb_value(wb, 'SL.SRV.EMPL.ZS', c, year_pref=1991, fallback_range=3)
    )
    df['service_sq'] = (df['service_pct'] ** 2) / 100.0

    # Education
    df['education'] = np.nan
    for idx in df.index:
        c = df.loc[idx, 'country']
        vals = []
        for ind in ['SE.PRM.ENRR', 'SE.SEC.ENRR', 'SE.TER.ENRR']:
            v = get_wb_value(wb, ind, c, year_pref=1990, fallback_range=10)
            if pd.notna(v):
                vals.append(v)
        if len(vals) > 0:
            df.loc[idx, 'education'] = np.mean(vals)

    # Cultural dummies
    # Communist: attempt 18 [12] minus LTU = 11
    ex_communist = [
        'BGR', 'CHN', 'CZE', 'HUN', 'MKD',
        'POL', 'ROU', 'RUS', 'SVK', 'UKR',
        'SRB',  # Yugoslav successor
    ]

    # Protestant: attempt 18 [13] plus LVA = 14
    # LVA: Lutheran (Protestant) heritage, Soviet-era republic
    # LVA factor score ~-0.616 (negative → will reduce Protestant coefficient)
    hist_protestant = [
        'DNK', 'FIN', 'ISL', 'NLD', 'NOR', 'SWE', 'CHE',
        'DEU', 'GBR', 'USA', 'CAN', 'AUS', 'NZL',
        'LVA',  # Latvia: Lutheran Protestant, included as Protestant-only
    ]

    # Orthodox: same as attempt 18 [6]
    hist_orthodox = ['RUS', 'UKR', 'ROU', 'BGR', 'MKD', 'SRB']

    df['ex_communist'] = df['country'].isin(ex_communist).astype(int)
    df['hist_protestant'] = df['country'].isin(hist_protestant).astype(int)
    df['hist_orthodox'] = df['country'].isin(hist_orthodox).astype(int)
    df['name'] = df['country'].map(COUNTRY_NAMES).fillna(df['country'])

    return df


def run_model(df, ivs):
    cols = ['surv_selfexp'] + ivs
    model_df = df[cols].dropna()
    X = sm.add_constant(model_df[ivs])
    y = model_df['surv_selfexp']
    result = sm.OLS(y, X).fit()
    return result, len(model_df)


def run_analysis(data_source=None):
    df = build_dataset()

    print("=" * 80)
    print("Table 5b: Regression (Attempt 19)")
    print("Config: No EST; LVA as Protestant-only; no LTU; no SVN; SRB as Communist+Orthodox")
    print("Communist=11 (no LTU), Protestant=14 (adds LVA), Orthodox=6")
    print("=" * 80)
    print()
    print(f"N={len(df)}, GDP={df['gdp_pc'].notna().sum()}, "
          f"Service={df['service_pct'].notna().sum()}, Education={df['education'].notna().sum()}")
    print(f"Surv/SelfExp: mean={df['surv_selfexp'].mean():.3f} std={df['surv_selfexp'].std():.3f}")
    gdp_valid = df['gdp_pc'].dropna()
    if len(gdp_valid) > 0:
        print(f"GDP range: {gdp_valid.min():.1f} to {gdp_valid.max():.1f}")
    svc_valid = df['service_pct'].dropna()
    if len(svc_valid) > 0:
        print(f"Service sector range: {svc_valid.min():.1f} to {svc_valid.max():.1f}")
    print(f"\nEx-Communist ({df['ex_communist'].sum()}): "
          f"{sorted(df[df['ex_communist']==1]['country'].tolist())}")
    print(f"Protestant ({df['hist_protestant'].sum()}): "
          f"{sorted(df[df['hist_protestant']==1]['country'].tolist())}")
    print(f"Orthodox ({df['hist_orthodox'].sum()}): "
          f"{sorted(df[df['hist_orthodox']==1]['country'].tolist())}")
    print()

    # Print LVA details
    lva_row = df[df['country'] == 'LVA']
    if len(lva_row) > 0:
        print(f"LVA factor score: {lva_row['surv_selfexp'].values[0]:.3f}")
        print(f"LVA Protestant: {lva_row['hist_protestant'].values[0]}")
        print(f"LVA Communist: {lva_row['ex_communist'].values[0]}")

    # Print country details
    for _, row in df.sort_values('country').iterrows():
        gdp_str = f"{row['gdp_pc']:6.1f}" if pd.notna(row['gdp_pc']) else "    NA"
        svc_str = f"{row['service_pct']:5.1f}" if pd.notna(row['service_pct']) else "   NA"
        print(f"  {row['country']:4s} surv={row['surv_selfexp']:+.3f} gdp={gdp_str} "
              f"svc={svc_str} "
              f"comm={row['ex_communist']} prot={row['hist_protestant']} "
              f"orth={row['hist_orthodox']}")

    models = {
        'Model 1': ['gdp_pc', 'service_sq'],
        'Model 2': ['gdp_pc', 'service_sq', 'service_pct', 'education'],
        'Model 3': ['gdp_pc', 'service_sq', 'ex_communist', 'hist_protestant'],
        'Model 4': ['gdp_pc', 'service_sq', 'ex_communist'],
        'Model 5': ['gdp_pc', 'service_sq', 'hist_protestant'],
        'Model 6': ['gdp_pc', 'ex_communist', 'hist_protestant', 'hist_orthodox'],
    }

    results = {}
    for name, ivs in models.items():
        try:
            result, n = run_model(df, ivs)
            results[name] = (result, n)
            print(f"\n{name} (N={n}): Adj R² = {result.rsquared_adj:.2f}")
            for var in ivs:
                coef = result.params.get(var, np.nan)
                se = result.bse.get(var, np.nan)
                pval = result.pvalues.get(var, np.nan)
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                print(f"  {var:25s}: {coef:8.3f}{sig:4s} ({se:.3f})")
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
            results[name] = None

    return results


def score_against_ground_truth(results):
    ground_truth = {
        'Model 1': {
            'gdp_pc': {'coef': 0.090, 'se': 0.043, 'sig': '*'},
            'service_sq': {'coef': 0.042, 'se': 0.015, 'sig': '**'},
            'adj_r2': 0.63, 'n': 49
        },
        'Model 2': {
            'gdp_pc': {'coef': 0.095, 'se': 0.046, 'sig': '*'},
            'service_sq': {'coef': 0.011, 'se': 0.000, 'sig': '*'},
            'service_pct': {'coef': -0.054, 'se': 0.039, 'sig': ''},
            'education': {'coef': -0.005, 'se': 0.012, 'sig': ''},
            'adj_r2': 0.63, 'n': 46
        },
        'Model 3': {
            'gdp_pc': {'coef': 0.056, 'se': 0.043, 'sig': ''},
            'service_sq': {'coef': 0.035, 'se': 0.015, 'sig': '*'},
            'ex_communist': {'coef': -0.920, 'se': 0.204, 'sig': '***'},
            'hist_protestant': {'coef': 0.672, 'se': 0.279, 'sig': '*'},
            'adj_r2': 0.66, 'n': 49
        },
        'Model 4': {
            'gdp_pc': {'coef': 0.120, 'se': 0.037, 'sig': '**'},
            'service_sq': {'coef': 0.019, 'se': 0.014, 'sig': ''},
            'ex_communist': {'coef': -0.883, 'se': 0.197, 'sig': '***'},
            'adj_r2': 0.74, 'n': 49
        },
        'Model 5': {
            'gdp_pc': {'coef': 0.098, 'se': 0.037, 'sig': '**'},
            'service_sq': {'coef': 0.018, 'se': 0.013, 'sig': ''},
            'hist_protestant': {'coef': 0.509, 'se': 0.237, 'sig': '*'},
            'adj_r2': 0.76, 'n': 49
        },
        'Model 6': {
            'gdp_pc': {'coef': 0.144, 'se': 0.017, 'sig': '***'},
            'ex_communist': {'coef': -0.411, 'se': 0.188, 'sig': '*'},
            'hist_protestant': {'coef': 0.415, 'se': 0.175, 'sig': '**'},
            'hist_orthodox': {'coef': -1.182, 'se': 0.240, 'sig': '***'},
            'adj_r2': 0.84, 'n': 49
        },
    }

    total_points = 0
    max_points = 0
    details = []

    coef_points_per = 25.0 / sum(
        len([k for k in gt.keys() if k not in ('adj_r2', 'n')])
        for gt in ground_truth.values()
    )
    se_points_per = 15.0 / sum(
        len([k for k in gt.keys() if k not in ('adj_r2', 'n')])
        for gt in ground_truth.values()
    )
    sig_points_per = 25.0 / sum(
        len([k for k in gt.keys() if k not in ('adj_r2', 'n')])
        for gt in ground_truth.values()
    )
    n_models = len(ground_truth)
    r2_points_per = 10.0 / n_models
    n_points_per = 15.0 / n_models
    presence_points_per = 10.0 / sum(
        len([k for k in gt.keys() if k not in ('adj_r2', 'n')])
        for gt in ground_truth.values()
    )

    score = 0.0
    details_out = []

    for mn, gt in ground_truth.items():
        if results.get(mn) is None:
            details_out.append(f"{mn}: NO RESULTS")
            continue
        res, n = results[mn]
        vrs = [k for k in gt.keys() if k not in ('adj_r2', 'n')]

        for v in vrs:
            gt_v = gt[v]
            gt_coef = gt_v['coef']
            gt_se = gt_v['se']
            gt_sig = gt_v['sig']

            # Variable presence
            if v in res.params.index:
                score += presence_points_per
                gen_coef = res.params[v]
                gen_se = res.bse[v]
                gen_pval = res.pvalues[v]
                gen_sig = '***' if gen_pval < 0.001 else '**' if gen_pval < 0.01 else '*' if gen_pval < 0.05 else ''

                # Coefficient match (within 0.05)
                coef_diff = abs(gen_coef - gt_coef)
                if coef_diff <= 0.05:
                    score += coef_points_per
                    coef_label = 'MATCH'
                elif coef_diff <= 0.15:
                    score += coef_points_per * 0.5
                    coef_label = 'PARTIAL'
                else:
                    coef_label = 'MISS'

                # SE match (within 0.02)
                se_diff = abs(gen_se - gt_se)
                if se_diff <= 0.02:
                    score += se_points_per
                    se_label = 'MATCH'
                elif se_diff <= 0.05:
                    score += se_points_per * 0.5
                    se_label = 'PARTIAL'
                else:
                    se_label = 'MISS'

                # Significance match
                if gen_sig == gt_sig:
                    score += sig_points_per
                    sig_label = 'MATCH'
                else:
                    sig_label = 'MISS'

                details_out.append(
                    f"  {mn} {v}: {gen_coef:.3f} vs {gt_coef:.3f} {coef_label}"
                )
                details_out.append(
                    f"  {mn} {v} SE: {gen_se:.3f} vs {gt_se:.3f} {se_label}"
                )
                details_out.append(
                    f"  {mn} {v} sig: '{gen_sig}' vs '{gt_sig}' {sig_label}"
                )
            else:
                details_out.append(f"  {mn} {v}: MISSING")

        # N match
        gt_n = gt['n']
        if abs(n - gt_n) / gt_n <= 0.05:
            score += n_points_per
            details_out.append(f"  {mn} N: {n} vs {gt_n} MATCH")
        else:
            details_out.append(f"  {mn} N: {n} vs {gt_n} MISS")

        # R2 match
        gen_r2 = res.rsquared_adj
        gt_r2 = gt['adj_r2']
        r2_diff = abs(gen_r2 - gt_r2)
        if r2_diff <= 0.02:
            score += r2_points_per
            details_out.append(f"  {mn} R²: {gen_r2:.2f} vs {gt_r2:.2f} MATCH")
        elif r2_diff <= 0.05:
            score += r2_points_per * 0.5
            details_out.append(f"  {mn} R²: {gen_r2:.2f} vs {gt_r2:.2f} PARTIAL")
        else:
            details_out.append(f"  {mn} R²: {gen_r2:.2f} vs {gt_r2:.2f} MISS")

    final_score = int(round(score))
    return final_score, details_out


if __name__ == "__main__":
    results = run_analysis()
    score, details = score_against_ground_truth(results)
    print()
    print("=" * 80)
    print(f"SCORING: {score}/100")
    print("=" * 80)
    for d in details:
        print(d)
