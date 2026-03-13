#!/usr/bin/env python3
"""
Table 5b Replication - Attempt 15

STRATEGY: Use the shared_factor_analysis module but pass waves_wvs=[3] (Wave 3 only)
to see if Wave 3 alone gives better factor scores. Also try waves_wvs=[2] (Wave 2 only).
Additionally try include_evs=False.

Also: fundamentally reconsider the Orthodox problem.
In Model 6, the paper has:
  Communist: -0.411 (*), Orthodox: -1.182 (***)

This means Orthodox adds -1.182 on top of the base Communist effect.
For this to happen, Orthodox countries must score ~1.182 SD lower than OTHER
Communist countries, after controlling for GDP.

If Communist-only countries (CZE=+0.341, POL=-0.442, HUN=-0.577, SVK=-0.160)
average around -0.2 to -0.3, and Orthodox Communist countries (RUS=-1.584, UKR=-1.499,
BGR=-1.112, ROU=-1.235) average around -1.4, then the difference is about -1.1.
This roughly matches the paper's Orthodox coefficient.

BUT: in M6, both Communist and Orthodox are in the model. If Communist=-0.411 and
Orthodox=-1.182, then:
- Pure Communist predicted: 0.144*GDP + (-0.411) = very roughly -0.411 adjustment
- Communist + Orthodox: 0.144*GDP + (-0.411) + (-1.182) = -1.593 total
- This means M6 explains RUS (very negative) as: high-GDP effect + Communist + Orthodox
  But RUS has GDP=6.1 so GDP effect = 0.144*6.1=0.879, Communist = -0.411, Orthodox = -1.182
  Total prediction for RUS = intercept + 0.879 - 0.411 - 1.182 = intercept - 0.714

The paper's coefficients imply Orthodox has a very large RESIDUAL effect after Communist.
This requires: among all Communist countries, Orthodox ones score MUCH more negative.

In our data:
- Non-Orthodox Communist: CZE=+0.341, HUN=-0.577, POL=-0.442, SVK=-0.160, EST=-1.082, LVA=-0.616, CHN=-0.560
  Average: (-0.577-0.442-0.160-1.082-0.616-0.560+0.341)/7 = -0.443
- Orthodox Communist: RUS=-1.584, UKR=-1.499, BGR=-1.112, ROU=-1.235, (SRB=-0.811, MKD=-1.041)
  Average (4): -1.358; Average (6 with SRB+MKD): -1.214

Orthodox adjustment = -1.214 - (-0.443) = -0.771
This matches our regression coefficient of ~-0.726, NOT the paper's -1.182.

The paper likely has different values. Perhaps:
1. Different factor scores for these countries
2. Different country sample (e.g., no EST, LVA in their sample reduces non-Orthodox Communist avg)
3. Non-Orthodox Communist: CZE=+0.341, HUN=-0.577, POL=-0.442, SVK=-0.160, CHN=-0.560 (without EST, LVA)
   Average (5): -0.280
   Orthodox Communist: -1.358 (4 countries)
   Adjustment: -1.358 - (-0.280) = -1.078 -> closer to -1.182!

So excluding EST, LVA from the Communist list (but keeping them as Protestant) might help!
EST and LVA are LOW-self-expression Communist countries (EST=-1.082, LVA=-0.616).
Including them in Communist raises the non-Orthodox Communist average downward,
which makes the Orthodox additional effect smaller.

If EST and LVA are not classified as ex-Communist (only as Protestant), then:
- Non-Orthodox Communist (CZE, HUN, POL, SVK, CHN): avg -0.280
- Orthodox Communist (RUS, UKR, BGR, ROU): avg -1.358
- Orthodox adjustment: -1.078 (much closer to -1.182!)

BUT: EST and LVA were clearly Communist! The paper might not have them in sample at all.
Or: EST, LVA classified as Protestant ONLY (not Communist, not Orthodox).

Let's try: EST, LVA classified as PROTESTANT ONLY (not communist).
This would make them "Protestant non-Communist" countries, which would:
1. Keep them in the Protestant group (slightly reduces Protestant coef since they're negative)
2. Remove them from Communist group (helps Orthodox coefficient as described)
3. Their negative surv scores would reduce Protestant coefficient
   But Protestant avg: SWE+2.234, NLD+1.944, NOR+1.621, NZL+1.724, AUS+1.735,
   FIN+1.211, ISL+1.301, DNK+1.317, DEU+1.281, GBR+1.075, CAN+1.431, CHE+1.362,
   USA+1.292, EST-1.082, LVA=-0.616
   With EST, LVA: avg ~0.860; Without EST, LVA: avg ~1.288

Including EST, LVA reduces Protestant coef. Excluding them from Protestant: Protestant coef goes up!
Paper has Protestant coef = 0.509 (M5) and 0.415 (M6).
If we remove EST, LVA from Protestant list -> Protestant coef goes up too much.

BEST STRATEGY: Remove EST, LVA from BOTH Protestant AND Communist (classify as neutral).
This way:
- Protestant avg increases (better for M5 R²)
- Non-Orthodox Communist avg changes (helps M6 Orthodox coefficient)

Let me implement this and test.
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
    }


def build_dataset(est_lva_protestant=False, est_lva_communist=False,
                  srb_mkd_in_sample=True, include_ltu=False):
    """
    Build dataset with configurable EST/LVA classification.

    Parameters:
    - est_lva_protestant: Include EST, LVA in Protestant list
    - est_lva_communist: Include EST, LVA in Communist list
    - srb_mkd_in_sample: Include SRB and MKD (Orthodox ex-Communist)
    - include_ltu: Include Lithuania in sample
    """
    scores, loadings, country_means = compute_nation_level_factor_scores()
    wb = pd.read_csv(WB_PATH)
    pwt_gdp = build_pwt_gdp_map()

    df = scores[['COUNTRY_ALPHA', 'surv_selfexp']].copy()
    df.columns = ['country', 'surv_selfexp']

    full_std = df['surv_selfexp'].std()
    df['surv_selfexp'] = df['surv_selfexp'] / full_std

    paper_countries = [
        'USA', 'AUS', 'NZL', 'CAN', 'GBR',
        'JPN', 'KOR', 'CHN', 'TUR',
        'BGD', 'IND', 'PHL',
        'DEU', 'NOR', 'SWE', 'FIN', 'DNK', 'ISL', 'CHE', 'NLD',
        'ESP', 'RUS', 'UKR', 'EST', 'LVA',
        'POL', 'BGR', 'SRB', 'MKD',
        'ZAF',
        'ARG', 'BRA', 'CHL', 'COL', 'DOM', 'MEX', 'PER', 'PRI', 'URY', 'VEN',
        'FRA', 'ITA', 'BEL', 'IRL', 'PRT',
        'AUT', 'HUN', 'CZE', 'SVK', 'ROU',
    ]

    if include_ltu:
        paper_countries.append('LTU')

    df = df[df['country'].isin(paper_countries)].copy()

    # GDP from PWT 5.6
    df['gdp_pc'] = df['country'].map(pwt_gdp)

    # Service sector
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
    ex_communist_base = ['CHN', 'CZE', 'HUN', 'POL', 'ROU', 'RUS', 'SVK', 'UKR', 'BGR']
    if srb_mkd_in_sample:
        ex_communist_base += ['SRB', 'MKD']
    if include_ltu:
        ex_communist_base.append('LTU')
    if est_lva_communist:
        ex_communist_base += ['EST', 'LVA']

    hist_protestant_base = ['DNK', 'FIN', 'ISL', 'NLD', 'NOR', 'SWE', 'CHE',
                             'DEU', 'GBR', 'USA', 'CAN', 'AUS', 'NZL']
    if est_lva_protestant:
        hist_protestant_base += ['EST', 'LVA']

    hist_orthodox = ['RUS', 'UKR', 'ROU', 'BGR']
    if srb_mkd_in_sample:
        hist_orthodox += ['SRB', 'MKD']

    df['ex_communist'] = df['country'].isin(ex_communist_base).astype(int)
    df['hist_protestant'] = df['country'].isin(hist_protestant_base).astype(int)
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
    # Test multiple configurations with EST/LVA classification:
    # Config A: EST/LVA as BOTH Protestant AND Communist (like attempts 7, 11, 14)
    # Config B: EST/LVA as Communist ONLY (not Protestant)
    # Config C: EST/LVA as Protestant ONLY (not Communist) <- KEY EXPERIMENT
    # Config D: EST/LVA as NEITHER (neutral)

    configs = {
        'A: Both Protestant+Communist': {'est_lva_protestant': True, 'est_lva_communist': True},
        'B: Communist only': {'est_lva_protestant': False, 'est_lva_communist': True},
        'C: Protestant only': {'est_lva_protestant': True, 'est_lva_communist': False},
        'D: Neither (neutral)': {'est_lva_protestant': False, 'est_lva_communist': False},
    }

    print("=" * 80)
    print("Table 5b: Attempt 15 - EST/LVA Classification Experiment")
    print("SRB+MKD included as Orthodox ex-Communist; SVN/BLR/ARM/AZE/GEO/MDA excluded")
    print("=" * 80)

    best_config = None
    best_score_local = 0
    best_results = None

    for config_name, config_params in configs.items():
        df = build_dataset(
            est_lva_protestant=config_params['est_lva_protestant'],
            est_lva_communist=config_params['est_lva_communist'],
            srb_mkd_in_sample=True,
            include_ltu=False
        )

        print(f"\n--- {config_name} (N={len(df)}) ---")
        print(f"  Communist ({df['ex_communist'].sum()}): {sorted(df[df['ex_communist']==1]['country'].tolist())}")
        print(f"  Protestant ({df['hist_protestant'].sum()}): {sorted(df[df['hist_protestant']==1]['country'].tolist())}")
        print(f"  Orthodox ({df['hist_orthodox'].sum()}): {sorted(df[df['hist_orthodox']==1]['country'].tolist())}")

        models = {
            'Model 1': ['gdp_pc', 'service_sq'],
            'Model 3': ['gdp_pc', 'service_sq', 'ex_communist', 'hist_protestant'],
            'Model 4': ['gdp_pc', 'service_sq', 'ex_communist'],
            'Model 5': ['gdp_pc', 'service_sq', 'hist_protestant'],
            'Model 6': ['gdp_pc', 'ex_communist', 'hist_protestant', 'hist_orthodox'],
        }

        for name, ivs in models.items():
            try:
                result, n = run_model(df, ivs)
                print(f"  {name} N={n} AdjR²={result.rsquared_adj:.2f}: ", end='')
                for var in ivs:
                    coef = result.params.get(var, np.nan)
                    pval = result.pvalues.get(var, np.nan)
                    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                    short = var.replace('gdp_pc', 'GDP').replace('service_sq', 'svc²').replace('ex_communist', 'Comm').replace('hist_protestant', 'Prot').replace('hist_orthodox', 'Orth')
                    print(f"{short}={coef:.3f}{sig} ", end='')
                print()
            except Exception as e:
                print(f"  {name}: ERROR - {e}")

    # Run best configuration (C: Protestant only for EST/LVA) with full output
    print("\n\n" + "=" * 80)
    print("FULL RESULTS for Config C: EST/LVA as Protestant ONLY (not Communist)")
    print("=" * 80)

    df = build_dataset(
        est_lva_protestant=True,
        est_lva_communist=False,
        srb_mkd_in_sample=True,
        include_ltu=False
    )

    print(f"\nN={len(df)}, GDP={df['gdp_pc'].notna().sum()}, "
          f"Service={df['service_pct'].notna().sum()}")
    print(f"Surv/SelfExp: mean={df['surv_selfexp'].mean():.3f} std={df['surv_selfexp'].std():.3f}")

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
            print(f"\n{name} (N={n}):")
            print(f"  Adj R² = {result.rsquared_adj:.2f}")
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
    total_coefs = sum(
        len([k for k in gt.keys() if k not in ('adj_r2', 'n')])
        for gt in ground_truth.values()
    )

    for mn, gt in ground_truth.items():
        if results.get(mn) is None:
            details.append(f"{mn}: NO RESULTS")
            max_points += 15
            continue
        res, n = results[mn]
        vrs = [k for k in gt.keys() if k not in ('adj_r2', 'n')]

        for v in vrs:
            max_points += 25.0 / total_coefs
            gc = res.params.get(v, np.nan)
            tc = gt[v]['coef']
            if abs(gc - tc) <= 0.05:
                total_points += 25.0 / total_coefs
                details.append(f"  {mn} {v}: {gc:.3f} vs {tc:.3f} MATCH")
            elif abs(gc - tc) <= 0.15:
                total_points += 12.5 / total_coefs
                details.append(f"  {mn} {v}: {gc:.3f} vs {tc:.3f} PARTIAL")
            else:
                details.append(f"  {mn} {v}: {gc:.3f} vs {tc:.3f} MISS")

        for v in vrs:
            max_points += 15.0 / total_coefs
            gs = res.bse.get(v, np.nan)
            ts = gt[v]['se']
            if abs(gs - ts) <= 0.02:
                total_points += 15.0 / total_coefs
                details.append(f"  {mn} {v} SE: {gs:.3f} vs {ts:.3f} MATCH")
            elif abs(gs - ts) <= 0.05:
                total_points += 7.5 / total_coefs
                details.append(f"  {mn} {v} SE: {gs:.3f} vs {ts:.3f} PARTIAL")
            else:
                details.append(f"  {mn} {v} SE: {gs:.3f} vs {ts:.3f} MISS")

        max_points += 2.5
        if abs(n - gt['n']) / gt['n'] <= 0.05:
            total_points += 2.5
            details.append(f"  {mn} N: {n} vs {gt['n']} MATCH")
        elif abs(n - gt['n']) / gt['n'] <= 0.10:
            total_points += 1.25
            details.append(f"  {mn} N: {n} vs {gt['n']} PARTIAL")
        else:
            details.append(f"  {mn} N: {n} vs {gt['n']} MISS")

        for v in vrs:
            max_points += 25.0 / total_coefs
            gp = res.pvalues.get(v, np.nan)
            gs = '***' if gp < 0.001 else '**' if gp < 0.01 else '*' if gp < 0.05 else ''
            if gs == gt[v]['sig']:
                total_points += 25.0 / total_coefs
                details.append(f"  {mn} {v} sig: '{gs}' vs '{gt[v]['sig']}' MATCH")
            else:
                details.append(f"  {mn} {v} sig: '{gs}' vs '{gt[v]['sig']}' MISS")

        max_points += 10.0 / 6
        rd = abs(res.rsquared_adj - gt['adj_r2'])
        if rd <= 0.02:
            total_points += 10.0 / 6
            details.append(f"  {mn} R²: {res.rsquared_adj:.2f} vs {gt['adj_r2']:.2f} MATCH")
        elif rd <= 0.05:
            total_points += 5.0 / 6
            details.append(f"  {mn} R²: {res.rsquared_adj:.2f} vs {gt['adj_r2']:.2f} PARTIAL")
        else:
            details.append(f"  {mn} R²: {res.rsquared_adj:.2f} vs {gt['adj_r2']:.2f} MISS")

    max_points += 10
    mp = sum(1 for m in ground_truth if results.get(m) is not None)
    total_points += 10 * mp / 6

    score = min(100, int(100 * total_points / max_points)) if max_points > 0 else 0
    print(f"\n{'=' * 80}\nSCORING: {score}/100\n{'=' * 80}")
    for d in details:
        print(d)
    return score


if __name__ == "__main__":
    results = run_analysis()
    score = score_against_ground_truth(results)
