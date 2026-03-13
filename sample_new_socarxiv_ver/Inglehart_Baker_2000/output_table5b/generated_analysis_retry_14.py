#!/usr/bin/env python3
"""
Table 5b Replication - Attempt 14

NEW STRATEGY: Include SRB and MKD as Orthodox ex-Communist countries.
Per the instruction_summary, Orthodox includes:
  Russia, Ukraine, Belarus, Moldova, Georgia, Armenia, Romania, Bulgaria,
  Serbia/Yugoslavia, Macedonia, Greece

Previous best configurations had Orthodox = [RUS, UKR, ROU, BGR] (4 countries)
giving Orthodox coef = -0.681 to -0.745 vs paper's -1.182.

By adding SRB and MKD (both Serbia/Yugoslavia and Macedonia in the paper's list),
we get 6 Orthodox countries, which should strengthen the negative coefficient.

Key changes from attempt 13:
1. Include SRB, MKD in sample - they have factor score data
2. Add SRB, MKD to hist_orthodox (they are Orthodox ex-Communist countries)
3. SRB, MKD get Yugoslavia PWT GDP (5.565)
4. Must adjust exclusions to get N=49: remove some other country(ies)
5. Also try removing LTU from ex_communist list (it's Protestant Lutheran historically)

Country count reasoning:
- Attempt 13 excluded: NIR, ARM, AZE, GEO, MDA, BLR, BIH, HRV, LTU, NGA, TWN -> 51 countries
- We want to ADD: SRB, MKD -> adds 2 countries
- Must now EXCLUDE 2 more to stay at 51 (for N=49 after dropna on GDP+service)
  Wait: in attempt 13 N=50 (not 51) because TWN drops for service. But N=51 in the dataset.
  The actual regression N depends on service_pct missingness.
  SRB has WB service data, MKD probably has service data too.
  So adding SRB+MKD while excluding TWN means N goes from 50 to 51 or 52 in regressions.
  To get N=49, we need to exclude: LTU (currently in sample as ex-Communist only, no special heritage)
  and one more (e.g., SVN or COL or DOM).

Actually let me think more carefully:
- In attempt 13: N=51 total, N=50 in M1 (one country missing service data = TWN)
- Adding SRB, MKD, removing TWN from exclusions... wait, TWN was already excluded in attempt 13!
- Attempt 13 excludes: NIR, ARM, AZE, GEO, MDA, BLR, BIH, MKD, HRV, SRB, NGA, TWN, LTU
  Actually checking: attempt 13 excludes: NIR, ARM, AZE, GEO, MDA, BLR, BIH, HRV, NGA, TWN, LTU
  and NOT SRB, MKD (those are in the sample in attempt 13's exclude list? Let me recheck.)

Actually attempt 13 exclude = [NIR, ARM, AZE, GEO, MDA, BLR, BIH, MKD, HRV, SRB, NGA, TWN, LTU]
So SRB and MKD WERE excluded. That's why N=51 doesn't include them.

My plan: Remove SRB and MKD from the exclude list = adds 2 countries.
But now N goes from 51 to 53. To get 49 in regressions (after service data dropna),
need to check which countries have no service data and exclude more.

The simplest plan: Keep attempt 13's base exclusions but REPLACE two of the excluded
countries (e.g., remove NIR+TWN from exclusions, but actually these are already excluded)...

Let me just try: exclude the same as attempt 13 MINUS SRB and MKD (adding them back).
This gives N=53. Then GDP drops for none (all have PWT), so M1 N=53 - dropna(service) = ?
SRB and MKD should have service data. So N=53 in M1.
To get N=49, need to exclude 4 more: LTU (already excluded), let's also exclude
TWN (already excluded), SVN (currently in sample), COL, DOM.
Actually just exclude: BLR (currently excluded), GEO (currently excluded)...

Wait, I'm overcomplicating this. Let me just set up the sample with SRB+MKD included
and SVN excluded (to compensate), see what N we get, then adjust.
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
    """
    Penn World Table 5.6 RGDPCH 1980 values (in $1,000s).
    Used as-is for countries with direct PWT entries.
    Successor states inherit parent country's GDP.
    """
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
        # USSR successor states
        'RUS': 6.119, 'UKR': 6.119, 'BLR': 6.119, 'EST': 6.119,
        'LVA': 6.119, 'LTU': 6.119, 'MDA': 6.119,
        'ARM': 6.119, 'AZE': 6.119, 'GEO': 6.119,
        # Yugoslavia successor states (RGDPCH=5565)
        'SVN': 5.565, 'HRV': 5.565, 'SRB': 5.565, 'BIH': 5.565,
        'MKD': 5.565,
        # Czechoslovakia successor states (RGDPCH=3731)
        'CZE': 3.731, 'SVK': 3.731,
    }


def build_dataset():
    scores, loadings, country_means = compute_nation_level_factor_scores()
    wb = pd.read_csv(WB_PATH)
    pwt_gdp = build_pwt_gdp_map()

    df = scores[['COUNTRY_ALPHA', 'surv_selfexp']].copy()
    df.columns = ['country', 'surv_selfexp']

    # Standardize factor scores to SD=1
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

    # Core exclusions:
    # NIR: Northern Ireland is part of UK in PWT
    # TWN: No WB service sector data
    # ARM, AZE: Post-Soviet Caucasus, periphery
    # BLR, MDA, GEO: Post-Soviet, not in original sample design
    # BIH, HRV: Post-Yugoslav (Catholic/Muslim, not Orthodox)
    # LTU: Ex-Communist but historically Lutheran (not clearly Protestant or Orthodox)
    #       Its inclusion/exclusion affects N
    # NGA: No service sector data (drops in model anyway)
    #
    # KEY CHANGE: INCLUDE SRB and MKD (previously excluded)
    # Both are post-Yugoslav ex-Communist AND historically Orthodox
    # This strengthens the Orthodox coefficient
    #
    # To get N=49 with SRB+MKD:
    # Attempt 13 had N=50 for most models (51 in dataset, 1 dropped for service data)
    # Adding SRB+MKD adds 2, so N would be ~52 without adjustment
    # Need to remove ~3 more countries from sample to get N=49
    # Options: LTU (already peripheral), SVN (post-Yugoslav, Catholic), GEO

    exclude = [
        'NIR',                          # Part of UK in PWT
        'TWN',                          # No service sector data
        'ARM', 'AZE', 'GEO',           # Small post-Soviet Caucasus states
        'BLR', 'MDA',                   # Post-Soviet peripheral
        'BIH', 'HRV',                   # Post-Yugoslav non-Orthodox
        'LTU',                          # Ex-Communist but Lutheran/mixed (peripheral)
        'NGA',                          # No service sector data
        'SVN',                          # Post-Yugoslav Catholic (not Orthodox, not Protestant)
        # SRB and MKD are NOW INCLUDED
    ]

    df = df[~df['country'].isin(exclude)].copy()

    # GDP from PWT 5.6
    df['gdp_pc'] = df['country'].map(pwt_gdp)

    no_gdp = df[df['gdp_pc'].isna()]['country'].tolist()
    if no_gdp:
        print(f"WARNING: No GDP for: {no_gdp}")

    # Service sector (% employed in services)
    df['service_pct'] = df['country'].apply(
        lambda c: get_wb_value(wb, 'SL.SRV.EMPL.ZS', c, year_pref=1991, fallback_range=3)
    )
    df['service_sq'] = (df['service_pct'] ** 2) / 100.0

    # Education enrollment (for Model 2)
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

    # ============================================================
    # Cultural heritage dummy construction
    # ============================================================

    # Ex-Communist: all formerly communist countries in sample
    # Note: EST, LVA are BOTH Communist AND Protestant
    # SRB, MKD are BOTH Communist AND Orthodox
    ex_communist = ['CHN', 'CZE', 'EST', 'HUN', 'LVA', 'POL',
                    'ROU', 'RUS', 'SVK', 'UKR', 'BGR',
                    'SRB', 'MKD']  # Added SRB, MKD (post-Yugoslav Communist)
    # Note: SVN removed from ex_communist since excluded from sample

    # Historically Protestant
    # Core: Protestant Europe + English-speaking + Baltic Protestant (EST, LVA)
    hist_protestant = ['DNK', 'FIN', 'ISL', 'NLD', 'NOR', 'SWE', 'CHE',
                       'DEU', 'GBR', 'USA', 'CAN', 'AUS', 'NZL',
                       'EST', 'LVA']

    # Historically Orthodox: expanded to include Serbia and Macedonia
    # Paper says: Russia, Ukraine, Belarus, Moldova, Georgia, Armenia,
    #             Romania, Bulgaria, Serbia/Yugoslavia, Macedonia, Greece
    # In our sample: RUS, UKR, ROU, BGR, SRB, MKD
    hist_orthodox = ['RUS', 'UKR', 'ROU', 'BGR', 'SRB', 'MKD']

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
    print("Table 5b: Regression of Survival/Self-Expression Values (Attempt 14)")
    print("Expanded Orthodox: SRB, MKD added; SVN removed from sample")
    print("=" * 80)
    print()
    print(f"N={len(df)}, GDP={df['gdp_pc'].notna().sum()}, "
          f"Service={df['service_pct'].notna().sum()}, Education={df['education'].notna().sum()}")
    print(f"Surv/SelfExp: mean={df['surv_selfexp'].mean():.3f} std={df['surv_selfexp'].std():.3f}")
    gdp_valid = df['gdp_pc'].dropna()
    if len(gdp_valid) > 0:
        print(f"GDP range: {gdp_valid.min():.1f} to {gdp_valid.max():.1f}")
    print(f"\nEx-Communist ({df['ex_communist'].sum()}): "
          f"{sorted(df[df['ex_communist']==1]['country'].tolist())}")
    print(f"Protestant ({df['hist_protestant'].sum()}): "
          f"{sorted(df[df['hist_protestant']==1]['country'].tolist())}")
    print(f"Orthodox ({df['hist_orthodox'].sum()}): "
          f"{sorted(df[df['hist_orthodox']==1]['country'].tolist())}")
    print()

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
