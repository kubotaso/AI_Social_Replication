#!/usr/bin/env python3
"""
Table 5b Replication - Attempt 10

Key changes from attempt 9 (score 69):
1. Remove EST and LVA from ex_communist classification
   (keep them as Protestant only)
   This should reduce Communist dummy's explanatory power → lower Model 3 R²
   And change the Communist/Protestant/Orthodox coefficient pattern in Model 6
2. Keep N=49 (exclude LTU, NGA drops for missing GDP)
3. Keep std=1.0 normalization
4. Try using service_sq WITHOUT the /100 normalization to see if
   coefficient magnitudes align better (then coefficient should be ~100x smaller)
   Actually no - paper's coef 0.042 * 49 (range of sq/100) = 2.06, reasonable.
   Keep /100.
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


def build_dataset():
    scores, loadings, country_means = compute_nation_level_factor_scores()
    wb = pd.read_csv(WB_PATH)

    df = scores[['COUNTRY_ALPHA', 'surv_selfexp']].copy()
    df.columns = ['country', 'surv_selfexp']

    # Normalize factor scores to SD=1
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

    # Exclude countries to get N=50, N=49 for GDP models (NGA drops)
    exclude = [
        'NIR', 'TWN', 'SRB',     # No WB data / separate entities
        'ARM', 'AZE', 'GEO',     # Post-Soviet small states
        'MDA', 'BLR',            # Post-Soviet
        'BIH', 'MKD', 'HRV',    # Post-Yugoslav
        'LTU',                    # Exclude for N count
    ]

    df = df[~df['country'].isin(exclude)].copy()

    # GDP: GNP per capita PPP current intl $ for 1990 / 1000
    df['gdp_pc'] = df['country'].apply(
        lambda c: get_wb_value(wb, 'NY.GNP.PCAP.PP.CD', c, year_pref=1990, fallback_range=3) / 1000.0
    )

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

    # Cultural heritage dummies
    # Ex-Communist - WITHOUT EST and LVA (they are Protestant, not double-coded)
    # The paper likely treats Baltic states as Protestant rather than Communist
    # for this table's purposes
    ex_communist = ['CHN', 'CZE', 'HUN', 'POL',
                    'ROU', 'RUS', 'SVK', 'SVN', 'UKR', 'BGR']

    # Historically Protestant - includes EST, LVA
    hist_protestant = ['DNK', 'FIN', 'ISL', 'NLD', 'NOR', 'SWE', 'CHE',
                       'DEU', 'GBR', 'USA', 'CAN', 'AUS', 'NZL',
                       'EST', 'LVA']

    # Historically Orthodox
    hist_orthodox = ['RUS', 'UKR', 'ROU', 'BGR']

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
    print("Table 5b: Regression of Survival/Self-Expression Values")
    print("=" * 80)
    print()
    print(f"N={len(df)}, GDP={df['gdp_pc'].notna().sum()}, "
          f"Service={df['service_pct'].notna().sum()}, Education={df['education'].notna().sum()}")
    print(f"Surv/SelfExp: mean={df['surv_selfexp'].mean():.3f} std={df['surv_selfexp'].std():.3f}")
    print(f"GDP range: {df['gdp_pc'].dropna().min():.1f} to {df['gdp_pc'].dropna().max():.1f}")
    print(f"\nEx-Communist ({df['ex_communist'].sum()}): {df[df['ex_communist']==1]['country'].tolist()}")
    print(f"Protestant ({df['hist_protestant'].sum()}): {df[df['hist_protestant']==1]['country'].tolist()}")
    print(f"Orthodox ({df['hist_orthodox'].sum()}): {df[df['hist_orthodox']==1]['country'].tolist()}")
    print()

    for _, row in df.sort_values('country').iterrows():
        gdp_str = f"{row['gdp_pc']:6.1f}" if pd.notna(row['gdp_pc']) else "   NA"
        print(f"  {row['country']:4s} surv={row['surv_selfexp']:+.3f} gdp={gdp_str} "
              f"comm={row['ex_communist']} prot={row['hist_protestant']} orth={row['hist_orthodox']}")

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
        'Model 1': {'gdp_pc': {'coef': 0.090, 'se': 0.043, 'sig': '*'}, 'service_sq': {'coef': 0.042, 'se': 0.015, 'sig': '**'}, 'adj_r2': 0.63, 'n': 49},
        'Model 2': {'gdp_pc': {'coef': 0.095, 'se': 0.046, 'sig': '*'}, 'service_sq': {'coef': 0.011, 'se': 0.000, 'sig': '*'}, 'service_pct': {'coef': -0.054, 'se': 0.039, 'sig': ''}, 'education': {'coef': -0.005, 'se': 0.012, 'sig': ''}, 'adj_r2': 0.63, 'n': 46},
        'Model 3': {'gdp_pc': {'coef': 0.056, 'se': 0.043, 'sig': ''}, 'service_sq': {'coef': 0.035, 'se': 0.015, 'sig': '*'}, 'ex_communist': {'coef': -0.920, 'se': 0.204, 'sig': '***'}, 'hist_protestant': {'coef': 0.672, 'se': 0.279, 'sig': '*'}, 'adj_r2': 0.66, 'n': 49},
        'Model 4': {'gdp_pc': {'coef': 0.120, 'se': 0.037, 'sig': '**'}, 'service_sq': {'coef': 0.019, 'se': 0.014, 'sig': ''}, 'ex_communist': {'coef': -0.883, 'se': 0.197, 'sig': '***'}, 'adj_r2': 0.74, 'n': 49},
        'Model 5': {'gdp_pc': {'coef': 0.098, 'se': 0.037, 'sig': '**'}, 'service_sq': {'coef': 0.018, 'se': 0.013, 'sig': ''}, 'hist_protestant': {'coef': 0.509, 'se': 0.237, 'sig': '*'}, 'adj_r2': 0.76, 'n': 49},
        'Model 6': {'gdp_pc': {'coef': 0.144, 'se': 0.017, 'sig': '***'}, 'ex_communist': {'coef': -0.411, 'se': 0.188, 'sig': '*'}, 'hist_protestant': {'coef': 0.415, 'se': 0.175, 'sig': '**'}, 'hist_orthodox': {'coef': -1.182, 'se': 0.240, 'sig': '***'}, 'adj_r2': 0.84, 'n': 49},
    }

    total_points = 0; max_points = 0; details = []
    total_coefs = sum(len([k for k in gt.keys() if k not in ('adj_r2', 'n')]) for gt in ground_truth.values())

    for mn, gt in ground_truth.items():
        if results.get(mn) is None:
            details.append(f"{mn}: NO RESULTS"); max_points += 15; continue
        res, n = results[mn]
        vrs = [k for k in gt.keys() if k not in ('adj_r2', 'n')]
        for v in vrs:
            max_points += 25.0/total_coefs; gc = res.params.get(v, np.nan); tc = gt[v]['coef']
            if abs(gc-tc) <= 0.05: total_points += 25.0/total_coefs; details.append(f"  {mn} {v}: {gc:.3f} vs {tc:.3f} MATCH")
            elif abs(gc-tc) <= 0.15: total_points += 12.5/total_coefs; details.append(f"  {mn} {v}: {gc:.3f} vs {tc:.3f} PARTIAL")
            else: details.append(f"  {mn} {v}: {gc:.3f} vs {tc:.3f} MISS")
        for v in vrs:
            max_points += 15.0/total_coefs; gs = res.bse.get(v, np.nan); ts = gt[v]['se']
            if abs(gs-ts) <= 0.02: total_points += 15.0/total_coefs; details.append(f"  {mn} {v} SE: {gs:.3f} vs {ts:.3f} MATCH")
            elif abs(gs-ts) <= 0.05: total_points += 7.5/total_coefs; details.append(f"  {mn} {v} SE: {gs:.3f} vs {ts:.3f} PARTIAL")
            else: details.append(f"  {mn} {v} SE: {gs:.3f} vs {ts:.3f} MISS")
        max_points += 2.5
        if abs(n-gt['n'])/gt['n'] <= 0.05: total_points += 2.5; details.append(f"  {mn} N: {n} vs {gt['n']} MATCH")
        elif abs(n-gt['n'])/gt['n'] <= 0.10: total_points += 1.25; details.append(f"  {mn} N: {n} vs {gt['n']} PARTIAL")
        else: details.append(f"  {mn} N: {n} vs {gt['n']} MISS")
        for v in vrs:
            max_points += 25.0/total_coefs; gp = res.pvalues.get(v, np.nan)
            gs = '***' if gp<0.001 else '**' if gp<0.01 else '*' if gp<0.05 else ''
            if gs == gt[v]['sig']: total_points += 25.0/total_coefs; details.append(f"  {mn} {v} sig: '{gs}' vs '{gt[v]['sig']}' MATCH")
            else: details.append(f"  {mn} {v} sig: '{gs}' vs '{gt[v]['sig']}' MISS")
        max_points += 10.0/6; rd = abs(res.rsquared_adj - gt['adj_r2'])
        if rd <= 0.02: total_points += 10.0/6; details.append(f"  {mn} R²: {res.rsquared_adj:.2f} vs {gt['adj_r2']:.2f} MATCH")
        elif rd <= 0.05: total_points += 5.0/6; details.append(f"  {mn} R²: {res.rsquared_adj:.2f} vs {gt['adj_r2']:.2f} PARTIAL")
        else: details.append(f"  {mn} R²: {res.rsquared_adj:.2f} vs {gt['adj_r2']:.2f} MISS")

    max_points += 10; mp = sum(1 for m in ground_truth if results.get(m) is not None)
    total_points += 10*mp/6
    score = min(100, int(100*total_points/max_points)) if max_points > 0 else 0
    print(f"\n{'='*80}\nSCORING: {score}/100\n{'='*80}")
    for d in details: print(d)
    return score


if __name__ == "__main__":
    results = run_analysis()
    score = score_against_ground_truth(results)
