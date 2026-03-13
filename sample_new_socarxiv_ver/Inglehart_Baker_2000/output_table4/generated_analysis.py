#!/usr/bin/env python3
"""
Table 4 replication: Standardized Coefficients from the Regression of
Traditional/Secular-Rational Values and Survival/Self-Expression Values
on Economic Development and Cultural Heritage Zone dummies.

8 separate regressions per DV (one per cultural zone dummy), each with:
- DV: Factor score (Traditional/Secular-Rational or Survival/Self-Expression)
- IVs: Zone dummy, GDP per capita 1980, Industrial employment 1980 (for Trad/Sec-Rat)
        or Service employment 1980 (for Surv/Self-Exp)
All variables standardized before regression (z-scored).
"""

import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from shared_factor_analysis import (
    compute_nation_level_factor_scores, load_world_bank_data,
    get_cultural_zones, COUNTRY_NAMES
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output_table4")
WB_PATH = os.path.join(BASE_DIR, "data/world_bank_indicators.csv")


def get_wb_data_for_country(wb, indicator, economy, preferred_year=1980):
    """Get WB indicator for a country, trying preferred year then nearest."""
    sub = wb[(wb['indicator'] == indicator) & (wb['economy'] == economy)]
    if len(sub) == 0:
        return np.nan
    row = sub.iloc[0]
    # Try preferred year first, then nearby years
    for delta in range(0, 20):
        for y in [preferred_year + delta, preferred_year - delta]:
            col = f'YR{y}'
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    try:
                        return float(val)
                    except:
                        pass
    return np.nan


def build_country_data():
    """Build country-level dataset with factor scores and economic indicators."""
    # Step 1: Compute factor scores (handling East/West Germany separately)
    scores, loadings, country_means = compute_nation_level_factor_scores()

    # Step 2: Load WB data
    wb = pd.read_csv(WB_PATH)

    # Step 3: Build economic indicators for each country
    econ_data = []

    # Map WVS country codes to WB economy codes
    wvs_to_wb = {
        'ALB': 'ALB', 'ARG': 'ARG', 'ARM': 'ARM', 'AUS': 'AUS',
        'AUT': 'AUT', 'AZE': 'AZE', 'BGD': 'BGD', 'BLR': 'BLR',
        'BEL': 'BEL', 'BIH': 'BIH', 'BRA': 'BRA', 'BGR': 'BGR',
        'CAN': 'CAN', 'CHL': 'CHL', 'CHN': 'CHN', 'COL': 'COL',
        'HRV': 'HRV', 'CZE': 'CZE', 'DNK': 'DNK', 'DOM': 'DOM',
        'EST': 'EST', 'FIN': 'FIN', 'FRA': 'FRA', 'GEO': 'GEO',
        'DEU': 'DEU', 'GHA': 'GHA', 'GBR': 'GBR', 'HUN': 'HUN',
        'ISL': 'ISL', 'IND': 'IND', 'IRL': 'IRL', 'ITA': 'ITA',
        'JPN': 'JPN', 'KOR': 'KOR', 'LVA': 'LVA', 'LTU': 'LTU',
        'MKD': 'MKD', 'MEX': 'MEX', 'MDA': 'MDA', 'NLD': 'NLD',
        'NZL': 'NZL', 'NGA': 'NGA', 'NIR': 'GBR', 'NOR': 'NOR',
        'PAK': 'PAK', 'PER': 'PER', 'PHL': 'PHL', 'POL': 'POL',
        'PRT': 'PRT', 'PRI': 'PRI', 'ROU': 'ROU', 'RUS': 'RUS',
        'SRB': 'SRB', 'SVK': 'SVK', 'SVN': 'SVN', 'ZAF': 'ZAF',
        'ESP': 'ESP', 'SWE': 'SWE', 'CHE': 'CHE', 'TWN': 'TWN',
        'TUR': 'TUR', 'UKR': 'UKR', 'USA': 'USA', 'URY': 'URY',
        'VEN': 'VEN'
    }

    for _, row in scores.iterrows():
        cc = row['COUNTRY_ALPHA']
        wb_cc = wvs_to_wb.get(cc, cc)

        # GDP per capita PPP (constant 2017 intl $) - use earliest available (1990)
        gdp = get_wb_data_for_country(wb, 'NY.GDP.PCAP.PP.KD', wb_cc, 1990)

        # Industrial employment (%) - earliest available is 1991
        ind_emp = get_wb_data_for_country(wb, 'SL.IND.EMPL.ZS', wb_cc, 1991)

        # Service employment (%) - earliest available is 1991
        srv_emp = get_wb_data_for_country(wb, 'SL.SRV.EMPL.ZS', wb_cc, 1991)

        econ_data.append({
            'COUNTRY_ALPHA': cc,
            'trad_secrat': row['trad_secrat'],
            'surv_selfexp': row['surv_selfexp'],
            'gdp_pc': gdp,
            'ind_emp': ind_emp,
            'srv_emp': srv_emp
        })

    df = pd.DataFrame(econ_data)

    # Convert GDP to thousands
    df['gdp_pc'] = df['gdp_pc'] / 1000.0

    return df


def assign_cultural_zones(df):
    """Assign cultural zone dummies to each country."""
    zones = get_cultural_zones()

    # The paper's 8 regressions use these zones (Table 4 lists them):
    zone_names = [
        'Ex-Communist', 'Protestant Europe', 'English-speaking',
        'Latin America', 'Africa', 'South Asia', 'Orthodox', 'Confucian'
    ]

    for zone_name in zone_names:
        col_name = zone_name.replace(' ', '_').replace('-', '_')
        countries_in_zone = zones.get(zone_name, [])
        df[col_name] = df['COUNTRY_ALPHA'].isin(countries_in_zone).astype(int)

    return df


def standardize(series):
    """Z-score standardization."""
    s = series.astype(float)
    return (s - s.mean()) / s.std()


def run_zone_regression(df, dv_col, zone_col, econ_cols, zone_name):
    """
    Run OLS regression with standardized variables.
    Returns dict with coefficients, t-values, pvalues, adj_r2, N.
    """
    # Select complete cases
    all_cols = [dv_col, zone_col] + econ_cols
    subset = df[all_cols].dropna()

    if len(subset) < 5:
        return None

    N = len(subset)

    # Standardize all variables
    y = standardize(subset[dv_col])
    X_data = {}
    for col in [zone_col] + econ_cols:
        X_data[col] = standardize(subset[col])

    X = pd.DataFrame(X_data)
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        return None

    results = {
        'zone_name': zone_name,
        'N': N,
        'adj_r2': model.rsquared_adj,
        'r2': model.rsquared,
        'coefficients': {},
        'tvalues': {},
        'pvalues': {}
    }

    for col in [zone_col] + econ_cols:
        results['coefficients'][col] = model.params[col]
        results['tvalues'][col] = model.tvalues[col]
        results['pvalues'][col] = model.pvalues[col]

    return results


def get_significance(pvalue):
    """Return significance stars."""
    if pvalue < 0.001:
        return '***'
    elif pvalue < 0.01:
        return '**'
    elif pvalue < 0.05:
        return '*'
    return ''


def run_analysis():
    """Run all Table 4 regressions."""
    df = build_country_data()
    df = assign_cultural_zones(df)

    print(f"Total countries with factor scores: {len(df)}")
    print(f"Countries with GDP data: {df['gdp_pc'].notna().sum()}")
    print(f"Countries with industrial emp data: {df['ind_emp'].notna().sum()}")
    print(f"Countries with service emp data: {df['srv_emp'].notna().sum()}")
    print()

    # Zone column mapping
    zone_info = [
        ('Ex-Communist', 'Ex_Communist'),
        ('Protestant Europe', 'Protestant_Europe'),
        ('English-speaking', 'English_speaking'),
        ('Latin America', 'Latin_America'),
        ('Africa', 'Africa'),
        ('South Asia', 'South_Asia'),
        ('Orthodox', 'Orthodox'),
        ('Confucian', 'Confucian'),
    ]

    all_results = []

    for zone_name, zone_col in zone_info:
        # Trad/Sec-Rat regression: zone dummy + GDP + industrial emp
        trad_result = run_zone_regression(
            df, 'trad_secrat', zone_col, ['gdp_pc', 'ind_emp'], zone_name
        )

        # Surv/Self-Exp regression: zone dummy + GDP + service emp
        surv_result = run_zone_regression(
            df, 'surv_selfexp', zone_col, ['gdp_pc', 'srv_emp'], zone_name
        )

        all_results.append((zone_name, zone_col, trad_result, surv_result))

    # Print results
    print("=" * 80)
    print("TABLE 4: Standardized Coefficients")
    print("=" * 80)
    print(f"{'':50s} {'Trad/Sec-Rat':>15s}  {'Surv/Self-Exp':>15s}")
    print(f"{'':50s} {'Coeff (t)':>15s}  {'Coeff (t)':>15s}")
    print("-" * 80)

    for zone_name, zone_col, trad_result, surv_result in all_results:
        if trad_result and surv_result:
            # Zone dummy
            t_coef = trad_result['coefficients'][zone_col]
            t_t = trad_result['tvalues'][zone_col]
            t_sig = get_significance(trad_result['pvalues'][zone_col])
            s_coef = surv_result['coefficients'][zone_col]
            s_t = surv_result['tvalues'][zone_col]
            s_sig = get_significance(surv_result['pvalues'][zone_col])
            print(f"{zone_name + ' zone (=1)':50s} {t_coef:+.3f}{t_sig} ({t_t:.2f})  {s_coef:+.3f}{s_sig} ({s_t:.2f})")

            # GDP
            t_coef = trad_result['coefficients']['gdp_pc']
            t_t = trad_result['tvalues']['gdp_pc']
            t_sig = get_significance(trad_result['pvalues']['gdp_pc'])
            s_coef = surv_result['coefficients']['gdp_pc']
            s_t = surv_result['tvalues']['gdp_pc']
            s_sig = get_significance(surv_result['pvalues']['gdp_pc'])
            print(f"{'  Real GDP per capita, 1980':50s} {t_coef:+.3f}{t_sig} ({t_t:.2f})  {s_coef:+.3f}{s_sig} ({s_t:.2f})")

            # Industrial emp (for trad)
            t_coef = trad_result['coefficients']['ind_emp']
            t_t = trad_result['tvalues']['ind_emp']
            t_sig = get_significance(trad_result['pvalues']['ind_emp'])
            print(f"{'  % employed in industrial sector, 1980':50s} {t_coef:+.3f}{t_sig} ({t_t:.2f})  {'---':>15s}")

            # Service emp (for surv)
            s_coef = surv_result['coefficients']['srv_emp']
            s_t = surv_result['tvalues']['srv_emp']
            s_sig = get_significance(surv_result['pvalues']['srv_emp'])
            print(f"{'  % employed in service sector, 1980':50s} {'---':>15s}  {s_coef:+.3f}{s_sig} ({s_t:.2f})")

            # Adj R2
            print(f"{'  Adjusted R²':50s} {trad_result['adj_r2']:.2f}            {surv_result['adj_r2']:.2f}")
            print(f"{'  N':50s} {trad_result['N']}               {surv_result['N']}")
            print()

    print("=" * 80)
    print("Note: Numbers in parentheses are t-values.")
    print("*p < .05  **p < .01  ***p < .001 (two-tailed tests)")

    return all_results


def score_against_ground_truth(all_results):
    """Score results against paper's ground truth values."""
    # Ground truth from paper (Table 4)
    ground_truth = {
        'Ex-Communist': {
            'trad': {'zone': (.424, 3.10, '**'), 'gdp': (.496, 3.57, '***'), 'ind': (.216, 1.43, ''), 'adj_r2': .50},
            'surv': {'zone': (-.393, -4.80, '***'), 'gdp': (.575, 4.13, '***'), 'srv': (.098, .67, ''), 'adj_r2': .73}
        },
        'Protestant Europe': {
            'trad': {'zone': (.370, 3.04, '**'), 'gdp': (.025, .19, ''), 'ind': (.553, 4.83, '***'), 'adj_r2': .50},
            'surv': {'zone': (.232, 2.24, '*'), 'gdp': (.362, 2.12, '*'), 'srv': (.331, 2.06, '*'), 'adj_r2': .63}
        },
        'English-speaking': {
            'trad': {'zone': (-.300, -2.65, '**'), 'gdp': (.394, 3.02, '**'), 'ind': (.468, 3.98, '***'), 'adj_r2': .47},
            'surv': {'zone': (.146, 1.48, ''), 'gdp': (.434, 2.56, '**'), 'srv': (.319, 1.93, '*'), 'adj_r2': .61}
        },
        'Latin America': {
            'trad': {'zone': (-.342, -3.29, '**'), 'gdp': (.195, 1.72, ''), 'ind': (.448, 3.94, '***'), 'adj_r2': .51},
            'surv': {'zone': (.108, .98, ''), 'gdp': (.602, 2.97, '**'), 'srv': (.224, 1.13, ''), 'adj_r2': .60}
        },
        'Africa': {
            'trad': {'zone': (-.189, -1.65, ''), 'gdp': (.211, 1.72, ''), 'ind': (.468, 3.79, '***'), 'adj_r2': .43},
            'surv': {'zone': (.021, .22, ''), 'gdp': (.502, 2.81, '**'), 'srv': (.320, 1.85, ''), 'adj_r2': .59}
        },
        'South Asia': {
            'trad': {'zone': (.070, .51, ''), 'gdp': (.258, 2.04, '*'), 'ind': (.542, 3.87, '***'), 'adj_r2': .40},
            'surv': {'zone': (.212, 2.08, '*'), 'gdp': (.469, 2.90, '**'), 'srv': (.455, 2.63, '**'), 'adj_r2': .62}
        },
        'Orthodox': {
            'trad': {'zone': (.152, 1.26, ''), 'gdp': (.304, 2.31, '*'), 'ind': (.432, 3.13, '**'), 'adj_r2': .42},
            'surv': {'zone': (-.457, -6.94, '***'), 'gdp': (.567, 4.77, '***'), 'srv': (.154, 1.28, ''), 'adj_r2': .80}
        },
        'Confucian': {
            'trad': {'zone': (.397, 4.15, '***'), 'gdp': (.304, 2.83, '**'), 'ind': (.505, 4.76, '***'), 'adj_r2': .56},
            'surv': {'zone': (-.020, -.21, ''), 'gdp': (.491, 2.90, '**'), 'srv': (.323, 1.95, '*'), 'adj_r2': .59}
        }
    }

    total_points = 0
    max_points = 0
    details = []

    for zone_name, zone_col, trad_result, surv_result in all_results:
        if zone_name not in ground_truth:
            continue
        gt = ground_truth[zone_name]

        if trad_result is None or surv_result is None:
            details.append(f"{zone_name}: NO RESULTS (regression failed)")
            max_points += 50  # 25 per DV approx
            continue

        # Score Trad/Sec-Rat
        for var_key, col_name in [('zone', zone_col), ('gdp', 'gdp_pc'), ('ind', 'ind_emp')]:
            gt_coef, gt_t, gt_sig = gt['trad'][var_key]
            gen_coef = trad_result['coefficients'].get(col_name, None)
            gen_t = trad_result['tvalues'].get(col_name, None)
            gen_p = trad_result['pvalues'].get(col_name, None)

            if gen_coef is not None:
                gen_sig = get_significance(gen_p)

                # Coefficient match (within 0.05 = full, within 0.15 = partial)
                max_points += 2
                diff = abs(gen_coef - gt_coef)
                if diff <= 0.05:
                    total_points += 2
                    details.append(f"  {zone_name} trad {var_key}: coef {gen_coef:.3f} vs {gt_coef:.3f} MATCH (diff={diff:.3f})")
                elif diff <= 0.15:
                    total_points += 1
                    details.append(f"  {zone_name} trad {var_key}: coef {gen_coef:.3f} vs {gt_coef:.3f} PARTIAL (diff={diff:.3f})")
                else:
                    details.append(f"  {zone_name} trad {var_key}: coef {gen_coef:.3f} vs {gt_coef:.3f} MISS (diff={diff:.3f})")

                # Significance match
                max_points += 1
                if gen_sig == gt_sig:
                    total_points += 1
                    details.append(f"  {zone_name} trad {var_key}: sig '{gen_sig}' vs '{gt_sig}' MATCH")
                else:
                    details.append(f"  {zone_name} trad {var_key}: sig '{gen_sig}' vs '{gt_sig}' MISS")

        # Adj R2 trad
        max_points += 2
        diff_r2 = abs(trad_result['adj_r2'] - gt['trad']['adj_r2'])
        if diff_r2 <= 0.02:
            total_points += 2
            details.append(f"  {zone_name} trad R2: {trad_result['adj_r2']:.2f} vs {gt['trad']['adj_r2']:.2f} MATCH")
        elif diff_r2 <= 0.08:
            total_points += 1
            details.append(f"  {zone_name} trad R2: {trad_result['adj_r2']:.2f} vs {gt['trad']['adj_r2']:.2f} PARTIAL")
        else:
            details.append(f"  {zone_name} trad R2: {trad_result['adj_r2']:.2f} vs {gt['trad']['adj_r2']:.2f} MISS")

        # Score Surv/Self-Exp
        for var_key, col_name in [('zone', zone_col), ('gdp', 'gdp_pc'), ('srv', 'srv_emp')]:
            gt_coef, gt_t, gt_sig = gt['surv'][var_key]
            gen_coef = surv_result['coefficients'].get(col_name, None)
            gen_t = surv_result['tvalues'].get(col_name, None)
            gen_p = surv_result['pvalues'].get(col_name, None)

            if gen_coef is not None:
                gen_sig = get_significance(gen_p)

                # Coefficient match
                max_points += 2
                diff = abs(gen_coef - gt_coef)
                if diff <= 0.05:
                    total_points += 2
                    details.append(f"  {zone_name} surv {var_key}: coef {gen_coef:.3f} vs {gt_coef:.3f} MATCH (diff={diff:.3f})")
                elif diff <= 0.15:
                    total_points += 1
                    details.append(f"  {zone_name} surv {var_key}: coef {gen_coef:.3f} vs {gt_coef:.3f} PARTIAL (diff={diff:.3f})")
                else:
                    details.append(f"  {zone_name} surv {var_key}: coef {gen_coef:.3f} vs {gt_coef:.3f} MISS (diff={diff:.3f})")

                # Significance match
                max_points += 1
                if gen_sig == gt_sig:
                    total_points += 1
                    details.append(f"  {zone_name} surv {var_key}: sig '{gen_sig}' vs '{gt_sig}' MATCH")
                else:
                    details.append(f"  {zone_name} surv {var_key}: sig '{gen_sig}' vs '{gt_sig}' MISS")

        # Adj R2 surv
        max_points += 2
        diff_r2 = abs(surv_result['adj_r2'] - gt['surv']['adj_r2'])
        if diff_r2 <= 0.02:
            total_points += 2
            details.append(f"  {zone_name} surv R2: {surv_result['adj_r2']:.2f} vs {gt['surv']['adj_r2']:.2f} MATCH")
        elif diff_r2 <= 0.08:
            total_points += 1
            details.append(f"  {zone_name} surv R2: {surv_result['adj_r2']:.2f} vs {gt['surv']['adj_r2']:.2f} PARTIAL")
        else:
            details.append(f"  {zone_name} surv R2: {surv_result['adj_r2']:.2f} vs {gt['surv']['adj_r2']:.2f} MISS")

    score = int(100 * total_points / max_points) if max_points > 0 else 0

    print("\n" + "=" * 80)
    print(f"SCORING: {total_points}/{max_points} raw points = {score}/100")
    print("=" * 80)
    for d in details:
        print(d)

    return score, details


if __name__ == "__main__":
    all_results = run_analysis()
    score, details = score_against_ground_truth(all_results)
    print(f"\nFINAL SCORE: {score}/100")
