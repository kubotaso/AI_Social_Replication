#!/usr/bin/env python3
"""
Table 4 replication - Attempt 3

Key improvements:
- Use 1980 GDP per capita (current USD) from WB API
- Use 1991 employment data (earliest available)
- Restrict sample to ~49 countries matching paper
- Better cultural zone assignments
- Handle East/West Germany issue
"""

import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from shared_factor_analysis import (
    compute_nation_level_factor_scores, FACTOR_ITEMS, COUNTRY_NAMES
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output_table4")
WB_PATH = os.path.join(BASE_DIR, "data/world_bank_indicators.csv")

# 1980 GDP per capita in current USD (from World Bank API)
GDP_1980_CURRENT_USD = {
    'ARG': 2747.50, 'AUS': 10223.29, 'AUT': 10826.97, 'BGD': 206.08,
    'BEL': 12864.00, 'BRA': 1958.57, 'BGR': 2238.80, 'CAN': 11208.20,
    'CHL': 2570.84, 'CHN': 195.15, 'COL': 1279.50, 'DNK': 13822.16,
    'DOM': 1180.18, 'FIN': 11224.94, 'FRA': 12565.16, 'DEU': 12182.78,
    'GHA': 372.25, 'GBR': 10032.06, 'HUN': 2158.22, 'ISL': 15339.91,
    'IND': 271.08, 'IRL': 6372.44, 'ITA': 8476.41, 'JPN': 9668.75,
    'KOR': 1745.58, 'MEX': 3054.62, 'NLD': 13812.16, 'NZL': 7467.17,
    'NGA': 870.36, 'NOR': 15772.24, 'PAK': 287.45, 'PER': 1044.69,
    'PHL': 766.97, 'POL': None,  # No 1980 data for Poland
    'PRT': 3368.37, 'PRI': 4502.84, 'ROU': None,  # No 1980 data
    'ZAF': 3029.97, 'ESP': 6204.14, 'SWE': 17073.04, 'CHE': 19393.88,
    'TUR': 1515.65, 'USA': 12574.79, 'URY': 3442.81, 'VEN': 3874.40,
    # Former Soviet/Yugoslav states - no 1980 data
    'ARM': None, 'AZE': None, 'BLR': None, 'BIH': None, 'HRV': None,
    'CZE': None, 'EST': None, 'GEO': None, 'LVA': None, 'LTU': None,
    'MKD': None, 'MDA': None, 'RUS': None, 'SVK': None, 'SVN': None,
    'UKR': None, 'SRB': None, 'ALB': None, 'TWN': None, 'NIR': None,
}

# 1991 Industrial employment (%) from WB API
IND_EMP_1991 = {
    'ARG': 27.38, 'ARM': 16.27, 'AUS': 23.92, 'AUT': 33.37,
    'AZE': 16.82, 'BGD': 17.18, 'BLR': 38.83, 'BEL': 30.33,
    'BIH': 30.22, 'BRA': 24.98, 'BGR': 45.09, 'CAN': 23.84,
    'CHL': 26.76, 'CHN': 21.43, 'COL': 24.04, 'HRV': 30.40,
    'CZE': 47.85, 'DNK': 26.84, 'DOM': 27.91, 'EST': 36.25,
    'FIN': 28.76, 'FRA': 28.42, 'GEO': 12.25, 'DEU': 38.43,
    'GHA': 8.97, 'GBR': 30.77, 'HUN': 36.13, 'ISL': 25.89,
    'IND': 14.58, 'IRL': 27.48, 'ITA': 34.82, 'JPN': 33.28,
    'KOR': 37.16, 'LVA': 30.61, 'LTU': 31.41, 'MKD': 35.84,
    'MEX': 22.06, 'MDA': 22.04, 'NLD': 24.83, 'NZL': 23.76,
    'NGA': 10.08, 'NOR': 23.52, 'PAK': 23.99, 'PER': 16.22,
    'PHL': 16.61, 'POL': 32.58, 'PRT': 34.55, 'PRI': 23.76,
    'ROU': 39.08, 'RUS': 40.09, 'SVK': 42.28, 'SVN': 40.27,
    'ZAF': 24.64, 'ESP': 32.04, 'SWE': 27.26, 'CHE': 27.52,
    'TUR': 22.81, 'UKR': 32.43, 'USA': 24.37, 'URY': 24.71,
    'VEN': 26.30, 'SRB': 21.98,
}

# 1991 Service employment (%) from WB API
SRV_EMP_1991 = {
    'ARG': 58.31, 'ARM': 37.58, 'AUS': 70.13, 'AUT': 59.29,
    'AZE': 38.26, 'BGD': 12.90, 'BLR': 38.75, 'BEL': 66.91,
    'BIH': 42.86, 'BRA': 58.92, 'BGR': 45.40, 'CAN': 72.67,
    'CHL': 56.52, 'CHN': 18.93, 'COL': 56.02, 'HRV': 47.60,
    'CZE': 44.48, 'DNK': 67.78, 'DOM': 53.88, 'EST': 43.67,
    'FIN': 62.34, 'FRA': 66.17, 'GEO': 40.10, 'DEU': 57.92,
    'GHA': 19.25, 'GBR': 66.81, 'HUN': 52.73, 'ISL': 63.92,
    'IND': 22.29, 'IRL': 58.61, 'ITA': 56.76, 'JPN': 59.52,
    'KOR': 47.36, 'LVA': 49.51, 'LTU': 45.29, 'MKD': 39.71,
    'MEX': 51.99, 'MDA': 40.14, 'NLD': 70.88, 'NZL': 65.47,
    'NGA': 36.08, 'NOR': 70.71, 'PAK': 35.35, 'PER': 50.63,
    'PHL': 39.31, 'POL': 41.15, 'PRT': 58.89, 'PRI': 72.30,
    'ROU': 30.34, 'RUS': 45.89, 'SVK': 45.57, 'SVN': 49.21,
    'ZAF': 65.93, 'ESP': 57.34, 'SWE': 69.22, 'CHE': 67.99,
    'TUR': 31.13, 'UKR': 52.13, 'USA': 72.69, 'URY': 62.48,
    'VEN': 60.76, 'SRB': 38.71,
}

# Countries to EXCLUDE: not in paper's dataset
EXCLUDE_COUNTRIES = {'ALB', 'MLT', 'MNE', 'SLV'}


def build_country_data():
    """Build country-level dataset."""
    scores, _, _ = compute_nation_level_factor_scores()
    scores = scores[~scores['COUNTRY_ALPHA'].isin(EXCLUDE_COUNTRIES)]

    records = []
    for _, row in scores.iterrows():
        cc = row['COUNTRY_ALPHA']

        # GDP per capita 1980 (current USD, in thousands)
        gdp = GDP_1980_CURRENT_USD.get(cc, None)
        if gdp is not None:
            gdp = gdp / 1000.0
        else:
            gdp = np.nan

        # Industrial employment 1991
        ind = IND_EMP_1991.get(cc, np.nan)

        # Service employment 1991
        srv = SRV_EMP_1991.get(cc, np.nan)

        records.append({
            'COUNTRY_ALPHA': cc,
            'trad_secrat': row['trad_secrat'],
            'surv_selfexp': row['surv_selfexp'],
            'gdp_pc': gdp,
            'ind_emp': ind,
            'srv_emp': srv
        })

    df = pd.DataFrame(records)

    # Print data availability
    n_gdp = df['gdp_pc'].notna().sum()
    n_ind = df['ind_emp'].notna().sum()
    n_srv = df['srv_emp'].notna().sum()
    n_complete = df.dropna(subset=['gdp_pc', 'ind_emp', 'srv_emp']).shape[0]

    print(f"Total societies: {len(df)}")
    print(f"  With 1980 GDP: {n_gdp}")
    print(f"  With ind_emp: {n_ind}")
    print(f"  With srv_emp: {n_srv}")
    print(f"  Complete cases (all 3): {n_complete}")

    missing = df[df[['gdp_pc', 'ind_emp', 'srv_emp']].isna().any(axis=1)]
    print(f"\n  Countries missing data:")
    for _, r in missing.iterrows():
        print(f"    {r['COUNTRY_ALPHA']}: GDP={'OK' if pd.notna(r['gdp_pc']) else 'MISS'}, "
              f"IND={'OK' if pd.notna(r['ind_emp']) else 'MISS'}, "
              f"SRV={'OK' if pd.notna(r['srv_emp']) else 'MISS'}")

    return df


def get_cultural_zones():
    """Cultural zone assignments."""
    return {
        'Ex-Communist': ['ARM', 'AZE', 'BLR', 'BIH', 'BGR', 'CHN', 'HRV', 'CZE',
                         'EST', 'GEO', 'HUN', 'LVA', 'LTU', 'MKD', 'MDA', 'POL',
                         'ROU', 'RUS', 'SVK', 'SVN', 'UKR', 'SRB'],
        'Protestant Europe': ['DNK', 'FIN', 'ISL', 'NLD', 'NOR', 'SWE', 'CHE', 'DEU'],
        'English-speaking': ['AUS', 'CAN', 'GBR', 'NIR', 'IRL', 'NZL', 'USA'],
        'Latin America': ['ARG', 'BRA', 'CHL', 'COL', 'DOM', 'MEX', 'PER', 'PRI', 'URY', 'VEN'],
        'Africa': ['GHA', 'NGA', 'ZAF'],
        'South Asia': ['BGD', 'IND', 'PAK', 'PHL', 'TUR'],
        'Orthodox': ['ARM', 'AZE', 'BLR', 'BGR', 'GEO', 'MDA', 'ROU', 'RUS', 'UKR', 'SRB', 'MKD'],
        'Confucian': ['CHN', 'JPN', 'KOR', 'TWN'],
        'Catholic Europe': ['FRA', 'BEL', 'ITA', 'ESP', 'PRT', 'AUT']
    }


def assign_zones(df):
    zones = get_cultural_zones()
    zone_cols = ['Ex_Communist', 'Protestant_Europe', 'English_speaking',
                 'Latin_America', 'Africa', 'South_Asia', 'Orthodox', 'Confucian']
    zone_map = {
        'Ex_Communist': 'Ex-Communist', 'Protestant_Europe': 'Protestant Europe',
        'English_speaking': 'English-speaking', 'Latin_America': 'Latin America',
        'Africa': 'Africa', 'South_Asia': 'South Asia',
        'Orthodox': 'Orthodox', 'Confucian': 'Confucian'
    }
    for col in zone_cols:
        df[col] = df['COUNTRY_ALPHA'].isin(zones[zone_map[col]]).astype(int)
    return df, zone_cols


def standardize(s):
    s = s.astype(float)
    m, sd = s.mean(), s.std()
    if sd == 0:
        return s - m
    return (s - m) / sd


def run_reg(df, dv, zone_col, econ_cols):
    cols = [dv, zone_col] + econ_cols
    sub = df[cols].dropna()
    if len(sub) < 5:
        return None
    N = len(sub)
    y = standardize(sub[dv])
    X_data = {c: standardize(sub[c]) for c in [zone_col] + econ_cols}
    X = pd.DataFrame(X_data)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    res = {'N': N, 'adj_r2': model.rsquared_adj}
    for c in [zone_col] + econ_cols:
        res[f'coef_{c}'] = model.params[c]
        res[f't_{c}'] = model.tvalues[c]
        res[f'p_{c}'] = model.pvalues[c]
    return res


def sig(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''


def run_analysis():
    df = build_country_data()
    df, zone_cols = assign_zones(df)

    labels = {
        'Ex_Communist': 'Ex-Communist', 'Protestant_Europe': 'Protestant Europe',
        'English_speaking': 'English-speaking', 'Latin_America': 'Latin America',
        'Africa': 'Africa', 'South_Asia': 'South Asia',
        'Orthodox': 'Orthodox', 'Confucian': 'Confucian'
    }

    all_results = []

    print(f"\n{'='*90}")
    print("TABLE 4: Standardized Coefficients")
    print(f"{'='*90}")
    print(f"{'':50s} {'Trad/Sec-Rat':>18s}  {'Surv/Self-Exp':>18s}")
    print(f"{'':50s} {'Coeff (t)':>18s}  {'Coeff (t)':>18s}")
    print("-" * 90)

    for zc in zone_cols:
        label = labels[zc]
        trad = run_reg(df, 'trad_secrat', zc, ['gdp_pc', 'ind_emp'])
        surv = run_reg(df, 'surv_selfexp', zc, ['gdp_pc', 'srv_emp'])
        all_results.append((label, zc, trad, surv))

        if trad and surv:
            def fmt(res, var):
                c = res[f'coef_{var}']
                t = res[f't_{var}']
                s = sig(res[f'p_{var}'])
                return f"{c:+.3f}{s:3s} ({t:+.2f})"

            print(f"{label + ' zone (=1)':50s} {fmt(trad, zc):>18s}   {fmt(surv, zc):>18s}")
            print(f"{'  Real GDP per capita, 1980':50s} {fmt(trad, 'gdp_pc'):>18s}   {fmt(surv, 'gdp_pc'):>18s}")
            print(f"{'  % industrial employment, 1980':50s} {fmt(trad, 'ind_emp'):>18s}   {'---':>18s}")
            print(f"{'  % service employment, 1980':50s} {'---':>18s}   {fmt(surv, 'srv_emp'):>18s}")
            print(f"{'  Adjusted R²':50s} {trad['adj_r2']:.2f}                {surv['adj_r2']:.2f}")
            print(f"{'  N':50s} {trad['N']}                  {surv['N']}")
            print()

    print(f"{'='*90}")
    print("*p < .05  **p < .01  ***p < .001")
    return all_results


def score_against_ground_truth(all_results):
    gt = {
        'Ex-Communist': {
            'trad': {'zone': (.424, '**'), 'gdp': (.496, '***'), 'ind': (.216, ''), 'r2': .50},
            'surv': {'zone': (-.393, '***'), 'gdp': (.575, '***'), 'srv': (.098, ''), 'r2': .73}
        },
        'Protestant Europe': {
            'trad': {'zone': (.370, '**'), 'gdp': (.025, ''), 'ind': (.553, '***'), 'r2': .50},
            'surv': {'zone': (.232, '*'), 'gdp': (.362, '*'), 'srv': (.331, '*'), 'r2': .63}
        },
        'English-speaking': {
            'trad': {'zone': (-.300, '**'), 'gdp': (.394, '**'), 'ind': (.468, '***'), 'r2': .47},
            'surv': {'zone': (.146, ''), 'gdp': (.434, '**'), 'srv': (.319, '*'), 'r2': .61}
        },
        'Latin America': {
            'trad': {'zone': (-.342, '**'), 'gdp': (.195, ''), 'ind': (.448, '***'), 'r2': .51},
            'surv': {'zone': (.108, ''), 'gdp': (.602, '**'), 'srv': (.224, ''), 'r2': .60}
        },
        'Africa': {
            'trad': {'zone': (-.189, ''), 'gdp': (.211, ''), 'ind': (.468, '***'), 'r2': .43},
            'surv': {'zone': (.021, ''), 'gdp': (.502, '**'), 'srv': (.320, ''), 'r2': .59}
        },
        'South Asia': {
            'trad': {'zone': (.070, ''), 'gdp': (.258, '*'), 'ind': (.542, '***'), 'r2': .40},
            'surv': {'zone': (.212, '*'), 'gdp': (.469, '**'), 'srv': (.455, '**'), 'r2': .62}
        },
        'Orthodox': {
            'trad': {'zone': (.152, ''), 'gdp': (.304, '*'), 'ind': (.432, '**'), 'r2': .42},
            'surv': {'zone': (-.457, '***'), 'gdp': (.567, '***'), 'srv': (.154, ''), 'r2': .80}
        },
        'Confucian': {
            'trad': {'zone': (.397, '***'), 'gdp': (.304, '**'), 'ind': (.505, '***'), 'r2': .56},
            'surv': {'zone': (-.020, ''), 'gdp': (.491, '**'), 'srv': (.323, '*'), 'r2': .59}
        }
    }

    total = 0
    maximum = 0
    details = []

    for label, zc, trad_res, surv_res in all_results:
        if label not in gt:
            continue
        g = gt[label]
        if trad_res is None or surv_res is None:
            maximum += 50
            details.append(f"{label}: NO RESULTS")
            continue

        # Trad
        for vk, col in [('zone', zc), ('gdp', 'gdp_pc'), ('ind', 'ind_emp')]:
            gc, gs = g['trad'][vk]
            rc = trad_res.get(f'coef_{col}')
            rp = trad_res.get(f'p_{col}')
            if rc is not None:
                rs = sig(rp)
                d = abs(rc - gc)
                maximum += 2
                if d <= 0.05: total += 2; m = 'MATCH'
                elif d <= 0.15: total += 1; m = 'PARTIAL'
                else: m = 'MISS'
                details.append(f"  {label} trad {vk}: {rc:.3f} vs {gc:.3f} ({m}, d={d:.3f})")
                maximum += 1
                if rs == gs: total += 1; details.append(f"    sig MATCH")
                else: details.append(f"    sig '{rs}' vs '{gs}' MISS")

        maximum += 2
        d = abs(trad_res['adj_r2'] - g['trad']['r2'])
        if d <= 0.02: total += 2; details.append(f"  {label} trad R2: {trad_res['adj_r2']:.2f} vs {g['trad']['r2']:.2f} MATCH")
        elif d <= 0.08: total += 1; details.append(f"  {label} trad R2: {trad_res['adj_r2']:.2f} vs {g['trad']['r2']:.2f} PARTIAL")
        else: details.append(f"  {label} trad R2: {trad_res['adj_r2']:.2f} vs {g['trad']['r2']:.2f} MISS")

        # Surv
        for vk, col in [('zone', zc), ('gdp', 'gdp_pc'), ('srv', 'srv_emp')]:
            gc, gs = g['surv'][vk]
            rc = surv_res.get(f'coef_{col}')
            rp = surv_res.get(f'p_{col}')
            if rc is not None:
                rs = sig(rp)
                d = abs(rc - gc)
                maximum += 2
                if d <= 0.05: total += 2; m = 'MATCH'
                elif d <= 0.15: total += 1; m = 'PARTIAL'
                else: m = 'MISS'
                details.append(f"  {label} surv {vk}: {rc:.3f} vs {gc:.3f} ({m}, d={d:.3f})")
                maximum += 1
                if rs == gs: total += 1; details.append(f"    sig MATCH")
                else: details.append(f"    sig '{rs}' vs '{gs}' MISS")

        maximum += 2
        d = abs(surv_res['adj_r2'] - g['surv']['r2'])
        if d <= 0.02: total += 2; details.append(f"  {label} surv R2: {surv_res['adj_r2']:.2f} vs {g['surv']['r2']:.2f} MATCH")
        elif d <= 0.08: total += 1; details.append(f"  {label} surv R2: {surv_res['adj_r2']:.2f} vs {g['surv']['r2']:.2f} PARTIAL")
        else: details.append(f"  {label} surv R2: {surv_res['adj_r2']:.2f} vs {g['surv']['r2']:.2f} MISS")

    score = int(100 * total / maximum) if maximum > 0 else 0
    print(f"\n{'='*80}")
    print(f"SCORING: {total}/{maximum} = {score}/100")
    print(f"{'='*80}")
    for d in details:
        print(d)
    return score, details


if __name__ == "__main__":
    results = run_analysis()
    score, _ = score_against_ground_truth(results)
    print(f"\nFINAL SCORE: {score}/100")
