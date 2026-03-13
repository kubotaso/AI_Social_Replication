#!/usr/bin/env python3
"""
Table 4 replication - Attempt 4

Key approach:
- Use GNP per capita PPP (current intl $) from WB for ~1990 as proxy for 1980
  The paper used "World Bank PPP estimates as of 1995, in U.S. dollars"
- Use employment data from 1991
- Restrict to paper's 65 societies minus those with missing econ data -> ~49
- East/West Germany handled by using DEU data for both but assigning to different zones
- The paper had 65 societies and got N=49 after dropping for missing econ data
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

# GNP per capita PPP (current international $) ~1990 from World Bank API
# Paper used "PPP estimates as of 1995, in U.S. dollars" (World Bank 1997)
# These 1990 values in current intl $ are our best approximation
GNP_PPP_1990 = {
    'ALB': 2650, 'ARG': 6840, 'ARM': 2660, 'AUS': 16780,
    'AUT': 19480, 'AZE': 5350, 'BGD': 950, 'BLR': 5220,
    'BEL': 18750, 'BIH': 900, 'BRA': 6510, 'BGR': 6950,
    'CAN': 19500, 'CHL': 4210, 'CHN': 990, 'COL': 4900,
    'HRV': None, 'CZE': 12680, 'DNK': 17590, 'DOM': 3510,
    'EST': 7190, 'FIN': 17820, 'FRA': 17550, 'GEO': 5850,
    'DEU': 19660, 'GHA': 1570, 'GBR': 16870, 'HUN': 8800,
    'ISL': 21170, 'IND': 1200, 'IRL': 12780, 'ITA': 18450,
    'JPN': 20130, 'KOR': 8560, 'LVA': 7820, 'LTU': 8330,
    'MKD': 5290, 'MEX': 8360, 'MDA': 6930, 'NLD': 18650,
    'NZL': 14190, 'NGA': None, 'NOR': 17940, 'PAK': 1870,
    'PER': 3160, 'PHL': 2570, 'POL': 5870, 'PRT': 11770,
    'PRI': 9470, 'ROU': 5260, 'RUS': 8010, 'SVK': 8030,
    'SVN': 12270, 'ZAF': 6170, 'ESP': 13520, 'SWE': 19890,
    'CHE': 29000, 'TUR': 8110, 'UKR': 7760, 'USA': 24030,
    'URY': 6460, 'VEN': 11480, 'SRB': None, 'TWN': None,
    'NIR': None, 'MLT': None, 'MNE': None,
}

# 1991 Industrial employment (%) from WB API
IND_EMP = {
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
    'VEN': 26.30, 'SRB': 21.98, 'ALB': 27.0,
}

# 1991 Service employment (%) from WB API
SRV_EMP = {
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
    'VEN': 60.76, 'SRB': 38.71, 'ALB': 41.0,
}

# Countries NOT in the paper's 65 societies
EXCLUDE = {'ALB', 'MLT', 'MNE', 'SLV'}

# Estimated values for missing countries that ARE in the paper's 65 societies
# NGA: Nigeria was in the paper. 1990 GNP PPP ~$1,200 (from WDI 1997 Atlas method)
# Actually from the API result, NGA was missing. Let me use estimates.
# TWN: Taiwan ~$12,000 GNP PPP 1990
# NIR: Northern Ireland - use UK values
# SRB: Serbia (Yugoslavia) ~$6,000
# HRV: Croatia ~$8,000 (before war)

ESTIMATED_GNP = {
    'NGA': 1200,  # Nigeria - low income
    'TWN': 12000, # Taiwan - industrializing
    'NIR': 16870, # N. Ireland - use UK
    'SRB': 6000,  # Serbia/Yugoslavia
    'HRV': 8000,  # Croatia
}

ESTIMATED_IND = {
    'TWN': 38.0,   # Taiwan - highly industrial
    'NIR': 30.77,  # N. Ireland - use UK
}

ESTIMATED_SRV = {
    'TWN': 47.0,   # Taiwan - similar to Korea
    'NIR': 66.81,  # N. Ireland - use UK
}


def build_country_data():
    """Build country-level dataset."""
    scores, _, _ = compute_nation_level_factor_scores()
    scores = scores[~scores['COUNTRY_ALPHA'].isin(EXCLUDE)]

    records = []
    for _, row in scores.iterrows():
        cc = row['COUNTRY_ALPHA']

        # GNP per capita PPP
        gnp = GNP_PPP_1990.get(cc, None)
        if gnp is None:
            gnp = ESTIMATED_GNP.get(cc, None)
        if gnp is not None:
            gnp = gnp / 1000.0  # Convert to thousands
        else:
            gnp = np.nan

        # Industrial employment
        ind = IND_EMP.get(cc, ESTIMATED_IND.get(cc, np.nan))

        # Service employment
        srv = SRV_EMP.get(cc, ESTIMATED_SRV.get(cc, np.nan))

        records.append({
            'COUNTRY_ALPHA': cc,
            'trad_secrat': row['trad_secrat'],
            'surv_selfexp': row['surv_selfexp'],
            'gnp_pc': gnp,
            'ind_emp': ind,
            'srv_emp': srv
        })

    df = pd.DataFrame(records)

    n_total = len(df)
    n_complete = df.dropna(subset=['gnp_pc', 'ind_emp', 'srv_emp']).shape[0]
    print(f"Total societies: {n_total}")
    print(f"Complete cases: {n_complete}")

    missing = df[df[['gnp_pc', 'ind_emp', 'srv_emp']].isna().any(axis=1)]
    if len(missing) > 0:
        print("Missing data:")
        for _, r in missing.iterrows():
            print(f"  {r['COUNTRY_ALPHA']}: GNP={'OK' if pd.notna(r['gnp_pc']) else 'MISS'}, "
                  f"IND={'OK' if pd.notna(r['ind_emp']) else 'MISS'}, "
                  f"SRV={'OK' if pd.notna(r['srv_emp']) else 'MISS'}")

    return df


def get_cultural_zones():
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
        trad = run_reg(df, 'trad_secrat', zc, ['gnp_pc', 'ind_emp'])
        surv = run_reg(df, 'surv_selfexp', zc, ['gnp_pc', 'srv_emp'])
        all_results.append((label, zc, trad, surv))

        if trad and surv:
            def fmt(res, var):
                c = res[f'coef_{var}']
                t = res[f't_{var}']
                s = sig(res[f'p_{var}'])
                return f"{c:+.3f}{s:3s} ({t:+.2f})"

            print(f"{label + ' zone (=1)':50s} {fmt(trad, zc):>18s}   {fmt(surv, zc):>18s}")
            print(f"{'  Real GDP per capita, 1980':50s} {fmt(trad, 'gnp_pc'):>18s}   {fmt(surv, 'gnp_pc'):>18s}")
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
            continue

        # Trad - zone, gdp, ind
        for vk, col in [('zone', zc), ('gdp', 'gnp_pc'), ('ind', 'ind_emp')]:
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
                if rs == gs: total += 1
                else: details.append(f"    sig '{rs}' vs '{gs}' MISS")

        maximum += 2
        d = abs(trad_res['adj_r2'] - g['trad']['r2'])
        if d <= 0.02: total += 2
        elif d <= 0.08: total += 1

        # Surv - zone, gdp, srv
        for vk, col in [('zone', zc), ('gdp', 'gnp_pc'), ('srv', 'srv_emp')]:
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
                if rs == gs: total += 1
                else: details.append(f"    sig '{rs}' vs '{gs}' MISS")

        maximum += 2
        d = abs(surv_res['adj_r2'] - g['surv']['r2'])
        if d <= 0.02: total += 2
        elif d <= 0.08: total += 1

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
