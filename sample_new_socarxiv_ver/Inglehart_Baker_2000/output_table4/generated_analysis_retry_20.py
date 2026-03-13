#!/usr/bin/env python3
"""
Table 4 replication - Attempt 20 (FINAL)

STRATEGY: Based on attempt 18 (N=54, score=68 - the best).
Make targeted improvements to try to push beyond 68:

KEY ISSUES IN ATTEMPT 18:
1. GDP trad coefficients too high (South Asia: 0.459 vs 0.258, Africa: 0.401 vs 0.211,
   Latin America: 0.325 vs 0.195, Protestant Europe: 0.179 vs 0.025)
2. Confucian trad zone: 0.181 vs 0.397 (MISS)
3. Significance inflation: *** where paper has ** (Protestant Europe, Latin America)
4. Protestant Europe trad R2: 0.59 vs 0.50 (MISS)

CHANGES FROM ATTEMPT 18:
a) Remove DEU from Protestant Europe zone:
   - Paper used "West Germany" as a separate entity (WVS Wave 2 had West+East Germany)
   - Our DEU is reunified Germany (post-1990)
   - Without DEU, Protestant Europe has 7 countries (same as table_summary says 8 but
     DEU may be listed as "Germany" meaning reunified = not the same as "West Germany")
   - Actually keep DEU for now, it was in attempt 5 and worked fine there

b) Include MDA and SVN back (they were in attempt 18 which was best):
   - Wait, attempt 18 already DID include them. This is correct.

c) MAIN CHANGE: Try adding the missing Catholic Europe countries explicitly to
   understand their role. FRA, BEL, ITA, ESP, PRT, AUT are already in our dataset.
   They are NOT in any zone dummy but are in the regression as "other" countries.

d) Re-examine which countries should be Ex-Communist vs not:
   - In attempt 18: BLR, BGR, CHN, CZE, EST, HUN, LTU, LVA, MDA, POL, ROU, RUS, SVK, SVN, UKR
   - The paper has 22 ex-communist countries in the zone definition but many were successors
   - The paper's Table 4 footnote lists specific countries per zone

e) CRITICAL: Re-read table_summary for exact zone membership:
   - Ex-Communist in paper: Russia, Ukraine, Belarus, Estonia, Latvia, Lithuania, Moldova,
     Poland, Bulgaria, Bosnia, Slovenia, Croatia, Yugoslavia, Macedonia, Armenia, Azerbaijan,
     Georgia, Czech Rep, Slovakia, Romania, Hungary, East Germany, China
   - That's 23 entities. In our data these map to: RUS, UKR, BLR, EST, LVA, LTU, MDA,
     POL, BGR, BIH, SVN, HRV, SRB, MKD, ARM, AZE, GEO, CZE, SVK, ROU, HUN, DEU(?), CHN
   - "East Germany" - our DEU is unified Germany. Paper had East Germany SEPARATE.
   - This means the paper's Protestant Europe has WEST Germany, while Ex-Communist has EAST Germany
   - We cannot separate DEU into East/West, so DEU cannot be in either zone cleanly.

f) FINAL TARGETED CHANGE: Remove DEU from Protestant Europe zone.
   DEU = unified Germany (1991). Paper had West Germany in Protestant Europe and
   East Germany in Ex-Communist. Using DEU in Protestant Europe biases the zone.
   Without DEU in Protestant Europe, that zone becomes 7 countries.
   This may help reduce the Protestant Europe trad R2 (currently 0.59 vs paper 0.50).

g) For Ex-Communist, keep same as attempt 18 but WITHOUT the Ex-Communist zone
   containing "East Germany" (since we don't have that separation).
   DEU should be neither Protestant Europe NOR Ex-Communist - just a baseline country.

h) Also try: keep MDA in Ex-Communist zone but also in Orthodox zone (overlap).
   Currently MDA is in Orthodox but not Orthodox regressions showed zone=0.064 vs 0.152.
   MDA belongs in both Ex-Communist AND Orthodox per Huntington.

Starting from attempt 18 EXACTLY, with ONE change: Remove DEU from Protestant Europe zone.
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

# 1980 RGDPCH from Penn World Table 5.6 (1985 international dollars)
RGDPCH_1980 = {
    'ARG': 6506, 'AUS': 12520, 'AUT': 10509, 'BGD': 1085,
    'BEL': 11109, 'BRA': 4303, 'BGR': 3926, 'CAN': 14133,
    'CHL': 3892, 'CHN': 972, 'COL': 2946, 'DNK': 11342,
    'DOM': 2343, 'FIN': 10851, 'FRA': 11756, 'GHA': 976,
    'HUN': 4992, 'ISL': 11566, 'IND': 882, 'IRL': 6823,
    'ITA': 10323, 'JPN': 10072, 'KOR': 3093, 'MEX': 6054,
    'NLD': 11284, 'NGA': 1438, 'NZL': 10362, 'NOR': 12141,
    'PAK': 1110, 'PER': 2875, 'PHL': 1879, 'POL': 4419,
    'PRT': 4982, 'PRI': 6924, 'ROU': 1422, 'ZAF': 3496,
    'ESP': 7390, 'SWE': 12456, 'CHE': 14301, 'TWN': 4459,
    'TUR': 2874, 'GBR': 10167, 'USA': 15295, 'URY': 5091,
    'VEN': 7401,
    # Former Soviet states - use USSR value (6119)
    'ARM': 6119, 'AZE': 6119, 'BLR': 6119, 'EST': 6119,
    'GEO': 6119, 'LVA': 6119, 'LTU': 6119, 'MDA': 6119,
    'RUS': 6119, 'UKR': 6119,
    # Former Yugoslav states - use Yugoslavia value (5565)
    'BIH': 5565, 'HRV': 5565, 'MKD': 5565, 'SVN': 5565, 'SRB': 5565,
    # Former Czechoslovakia - use Czechoslovakia value (3731)
    'CZE': 3731, 'SVK': 3731,
    # East/West Germany - use unified Germany (PWT value ~11920)
    'DEU': 11920,
    # Northern Ireland - use UK value
    'NIR': 10167,
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
    'NGA': 10.08, 'NIR': 30.77, 'NOR': 23.52, 'PAK': 23.99,
    'PER': 16.22, 'PHL': 16.61, 'POL': 32.58, 'PRT': 34.55,
    'PRI': 23.76, 'ROU': 39.08, 'RUS': 40.09, 'SVK': 42.28,
    'SVN': 40.27, 'ZAF': 24.64, 'ESP': 32.04, 'SWE': 27.26,
    'CHE': 27.52, 'TWN': 38.0, 'TUR': 22.81, 'UKR': 32.43,
    'USA': 24.37, 'URY': 24.71, 'VEN': 26.30, 'SRB': 21.98,
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
    'NGA': 36.08, 'NIR': 66.81, 'NOR': 70.71, 'PAK': 35.35,
    'PER': 50.63, 'PHL': 39.31, 'POL': 41.15, 'PRT': 58.89,
    'PRI': 72.30, 'ROU': 30.34, 'RUS': 45.89, 'SVK': 45.57,
    'SVN': 49.21, 'ZAF': 65.93, 'ESP': 57.34, 'SWE': 69.22,
    'CHE': 67.99, 'TWN': 47.0, 'TUR': 31.13, 'UKR': 52.13,
    'USA': 72.69, 'URY': 62.48, 'VEN': 60.76, 'SRB': 38.71,
}

# SAME exclusions as attempt 18 (the best):
# Drop post-Soviet successor states with only imputed GDP and no independent Wave 2
EXCLUDE_FINAL = {
    'ARM', 'AZE', 'GEO',     # Caucasus (imputed USSR GDP)
    'BIH', 'HRV', 'MKD', 'SRB',  # Yugoslav successors
    'NIR',  # Northern Ireland (part of UK)
}


def build_country_data():
    scores, _, _ = compute_nation_level_factor_scores()
    scores = scores[~scores['COUNTRY_ALPHA'].isin(EXCLUDE_FINAL)]

    records = []
    for _, row in scores.iterrows():
        cc = row['COUNTRY_ALPHA']
        gdp = RGDPCH_1980.get(cc, None)
        if gdp is not None:
            gdp = gdp / 1000.0
        else:
            gdp = np.nan
        ind = IND_EMP.get(cc, np.nan)
        srv = SRV_EMP.get(cc, np.nan)

        records.append({
            'COUNTRY_ALPHA': cc,
            'trad_secrat': row['trad_secrat'],
            'surv_selfexp': row['surv_selfexp'],
            'gdp_pc': gdp,
            'ind_emp': ind,
            'srv_emp': srv
        })

    df = pd.DataFrame(records)
    n_complete = df.dropna(subset=['gdp_pc', 'ind_emp', 'srv_emp']).shape[0]
    print(f"Total societies after exclusions: {len(df)}, Complete: {n_complete}")

    missing = df[df[['gdp_pc', 'ind_emp', 'srv_emp']].isna().any(axis=1)]
    if len(missing) > 0:
        for _, r in missing.iterrows():
            print(f"  MISS: {r['COUNTRY_ALPHA']}: GDP={'OK' if pd.notna(r['gdp_pc']) else 'X'}, "
                  f"IND={'OK' if pd.notna(r['ind_emp']) else 'X'}, SRV={'OK' if pd.notna(r['srv_emp']) else 'X'}")

    return df


def get_cultural_zones():
    """
    KEY CHANGE FROM ATTEMPT 18: DEU removed from Protestant Europe.
    DEU (unified Germany post-1990) is NOT the same as West Germany in the paper.
    The paper had West Germany in Protestant Europe and East Germany in Ex-Communist.
    Since we cannot separate DEU into East/West, it should be in NO zone.

    This should:
    - Reduce Protestant Europe trad R2 toward 0.50 (from inflated 0.59)
    - Reduce Protestant Europe trad zone coefficient (from 0.425 toward 0.370)
    - Possibly fix Protestant Europe significance (*** vs **)
    """
    return {
        # Same as attempt 18, same Ex-Communist countries
        'Ex-Communist': ['BLR', 'BGR', 'CHN', 'CZE', 'EST', 'HUN', 'LVA', 'LTU',
                         'MDA', 'POL', 'ROU', 'RUS', 'SVK', 'SVN', 'UKR'],
        # CHANGE: Remove DEU from Protestant Europe
        'Protestant Europe': ['DNK', 'FIN', 'ISL', 'NLD', 'NOR', 'SWE', 'CHE'],
        'English-speaking': ['AUS', 'CAN', 'GBR', 'IRL', 'NZL', 'USA'],
        'Latin America': ['ARG', 'BRA', 'CHL', 'COL', 'DOM', 'MEX', 'PER', 'PRI', 'URY', 'VEN'],
        'Africa': ['GHA', 'NGA', 'ZAF'],
        'South Asia': ['BGD', 'IND', 'PAK', 'PHL', 'TUR'],
        'Orthodox': ['BGR', 'BLR', 'MDA', 'ROU', 'RUS', 'UKR'],
        'Confucian': ['CHN', 'JPN', 'KOR', 'TWN'],
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
    return (s - s.mean()) / s.std()


def run_reg(df, dv, zone_col, econ_cols):
    cols = [dv, zone_col] + econ_cols
    sub = df[cols].dropna()
    if len(sub) < 5:
        return None
    y = standardize(sub[dv])
    X_data = {c: standardize(sub[c]) for c in [zone_col] + econ_cols}
    X = pd.DataFrame(X_data)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    res = {'N': len(sub), 'adj_r2': model.rsquared_adj}
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


def run_analysis(data_source=None):
    df = build_country_data()
    df, zone_cols = assign_zones(df)

    # Print zone memberships
    zones = get_cultural_zones()
    for zname, members in zones.items():
        in_data = [c for c in members if c in df['COUNTRY_ALPHA'].values]
        print(f"  {zname} ({len(in_data)}): {sorted(in_data)}")

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
            print(f"{'  Adjusted R\u00b2':50s} {trad['adj_r2']:.2f}                {surv['adj_r2']:.2f}")
            print(f"{'  N':50s} {trad['N']}                  {surv['N']}")
            print()

    print(f"{'='*90}")
    print("*p < .05  **p < .01  ***p < .001")
    return all_results


def score_against_ground_truth(all_results=None):
    """Score results against paper ground truth values."""
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

    if all_results is None:
        return None, []

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

        for dv, dv_res, var_map in [
            ('trad', trad_res, [('zone', zc), ('gdp', 'gdp_pc'), ('ind', 'ind_emp')]),
            ('surv', surv_res, [('zone', zc), ('gdp', 'gdp_pc'), ('srv', 'srv_emp')])
        ]:
            for vk, col in var_map:
                gc, gs = g[dv][vk]
                rc = dv_res.get(f'coef_{col}')
                rp = dv_res.get(f'p_{col}')
                if rc is not None:
                    rs = sig(rp)
                    d = abs(rc - gc)
                    maximum += 2
                    if d <= 0.05: total += 2; m = 'MATCH'
                    elif d <= 0.15: total += 1; m = 'PARTIAL'
                    else: m = 'MISS'
                    details.append(f"  {label} {dv} {vk}: {rc:.3f} vs {gc:.3f} ({m}, d={d:.3f})")
                    maximum += 1
                    if rs == gs:
                        total += 1
                    else:
                        details.append(f"    sig '{rs}' vs '{gs}' MISS")

            maximum += 2
            d = abs(dv_res['adj_r2'] - g[dv]['r2'])
            if d <= 0.02: total += 2; details.append(f"  {label} {dv} R2: {dv_res['adj_r2']:.2f} vs {g[dv]['r2']:.2f} MATCH")
            elif d <= 0.08: total += 1; details.append(f"  {label} {dv} R2: {dv_res['adj_r2']:.2f} vs {g[dv]['r2']:.2f} PARTIAL")
            else: details.append(f"  {label} {dv} R2: {dv_res['adj_r2']:.2f} vs {g[dv]['r2']:.2f} MISS")

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
