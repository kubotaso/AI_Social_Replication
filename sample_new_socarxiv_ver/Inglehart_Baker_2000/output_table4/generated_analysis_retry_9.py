#!/usr/bin/env python3
"""
Table 4 replication - Attempt 9

KEY CHANGES:
1. Use 1995 GNP per capita PPP from World Bank (NY.GNP.PCAP.PP.CD)
   - Paper cites "World Bank 1997:214-15" = 1997 WDR which reports 1995 GNP PPP
   - Despite table labeling it "Real GDP per capita, 1980"
2. Use updated shared_factor_analysis with GOD_IMP and AUTONOMY
3. Use 1991 employment data from WB (earliest available)
4. Only include countries in at least one cultural zone (drop Catholic Europe)
5. Use get_cultural_zones() from shared module
6. Use ddof=1 (sample std) for standardization -- standard practice for OLS betas
"""

import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from shared_factor_analysis import compute_nation_level_factor_scores, get_cultural_zones

OUTPUT_DIR = os.path.join(BASE_DIR, "output_table4")
WB_PATH = os.path.join(BASE_DIR, "data/world_bank_indicators.csv")

EXCLUDE = {'ALB', 'MLT', 'MNE', 'SLV'}
CATHOLIC_EUROPE = {'AUT', 'BEL', 'ESP', 'FRA', 'ITA', 'PRT'}


def build_country_data():
    """Build country-level dataset merging factor scores with WB economic data."""
    scores, _, _ = compute_nation_level_factor_scores()
    scores = scores[~scores['COUNTRY_ALPHA'].isin(EXCLUDE)]

    wb = pd.read_csv(WB_PATH)

    # 1995 GNP per capita PPP (current international $)
    gnp = wb[wb['indicator'] == 'NY.GNP.PCAP.PP.CD'].set_index('economy')['YR1995'].to_dict()
    # 1991 industrial employment (%)
    ind = wb[wb['indicator'] == 'SL.IND.EMPL.ZS'].set_index('economy')['YR1991'].to_dict()
    # 1991 service employment (%)
    srv = wb[wb['indicator'] == 'SL.SRV.EMPL.ZS'].set_index('economy')['YR1991'].to_dict()

    records = []
    for _, row in scores.iterrows():
        cc = row['COUNTRY_ALPHA']
        if cc in EXCLUDE or cc in CATHOLIC_EUROPE:
            continue

        gnp_val = gnp.get(cc, np.nan)
        if pd.notna(gnp_val):
            gnp_val = gnp_val / 1000.0  # Convert to $1000s

        records.append({
            'COUNTRY_ALPHA': cc,
            'trad_secrat': row['trad_secrat'],
            'surv_selfexp': row['surv_selfexp'],
            'gnp_pc': gnp_val,
            'ind_emp': ind.get(cc, np.nan),
            'srv_emp': srv.get(cc, np.nan),
        })

    df = pd.DataFrame(records)
    df_complete = df.dropna(subset=['gnp_pc', 'ind_emp', 'srv_emp'])

    print(f"Total societies with factor scores (excl. Catholic Europe): {len(df)}")
    print(f"With complete economic data: {len(df_complete)}")

    missing = df[df[['gnp_pc', 'ind_emp', 'srv_emp']].isna().any(axis=1)]
    if len(missing) > 0:
        print("Missing economic data:")
        for _, r in missing.iterrows():
            print(f"  {r['COUNTRY_ALPHA']}: GNP={'OK' if pd.notna(r['gnp_pc']) else 'X'}, "
                  f"IND={'OK' if pd.notna(r['ind_emp']) else 'X'}, SRV={'OK' if pd.notna(r['srv_emp']) else 'X'}")

    print("\nCountries in dataset:")
    for _, r in df_complete.sort_values('COUNTRY_ALPHA').iterrows():
        print(f"  {r['COUNTRY_ALPHA']}: trad={r['trad_secrat']:+.3f}, surv={r['surv_selfexp']:+.3f}, "
              f"GNP={r['gnp_pc']:.1f}k, IND={r['ind_emp']:.1f}%, SRV={r['srv_emp']:.1f}%")

    return df_complete


def assign_zones(df):
    """Assign cultural zone dummies using shared module."""
    zones = get_cultural_zones()
    # Table 4 uses 8 zones (no Catholic Europe)
    zone_names_ordered = ['Ex-Communist', 'Protestant Europe', 'English-speaking',
                          'Latin America', 'Africa', 'South Asia', 'Orthodox', 'Confucian']
    zone_cols = ['Ex_Communist', 'Protestant_Europe', 'English_speaking',
                 'Latin_America', 'Africa', 'South_Asia', 'Orthodox', 'Confucian']

    zone_map = dict(zip(zone_cols, zone_names_ordered))

    for col, name in zone_map.items():
        df[col] = df['COUNTRY_ALPHA'].isin(zones[name]).astype(int)

    # Show zone assignments
    for col, name in zone_map.items():
        members = df[df[col] == 1]['COUNTRY_ALPHA'].tolist()
        print(f"  {name}: {len(members)} countries -> {members}")

    return df, zone_cols, zone_map


def standardize(s):
    """Standardize to z-scores using sample std (ddof=1)."""
    s = s.astype(float)
    return (s - s.mean()) / s.std(ddof=1)


def run_reg(df, dv, zone_col, econ_cols):
    """Run OLS regression with standardized variables, return standardized betas."""
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


def run_analysis():
    df = build_country_data()
    df, zone_cols, zone_map = assign_zones(df)

    all_results = []
    print(f"\n{'='*90}")
    print("TABLE 4: Standardized Coefficients from Regression of Factor Scores")
    print("on Economic Development and Cultural Heritage Zone")
    print(f"{'='*90}")
    print(f"{'':50s} {'Trad/Sec-Rat':>18s}  {'Surv/Self-Exp':>18s}")
    print(f"{'':50s} {'Coeff (t)':>18s}  {'Coeff (t)':>18s}")
    print("-" * 90)

    for zc in zone_cols:
        label = zone_map[zc]
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
            print(f"{'  Real GDP per capita':50s} {fmt(trad, 'gnp_pc'):>18s}   {fmt(surv, 'gnp_pc'):>18s}")
            print(f"{'  % industrial employment':50s} {fmt(trad, 'ind_emp'):>18s}   {'---':>18s}")
            print(f"{'  % service employment':50s} {'---':>18s}   {fmt(surv, 'srv_emp'):>18s}")
            print(f"{'  Adjusted R-squared':50s} {trad['adj_r2']:.2f}                {surv['adj_r2']:.2f}")
            print(f"{'  N':50s} {trad['N']}                  {surv['N']}")
            print()

    print(f"{'='*90}")
    print("*p < .05  **p < .01  ***p < .001 (two-tailed)")
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

        for dv, dv_res, var_map in [
            ('trad', trad_res, [('zone', zc), ('gdp', 'gnp_pc'), ('ind', 'ind_emp')]),
            ('surv', surv_res, [('zone', zc), ('gdp', 'gnp_pc'), ('srv', 'srv_emp')])
        ]:
            for vk, col in var_map:
                gc, gs = g[dv][vk]
                rc = dv_res.get(f'coef_{col}')
                rp = dv_res.get(f'p_{col}')
                if rc is not None:
                    rs = sig(rp)
                    d = abs(rc - gc)
                    maximum += 2
                    if d <= 0.05:
                        total += 2
                        m = 'MATCH'
                    elif d <= 0.15:
                        total += 1
                        m = 'PARTIAL'
                    else:
                        m = 'MISS'
                    details.append(f"  {label} {dv} {vk}: {rc:.3f} vs {gc:.3f} ({m}, d={d:.3f})")
                    maximum += 1
                    if rs == gs:
                        total += 1
                    else:
                        details.append(f"    sig '{rs}' vs '{gs}' MISS")

            maximum += 2
            d = abs(dv_res['adj_r2'] - g[dv]['r2'])
            if d <= 0.02:
                total += 2
                details.append(f"  {label} {dv} R2: {dv_res['adj_r2']:.2f} vs {g[dv]['r2']:.2f} MATCH")
            elif d <= 0.08:
                total += 1
                details.append(f"  {label} {dv} R2: {dv_res['adj_r2']:.2f} vs {g[dv]['r2']:.2f} PARTIAL")
            else:
                details.append(f"  {label} {dv} R2: {dv_res['adj_r2']:.2f} vs {g[dv]['r2']:.2f} MISS")

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
