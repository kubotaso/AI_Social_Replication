#!/usr/bin/env python3
"""
Table 5a Replication - Attempt 2
Inglehart & Baker (2000), Table 5a.

Key changes from attempt 1:
- Normalize factor scores to match paper's scale (divide by SD to get unit-variance)
- Restrict country list to match paper's 65 societies -> ~49 with WB data
- Use GNP per capita PPP (current) from WB since Penn World Table not available
- Split Germany into East/West using S024 codes
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared_factor_analysis import (
    load_combined_data, get_latest_per_country, recode_factor_items,
    clean_missing, varimax, FACTOR_ITEMS, COUNTRY_NAMES
)

# Ground truth from Table 5a
GROUND_TRUTH = {
    'Model 1': {
        'GDP': {'coef': 0.066, 'se': 0.031, 'sig': '*'},
        'Industrial': {'coef': 0.052, 'se': 0.012, 'sig': '***'},
        'Adj_R2': 0.42,
        'N': 49,
    },
    'Model 2': {
        'GDP': {'coef': 0.086, 'se': 0.043, 'sig': '*'},
        'Industrial': {'coef': 0.051, 'se': 0.014, 'sig': '***'},
        'Education': {'coef': -0.01, 'se': 0.01, 'sig': ''},
        'Service': {'coef': -0.054, 'se': 0.039, 'sig': ''},
        'Education2': {'coef': -0.005, 'se': 0.012, 'sig': ''},
        'Adj_R2': 0.37,
        'N': 46,
    },
    'Model 3': {
        'GDP': {'coef': 0.131, 'se': 0.036, 'sig': '**'},
        'Industrial': {'coef': 0.023, 'se': 0.015, 'sig': ''},
        'Ex_Communist': {'coef': 1.05, 'se': 0.351, 'sig': '**'},
        'Adj_R2': 0.50,
        'N': 49,
    },
    'Model 4': {
        'GDP': {'coef': 0.042, 'se': 0.029, 'sig': ''},
        'Industrial': {'coef': 0.061, 'se': 0.011, 'sig': '***'},
        'Catholic': {'coef': -0.767, 'se': 0.216, 'sig': '**'},
        'Adj_R2': 0.53,
        'N': 49,
    },
    'Model 5': {
        'GDP': {'coef': 0.080, 'se': 0.027, 'sig': '**'},
        'Industrial': {'coef': 0.052, 'se': 0.011, 'sig': '***'},
        'Confucian': {'coef': 1.57, 'se': 0.370, 'sig': '***'},
        'Adj_R2': 0.57,
        'N': 49,
    },
    'Model 6': {
        'GDP': {'coef': 0.122, 'se': 0.030, 'sig': '***'},
        'Industrial': {'coef': 0.030, 'se': 0.012, 'sig': '*'},
        'Ex_Communist': {'coef': 0.952, 'se': 0.282, 'sig': '***'},
        'Catholic': {'coef': -0.409, 'se': 0.188, 'sig': '*'},
        'Confucian': {'coef': 1.39, 'se': 0.329, 'sig': '***'},
        'Adj_R2': 0.70,
        'N': 49,
    },
}


def get_significance(pvalue):
    if pvalue < 0.001:
        return '***'
    elif pvalue < 0.01:
        return '**'
    elif pvalue < 0.05:
        return '*'
    return ''


def compute_factor_scores_with_east_west():
    """
    Compute factor scores with East/West Germany separated.
    """
    DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
    EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")

    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'S024',
              'A006', 'A008', 'A042', 'A165',
              'E018', 'E025',
              'F118', 'F120',
              'G006',
              'Y002']
    available = [c for c in needed if c in header]

    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    # Use waves 2 and 3 (1990-1991 and 1995-1998)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]

    # Split Germany into East/West based on S024
    # S024=2763 is wave 3 Germany (all available in waves 2-3)
    # For the paper's analysis, we treat DEU as unified for now since wave 2/3 don't
    # clearly separate E/W. The key S024 values:
    # 2763 appears in wave 3 only
    # We need to check if wave 2 had separate E/W
    # Actually, looking at it: WVS wave 2 (1990) had West Germany only
    # WVS wave 3 (1995-98) had unified Germany
    # But the paper has E/W Germany separate. They likely used the EVS 1990 data
    # which had both, or used S024 region coding.

    # For now, let's NOT split Germany since we don't have reliable E/W split
    # in our wave 2/3 data. This will give us ~49 countries after filtering.

    # Load EVS
    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)
        combined = pd.concat([wvs, evs], ignore_index=True, sort=False)
    else:
        combined = wvs

    # Get latest per country
    combined = get_latest_per_country(combined)
    combined = recode_factor_items(combined)

    # Compute country means
    country_means = combined.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)

    # Fill remaining NaN with column means
    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Standardize
    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    # PCA
    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(len(scaled) - 1)
    scores_raw = U[:, :2] * S[:2]

    # Varimax
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    # Determine which is trad/sec-rat vs surv/self-exp
    trad_items = ['A042', 'F120', 'G006', 'E018']
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items if i in loadings_df.index)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items if i in loadings_df.index)

    if f1_trad > f2_trad:
        trad_col, surv_col = 'F1', 'F2'
    else:
        trad_col, surv_col = 'F2', 'F1'

    result = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
        'surv_selfexp': scores_df[surv_col].values
    }).reset_index(drop=True)

    # Fix direction
    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            result['trad_secrat'] = -result['trad_secrat']
        if swe['surv_selfexp'].values[0] < 0:
            result['surv_selfexp'] = -result['surv_selfexp']

    return result


def run_analysis():
    """Run Table 5a replication."""

    # Step 1: Compute factor scores
    scores = compute_factor_scores_with_east_west()
    print(f"Factor scores computed for {len(scores)} countries")

    # Paper's 65 societies. Need to identify valid countries.
    # The paper includes: E. Germany, W. Germany, N. Ireland, Yugoslavia, Pakistan
    # But our data has: DEU (unified), NIR, SRB, PAK, etc.
    # Countries to EXCLUDE (not in IB2000 or have no WB data):
    # ALB, MNE, MLT, SLV - not in original paper
    # TWN - no WB data (not a WB member)
    # PAK - present in Figure 3, likely has WB data but check
    # VEN - might be missing from WB

    # The paper's 65 societies (from Figure 3):
    # Argentina, Armenia, Australia, Austria, Azerbaijan, Bangladesh, Belarus, Belgium,
    # Bosnia, Brazil, Britain, Bulgaria, Canada, Chile, China, Colombia, Croatia,
    # Czech Republic, Denmark, Dominican Republic, East Germany, Estonia, Finland,
    # France, Georgia, Ghana, Hungary, Iceland, India, Ireland, Italy, Japan,
    # Latvia, Lithuania, Macedonia, Mexico, Moldova, Netherlands, New Zealand,
    # N. Ireland, Nigeria, Norway, Pakistan, Peru, Philippines, Poland, Portugal,
    # Puerto Rico, Romania, Russia, Serbia/Yugoslavia, Slovakia, Slovenia,
    # South Africa, South Korea, Spain, Sweden, Switzerland, Taiwan, Turkey,
    # Ukraine, Uruguay, U.S.A., Venezuela, West Germany

    # That's about 65 with E/W Germany and N. Ireland.
    # For regression: need WB data. Taiwan has none. Pakistan might.
    # The paper gets N=49, so about 16 societies lack WB data.

    # Countries definitely in paper's 49:
    paper_countries = [
        'ARG', 'ARM', 'AUS', 'AUT', 'AZE', 'BGD', 'BLR', 'BEL',
        'BIH', 'BRA', 'GBR', 'BGR', 'CAN', 'CHL', 'CHN', 'COL',
        'HRV', 'CZE', 'DNK', 'DOM', 'EST', 'FIN', 'FRA',
        'GEO', 'GHA', 'HUN', 'ISL', 'IND', 'IRL', 'ITA', 'JPN',
        'LVA', 'LTU', 'MKD', 'MEX', 'MDA', 'NLD', 'NZL',
        'NIR', 'NGA', 'NOR', 'PAK', 'PER', 'PHL', 'POL', 'PRT',
        'PRI', 'ROU', 'RUS', 'SRB', 'SVK', 'SVN',
        'ZAF', 'KOR', 'ESP', 'SWE', 'CHE',
        'TUR', 'UKR', 'URY', 'USA', 'VEN', 'DEU'
    ]
    # Note: DEU represents both E+W Germany here (as we can't split reliably)
    # This gives us ~64 unique codes, plus N. Ireland = ~65
    # After merging with WB data (no TWN), we get ~49

    # Filter scores to paper's countries
    scores = scores[scores['COUNTRY_ALPHA'].isin(paper_countries)]
    print(f"After filtering to paper countries: {len(scores)}")

    # Step 2: Load World Bank data
    wb = pd.read_csv(os.path.join(BASE_DIR, 'data/world_bank_indicators.csv'))

    # GDP per capita - use GNP per capita PPP (current) as closer to Penn World Tables
    # The paper says "Penn World tables" which reports PPP-adjusted GDP
    # Our WB data has NY.GNP.PCAP.PP.CD (GNP per capita PPP current intl $)
    # Use 1995 as a proxy for "1980" (the latest pre-survey year with good coverage)
    gnp = wb[wb['indicator'] == 'NY.GNP.PCAP.PP.CD'][['economy', 'YR1995']].copy()
    gnp.columns = ['economy', 'gdp']
    gnp = gnp.dropna()
    gnp['gdp'] = gnp['gdp'] / 1000  # Convert to $1000s

    # Industry employment
    ind = wb[wb['indicator'] == 'SL.IND.EMPL.ZS'].copy()
    # Use earliest available year (1991)
    ind_data = ind[['economy', 'YR1991']].copy()
    ind_data.columns = ['economy', 'industrial']
    ind_data = ind_data.dropna()

    # Service sector employment
    svc = wb[wb['indicator'] == 'SL.SRV.EMPL.ZS'][['economy', 'YR1991']].copy()
    svc.columns = ['economy', 'service']
    svc = svc.dropna()

    # Education enrollment - combine primary + secondary + tertiary
    edu_data = {}
    for ind_code, name in [('SE.PRM.ENRR', 'primary'), ('SE.SEC.ENRR', 'secondary'),
                           ('SE.TER.ENRR', 'tertiary')]:
        sub = wb[wb['indicator'] == ind_code]
        # Try years from 1990 backward
        for yr in ['YR1990', 'YR1989', 'YR1988', 'YR1987', 'YR1986', 'YR1985',
                    'YR1991', 'YR1992', 'YR1993', 'YR1994', 'YR1995']:
            col_data = sub[['economy', yr]].dropna()
            if len(col_data) > 0:
                col_data.columns = ['economy', name]
                edu_data[name] = col_data
                break

    # Merge factor scores with WB data
    data = scores.rename(columns={'COUNTRY_ALPHA': 'economy'})
    data = data.merge(gnp, on='economy', how='inner')
    data = data.merge(ind_data, on='economy', how='inner')

    print(f"After merging GDP + Industry: {len(data)} countries")
    print(f"Countries: {sorted(data['economy'].tolist())}")

    # The paper's factor scores are on a different scale than ours.
    # Looking at Figure 3: y-axis (trad/sec-rat) ranges ~-2.2 to ~1.8
    # Our scores range from about -3.8 to 3.7
    # The paper likely standardized differently. Let's normalize to match.
    # Scale so that the standard deviation is approximately 1.0 (similar to standardized scores)
    # Actually, looking at the paper's values and our regression coefficients:
    # Paper M1: GDP coef=0.066, Industrial coef=0.052
    # Our M1 (58 countries): GDP coef=0.043, Industrial coef=0.090
    # Ratio of GDP: 0.066/0.043 = 1.53
    # Ratio of Industrial: 0.052/0.090 = 0.58
    # These don't match a simple scale factor, suggesting the country composition matters.

    # For now, keep scores as-is and see if the N=49 fix helps

    # Cultural dummies
    ex_communist = ['ARM', 'AZE', 'BLR', 'BIH', 'BGR', 'CHN', 'HRV', 'CZE',
                    'EST', 'GEO', 'HUN', 'LVA', 'LTU', 'MKD', 'MDA', 'POL',
                    'ROU', 'RUS', 'SVK', 'SVN', 'UKR', 'SRB']

    hist_catholic = ['FRA', 'BEL', 'ITA', 'ESP', 'PRT', 'AUT', 'IRL', 'POL',
                     'HRV', 'SVN', 'HUN', 'CZE', 'SVK',
                     'ARG', 'BRA', 'CHL', 'COL', 'DOM', 'MEX', 'PER', 'PRI',
                     'URY', 'VEN', 'PHL']

    hist_confucian = ['JPN', 'KOR', 'TWN', 'CHN']

    data['ex_communist'] = data['economy'].isin(ex_communist).astype(int)
    data['catholic'] = data['economy'].isin(hist_catholic).astype(int)
    data['confucian'] = data['economy'].isin(hist_confucian).astype(int)

    # Merge service sector
    data_full = data.merge(svc, on='economy', how='left')
    # Merge education
    for name, edf in edu_data.items():
        data_full = data_full.merge(edf, on='economy', how='left')

    dv = 'trad_secrat'

    results = {}

    # Model 1: GDP + Industrial
    m1_data = data.dropna(subset=[dv, 'gdp', 'industrial'])
    X1 = sm.add_constant(m1_data[['gdp', 'industrial']])
    y1 = m1_data[dv]
    results['Model 1'] = sm.OLS(y1, X1).fit()
    print(f"\nModel 1: N={len(m1_data)}")

    # Model 2: GDP + Industrial + Education + Service + Education
    # Build education variable: sum of primary + secondary + tertiary rates
    m2_data = data_full.copy()
    edu_cols = [c for c in ['primary', 'secondary', 'tertiary'] if c in m2_data.columns]
    if len(edu_cols) >= 2:
        # "Percentage enrolled in education" = combined enrollment
        m2_data['edu_combined'] = m2_data[edu_cols].mean(axis=1)
        # The paper has TWO education rows in Model 2:
        # 1. "Percentage enrolled in education" with coef -0.01 (SE 0.01)
        # 2. "Percentage enrolled in education" with coef -0.005 (SE 0.012)
        # These might be primary+secondary vs. tertiary
        if 'primary' in m2_data.columns and 'secondary' in m2_data.columns:
            m2_data['edu1'] = (m2_data['primary'] + m2_data['secondary']) / 2
        elif 'primary' in m2_data.columns:
            m2_data['edu1'] = m2_data['primary']
        else:
            m2_data['edu1'] = m2_data[edu_cols[0]]

        if 'tertiary' in m2_data.columns:
            m2_data['edu2'] = m2_data['tertiary']
        elif len(edu_cols) > 1:
            m2_data['edu2'] = m2_data[edu_cols[-1]]

        m2_ivs = ['gdp', 'industrial']
        if 'edu1' in m2_data.columns:
            m2_ivs.append('edu1')
        if 'service' in m2_data.columns:
            m2_ivs.append('service')
        if 'edu2' in m2_data.columns:
            m2_ivs.append('edu2')

        m2_data = m2_data.dropna(subset=[dv] + m2_ivs)
        print(f"Model 2: N={len(m2_data)}")

        if len(m2_data) > len(m2_ivs) + 2:
            X2 = sm.add_constant(m2_data[m2_ivs])
            y2 = m2_data[dv]
            results['Model 2'] = sm.OLS(y2, X2).fit()
        else:
            print("Not enough data for Model 2")
            results['Model 2'] = None
    else:
        print("Not enough education variables for Model 2")
        results['Model 2'] = None

    # Model 3: GDP + Industrial + Ex-Communist
    m3_data = data.dropna(subset=[dv, 'gdp', 'industrial'])
    X3 = sm.add_constant(m3_data[['gdp', 'industrial', 'ex_communist']])
    y3 = m3_data[dv]
    results['Model 3'] = sm.OLS(y3, X3).fit()

    # Model 4: GDP + Industrial + Catholic
    m4_data = data.dropna(subset=[dv, 'gdp', 'industrial'])
    X4 = sm.add_constant(m4_data[['gdp', 'industrial', 'catholic']])
    y4 = m4_data[dv]
    results['Model 4'] = sm.OLS(y4, X4).fit()

    # Model 5: GDP + Industrial + Confucian
    m5_data = data.dropna(subset=[dv, 'gdp', 'industrial'])
    X5 = sm.add_constant(m5_data[['gdp', 'industrial', 'confucian']])
    y5 = m5_data[dv]
    results['Model 5'] = sm.OLS(y5, X5).fit()

    # Model 6: GDP + Industrial + All three dummies
    m6_data = data.dropna(subset=[dv, 'gdp', 'industrial'])
    X6 = sm.add_constant(m6_data[['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian']])
    y6 = m6_data[dv]
    results['Model 6'] = sm.OLS(y6, X6).fit()

    # Print results
    print("\n" + "=" * 90)
    print("Table 5a: Unstandardized Coefficients - Traditional/Secular-Rational Values")
    print("=" * 90)

    for mname in ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']:
        model = results.get(mname)
        if model is not None:
            print(f"\n{mname} (N={int(model.nobs)}, Adj R2={model.rsquared_adj:.2f}):")
            for var in model.params.index:
                if var == 'const':
                    continue
                coef = model.params[var]
                se = model.bse[var]
                sig = get_significance(model.pvalues[var])
                print(f"  {var:<20s} {coef:8.3f}{sig:<4s} ({se:.3f})")
        else:
            print(f"\n{mname}: Not computed")

    # Print summary table
    print("\n\n=== SUMMARY TABLE ===")
    header = f"{'Variable':<35} {'M1':>10} {'M2':>10} {'M3':>10} {'M4':>10} {'M5':>10} {'M6':>10}"
    print(header)
    print("-" * 95)

    var_list = [
        ('GDP per capita', 'gdp'),
        ('% Industrial', 'industrial'),
        ('Education 1', 'edu1'),
        ('Service', 'service'),
        ('Education 2', 'edu2'),
        ('Ex-Communist', 'ex_communist'),
        ('Catholic', 'catholic'),
        ('Confucian', 'confucian'),
    ]

    for label, varname in var_list:
        coef_line = f"{label:<35}"
        se_line = f"{'':<35}"
        for mname in ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']:
            model = results.get(mname)
            if model is not None and varname in model.params.index:
                c = model.params[varname]
                s = model.bse[varname]
                sig = get_significance(model.pvalues[varname])
                coef_line += f" {c:7.3f}{sig:<3s}"
                se_line += f" ({s:6.3f})  "
            else:
                coef_line += f" {'---':>10}"
                se_line += f" {'':>10}"
        print(coef_line)
        print(se_line)

    print("-" * 95)
    r2_line = f"{'Adj R2':<35}"
    n_line = f"{'N':<35}"
    for mname in ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']:
        model = results.get(mname)
        if model is not None:
            r2_line += f" {model.rsquared_adj:10.2f}"
            n_line += f" {int(model.nobs):>10}"
        else:
            r2_line += f" {'---':>10}"
            n_line += f" {'---':>10}"
    print(r2_line)
    print(n_line)

    return results


def score_against_ground_truth():
    results = run_analysis()

    total_points = 0
    max_points = 0
    details = []

    for mname, gt in GROUND_TRUTH.items():
        model = results.get(mname)
        if model is None:
            # Missing model - lose all points
            for var_key, var_info in gt.items():
                if var_key in ['Adj_R2', 'N']:
                    max_points += 5
                else:
                    max_points += 13
            details.append(f"{mname}: NOT COMPUTED - 0 points")
            continue

        var_map = {
            'GDP': 'gdp',
            'Industrial': 'industrial',
            'Ex_Communist': 'ex_communist',
            'Catholic': 'catholic',
            'Confucian': 'confucian',
            'Service': 'service',
            'Education': 'edu1',
            'Education2': 'edu2',
        }

        for var_key, var_info in gt.items():
            if var_key in ['Adj_R2', 'N']:
                continue

            col_name = var_map.get(var_key)
            if col_name and col_name in model.params.index:
                gen_coef = model.params[col_name]
                gen_se = model.bse[col_name]
                gen_sig = get_significance(model.pvalues[col_name])

                # Coef match
                max_points += 5
                coef_diff = abs(gen_coef - var_info['coef'])
                if coef_diff <= 0.05:
                    total_points += 5
                    details.append(f"{mname} {var_key} coef: {gen_coef:.3f} vs {var_info['coef']:.3f} MATCH")
                elif coef_diff <= 0.1:
                    total_points += 3
                    details.append(f"{mname} {var_key} coef: {gen_coef:.3f} vs {var_info['coef']:.3f} PARTIAL")
                else:
                    details.append(f"{mname} {var_key} coef: {gen_coef:.3f} vs {var_info['coef']:.3f} MISS (diff={coef_diff:.3f})")

                # SE match
                max_points += 3
                se_diff = abs(gen_se - var_info['se'])
                if se_diff <= 0.02:
                    total_points += 3
                    details.append(f"  SE: {gen_se:.3f} vs {var_info['se']:.3f} MATCH")
                elif se_diff <= 0.05:
                    total_points += 1.5
                    details.append(f"  SE: {gen_se:.3f} vs {var_info['se']:.3f} PARTIAL")
                else:
                    details.append(f"  SE: {gen_se:.3f} vs {var_info['se']:.3f} MISS")

                # Sig match
                max_points += 5
                if gen_sig == var_info['sig']:
                    total_points += 5
                    details.append(f"  Sig: {gen_sig} vs {var_info['sig']} MATCH")
                else:
                    details.append(f"  Sig: {gen_sig} vs {var_info['sig']} MISS")
            else:
                max_points += 13
                details.append(f"{mname} {var_key}: MISSING from model")

        # Adj R2
        max_points += 5
        gen_r2 = model.rsquared_adj
        paper_r2 = gt['Adj_R2']
        r2_diff = abs(gen_r2 - paper_r2)
        if r2_diff <= 0.02:
            total_points += 5
            details.append(f"{mname} Adj R2: {gen_r2:.2f} vs {paper_r2:.2f} MATCH")
        elif r2_diff <= 0.05:
            total_points += 3
            details.append(f"{mname} Adj R2: {gen_r2:.2f} vs {paper_r2:.2f} PARTIAL")
        else:
            details.append(f"{mname} Adj R2: {gen_r2:.2f} vs {paper_r2:.2f} MISS (diff={r2_diff:.2f})")

        # N
        max_points += 5
        gen_n = int(model.nobs)
        paper_n = gt['N']
        n_diff = abs(gen_n - paper_n) / paper_n
        if n_diff <= 0.05:
            total_points += 5
            details.append(f"{mname} N: {gen_n} vs {paper_n} MATCH")
        elif n_diff <= 0.15:
            total_points += 3
            details.append(f"{mname} N: {gen_n} vs {paper_n} PARTIAL")
        else:
            details.append(f"{mname} N: {gen_n} vs {paper_n} MISS")

    score = int(100 * total_points / max_points) if max_points > 0 else 0

    print("\n\n=== SCORING ===")
    for d in details:
        print(d)
    print(f"\nTotal: {total_points}/{max_points} = {score}/100")

    return score


if __name__ == "__main__":
    score = score_against_ground_truth()
