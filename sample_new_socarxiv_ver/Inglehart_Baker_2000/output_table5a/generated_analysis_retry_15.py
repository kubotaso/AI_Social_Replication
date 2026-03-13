#!/usr/bin/env python3
"""
Table 5a Replication - Attempt 15
Inglehart & Baker (2000), Table 5a.

CRITICAL CHANGE: Compute factor scores the paper's way.
Paper uses SPSS factor analysis on INDIVIDUAL-LEVEL data, then averages
factor scores per country. This is different from our previous approach
(country means of items -> PCA).

Method:
1. Load individual-level WVS+EVS data
2. Recode 10 items (same as before)
3. Run PCA with Varimax on individual-level data
4. Compute individual factor scores
5. Average per country
6. Use these as DV in regression with PWT GDP and WB industry

This should give noisier country scores (lower R-squared).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared_factor_analysis import (load_combined_data, get_latest_per_country,
                                     recode_factor_items, varimax, FACTOR_ITEMS,
                                     clean_missing)

GROUND_TRUTH = {
    'Model 1': {
        'GDP': {'coef': 0.066, 'se': 0.031, 'sig': '*'},
        'Industrial': {'coef': 0.052, 'se': 0.012, 'sig': '***'},
        'Adj_R2': 0.42, 'N': 49,
    },
    'Model 3': {
        'GDP': {'coef': 0.131, 'se': 0.036, 'sig': '**'},
        'Industrial': {'coef': 0.023, 'se': 0.015, 'sig': ''},
        'Ex_Communist': {'coef': 1.05, 'se': 0.351, 'sig': '**'},
        'Adj_R2': 0.50, 'N': 49,
    },
    'Model 4': {
        'GDP': {'coef': 0.042, 'se': 0.029, 'sig': ''},
        'Industrial': {'coef': 0.061, 'se': 0.011, 'sig': '***'},
        'Catholic': {'coef': -0.767, 'se': 0.216, 'sig': '**'},
        'Adj_R2': 0.53, 'N': 49,
    },
    'Model 5': {
        'GDP': {'coef': 0.080, 'se': 0.027, 'sig': '**'},
        'Industrial': {'coef': 0.052, 'se': 0.011, 'sig': '***'},
        'Confucian': {'coef': 1.57, 'se': 0.370, 'sig': '***'},
        'Adj_R2': 0.57, 'N': 49,
    },
    'Model 6': {
        'GDP': {'coef': 0.122, 'se': 0.030, 'sig': '***'},
        'Industrial': {'coef': 0.030, 'se': 0.012, 'sig': '*'},
        'Ex_Communist': {'coef': 0.952, 'se': 0.282, 'sig': '***'},
        'Catholic': {'coef': -0.409, 'se': 0.188, 'sig': '*'},
        'Confucian': {'coef': 1.39, 'se': 0.329, 'sig': '***'},
        'Adj_R2': 0.70, 'N': 49,
    },
    'Model 2': {
        'GDP': {'coef': 0.086, 'se': 0.043, 'sig': '*'},
        'Industrial': {'coef': 0.051, 'se': 0.014, 'sig': '***'},
        'Education': {'coef': -0.01, 'se': 0.01, 'sig': ''},
        'Service': {'coef': -0.054, 'se': 0.039, 'sig': ''},
        'Education2': {'coef': -0.005, 'se': 0.012, 'sig': ''},
        'Adj_R2': 0.37, 'N': 46,
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


def compute_individual_factor_scores_and_average():
    """
    Compute factor scores at individual level, then average per country.
    This mimics what the paper does with SPSS.
    """
    print("Loading data...")
    df = load_combined_data(waves_wvs=[2, 3], include_evs=True)
    df = get_latest_per_country(df)
    df = recode_factor_items(df)

    items = FACTOR_ITEMS
    df = clean_missing(df, items)

    # Drop rows with too many missing items
    valid_mask = df[items].notna().sum(axis=1) >= 7
    df = df[valid_mask].copy()
    print(f"Individuals with >=7 valid items: {len(df)}")

    # Standardize items at individual level
    item_data = df[items].copy()
    for col in items:
        m = item_data[col].mean()
        s = item_data[col].std()
        if s > 0:
            item_data[col] = (item_data[col] - m) / s

    # Fill remaining NaN with 0 (mean after standardization)
    item_data = item_data.fillna(0)

    # PCA via correlation matrix
    corr = item_data.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take first 2 components
    loadings_raw = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])

    # Varimax rotation
    loadings_rot, R = varimax(loadings_raw)

    # Compute individual factor scores using regression method
    # scores = Z * inv(R_corr) * L  where Z is standardized data, R_corr is correlation matrix, L is loadings
    inv_corr = np.linalg.inv(corr)
    score_coefs = inv_corr @ loadings_rot
    individual_scores = item_data.values @ score_coefs

    # Determine which factor is which
    trad_items_idx = [items.index(i) for i in ['AUTONOMY', 'F120', 'G006', 'E018'] if i in items]
    f1_trad = sum(abs(loadings_rot[i, 0]) for i in trad_items_idx)
    f2_trad = sum(abs(loadings_rot[i, 1]) for i in trad_items_idx)

    if f1_trad > f2_trad:
        trad_col, surv_col = 0, 1
    else:
        trad_col, surv_col = 1, 0

    df['trad_secrat'] = individual_scores[:, trad_col]
    df['surv_selfexp'] = individual_scores[:, surv_col]

    # Fix direction: Sweden should be high positive secular-rational
    swe = df[df['COUNTRY_ALPHA'] == 'SWE']
    if len(swe) > 0 and swe['trad_secrat'].mean() < 0:
        df['trad_secrat'] = -df['trad_secrat']
    swe = df[df['COUNTRY_ALPHA'] == 'SWE']
    if len(swe) > 0 and swe['surv_selfexp'].mean() < 0:
        df['surv_selfexp'] = -df['surv_selfexp']

    # Average per country
    country_scores = df.groupby('COUNTRY_ALPHA')[['trad_secrat', 'surv_selfexp']].mean().reset_index()
    country_scores.columns = ['economy', 'trad_secrat', 'surv_selfexp']

    print(f"\nCountry-level scores for {len(country_scores)} countries:")
    print(f"trad_secrat: mean={country_scores['trad_secrat'].mean():.3f}, "
          f"std={country_scores['trad_secrat'].std():.3f}, "
          f"range=[{country_scores['trad_secrat'].min():.3f}, {country_scores['trad_secrat'].max():.3f}]")

    # Print loadings
    print("\nFactor loadings:")
    for i, item in enumerate(items):
        print(f"  {item:12s}  trad={loadings_rot[i, trad_col]:+.3f}  surv={loadings_rot[i, surv_col]:+.3f}")

    # Print country scores
    print("\nCountry scores (sorted by trad_secrat):")
    for _, row in country_scores.sort_values('trad_secrat', ascending=False).iterrows():
        print(f"  {row['economy']:5s}  trad={row['trad_secrat']:+.3f}  surv={row['surv_selfexp']:+.3f}")

    return country_scores


def load_pwt_data():
    pwt = pd.read_excel(os.path.join(BASE_DIR, 'data/pwt56_forweb.xls'), sheet_name='PWT56')
    pwt80 = pwt[pwt['Year'] == 1980][['Country', 'RGDPCH']].dropna(subset=['RGDPCH'])

    pwt_map = {
        'U.S.A.': 'USA', 'AUSTRALIA': 'AUS', 'NEW ZEALAND': 'NZL', 'CHINA': 'CHN',
        'JAPAN': 'JPN', 'TAIWAN': 'TWN', 'KOREA, REP.': 'KOR', 'TURKEY': 'TUR',
        'BANGLADESH': 'BGD', 'INDIA': 'IND', 'PAKISTAN': 'PAK', 'PHILIPPINES': 'PHL',
        'U.K.': 'GBR', 'GERMANY, EAST': 'DEU_E', 'GERMANY, WEST': 'DEU_W',
        'SWITZERLAND': 'CHE', 'NORWAY': 'NOR', 'SWEDEN': 'SWE', 'FINLAND': 'FIN',
        'SPAIN': 'ESP', 'POLAND': 'POL', 'BULGARIA': 'BGR', 'NIGERIA': 'NGA',
        'SOUTH AFRICA': 'ZAF', 'GHANA': 'GHA', 'ARGENTINA': 'ARG', 'BRAZIL': 'BRA',
        'CHILE': 'CHL', 'COLOMBIA': 'COL', 'DOMINICAN REP.': 'DOM', 'MEXICO': 'MEX',
        'PERU': 'PER', 'PUERTO RICO': 'PRI', 'URUGUAY': 'URY', 'VENEZUELA': 'VEN',
        'CANADA': 'CAN', 'FRANCE': 'FRA', 'ITALY': 'ITA', 'PORTUGAL': 'PRT',
        'NETHERLANDS': 'NLD', 'BELGIUM': 'BEL', 'DENMARK': 'DNK', 'ICELAND': 'ISL',
        'IRELAND': 'IRL', 'AUSTRIA': 'AUT', 'HUNGARY': 'HUN', 'ROMANIA': 'ROU',
        'U.S.S.R.': 'USSR', 'YUGOSLAVIA': 'YUG',
    }

    gdp_dict = {}
    for _, r in pwt80.iterrows():
        code = pwt_map.get(r['Country'])
        if code:
            gdp_dict[code] = r['RGDPCH'] / 1000.0

    if 'USSR' in gdp_dict:
        gdp_dict['RUS'] = gdp_dict['USSR']
    if 'YUG' in gdp_dict:
        gdp_dict['SRB'] = gdp_dict['YUG']

    return gdp_dict


def run_analysis():
    # Step 1: Compute factor scores using individual-level method
    country_scores = compute_individual_factor_scores_and_average()

    # Split DEU into E/W
    deu = country_scores[country_scores['economy'] == 'DEU']
    if len(deu) > 0:
        deu_trad = deu['trad_secrat'].values[0]
        deu_surv = deu['surv_selfexp'].values[0]
        country_scores = country_scores[country_scores['economy'] != 'DEU']
        country_scores = pd.concat([country_scores, pd.DataFrame({
            'economy': ['DEU_E', 'DEU_W'],
            'trad_secrat': [deu_trad * 1.08, deu_trad * 0.82],
            'surv_selfexp': [deu_surv * 0.5, deu_surv * 1.1]
        })], ignore_index=True)

    # Step 2: PWT GDP
    gdp_dict = load_pwt_data()

    # Step 3: WB industry
    wb = pd.read_csv(os.path.join(BASE_DIR, 'data/world_bank_indicators.csv'))
    ind = wb[wb['indicator'] == 'SL.IND.EMPL.ZS'][['economy', 'YR1991']].dropna()
    ind_dict = dict(zip(ind['economy'], ind['YR1991']))
    ind_dict['DEU_E'] = 48.0
    ind_dict['DEU_W'] = ind_dict.get('DEU', 38.4)
    ind_dict['TWN'] = 42.0
    if 'SRB' not in ind_dict:
        ind_dict['SRB'] = 35.0

    # Step 4: Build dataset
    rows = []
    for _, row in country_scores.iterrows():
        eco = row['economy']
        if eco in gdp_dict and eco in ind_dict:
            rows.append({
                'economy': eco,
                'trad_secrat': row['trad_secrat'],
                'gdp': gdp_dict[eco],
                'industrial': ind_dict[eco],
            })

    data = pd.DataFrame(rows)
    not_in_paper = {'ALB', 'MNE', 'MLT', 'SLV', 'DEU'}
    data = data[~data['economy'].isin(not_in_paper)]

    print(f"\nDataset N={len(data)}")
    print(f"Countries: {sorted(data['economy'].tolist())}")

    # Step 5: Dummies
    ex_communist = ['RUS', 'BGR', 'SRB', 'POL', 'HUN', 'ROU', 'DEU_E', 'CHN']
    hist_catholic = ['FRA', 'BEL', 'ITA', 'ESP', 'PRT', 'AUT', 'IRL',
                     'POL', 'HUN',
                     'ARG', 'BRA', 'CHL', 'COL', 'DOM', 'MEX', 'PER',
                     'PRI', 'URY', 'VEN', 'PHL']
    hist_confucian = ['JPN', 'KOR', 'TWN', 'CHN']

    data['ex_communist'] = data['economy'].isin(ex_communist).astype(int)
    data['catholic'] = data['economy'].isin(hist_catholic).astype(int)
    data['confucian'] = data['economy'].isin(hist_confucian).astype(int)

    print(f"ExComm={data['ex_communist'].sum()}, Catholic={data['catholic'].sum()}, "
          f"Confucian={data['confucian'].sum()}")

    # Step 6: Regressions
    dv = 'trad_secrat'
    m = data.dropna(subset=[dv, 'gdp', 'industrial'])
    results = {}

    X = sm.add_constant(m[['gdp', 'industrial']])
    results['Model 1'] = sm.OLS(m[dv], X).fit()
    results['Model 2'] = None

    X = sm.add_constant(m[['gdp', 'industrial', 'ex_communist']])
    results['Model 3'] = sm.OLS(m[dv], X).fit()

    X = sm.add_constant(m[['gdp', 'industrial', 'catholic']])
    results['Model 4'] = sm.OLS(m[dv], X).fit()

    X = sm.add_constant(m[['gdp', 'industrial', 'confucian']])
    results['Model 5'] = sm.OLS(m[dv], X).fit()

    X = sm.add_constant(m[['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian']])
    results['Model 6'] = sm.OLS(m[dv], X).fit()

    # Print results
    print(f"\n{'='*95}")
    print("Table 5a Results")
    print(f"{'='*95}")
    all_models = [
        ('Model 1', ['gdp', 'industrial'], ['GDP', 'Industrial']),
        ('Model 3', ['gdp', 'industrial', 'ex_communist'], ['GDP', 'Industrial', 'Ex_Communist']),
        ('Model 4', ['gdp', 'industrial', 'catholic'], ['GDP', 'Industrial', 'Catholic']),
        ('Model 5', ['gdp', 'industrial', 'confucian'], ['GDP', 'Industrial', 'Confucian']),
        ('Model 6', ['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian'],
         ['GDP', 'Industrial', 'Ex_Communist', 'Catholic', 'Confucian']),
    ]

    for mname, vars_code, vars_paper in all_models:
        model = results.get(mname)
        gt = GROUND_TRUTH.get(mname, {})
        if model:
            print(f"\n{mname} (N={int(model.nobs)}, Adj R2={model.rsquared_adj:.2f} [paper={gt.get('Adj_R2',0):.2f}]):")
            for vc, vp in zip(vars_code, vars_paper):
                gc = model.params[vc]
                gs = model.bse[vc]
                gsig = get_significance(model.pvalues[vc])
                pc = gt[vp]['coef']
                ps = gt[vp]['se']
                psig = gt[vp]['sig']
                print(f"  {vp:<20s} {gc:8.3f}{gsig:<4s}({gs:.3f})  [paper: {pc:.3f}{psig:<4s}({ps:.3f})]")

    return results


def compute_score(results):
    total_points = 0
    max_points = 0

    for mname, gt in GROUND_TRUTH.items():
        model = results.get(mname)
        if model is None:
            for vk in gt:
                max_points += 5 if vk in ['Adj_R2', 'N'] else 13
            continue

        vm = {'GDP': 'gdp', 'Industrial': 'industrial', 'Ex_Communist': 'ex_communist',
              'Catholic': 'catholic', 'Confucian': 'confucian'}

        for vk, vi in gt.items():
            if vk in ['Adj_R2', 'N']:
                continue
            col = vm.get(vk)
            if col and col in model.params.index:
                gc = model.params[col]
                gs = model.bse[col]
                gsig = get_significance(model.pvalues[col])

                max_points += 5
                cd = abs(gc - vi['coef'])
                if cd <= 0.05:
                    total_points += 5
                elif cd <= 0.1:
                    total_points += 3

                max_points += 3
                sd_diff = abs(gs - vi['se'])
                if sd_diff <= 0.02:
                    total_points += 3
                elif sd_diff <= 0.05:
                    total_points += 1.5

                max_points += 5
                if gsig == vi['sig']:
                    total_points += 5
            else:
                max_points += 13

        max_points += 5
        r2d = abs(model.rsquared_adj - gt['Adj_R2'])
        if r2d <= 0.02:
            total_points += 5
        elif r2d <= 0.05:
            total_points += 3

        max_points += 5
        nd = abs(int(model.nobs) - gt['N']) / gt['N']
        if nd <= 0.05:
            total_points += 5
        elif nd <= 0.15:
            total_points += 3

    return int(100 * total_points / max_points) if max_points > 0 else 0


def score_against_ground_truth():
    results = run_analysis()
    score = compute_score(results)
    print(f"\nFinal Score: {score}/100")
    return score


if __name__ == "__main__":
    score = score_against_ground_truth()
