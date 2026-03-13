#!/usr/bin/env python3
"""
Table 5a Replication - Attempt 29
Inglehart & Baker (2000), Table 5a.

KEY INSIGHT: The paper uses INDIVIDUAL-LEVEL PCA (pooling all respondents from all
countries), then computes country means of individual factor scores. Our prior
attempts used ecological PCA (PCA of country means), giving different results.
Individual-level PCA should better match the paper's Figure 1 positions.

This is the methodologically correct approach per the paper's description:
"factor analysis was conducted using data from all surveys combined."
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared_factor_analysis import (load_combined_data, get_latest_per_country,
                                      recode_factor_items, FACTOR_ITEMS, varimax)

GROUND_TRUTH = {
    'Model 1': {'GDP': {'coef': 0.066, 'se': 0.031, 'sig': '*'},
                'Industrial': {'coef': 0.052, 'se': 0.012, 'sig': '***'},
                'Adj_R2': 0.42, 'N': 49},
    'Model 3': {'GDP': {'coef': 0.131, 'se': 0.036, 'sig': '**'},
                'Industrial': {'coef': 0.023, 'se': 0.015, 'sig': ''},
                'Ex_Communist': {'coef': 1.05, 'se': 0.351, 'sig': '**'},
                'Adj_R2': 0.50, 'N': 49},
    'Model 4': {'GDP': {'coef': 0.042, 'se': 0.029, 'sig': ''},
                'Industrial': {'coef': 0.061, 'se': 0.011, 'sig': '***'},
                'Catholic': {'coef': -0.767, 'se': 0.216, 'sig': '**'},
                'Adj_R2': 0.53, 'N': 49},
    'Model 5': {'GDP': {'coef': 0.080, 'se': 0.027, 'sig': '**'},
                'Industrial': {'coef': 0.052, 'se': 0.011, 'sig': '***'},
                'Confucian': {'coef': 1.57, 'se': 0.370, 'sig': '***'},
                'Adj_R2': 0.57, 'N': 49},
    'Model 6': {'GDP': {'coef': 0.122, 'se': 0.030, 'sig': '***'},
                'Industrial': {'coef': 0.030, 'se': 0.012, 'sig': '*'},
                'Ex_Communist': {'coef': 0.952, 'se': 0.282, 'sig': '***'},
                'Catholic': {'coef': -0.409, 'se': 0.188, 'sig': '*'},
                'Confucian': {'coef': 1.39, 'se': 0.329, 'sig': '***'},
                'Adj_R2': 0.70, 'N': 49},
}

EX_COMMUNIST = ['BGR', 'CHN', 'CZE', 'DEU_E', 'HUN', 'POL', 'ROU', 'RUS',
                'SVK', 'SVN', 'SRB', 'BIH', 'MKD', 'HRV', 'BLR', 'UKR',
                'ARM', 'AZE', 'GEO', 'MDA', 'LVA', 'LTU', 'EST']
HIST_CATHOLIC = ['FRA', 'BEL', 'ITA', 'ESP', 'PRT', 'AUT', 'IRL', 'POL',
                 'HRV', 'SVN', 'HUN', 'CZE', 'SVK', 'ARG', 'BRA', 'CHL',
                 'COL', 'DOM', 'MEX', 'PER', 'PRI', 'URY', 'VEN', 'PHL', 'SLV']
HIST_CONFUCIAN = ['JPN', 'KOR', 'TWN', 'CHN']

EXCLUDE = {'ARM', 'AZE', 'BIH', 'BLR', 'COL', 'EST', 'GEO', 'GHA',
           'LTU', 'LVA', 'MDA', 'MKD', 'NGA', 'SVN', 'VEN'}

ISO_MAP = {'DEU': 'DEU_W', 'NIR': 'GBR'}


def get_significance(pvalue):
    if pvalue < 0.001: return '***'
    elif pvalue < 0.01: return '**'
    elif pvalue < 0.05: return '*'
    return ''


def compute_individual_pca_scores():
    """
    Compute factor scores using individual-level PCA (paper's method).
    1. Pool all individuals from all countries (waves 2-3 + EVS)
    2. Standardize items across all individuals
    3. Run PCA on individuals -> get factor loadings
    4. Apply loadings to get individual factor scores
    5. Average within each country to get national scores
    """
    df = load_combined_data(waves_wvs=[2, 3], include_evs=True)
    df = get_latest_per_country(df)
    df = recode_factor_items(df)

    # Use only complete cases (all 10 items present)
    items_data = df[['COUNTRY_ALPHA'] + FACTOR_ITEMS].copy()

    # Handle missing: use items with at least 7 valid
    valid_counts = items_data[FACTOR_ITEMS].notna().sum(axis=1)
    items_data = items_data[valid_counts >= 7].copy()

    # Fill remaining NaN with column means (within the pooled sample)
    for col in FACTOR_ITEMS:
        items_data[col] = items_data[col].fillna(items_data[col].mean())

    print(f"Individual-level PCA: {len(items_data)} respondents from {items_data['COUNTRY_ALPHA'].nunique()} countries")

    # Standardize across all individuals (individual-level standardization)
    X = items_data[FACTOR_ITEMS].values.astype(float)
    col_means = X.mean(axis=0)
    col_stds = X.std(axis=0)
    col_stds = np.where(col_stds == 0, 1, col_stds)
    X_std = (X - col_means) / col_stds

    # PCA via covariance matrix (individual-level)
    # Use correlation matrix approach (X already standardized)
    n = X_std.shape[0]
    cov = X_std.T @ X_std / (n - 1)  # correlation matrix (since X is standardized)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take first 2 components
    loadings = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])  # factor loadings

    # Varimax rotation
    loadings_rot, R = varimax(loadings)

    # Compute individual factor scores
    scores = X_std @ eigenvectors[:, :2]  # N x 2
    scores_rot = scores @ R  # apply rotation

    # Create individual score columns
    items_data = items_data.copy()
    items_data['F1'] = scores_rot[:, 0]
    items_data['F2'] = scores_rot[:, 1]

    # Compute country means
    country_scores = items_data.groupby('COUNTRY_ALPHA')[['F1', 'F2']].mean()

    # Determine which factor is Traditional vs. Survival
    # Check F1 loadings: items 0-4 are Traditional (should load positively)
    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    trad_items = ['AUTONOMY', 'F120', 'G006', 'E018']
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items if i in loadings_df.index)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items if i in loadings_df.index)

    if f1_trad > f2_trad:
        trad_col, surv_col = 'F1', 'F2'
    else:
        trad_col, surv_col = 'F2', 'F1'

    result = pd.DataFrame({
        'COUNTRY_ALPHA': country_scores.index,
        'trad_secrat': country_scores[trad_col].values,
        'surv_selfexp': country_scores[surv_col].values
    }).reset_index(drop=True)

    # Fix direction: Sweden should be high positive on secular-rational
    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe_val = result.loc[result['COUNTRY_ALPHA'] == 'SWE', 'trad_secrat'].values[0]
        if swe_val < 0:
            result['trad_secrat'] = -result['trad_secrat']
        swe_surv = result.loc[result['COUNTRY_ALPHA'] == 'SWE', 'surv_selfexp'].values[0]
        if swe_surv < 0:
            result['surv_selfexp'] = -result['surv_selfexp']

    # Normalize to SD=1 to match paper scale
    trad_sd = result['trad_secrat'].std()
    result['trad_secrat_norm'] = result['trad_secrat'] / trad_sd

    print(f"Individual PCA - Factor score SD: {trad_sd:.3f}")
    for iso in ['JPN', 'KOR', 'TWN', 'CHN', 'SWE', 'NOR', 'DEU', 'USA', 'IRL']:
        row = result[result['COUNTRY_ALPHA'] == iso]
        if len(row) > 0:
            raw_v = row['trad_secrat'].values[0]
            norm_v = row['trad_secrat_norm'].values[0]
            print(f"  {iso}: raw={raw_v:.3f} norm={norm_v:.3f}")

    return result


def load_wdr_industry():
    csv_path = os.path.join(BASE_DIR, 'data/wdr_industry_1980.csv')
    df = pd.read_csv(csv_path)
    return dict(zip(df['iso'], df['industry_pct_1980']))


def load_pwt_gdp():
    pwt_path = os.path.join(BASE_DIR, 'data/pwt56_forweb.xls')
    pwt = pd.read_excel(pwt_path, sheet_name='PWT56')
    pwt80 = pwt[pwt['Year'] == 1980][['Country', 'RGDPCH']].dropna(subset=['RGDPCH'])
    pwt_to_iso = {
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
        'U.S.S.R.': 'USSR', 'YUGOSLAVIA': 'YUG', 'CZECHOSLOVAKIA': 'CSK',
        'EL SALVADOR': 'SLV', 'GREECE': 'GRC', 'ISRAEL': 'ISR',
    }
    gdp_dict = {}
    for _, r in pwt80.iterrows():
        code = pwt_to_iso.get(r['Country'])
        if code:
            gdp_dict[code] = r['RGDPCH'] / 1000.0
    if 'USSR' in gdp_dict:
        u = gdp_dict['USSR']
        for c in ['RUS', 'BLR', 'UKR', 'EST', 'LVA', 'LTU']: gdp_dict[c] = u
        for c in ['ARM', 'AZE', 'GEO', 'MDA']: gdp_dict[c] = u * 0.8
    if 'YUG' in gdp_dict:
        y = gdp_dict['YUG']
        for c in ['SRB', 'HRV', 'SVN', 'BIH', 'MKD']: gdp_dict[c] = y
    if 'CSK' in gdp_dict:
        gdp_dict['CZE'] = gdp_dict['SVK'] = gdp_dict['CSK']
    if 'DEU_W' in gdp_dict:
        gdp_dict['DEU'] = gdp_dict['DEU_W']
    if 'GBR' in gdp_dict:
        gdp_dict['NIR'] = gdp_dict['GBR']
    return gdp_dict


def run_analysis():
    # Step 1: Individual-level PCA for factor scores
    scores_df = compute_individual_pca_scores()

    # Step 2: External data
    gdp_dict = load_pwt_gdp()
    industry_dict = load_wdr_industry()

    # Step 3: Build dataset
    data_rows = []
    for _, row in scores_df.iterrows():
        iso = row['COUNTRY_ALPHA']
        if iso in EXCLUDE:
            continue
        lookup_iso = ISO_MAP.get(iso, iso)
        gdp_val = gdp_dict.get(lookup_iso)
        ind_val = industry_dict.get(iso, industry_dict.get(lookup_iso))
        if gdp_val is None or ind_val is None:
            print(f"  Missing: {iso}, GDP={gdp_val is not None}, Ind={ind_val is not None}")
            continue
        data_rows.append({
            'economy': iso,
            'trad_secrat': row['trad_secrat_norm'],
            'gdp': gdp_val,
            'industrial': ind_val,
        })

    df = pd.DataFrame(data_rows)
    df['ex_communist'] = df['economy'].isin(EX_COMMUNIST).astype(int)
    df['catholic'] = df['economy'].isin(HIST_CATHOLIC).astype(int)
    df['confucian'] = df['economy'].isin(HIST_CONFUCIAN).astype(int)
    print(f"\nFinal N={len(df)}")

    # Step 4: Regressions
    y = df['trad_secrat']
    results = {}
    results['Model 1'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial']])).fit()
    results['Model 3'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'ex_communist']])).fit()
    results['Model 4'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'catholic']])).fit()
    results['Model 5'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'confucian']])).fit()
    results['Model 6'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian']])).fit()

    var_map = {'GDP': 'gdp', 'Industrial': 'industrial', 'Ex_Communist': 'ex_communist',
               'Catholic': 'catholic', 'Confucian': 'confucian'}

    print("\n=== RESULTS vs GROUND TRUTH ===")
    for mname, gt in GROUND_TRUTH.items():
        res = results[mname]
        r2d = abs(res.rsquared_adj - gt['Adj_R2'])
        r2m = 'MATCH' if r2d <= 0.02 else ('PARTIAL' if r2d <= 0.05 else 'MISS')
        print(f"\n{mname} (N={len(res.resid)}, AdjR2={res.rsquared_adj:.2f} vs {gt['Adj_R2']} {r2m}):")
        for vname, vgt in gt.items():
            if vname in ('Adj_R2', 'N'): continue
            vkey = var_map[vname]
            if vkey not in res.params.index: continue
            c = res.params[vkey]; s = res.bse[vkey]
            sg = get_significance(res.pvalues[vkey])
            label = 'MATCH' if abs(c - vgt['coef']) <= 0.05 else (
                    'PARTIAL' if abs(c - vgt['coef']) <= 0.15 else 'MISS')
            print(f"  {vname}: {c:.3f}{sg} SE={s:.3f} vs {vgt['coef']:.3f}{vgt['sig']} {label}")

    score = score_against_ground_truth_with_results(results)
    print(f"\nFinal Score: {score:.1f}/100")
    return score


def score_against_ground_truth_with_results(results):
    var_map = {'GDP': 'gdp', 'Industrial': 'industrial', 'Ex_Communist': 'ex_communist',
               'Catholic': 'catholic', 'Confucian': 'confucian'}
    total = 0.0; max_pts = 0.0
    for mname, gt in GROUND_TRUTH.items():
        res = results.get(mname)
        if res is None: continue
        max_pts += 1.5
        total += 1.5 if abs(len(res.resid) - gt['N']) <= int(gt['N'] * 0.05) + 1 else 0
        max_pts += 2.0
        r2d = abs(res.rsquared_adj - gt['Adj_R2'])
        total += 2.0 if r2d <= 0.02 else (1.0 if r2d <= 0.05 else 0)
        for vname, vgt in gt.items():
            if vname in ('Adj_R2', 'N'): continue
            vkey = var_map[vname]
            if vkey not in res.params.index: max_pts += 6.0; continue
            c = res.params[vkey]; s = res.bse[vkey]
            sg = get_significance(res.pvalues[vkey])
            max_pts += 2.0; dc = abs(c - vgt['coef'])
            total += 2.0 if dc <= 0.05 else (1.0 if dc <= 0.15 else 0)
            max_pts += 2.0
            total += 2.0 if abs(s - vgt['se']) <= max(0.01, 0.2 * vgt['se']) else (
                     1.0 if abs(s - vgt['se']) <= max(0.02, 0.4 * vgt['se']) else 0)
            max_pts += 2.0
            total += 2.0 if sg == vgt['sig'] else (0.5 if bool(sg) == bool(vgt['sig']) else 0)
    return 100.0 * total / max_pts if max_pts > 0 else 0


def score_against_ground_truth():
    return run_analysis()


if __name__ == "__main__":
    run_analysis()
