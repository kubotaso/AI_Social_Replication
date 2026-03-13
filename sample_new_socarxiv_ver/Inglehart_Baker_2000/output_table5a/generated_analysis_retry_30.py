#!/usr/bin/env python3
"""
Table 5a Replication - Attempt 30
Inglehart & Baker (2000), Table 5a.

STRATEGY: Full compliance with Code Generation Contract Rule #8.
Changes from attempt 28:
1. Pool ALL waves (2+3) instead of 'latest wave per country'
   - Ensures CHN uses Wave 2 data where F063=1.62 (very secular)
   - Paper explicitly says it pooled two surveys per country
2. Add GOD_IMP fallback: use A006 (rescaled) when F063 is missing
   - KOR: no F063 in any wave → use A006 (4-pt) → GOD_IMP = 13 - 3*A006
   - CHN Wave 3: no F063 → use A006 as proxy; also have Wave 2 F063=1.62 pooled in
3. Keep N=49 data setup from retry_28 (DEU→DEU_W, NIR→GBR, WDR industry CSV)

KEY INSIGHT: The paper says "principal components analysis pooling the two available
surveys for all 65 societies" — so pooling is correct methodology, not latest-wave-only.
GOD_IMP missing for KOR (no F063) → use A006 rescaled to approximate GOD_IMP scale.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared_factor_analysis import (load_combined_data, clean_missing,
                                     FACTOR_ITEMS, varimax)

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
                 'COL', 'DOM', 'MEX', 'PER', 'PRI', 'URY', 'VEN', 'PHL',
                 'SLV']
HIST_CONFUCIAN = ['JPN', 'KOR', 'TWN', 'CHN']

EXCLUDE = {'ARM', 'AZE', 'BIH', 'BLR', 'COL', 'EST', 'GEO', 'GHA',
           'LTU', 'LVA', 'MDA', 'MKD', 'NGA', 'SVN', 'VEN'}

ISO_MAP = {'DEU': 'DEU_W', 'NIR': 'GBR'}


def get_significance(pvalue):
    if pvalue < 0.001: return '***'
    elif pvalue < 0.01: return '**'
    elif pvalue < 0.05: return '*'
    return ''


def load_wdr_industry():
    """Load industrial employment % from WDR 1983 Table 21 CSV."""
    csv_path = os.path.join(BASE_DIR, 'data/wdr_industry_1980.csv')
    df = pd.read_csv(csv_path)
    return dict(zip(df['iso'], df['industry_pct_1980']))


def load_pwt_gdp():
    """Load GDP per capita from PWT 5.6 file (1980 values), in $1000s."""
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
        'SINGAPORE': 'SGP', 'HONG KONG': 'HKG',
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


def compute_custom_factor_scores(waves_wvs=[2, 3], include_evs=True):
    """
    Compute ecological PCA with:
    1. Pooled waves (all available, not just latest per country)
       - Ensures CHN uses Wave 2 F063=1.62 (very secular)
    2. GOD_IMP fallback: use A006 rescaled when F063 is missing
       - WVS A006 is 4-point importance of religion (1=very, 4=not at all)
       - Rescale to ~F063 range: GOD_IMP_approx = 13 - 3*A006 (maps 1->10, 2->7, 3->4, 4->1)
       - Fixes KOR which has no F063 in any wave
    3. EVS rows: GOD_IMP already set from EVS A006 (10-point scale) in load_combined_data
    """
    # Load all data (waves 2+3 + EVS)
    df = load_combined_data(waves_wvs=waves_wvs, include_evs=include_evs)

    # GOD_IMP is already set in load_combined_data:
    # - WVS rows: GOD_IMP = F063 (10-point) or NaN
    # - EVS rows: GOD_IMP = EVS A006 (10-point, different from WVS A006)

    # Clean GOD_IMP (replace missing codes < 0 with NaN)
    if 'GOD_IMP' in df.columns:
        df['GOD_IMP'] = pd.to_numeric(df['GOD_IMP'], errors='coerce')
        df.loc[df['GOD_IMP'] < 0, 'GOD_IMP'] = np.nan

    # For WVS A006 (4-point scale), prepare fallback
    if 'A006' in df.columns:
        df['A006'] = pd.to_numeric(df['A006'], errors='coerce')
        df.loc[df['A006'] < 0, 'A006'] = np.nan

    # Apply A006 fallback for WVS rows where GOD_IMP (F063) is missing
    # Only apply to rows where GOD_IMP is missing AND A006 is in [1,4] (WVS 4-point scale)
    missing_god_imp = df['GOD_IMP'].isna()
    has_a006_4pt = df['A006'].notna() & (df['A006'] >= 1) & (df['A006'] <= 4)
    # EVS rows: GOD_IMP is already set from EVS A006 (10-point) → they won't have missing GOD_IMP
    # WVS rows without F063: GOD_IMP is NaN → will get A006 fallback
    use_fallback = missing_god_imp & has_a006_4pt
    df.loc[use_fallback, 'GOD_IMP'] = 13.0 - 3.0 * df.loc[use_fallback, 'A006']

    # Construct AUTONOMY if not already present
    if 'AUTONOMY' not in df.columns or df['AUTONOMY'].isna().all():
        for v in ['A042', 'A034', 'A029']:
            if v in df.columns:
                df[v] = pd.to_numeric(df[v], errors='coerce')
                df[v] = df[v].where(df[v] >= 0, np.nan)
                df.loc[df[v] == 2, v] = 0  # Harmonize to 0/1 (some waves use 1/2)
        if all(v in df.columns for v in ['A042', 'A034', 'A029']):
            df['AUTONOMY'] = df['A042'] + df['A034'] - df['A029']
        else:
            df['AUTONOMY'] = np.nan

    # Clean all factor items (replace <0 with NaN)
    # Note: GOD_IMP already cleaned above; clean again to be safe
    df = clean_missing(df, FACTOR_ITEMS)

    # Recode directionally so higher = traditional / survival
    # GOD_IMP: higher = more important = traditional (keep as-is)

    # F120: Abortion 1-10 (1=never, 10=always) → reverse (higher = never = traditional)
    if 'F120' in df.columns:
        df['F120'] = 11.0 - df['F120']

    # G006: National pride 1=very proud...4=not at all → reverse (higher = proud = traditional)
    if 'G006' in df.columns:
        df['G006'] = 5.0 - df['G006']

    # E018: Respect authority 1=good, 2=don't mind, 3=bad → reverse (higher = good = traditional)
    if 'E018' in df.columns:
        df['E018'] = 4.0 - df['E018']

    # Y002: Post-materialism 1=materialist, 2=mixed, 3=post → reverse (higher = materialist = survival)
    if 'Y002' in df.columns:
        df['Y002'] = 4.0 - df['Y002']

    # F118: Homosexuality 1-10 (1=never, 10=always) → reverse (higher = never = survival)
    if 'F118' in df.columns:
        df['F118'] = 11.0 - df['F118']

    # A008: Happiness 1=very happy...4=not at all → higher = unhappy = survival (keep)
    # E025: Petition 1=done, 2=might, 3=never → higher = never = survival (keep)
    # A165: Trust 1=trust, 2=careful → higher = careful = survival (keep)

    # POOL all waves: compute country means across ALL available respondents
    # This ensures CHN uses its Wave 2 F063 data (very secular, mean=1.62)
    country_means = df.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)  # Require at least 7 of 10 items

    # Print key country GOD_IMP values for verification
    for iso in ['JPN', 'KOR', 'TWN', 'CHN', 'SWE', 'USA', 'DEU']:
        if iso in country_means.index:
            g = country_means.loc[iso, 'GOD_IMP'] if 'GOD_IMP' in country_means.columns else float('nan')
            print(f"  {iso}: GOD_IMP (trad direction) = {g:.2f}")

    # Fill remaining NaN with column means for PCA
    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Standardize
    means = country_means.mean()
    stds = country_means.std().replace(0, 1)
    scaled = (country_means - means) / stds

    # PCA via SVD
    n = len(scaled)
    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)

    # Take first 2 components
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(n - 1)
    scores_raw = U[:, :2] * S[:2]

    # Varimax rotation
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    # Identify which factor is Traditional/Secular-Rational vs Survival/Self-Expression
    trad_items = ['AUTONOMY', 'F120', 'G006', 'E018']
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items if i in loadings_df.index)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items if i in loadings_df.index)

    trad_col = 'F1' if f1_trad > f2_trad else 'F2'
    surv_col = 'F2' if trad_col == 'F1' else 'F1'

    result = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
        'surv_selfexp': scores_df[surv_col].values
    }).reset_index(drop=True)

    # Fix direction: Sweden should be positive on secular-rational (highly secular)
    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe_trad = result[result['COUNTRY_ALPHA'] == 'SWE']['trad_secrat'].values[0]
        if swe_trad < 0:
            result['trad_secrat'] = -result['trad_secrat']
        swe_surv = result[result['COUNTRY_ALPHA'] == 'SWE']['surv_selfexp'].values[0]
        if swe_surv < 0:
            result['surv_selfexp'] = -result['surv_selfexp']

    return result, loadings_df, country_means


def run_analysis():
    # Step 1: Compute factor scores with pooled waves + GOD_IMP fallback
    print("Computing custom factor scores (pooled waves + A006 fallback)...")
    scores_df, loadings_df, country_means = compute_custom_factor_scores(
        waves_wvs=[2, 3], include_evs=True)
    print(f"Got scores for {len(scores_df)} countries")

    # Normalize to SD=1 to match paper scale
    trad_sd = scores_df['trad_secrat'].std()
    scores_df['trad_secrat_norm'] = scores_df['trad_secrat'] / trad_sd
    print(f"Factor score SD={trad_sd:.3f}")

    print("\nKey country factor scores (normalized):")
    for iso in ['JPN', 'KOR', 'TWN', 'CHN', 'SWE', 'NOR', 'DEU', 'USA', 'IND', 'NGA']:
        row = scores_df[scores_df['COUNTRY_ALPHA'] == iso]
        if len(row) > 0:
            print(f"  {iso}: norm={row['trad_secrat_norm'].values[0]:.3f}")

    # Step 2: Load external data
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
            print(f"  Missing: {iso} (lookup={lookup_iso}), "
                  f"GDP={'ok' if gdp_val else 'MISS'}, Ind={'ok' if ind_val else 'MISS'}")
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
    print("Countries:", sorted(df['economy'].tolist()))

    print("\nConfucian country details:")
    for iso in HIST_CONFUCIAN:
        row = df[df['economy'] == iso]
        if len(row) > 0:
            print(f"  {iso}: trad={row['trad_secrat'].values[0]:.3f}, "
                  f"gdp={row['gdp'].values[0]:.2f}, ind={row['industrial'].values[0]}")

    # Step 4: Regressions
    y = df['trad_secrat']
    results = {}
    results['Model 1'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial']])).fit()
    results['Model 3'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'ex_communist']])).fit()
    results['Model 4'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'catholic']])).fit()
    results['Model 5'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'confucian']])).fit()
    results['Model 6'] = sm.OLS(y, sm.add_constant(
        df[['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian']])).fit()

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
    return score, results


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
    """Standalone scoring function (required by Code Generation Contract)."""
    scores_df, _, _ = compute_custom_factor_scores(waves_wvs=[2, 3], include_evs=True)
    trad_sd = scores_df['trad_secrat'].std()
    scores_df['trad_secrat_norm'] = scores_df['trad_secrat'] / trad_sd
    gdp_dict = load_pwt_gdp()
    industry_dict = load_wdr_industry()
    rows = []
    for _, row in scores_df.iterrows():
        iso = row['COUNTRY_ALPHA']
        if iso in EXCLUDE: continue
        lookup_iso = ISO_MAP.get(iso, iso)
        gdp_val = gdp_dict.get(lookup_iso)
        ind_val = industry_dict.get(iso, industry_dict.get(lookup_iso))
        if gdp_val is None or ind_val is None: continue
        rows.append({'economy': iso, 'trad_secrat': row['trad_secrat_norm'],
                     'gdp': gdp_val, 'industrial': ind_val})
    df = pd.DataFrame(rows)
    if len(df) < 10: return 0
    df['ex_communist'] = df['economy'].isin(EX_COMMUNIST).astype(int)
    df['catholic'] = df['economy'].isin(HIST_CATHOLIC).astype(int)
    df['confucian'] = df['economy'].isin(HIST_CONFUCIAN).astype(int)
    y = df['trad_secrat']
    results = {}
    results['Model 1'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial']])).fit()
    results['Model 3'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'ex_communist']])).fit()
    results['Model 4'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'catholic']])).fit()
    results['Model 5'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'confucian']])).fit()
    results['Model 6'] = sm.OLS(y, sm.add_constant(
        df[['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian']])).fit()
    return score_against_ground_truth_with_results(results)


if __name__ == "__main__":
    run_analysis()
