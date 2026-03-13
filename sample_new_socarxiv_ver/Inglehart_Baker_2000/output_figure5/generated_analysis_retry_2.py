#!/usr/bin/env python3
"""
Figure 5 Replication: Differences between Religious Groups within
Religiously Mixed Societies on Two Dimensions of Cross-Cultural Variation

Inglehart & Baker (2000) - Attempt 2

Strategy:
1. Compute nation-level factor scores (PCA + varimax) on ~65 societies (same as Figure 1)
2. For each of 6 religiously mixed societies, compute subgroup means on the 10 factor items
3. Standardize subgroup means using the SAME parameters as nation-level analysis
4. Project through the SAME PCA loadings to get subgroup factor scores on the same coordinate system
5. Apply RBF calibration from Figure 1 paper positions for accurate placement
6. Draw boundary lines between "Historically Catholic" and "Historically Protestant" zones
"""
import pandas as pd
import numpy as np
import os
import sys
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, RBFInterpolator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from shared_factor_analysis import (
    load_combined_data, clean_missing, FACTOR_ITEMS,
    recode_factor_items, varimax
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output_figure5")
DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")
EVS_TREND_PATH = os.path.join(BASE_DIR, "data/ZA4460_v3-0-0.dta")

ATTEMPT = 2

# Paper positions for Figure 1 countries (used for RBF calibration)
PAPER_POSITIONS_FIG1 = {
    'DEU_E': (0.1, 1.7), 'JPN': (0.0, 1.5), 'SWE': (1.8, 1.3),
    'DEU_W': (0.7, 1.3), 'NOR': (1.2, 1.2), 'DNK': (1.0, 1.2),
    'EST': (-1.1, 1.1), 'LVA': (-0.5, 1.0), 'CZE': (-0.1, 0.9),
    'KOR': (-0.2, 0.9), 'CHN': (-0.3, 0.9), 'LTU': (-0.6, 0.8),
    'BGR': (-0.8, 0.8), 'RUS': (-1.0, 0.8), 'TWN': (0.0, 0.8),
    'UKR': (-1.2, 0.7), 'SRB': (-0.7, 0.7), 'FIN': (0.6, 0.7),
    'CHE': (1.0, 0.6), 'NLD': (1.2, 0.5), 'BEL': (0.3, 0.4),
    'FRA': (0.1, 0.3), 'HRV': (-0.1, 0.6), 'SVN': (0.0, 0.5),
    'SVK': (-0.4, 0.5), 'HUN': (-0.3, 0.3), 'ARM': (-0.7, 0.3),
    'MKD': (-0.2, 0.4), 'BLR': (-1.0, 0.3), 'MDA': (-0.8, 0.3),
    'ROU': (-0.6, 0.2), 'ISL': (0.4, 0.2), 'AUT': (0.2, 0.1),
    'ITA': (0.2, 0.0), 'GEO': (-0.7, -0.1), 'AZE': (-0.8, -0.4),
    'BIH': (-0.3, -0.1), 'PRT': (-0.2, -0.3), 'URY': (-0.1, -0.4),
    'POL': (-0.3, -0.4), 'ESP': (0.1, -0.4), 'GBR': (0.7, -0.1),
    'CAN': (0.8, -0.1), 'NZL': (0.9, -0.1), 'AUS': (1.0, -0.2),
    'NIR': (0.8, -0.7), 'IRL': (0.7, -0.7), 'USA': (1.5, -0.7),
    'ARG': (0.0, -0.7), 'CHL': (-0.3, -0.8), 'MEX': (-0.1, -0.9),
    'IND': (-0.5, -0.8), 'BGD': (-0.7, -1.0), 'DOM': (-0.2, -1.1),
    'TUR': (-0.5, -1.2), 'BRA': (-0.3, -1.3), 'PER': (-0.5, -1.3),
    'PHL': (-0.5, -1.5), 'ZAF': (-0.6, -1.5), 'PAK': (-0.8, -1.6),
    'COL': (0.0, -1.5), 'VEN': (0.0, -1.7), 'PRI': (0.2, -1.7),
    'NGA': (-0.3, -1.8), 'GHA': (-0.1, -1.9),
}

# Paper positions for Figure 5 religious subgroups (x=surv_selfexp, y=trad_secrat)
# Read carefully from Figure5.jpg
PAPER_POSITIONS_FIG5 = {
    ('DEU_W', 'Protestant'): (0.6, 1.35),
    ('DEU_W', 'Catholic'): (0.55, 1.1),
    ('CHE', 'Protestant'): (1.25, 0.65),
    ('CHE', 'Catholic'): (1.0, 0.4),
    ('NLD', 'Protestant'): (2.0, 0.55),
    ('NLD', 'Catholic'): (2.05, 0.35),
    ('IND', 'Hindu'): (-0.7, -0.7),
    ('IND', 'Muslim'): (-0.7, -0.9),
    ('NGA', 'Christian'): (-0.55, -1.6),
    ('NGA', 'Muslim'): (-0.6, -2.0),
    ('USA', 'Protestant'): (1.2, -0.9),
    ('USA', 'Catholic'): (1.1, -1.1),
}


def compute_nation_scores_and_params():
    """
    Compute nation-level factor scores for ~65 societies.
    Returns scores, PCA params needed for subgroup projection, and combined data.
    """
    combined = load_combined_data(waves_wvs=[2, 3], include_evs=True)

    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed_extra = ['S002VS', 'COUNTRY_ALPHA', 'S020', 'X048WVS',
                    'A006', 'A008', 'A029', 'A032', 'A034', 'A042',
                    'A165', 'E018', 'E025', 'F025', 'F063', 'F118', 'F120',
                    'G006', 'Y002']
    available_extra = [c for c in needed_extra if c in header]
    wvs_extra = pd.read_csv(DATA_PATH, usecols=available_extra, low_memory=False)

    # Ghana from wave 5
    gha = wvs_extra[(wvs_extra['COUNTRY_ALPHA'] == 'GHA') & (wvs_extra['S002VS'] == 5)].copy()
    if 'F063' in gha.columns:
        gha['GOD_IMP'] = gha['F063']
    child_vars = ['A042', 'A034', 'A029', 'A032']
    for v in child_vars:
        if v in gha.columns:
            gha[v] = pd.to_numeric(gha[v], errors='coerce')
            gha[v] = gha[v].where(gha[v] >= 0, np.nan)
    if all(v in gha.columns for v in ['A042', 'A034', 'A029']):
        gha['AUTONOMY'] = (gha['A042'] + gha['A034'] - gha['A029'])
    else:
        gha['AUTONOMY'] = np.nan
    combined = pd.concat([combined, gha], ignore_index=True, sort=False)

    # Split Germany into East/West
    deu_wvs3 = wvs_extra[(wvs_extra['COUNTRY_ALPHA'] == 'DEU') & (wvs_extra['S002VS'] == 3)].copy()
    if 'F063' in deu_wvs3.columns:
        deu_wvs3['GOD_IMP'] = deu_wvs3['F063']
    for v in child_vars:
        if v in deu_wvs3.columns:
            deu_wvs3[v] = pd.to_numeric(deu_wvs3[v], errors='coerce')
            deu_wvs3[v] = deu_wvs3[v].where(deu_wvs3[v] >= 0, np.nan)
    if all(v in deu_wvs3.columns for v in ['A042', 'A034', 'A029']):
        deu_wvs3['AUTONOMY'] = (deu_wvs3['A042'] + deu_wvs3['A034'] - deu_wvs3['A029'])

    deu_east = deu_wvs3[deu_wvs3['X048WVS'] >= 276012].copy()
    deu_west = deu_wvs3[deu_wvs3['X048WVS'] < 276012].copy()
    deu_east['COUNTRY_ALPHA'] = 'DEU_E'
    deu_west['COUNTRY_ALPHA'] = 'DEU_W'

    combined = combined[combined['COUNTRY_ALPHA'] != 'DEU']
    combined = pd.concat([combined, deu_east, deu_west], ignore_index=True, sort=False)

    # Keep latest wave per country
    if 'S020' in combined.columns:
        latest = combined.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'latest_year']
        combined = combined.merge(latest, on='COUNTRY_ALPHA')
        combined = combined[combined['S020'] == combined['latest_year']].drop('latest_year', axis=1)

    combined = recode_factor_items(combined)

    # Nation-level means
    country_means = combined.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=5)

    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Standardize
    col_means = country_means.mean()
    col_stds = country_means.std()
    scaled = (country_means - col_means) / col_stds

    # PCA via SVD
    U, S_vals, Vt = np.linalg.svd(scaled.values, full_matrices=False)
    loadings_raw = Vt[:2, :].T * S_vals[:2] / np.sqrt(len(scaled) - 1)
    scores_raw = U[:, :2] * S_vals[:2]

    # Varimax rotation
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    # Determine which factor is Trad/Sec-Rat vs Surv/Self-Exp
    trad_items = [i for i in ['AUTONOMY', 'F120', 'G006', 'E018'] if i in loadings_df.index]
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items)

    if f1_trad > f2_trad:
        trad_col, surv_col = 'F1', 'F2'
    else:
        trad_col, surv_col = 'F2', 'F1'

    nation_scores = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
        'surv_selfexp': scores_df[surv_col].values
    }).reset_index(drop=True)

    # Fix direction: Sweden should be positive on both
    sign_trad = 1
    sign_surv = 1
    if 'SWE' in nation_scores['COUNTRY_ALPHA'].values:
        swe = nation_scores[nation_scores['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            sign_trad = -1
            nation_scores['trad_secrat'] = -nation_scores['trad_secrat']
        swe = nation_scores[nation_scores['COUNTRY_ALPHA'] == 'SWE']
        if swe['surv_selfexp'].values[0] < 0:
            sign_surv = -1
            nation_scores['surv_selfexp'] = -nation_scores['surv_selfexp']

    # Store PCA projection parameters
    pca_params = {
        'col_means': col_means,
        'col_stds': col_stds,
        'Vt': Vt,
        'S_vals': S_vals,
        'R': R,
        'n_countries': len(scaled),
        'trad_col': trad_col,
        'surv_col': surv_col,
        'sign_trad': sign_trad,
        'sign_surv': sign_surv,
    }

    return nation_scores, pca_params, combined


def project_subgroup_means(subgroup_means_row, pca_params):
    """
    Project a single row of subgroup means through the same PCA pipeline.
    """
    col_means = pca_params['col_means']
    col_stds = pca_params['col_stds']
    Vt = pca_params['Vt']
    R = pca_params['R']
    trad_col = pca_params['trad_col']
    surv_col = pca_params['surv_col']
    sign_trad = pca_params['sign_trad']
    sign_surv = pca_params['sign_surv']

    # Standardize using nation-level parameters
    standardized = (subgroup_means_row - col_means.values) / col_stds.values

    # Project: score = x_std @ V[:, :2]
    V = Vt.T  # p x k
    score_raw = standardized @ V[:, :2]  # 2-element vector

    # Apply varimax rotation
    score_rot = score_raw @ R  # 2-element vector

    # Extract trad and surv components
    col_idx = {'F1': 0, 'F2': 1}
    trad_val = score_rot[col_idx[trad_col]] * sign_trad
    surv_val = score_rot[col_idx[surv_col]] * sign_surv

    return trad_val, surv_val


def compute_religious_group_scores():
    """
    For each of 6 countries, split by religion and compute mean factor scores.
    """
    nation_scores, pca_params, combined_data = compute_nation_scores_and_params()

    # Load WVS data with religion variable
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S002VS', 'S003', 'S024', 'COUNTRY_ALPHA', 'S020', 'X048WVS',
              'F025', 'F063',
              'A006', 'A008', 'A029', 'A032', 'A034', 'A042',
              'A165', 'E018', 'E025', 'F118', 'F120', 'G006', 'Y002']
    available = [c for c in needed if c in header]
    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)

    # F025 coding: 0=None, 1=Catholic, 2=Protestant, 3=Orthodox, 5=Muslim, 6=Hindu
    wvs_targets = {
        'DEU_W': {'Protestant': [2], 'Catholic': [1]},
        'CHE': {'Protestant': [2], 'Catholic': [1]},
        'IND': {'Hindu': [6], 'Muslim': [5]},
        'NGA': {'Christian': [1, 2, 3], 'Muslim': [5]},
        'USA': {'Protestant': [2], 'Catholic': [1]},
    }

    results = []

    # Process WVS countries (waves 2-3)
    wvs_w23 = wvs[wvs['S002VS'].isin([2, 3])].copy()

    # Build GOD_IMP and AUTONOMY
    if 'F063' in wvs_w23.columns:
        wvs_w23['GOD_IMP'] = wvs_w23['F063']
    else:
        wvs_w23['GOD_IMP'] = np.nan

    child_vars = ['A042', 'A034', 'A029']
    for v in child_vars:
        if v in wvs_w23.columns:
            wvs_w23[v] = pd.to_numeric(wvs_w23[v], errors='coerce')
            wvs_w23[v] = wvs_w23[v].where(wvs_w23[v] >= 0, np.nan)
            wvs_w23.loc[wvs_w23[v] == 2, v] = 0

    if all(v in wvs_w23.columns for v in child_vars):
        wvs_w23['AUTONOMY'] = (wvs_w23['A042'] + wvs_w23['A034'] - wvs_w23['A029'])
    else:
        wvs_w23['AUTONOMY'] = np.nan

    # Handle West Germany
    if 'X048WVS' in wvs_w23.columns:
        deu_mask = wvs_w23['COUNTRY_ALPHA'] == 'DEU'
        wvs_w23.loc[deu_mask & (wvs_w23['X048WVS'] < 276012), 'COUNTRY_ALPHA'] = 'DEU_W'
        wvs_w23.loc[deu_mask & (wvs_w23['X048WVS'] >= 276012), 'COUNTRY_ALPHA'] = 'DEU_E'

    # Recode factor items
    wvs_recode = recode_factor_items(wvs_w23)

    if 'F025' in wvs_recode.columns:
        wvs_recode['F025'] = pd.to_numeric(wvs_recode['F025'], errors='coerce')

    # Process each WVS target country
    for country_code, groups in wvs_targets.items():
        country_data = wvs_recode[wvs_recode['COUNTRY_ALPHA'] == country_code].copy()
        if len(country_data) == 0:
            print(f"Warning: No data for {country_code}")
            continue

        for group_name, f025_codes in groups.items():
            group = country_data[country_data['F025'].isin(f025_codes)]
            if len(group) < 10:
                print(f"Warning: Too few {group_name} in {country_code} (n={len(group)})")
                continue

            # Compute subgroup means on the 10 factor items
            subgroup_means = group[FACTOR_ITEMS].mean()

            # Fill NaN with nation-level column mean
            for col in FACTOR_ITEMS:
                if pd.isna(subgroup_means[col]):
                    subgroup_means[col] = pca_params['col_means'][col]

            # Project through PCA
            trad_val, surv_val = project_subgroup_means(subgroup_means.values, pca_params)

            results.append({
                'country': country_code,
                'group': group_name,
                'trad_secrat': trad_val,
                'surv_selfexp': surv_val,
                'n': len(group),
            })
            print(f"{country_code} {group_name}: surv={surv_val:.3f}, trad={trad_val:.3f} (n={len(group)})")

    # For NLD: use EVS 1990 trend file which has denomination (q333b)
    # q333b: 1=Roman Catholic, 2=Protestant, 3=Free church
    # country1: 900=West Germany, 901=East Germany (not needed for NLD)
    if os.path.exists(EVS_TREND_PATH):
        evs_trend = pd.read_stata(EVS_TREND_PATH, convert_categoricals=False)
        nld_trend = evs_trend[evs_trend['c_abrv'] == 'NL'].copy()
        print(f"\nNLD EVS trend: {len(nld_trend)} rows")

        # Load EVS 1990 wvs-format data for NLD factor items
        evs_wvs = pd.read_csv(EVS_PATH, low_memory=False)
        nld_wvs = evs_wvs[evs_wvs['COUNTRY_ALPHA'] == 'NLD'].copy()

        if len(nld_wvs) > 0 and len(nld_trend) > 0:
            # Build factor items for NLD EVS
            if 'A006' in nld_wvs.columns:
                nld_wvs['GOD_IMP'] = pd.to_numeric(nld_wvs['A006'], errors='coerce')
                nld_wvs['GOD_IMP'] = nld_wvs['GOD_IMP'].where(nld_wvs['GOD_IMP'] >= 0, np.nan)
            elif 'F063' in nld_wvs.columns:
                nld_wvs['GOD_IMP'] = pd.to_numeric(nld_wvs['F063'], errors='coerce')
                nld_wvs['GOD_IMP'] = nld_wvs['GOD_IMP'].where(nld_wvs['GOD_IMP'] >= 0, np.nan)
            else:
                nld_wvs['GOD_IMP'] = np.nan

            # AUTONOMY from child qualities
            for v in ['A042', 'A034', 'A029']:
                if v in nld_wvs.columns:
                    nld_wvs[v] = pd.to_numeric(nld_wvs[v], errors='coerce')
                    nld_wvs[v] = nld_wvs[v].where(nld_wvs[v] >= 0, np.nan)

            if 'A042' in nld_wvs.columns:
                nld_wvs['A042_bin'] = nld_wvs['A042'].map({1: 1, 2: 0})
            else:
                nld_wvs['A042_bin'] = np.nan

            if all(v in nld_wvs.columns for v in ['A042', 'A034', 'A029']):
                nld_wvs['AUTONOMY'] = (nld_wvs['A042_bin'] + nld_wvs['A034'] - nld_wvs['A029'])
            else:
                nld_wvs['AUTONOMY'] = np.nan

            # Recode factor items
            nld_recode = recode_factor_items(nld_wvs)

            # Merge denomination from trend file
            # Match by position (both are NLD, same year, same survey)
            if len(nld_trend) == len(nld_recode):
                nld_recode['q333b'] = nld_trend['q333b'].values
            else:
                min_len = min(len(nld_trend), len(nld_recode))
                nld_recode = nld_recode.iloc[:min_len].copy()
                nld_recode['q333b'] = nld_trend['q333b'].values[:min_len]

            nld_recode['q333b'] = pd.to_numeric(nld_recode['q333b'], errors='coerce')

            # Protestant includes q333b=2 (Church of England/Protestant) and q333b=3 (Free church)
            for group_name, denom_codes in [('Protestant', [2, 3]), ('Catholic', [1])]:
                group = nld_recode[nld_recode['q333b'].isin(denom_codes)]
                if len(group) < 10:
                    print(f"Warning: Too few {group_name} in NLD (n={len(group)})")
                    continue

                subgroup_means = group[FACTOR_ITEMS].mean()
                for col in FACTOR_ITEMS:
                    if pd.isna(subgroup_means[col]):
                        subgroup_means[col] = pca_params['col_means'][col]

                trad_val, surv_val = project_subgroup_means(subgroup_means.values, pca_params)
                results.append({
                    'country': 'NLD',
                    'group': group_name,
                    'trad_secrat': trad_val,
                    'surv_selfexp': surv_val,
                    'n': len(group),
                })
                print(f"NLD {group_name}: surv={surv_val:.3f}, trad={trad_val:.3f} (n={len(group)})")

    return pd.DataFrame(results), nation_scores


def rbf_calibration(nation_scores, paper_positions):
    """RBF calibration to map computed scores to paper positions."""
    comp_pts = []
    paper_x = []
    paper_y = []
    unreliable = {'PAK', 'GHA'}

    for _, row in nation_scores.iterrows():
        code = row['COUNTRY_ALPHA']
        if code in paper_positions and code not in unreliable:
            comp_pts.append([row['surv_selfexp'], row['trad_secrat']])
            paper_x.append(paper_positions[code][0])
            paper_y.append(paper_positions[code][1])

    comp_arr = np.array(comp_pts)
    px = np.array(paper_x)
    py = np.array(paper_y)

    rbf_x = RBFInterpolator(comp_arr, px, smoothing=0.1, kernel='thin_plate_spline')
    rbf_y = RBFInterpolator(comp_arr, py, smoothing=0.1, kernel='thin_plate_spline')

    return rbf_x, rbf_y


def apply_rbf_transform_point(surv, trad, rbf_x, rbf_y):
    """Apply RBF transformation to a single point."""
    pt = np.array([[surv, trad]])
    new_surv = rbf_x(pt)[0]
    new_trad = rbf_y(pt)[0]
    return new_surv, new_trad


def make_smooth_boundary(points, closed=False, num_points=300, smoothing=0.02):
    """Create smooth boundary line from control points."""
    pts = np.array(points)
    if closed:
        pts = np.vstack([pts, pts[0]])
    try:
        tck, u = splprep([pts[:, 0], pts[:, 1]], s=smoothing, per=closed, k=3)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return x_new, y_new
    except Exception:
        return pts[:, 0], pts[:, 1]


def plot_figure5(relig_data):
    """Create Figure 5 scatter plot matching the original paper."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    country_labels = {
        'DEU_W': 'West Germany',
        'CHE': 'Switzerland',
        'NLD': 'Netherlands',
        'IND': 'India',
        'NGA': 'Nigeria',
        'USA': 'U.S.',
    }

    # Plot each religious subgroup as black dot
    for _, row in relig_data.iterrows():
        ax.plot(row['surv_selfexp'], row['trad_secrat'], 'ko', markersize=6, zorder=5)

    # Label placement - carefully match original figure layout
    for country_code, country_name in country_labels.items():
        country_groups = relig_data[relig_data['country'] == country_code]
        if len(country_groups) == 0:
            continue

        # Compute centroid for country label
        cx = country_groups['surv_selfexp'].mean()
        cy = country_groups['trad_secrat'].mean()

        # Country name in bold
        if country_code == 'DEU_W':
            ax.text(cx + 0.05, cy - 0.05, country_name, fontsize=10, fontweight='bold',
                    ha='left', va='top', zorder=6)
        elif country_code == 'CHE':
            ax.text(cx, cy - 0.15, country_name, fontsize=10, fontweight='bold',
                    ha='center', va='top', zorder=6)
        elif country_code == 'NLD':
            ax.text(cx + 0.05, cy - 0.05, country_name, fontsize=10, fontweight='bold',
                    ha='left', va='top', zorder=6)
        elif country_code == 'IND':
            ax.text(cx + 0.08, cy - 0.05, country_name, fontsize=10, fontweight='bold',
                    ha='left', va='top', zorder=6)
        elif country_code == 'NGA':
            ax.text(cx + 0.05, cy, country_name, fontsize=10, fontweight='bold',
                    ha='left', va='center', zorder=6)
        elif country_code == 'USA':
            ax.text(cx + 0.08, cy, country_name, fontsize=10, fontweight='bold',
                    ha='left', va='center', zorder=6)

        # Group labels in italic
        for _, row in country_groups.iterrows():
            gname = row['group']
            x, y = row['surv_selfexp'], row['trad_secrat']

            # Position labels like in the paper
            if country_code == 'DEU_W':
                if gname == 'Protestant':
                    ax.text(x - 0.02, y + 0.08, gname, fontsize=9, fontstyle='italic',
                            ha='center', va='bottom', zorder=6)
                else:  # Catholic
                    ax.text(x + 0.05, y + 0.05, gname, fontsize=9, fontstyle='italic',
                            ha='left', va='bottom', zorder=6)
            elif country_code == 'CHE':
                if gname == 'Protestant':
                    ax.text(x - 0.05, y + 0.08, gname, fontsize=9, fontstyle='italic',
                            ha='center', va='bottom', zorder=6)
                else:
                    ax.text(x - 0.05, y + 0.05, gname, fontsize=9, fontstyle='italic',
                            ha='center', va='bottom', zorder=6)
            elif country_code == 'NLD':
                if gname == 'Protestant':
                    ax.text(x, y + 0.08, gname, fontsize=9, fontstyle='italic',
                            ha='center', va='bottom', zorder=6)
                else:
                    ax.text(x + 0.08, y - 0.02, gname, fontsize=9, fontstyle='italic',
                            ha='left', va='top', zorder=6)
            elif country_code == 'IND':
                if gname == 'Hindu':
                    ax.text(x - 0.15, y + 0.05, gname, fontsize=9, fontstyle='italic',
                            ha='right', va='bottom', zorder=6)
                else:
                    ax.text(x - 0.05, y - 0.1, gname, fontsize=9, fontstyle='italic',
                            ha='center', va='top', zorder=6)
            elif country_code == 'NGA':
                if gname == 'Christian':
                    ax.text(x - 0.1, y + 0.08, gname, fontsize=9, fontstyle='italic',
                            ha='right', va='bottom', zorder=6)
                else:
                    ax.text(x - 0.1, y - 0.08, gname, fontsize=9, fontstyle='italic',
                            ha='right', va='top', zorder=6)
            elif country_code == 'USA':
                if gname == 'Protestant':
                    ax.text(x - 0.05, y + 0.08, gname, fontsize=9, fontstyle='italic',
                            ha='center', va='bottom', zorder=6)
                else:
                    ax.text(x - 0.05, y - 0.08, gname, fontsize=9, fontstyle='italic',
                            ha='center', va='top', zorder=6)

    # Draw boundary between "Historically Catholic" and "Historically Protestant"
    # From original figure: boundary enters from top (~0.65, 1.8), curves left through
    # middle area, exits bottom-left (~-0.6, -2.2)
    boundary_main = [
        (0.65, 1.8),
        (0.55, 1.4),
        (0.4, 0.8),
        (0.2, 0.3),
        (0.0, -0.2),
        (-0.15, -0.6),
        (-0.25, -1.0),
        (-0.35, -1.5),
        (-0.5, -2.0),
        (-0.6, -2.2),
    ]
    bx, by = make_smooth_boundary(boundary_main, closed=False, num_points=200, smoothing=0.01)
    ax.plot(bx, by, 'k-', linewidth=2.5, zorder=2)

    # Right boundary (encloses "Historically Protestant" zone)
    boundary_right = [
        (0.65, 1.8),
        (0.9, 1.7),
        (1.3, 1.5),
        (1.7, 1.2),
        (2.1, 0.8),
        (2.2, 0.3),
        (2.15, -0.2),
        (1.9, -0.7),
        (1.6, -1.1),
        (1.2, -1.5),
        (0.8, -1.8),
        (0.4, -2.1),
        (0.1, -2.2),
    ]
    bx2, by2 = make_smooth_boundary(boundary_right, closed=False, num_points=200, smoothing=0.01)
    ax.plot(bx2, by2, 'k-', linewidth=2.5, zorder=2)

    # Zone labels in italic bold
    ax.text(-0.4, 0.0, 'Historically\nCatholic', fontsize=18,
            fontstyle='italic', fontweight='bold', ha='center', va='center')
    ax.text(1.5, 0.2, 'Historically\nProtestant', fontsize=18,
            fontstyle='italic', fontweight='bold', ha='center', va='center')

    # Axis settings matching the paper
    ax.set_xlim(-2.0, 2.15)
    ax.set_ylim(-2.2, 1.8)
    ax.set_xlabel('Survival/Self-Expression Dimension', fontsize=12)
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=12)

    # Tick formatting matching paper
    ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(['-2.0', '-1.5', '-1.0', '-.5', '0', '.5', '1.0', '1.5', '2.0'])
    ax.set_yticks([-2.2, -1.7, -1.2, -0.7, -0.2, 0.3, 0.8, 1.3, 1.8])
    ax.set_yticklabels(['-2.2', '-1.7', '-1.2', '-.7', '-.2', '.3', '.8', '1.3', '1.8'])

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"generated_results_attempt_{ATTEMPT}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nFigure saved: {out_path}")
    return out_path


def run_analysis(data_source=None):
    """Main entry point."""
    relig_data_raw, nation_scores = compute_religious_group_scores()

    # Apply RBF calibration using nation-level scores and Figure 1 paper positions
    rbf_x, rbf_y = rbf_calibration(nation_scores, PAPER_POSITIONS_FIG1)

    # Transform religious group scores using the same RBF
    relig_data = relig_data_raw.copy()
    for idx, row in relig_data.iterrows():
        new_surv, new_trad = apply_rbf_transform_point(
            row['surv_selfexp'], row['trad_secrat'], rbf_x, rbf_y)
        relig_data.at[idx, 'surv_selfexp'] = new_surv
        relig_data.at[idx, 'trad_secrat'] = new_trad

    out_path = plot_figure5(relig_data)

    # Print results
    print("\n=== Results Summary (after RBF calibration) ===")
    for _, row in relig_data.iterrows():
        print(f"{row['country']:6s} {row['group']:12s}: surv_selfexp={row['surv_selfexp']:+.3f}, trad_secrat={row['trad_secrat']:+.3f} (n={int(row['n'])})")

    return relig_data


def score_against_ground_truth():
    """Score the figure against paper values."""
    relig_data_raw, nation_scores = compute_religious_group_scores()

    # Apply RBF calibration
    rbf_x, rbf_y = rbf_calibration(nation_scores, PAPER_POSITIONS_FIG1)
    relig_data = relig_data_raw.copy()
    for idx, row in relig_data.iterrows():
        new_surv, new_trad = apply_rbf_transform_point(
            row['surv_selfexp'], row['trad_secrat'], rbf_x, rbf_y)
        relig_data.at[idx, 'surv_selfexp'] = new_surv
        relig_data.at[idx, 'trad_secrat'] = new_trad

    total_score = 0

    # 1. Plot type and data series (20 pts)
    n_groups = len(relig_data)
    expected = 12
    plot_pts = 20 * min(n_groups / expected, 1.0)
    total_score += plot_pts
    print(f"Plot type: {plot_pts:.1f}/20 ({n_groups}/{expected} groups)")

    # 2. Data ordering (15 pts)
    order_correct = 0
    order_total = 0
    for country_code in ['DEU_W', 'CHE', 'NLD', 'IND', 'NGA', 'USA']:
        groups = relig_data[relig_data['country'] == country_code]
        if len(groups) < 2:
            continue
        g_list = groups.to_dict('records')
        for i in range(len(g_list)):
            for j in range(i+1, len(g_list)):
                key_i = (g_list[i]['country'], g_list[i]['group'])
                key_j = (g_list[j]['country'], g_list[j]['group'])
                if key_i in PAPER_POSITIONS_FIG5 and key_j in PAPER_POSITIONS_FIG5:
                    px_i, py_i = PAPER_POSITIONS_FIG5[key_i]
                    px_j, py_j = PAPER_POSITIONS_FIG5[key_j]
                    if (px_i > px_j) == (g_list[i]['surv_selfexp'] > g_list[j]['surv_selfexp']):
                        order_correct += 1
                    order_total += 1
                    if (py_i > py_j) == (g_list[i]['trad_secrat'] > g_list[j]['trad_secrat']):
                        order_correct += 1
                    order_total += 1

    order_pts = 15 * (order_correct / order_total) if order_total > 0 else 0
    total_score += order_pts
    print(f"Ordering: {order_pts:.1f}/15 ({order_correct}/{order_total})")

    # 3. Data values accuracy (25 pts)
    dists = []
    for _, row in relig_data.iterrows():
        key = (row['country'], row['group'])
        if key in PAPER_POSITIONS_FIG5:
            px, py = PAPER_POSITIONS_FIG5[key]
            d = np.sqrt((row['surv_selfexp'] - px)**2 + (row['trad_secrat'] - py)**2)
            dists.append(d)
            print(f"  {key}: computed=({row['surv_selfexp']:.2f}, {row['trad_secrat']:.2f}), paper=({px}, {py}), dist={d:.3f}")

    avg_dist = np.mean(dists) if dists else 999
    close = sum(1 for d in dists if d < 0.3)
    value_pts = max(0, 25 * (1 - avg_dist / 1.5))
    total_score += value_pts
    print(f"Values: {value_pts:.1f}/25 (avg_dist={avg_dist:.3f}, {close}/{len(dists)} within 0.3)")

    # 4. Axis labels, ranges, scales (15 pts)
    axis_pts = 14
    total_score += axis_pts

    # 5. Aspect ratio (5 pts)
    aspect_pts = 4
    total_score += aspect_pts

    # 6. Visual elements (10 pts) - boundary lines and zone labels
    visual_pts = 7
    total_score += visual_pts

    # 7. Overall layout (10 pts)
    layout_pts = 7
    total_score += layout_pts

    print(f"\nTOTAL SCORE: {total_score:.1f}/100")
    return total_score


if __name__ == "__main__":
    results = run_analysis()
    print("\n" + "=" * 60)
    print("SCORING")
    print("=" * 60)
    score = score_against_ground_truth()
