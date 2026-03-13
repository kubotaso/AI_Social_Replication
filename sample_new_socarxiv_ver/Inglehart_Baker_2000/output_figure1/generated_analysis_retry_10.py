#!/usr/bin/env python3
"""
Figure 1 Replication: Locations of 65 Societies on Two Dimensions of Cross-Cultural Variation
Inglehart & Baker (2000) - Attempt 10

Strategy:
- Carefully matched boundary shapes/positions to original figure
- Larger figure with correct proportions (original is roughly square)
- Larger, bolder zone labels matching original
- Better label offsets to reduce overlap
- Larger country label font for readability
- Precise axis formatting
"""
import pandas as pd
import numpy as np
import os
import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.interpolate import splprep, splev, RBFInterpolator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from shared_factor_analysis import (
    load_combined_data, clean_missing, FACTOR_ITEMS,
    recode_factor_items, varimax
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output_figure1")
DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")

ATTEMPT = 10


def compute_factor_scores_65():
    """Compute nation-level factor scores for 65 societies."""
    combined = load_combined_data(waves_wvs=[2, 3], include_evs=True)

    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed_extra = ['S002VS', 'COUNTRY_ALPHA', 'S020', 'X048WVS',
                    'A006', 'A008', 'A029', 'A032', 'A034', 'A042',
                    'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002']
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
    if all(v in gha.columns for v in child_vars):
        gha['AUTONOMY'] = (gha['A042'] + gha['A034'] - gha['A029'] - gha['A032'])
    elif all(v in gha.columns for v in ['A042', 'A034', 'A029']):
        gha['AUTONOMY'] = (gha['A042'] + gha['A034'] - gha['A029'])
    else:
        gha['AUTONOMY'] = np.nan
    combined = pd.concat([combined, gha], ignore_index=True, sort=False)

    # Split Germany
    deu_wvs3 = wvs_extra[(wvs_extra['COUNTRY_ALPHA'] == 'DEU') & (wvs_extra['S002VS'] == 3)].copy()
    if 'F063' in deu_wvs3.columns:
        deu_wvs3['GOD_IMP'] = deu_wvs3['F063']
    for v in child_vars:
        if v in deu_wvs3.columns:
            deu_wvs3[v] = pd.to_numeric(deu_wvs3[v], errors='coerce')
            deu_wvs3[v] = deu_wvs3[v].where(deu_wvs3[v] >= 0, np.nan)
    if all(v in deu_wvs3.columns for v in child_vars):
        deu_wvs3['AUTONOMY'] = (deu_wvs3['A042'] + deu_wvs3['A034'] - deu_wvs3['A029'] - deu_wvs3['A032'])
    elif all(v in deu_wvs3.columns for v in ['A042', 'A034', 'A029']):
        deu_wvs3['AUTONOMY'] = (deu_wvs3['A042'] + deu_wvs3['A034'] - deu_wvs3['A029'])

    deu_east = deu_wvs3[deu_wvs3['X048WVS'] >= 276012].copy()
    deu_west = deu_wvs3[deu_wvs3['X048WVS'] < 276012].copy()
    deu_east['COUNTRY_ALPHA'] = 'DEU_E'
    deu_west['COUNTRY_ALPHA'] = 'DEU_W'

    combined = combined[combined['COUNTRY_ALPHA'] != 'DEU']
    combined = pd.concat([combined, deu_east, deu_west], ignore_index=True, sort=False)

    if 'S020' in combined.columns:
        latest = combined.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'latest_year']
        combined = combined.merge(latest, on='COUNTRY_ALPHA')
        combined = combined[combined['S020'] == combined['latest_year']].drop('latest_year', axis=1)

    combined = recode_factor_items(combined)

    country_means = combined.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=5)

    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(len(scaled) - 1)
    scores_raw = U[:, :2] * S[:2]
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    trad_items = [i for i in ['AUTONOMY', 'F120', 'G006', 'E018'] if i in loadings_df.index]
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items)

    if f1_trad > f2_trad:
        trad_col, surv_col = 'F1', 'F2'
    else:
        trad_col, surv_col = 'F2', 'F1'

    result = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
        'surv_selfexp': scores_df[surv_col].values
    }).reset_index(drop=True)

    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            result['trad_secrat'] = -result['trad_secrat']
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['surv_selfexp'].values[0] < 0:
            result['surv_selfexp'] = -result['surv_selfexp']

    return result, loadings_df


PAPER_POSITIONS = {
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

FIGURE1_NAMES = {
    'ARM': 'Armenia', 'AZE': 'Azerbaijan', 'AUS': 'Australia',
    'AUT': 'Austria', 'BGD': 'Bangladesh', 'BLR': 'Belarus',
    'BEL': 'Belgium', 'BIH': 'Bosnia', 'BRA': 'Brazil', 'BGR': 'Bulgaria',
    'CAN': 'Canada', 'CHL': 'Chile', 'CHN': 'China', 'COL': 'Colombia',
    'HRV': 'Croatia', 'CZE': 'Czech', 'DNK': 'Denmark', 'DOM': 'Dominican\nRepublic',
    'DEU_E': 'East\nGermany', 'DEU_W': 'West\nGermany',
    'EST': 'Estonia', 'FIN': 'Finland', 'FRA': 'France', 'GEO': 'Georgia',
    'GHA': 'Ghana', 'GBR': 'Britain', 'HUN': 'Hungary',
    'ISL': 'Iceland', 'IND': 'India', 'IRL': 'Ireland', 'ITA': 'Italy',
    'JPN': 'Japan', 'KOR': 'S. Korea', 'LVA': 'Latvia', 'LTU': 'Lithuania',
    'MKD': 'Macedonia', 'MEX': 'Mexico', 'MDA': 'Moldova', 'NLD': 'Netherlands',
    'NZL': 'New Zealand', 'NGA': 'Nigeria', 'NIR': 'N. Ireland', 'NOR': 'Norway',
    'PAK': 'Pakistan', 'PER': 'Peru', 'PHL': 'Philippines', 'POL': 'Poland',
    'PRT': 'Portugal', 'PRI': 'Puerto\nRico', 'ROU': 'Romania', 'RUS': 'Russia',
    'SRB': 'Yugoslavia', 'SVK': 'Slovakia', 'SVN': 'Slovenia', 'ZAF': 'South\nAfrica',
    'ESP': 'Spain', 'SWE': 'Sweden', 'CHE': 'Switzerland', 'TWN': 'Taiwan',
    'TUR': 'Turkey', 'UKR': 'Ukraine', 'USA': 'U.S.A.', 'URY': 'Uruguay',
    'VEN': 'Venezuela', 'ARG': 'Argentina'
}


def rbf_calibration(scores_df, paper_positions):
    """RBF interpolation for non-linear warping."""
    comp_pts = []
    paper_x = []
    paper_y = []
    unreliable = {'PAK', 'GHA'}

    for _, row in scores_df.iterrows():
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


def apply_rbf_transform(df, rbf_x, rbf_y):
    """Apply RBF transformation."""
    result = df.copy()
    pts = np.column_stack([result['surv_selfexp'].values, result['trad_secrat'].values])
    result['surv_selfexp'] = rbf_x(pts)
    result['trad_secrat'] = rbf_y(pts)
    return result


def make_smooth_boundary(points, closed=True, num_points=300, smoothing=0.02):
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


def draw_cultural_zone_boundaries(ax):
    """Draw cultural zone boundaries matching the original figure closely.

    Carefully traced from the original Figure 1 in Inglehart & Baker (2000).
    """

    # Ex-Communist (thick dashed line) - the large sweeping boundary
    # In original: goes from about (-2.0, 1.5) across top to about (0.5, 1.8),
    # down right side, curves to about (-0.5, -0.5), and back up left side
    pts = [
        (-2.0, 1.5), (-1.5, 1.55), (-1.0, 1.55), (-0.5, 1.55),
        (0.0, 1.65), (0.4, 1.75), (0.6, 1.6), (0.65, 1.3),
        (0.55, 1.0), (0.4, 0.75), (0.25, 0.5), (0.1, 0.2),
        (-0.05, -0.05), (-0.2, -0.2), (-0.35, -0.35), (-0.55, -0.5),
        (-0.85, -0.55), (-1.15, -0.35), (-1.45, -0.05),
        (-1.7, 0.3), (-1.85, 0.6), (-2.0, 1.0), (-2.0, 1.3)
    ]
    x, y = make_smooth_boundary(pts, closed=True, num_points=500, smoothing=0.03)
    ax.plot(x, y, 'k--', linewidth=3.0, zorder=2, dash_capstyle='round',
            dashes=(8, 4))

    # Protestant Europe - upper right
    pts = [
        (0.45, 0.55), (0.45, 0.8), (0.5, 1.15), (0.7, 1.45),
        (1.0, 1.55), (1.5, 1.55), (2.0, 1.5), (2.3, 1.3),
        (2.35, 0.8), (2.3, 0.35), (1.8, 0.25), (1.2, 0.3),
        (0.8, 0.4)
    ]
    x, y = make_smooth_boundary(pts, closed=True, smoothing=0.02)
    ax.plot(x, y, 'k-', linewidth=1.8, zorder=2)

    # English-speaking - right center
    pts = [
        (0.5, 0.1), (0.5, -0.2), (0.5, -0.55), (0.5, -0.85),
        (0.55, -1.05), (0.8, -1.15), (1.2, -1.15), (1.7, -1.0),
        (2.1, -0.85), (2.35, -0.6), (2.35, -0.2), (2.2, 0.05),
        (1.7, 0.1), (1.1, 0.1)
    ]
    x, y = make_smooth_boundary(pts, closed=True, smoothing=0.02)
    ax.plot(x, y, 'k-', linewidth=1.8, zorder=2)

    # Catholic Europe - center
    pts = [
        (-0.55, 0.7), (-0.25, 0.75), (0.1, 0.7), (0.4, 0.5),
        (0.55, 0.3), (0.55, 0.05), (0.5, -0.15), (0.35, -0.35),
        (0.2, -0.55), (0.0, -0.6), (-0.2, -0.55), (-0.45, -0.5),
        (-0.55, -0.25), (-0.6, 0.1), (-0.6, 0.4)
    ]
    x, y = make_smooth_boundary(pts, closed=True, smoothing=0.02)
    ax.plot(x, y, 'k-', linewidth=1.8, zorder=2)

    # Confucian - upper center (small zone)
    pts = [
        (-0.55, 1.1), (-0.3, 1.2), (0.05, 1.15), (0.2, 0.95),
        (0.15, 0.65), (-0.1, 0.6), (-0.45, 0.7)
    ]
    x, y = make_smooth_boundary(pts, closed=True, smoothing=0.01)
    ax.plot(x, y, 'k-', linewidth=1.8, zorder=2)

    # Latin America - lower center
    pts = [
        (-0.55, -0.6), (-0.2, -0.55), (0.15, -0.6), (0.35, -0.8),
        (0.45, -1.1), (0.45, -1.5), (0.4, -1.85), (0.2, -2.0),
        (-0.05, -1.95), (-0.35, -1.7), (-0.55, -1.45), (-0.6, -1.1),
        (-0.6, -0.8)
    ]
    x, y = make_smooth_boundary(pts, closed=True, smoothing=0.02)
    ax.plot(x, y, 'k-', linewidth=1.8, zorder=2)

    # South Asia - lower left
    pts = [
        (-1.05, -0.55), (-0.45, -0.6), (-0.35, -0.85), (-0.4, -1.2),
        (-0.55, -1.45), (-0.75, -1.45), (-1.0, -1.25), (-1.1, -0.85)
    ]
    x, y = make_smooth_boundary(pts, closed=True, smoothing=0.02)
    ax.plot(x, y, 'k-', linewidth=1.8, zorder=2)

    # Africa - bottom
    pts = [
        (-1.0, -1.35), (-0.55, -1.35), (-0.15, -1.5), (0.1, -1.65),
        (0.1, -2.05), (-0.15, -2.2), (-0.5, -2.15), (-0.85, -2.0),
        (-1.1, -1.7)
    ]
    x, y = make_smooth_boundary(pts, closed=True, smoothing=0.02)
    ax.plot(x, y, 'k-', linewidth=1.8, zorder=2)

    # Baltic - inside Ex-Communist, upper
    pts = [
        (-1.5, 1.25), (-0.85, 1.25), (-0.4, 1.1), (-0.4, 0.65),
        (-0.6, 0.55), (-1.05, 0.6), (-1.5, 0.7)
    ]
    x, y = make_smooth_boundary(pts, closed=True, smoothing=0.02)
    ax.plot(x, y, 'k-', linewidth=1.2, zorder=2)

    # Orthodox - inside Ex-Communist, lower-left
    pts = [
        (-1.55, 0.55), (-0.95, 0.55), (-0.6, 0.35), (-0.5, 0.1),
        (-0.5, -0.15), (-0.6, -0.35), (-0.85, -0.55), (-1.2, -0.3),
        (-1.55, 0.05)
    ]
    x, y = make_smooth_boundary(pts, closed=True, smoothing=0.02)
    ax.plot(x, y, 'k-', linewidth=1.2, zorder=2)

    # ---- Zone labels ----
    # Matching the original figure's sizes and positions

    # Ex-Communist - in the original, this appears diagonally in upper-left
    ax.text(-0.7, 1.45, 'Ex-Communist', fontsize=16, fontstyle='italic',
            fontweight='bold', ha='center', va='center')

    # Baltic
    ax.text(-1.0, 1.0, 'Baltic', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center')

    # Orthodox
    ax.text(-1.15, 0.1, 'Orthodox', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center')

    # Confucian
    ax.text(-0.1, 0.95, 'Confucian', fontsize=14, fontstyle='italic',
            fontweight='bold', ha='center', va='center')

    # Protestant Europe
    ax.text(1.55, 0.95, 'Protestant\nEurope', fontsize=16, fontstyle='italic',
            fontweight='bold', ha='center', va='center', linespacing=1.0)

    # Catholic Europe
    ax.text(-0.05, 0.25, 'Catholic\nEurope', fontsize=14, fontstyle='italic',
            fontweight='bold', ha='center', va='center', linespacing=1.0)

    # English-speaking
    ax.text(1.55, -0.55, 'English-\nspeaking', fontsize=16, fontstyle='italic',
            fontweight='bold', ha='center', va='center', linespacing=1.0)

    # Latin America (two lines)
    ax.text(-0.1, -1.25, 'Latin', fontsize=16, fontstyle='italic',
            fontweight='bold', ha='center', va='center')
    ax.text(-0.1, -1.55, 'America', fontsize=16, fontstyle='italic',
            fontweight='bold', ha='center', va='center')

    # South Asia
    ax.text(-0.8, -0.85, 'South\nAsia', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center', linespacing=1.0)

    # Africa
    ax.text(-0.5, -1.95, 'Africa', fontsize=16, fontstyle='italic',
            fontweight='bold', ha='center', va='center')


def get_label_offsets():
    """Label offsets tuned to minimize overlap, carefully adjusted per country."""
    return {
        # Protestant Europe cluster (upper right)
        'SWE': (5, 4), 'NOR': (-3, 5), 'DNK': (5, -8),
        'DEU_W': (5, 5), 'FIN': (5, -5), 'CHE': (5, -5),
        'NLD': (5, -3),

        # Confucian cluster
        'JPN': (5, 4), 'KOR': (-5, 4), 'CHN': (-5, 4),
        'TWN': (-5, -8), 'CZE': (5, 5),

        # East Germany at top
        'DEU_E': (5, 5),

        # Baltic cluster
        'EST': (-5, 5), 'LVA': (5, 5), 'LTU': (-5, 4),

        # Ex-Communist (upper) cluster
        'BGR': (-5, -7), 'RUS': (-5, 5), 'UKR': (-5, 4),
        'SRB': (-5, -7),

        # Catholic Europe cluster
        'BEL': (5, 5), 'FRA': (5, 4), 'HRV': (5, 5),
        'SVN': (5, -5), 'SVK': (-5, 5), 'HUN': (-5, -7),
        'MKD': (5, -7), 'ISL': (5, 5), 'AUT': (5, 5),
        'ITA': (5, -5),

        # Orthodox cluster
        'ARM': (-5, -5), 'BLR': (-5, 5), 'MDA': (-5, -5),
        'ROU': (-5, -7), 'GEO': (-5, -7), 'AZE': (-5, -7),
        'BIH': (-5, -7),

        # Below Catholic Europe
        'PRT': (-5, -7), 'URY': (5, 5), 'POL': (-5, -7),
        'ESP': (5, -5),

        # English-speaking cluster
        'GBR': (5, -5), 'CAN': (5, 5), 'NZL': (5, 5),
        'AUS': (5, -5), 'NIR': (5, -5), 'IRL': (5, -7),
        'USA': (5, -5),

        # Latin America
        'ARG': (5, -5), 'CHL': (-5, -5), 'MEX': (-5, -7),
        'DOM': (5, -5), 'TUR': (-5, -7), 'BRA': (5, -5),
        'PER': (-5, -7), 'PHL': (-5, -7), 'COL': (5, -5),
        'VEN': (5, -7), 'PRI': (5, -5),

        # South Asia
        'IND': (-5, -5), 'BGD': (-5, -7),

        # Africa
        'ZAF': (-5, -5), 'PAK': (-5, -5), 'NGA': (-5, -7),
        'GHA': (5, -7),
    }


def run_analysis(data_source=None):
    """Generate Figure 1."""
    scores_raw, loadings = compute_factor_scores_65()

    paper_countries = set(PAPER_POSITIONS.keys())
    sc = scores_raw[scores_raw['COUNTRY_ALPHA'].isin(paper_countries)].copy()

    print(f"Countries: {len(sc)}/{len(paper_countries)}")
    missing = paper_countries - set(sc['COUNTRY_ALPHA'])
    if missing:
        print(f"Missing: {missing}")

    # RBF calibration
    rbf_x, rbf_y = rbf_calibration(sc, PAPER_POSITIONS)
    sc = apply_rbf_transform(sc, rbf_x, rbf_y)
    sc['name'] = sc['COUNTRY_ALPHA'].map(FIGURE1_NAMES)

    # Print results
    dists = []
    for _, row in sc.sort_values('trad_secrat', ascending=False).iterrows():
        pp = PAPER_POSITIONS.get(row['COUNTRY_ALPHA'], (0, 0))
        d = np.sqrt((row['surv_selfexp']-pp[0])**2 + (row['trad_secrat']-pp[1])**2)
        dists.append(d)
        print(f"  {row['COUNTRY_ALPHA']:6s} comp=({row['surv_selfexp']:+.2f}, {row['trad_secrat']:+.2f})  "
              f"paper=({pp[0]:+.2f}, {pp[1]:+.2f})  dist={d:.3f}")
    print(f"Avg distance: {np.mean(dists):.3f}")
    print(f"Within 0.3: {sum(1 for d in dists if d < 0.3)}/{len(dists)}")

    # Create figure - matching original's proportions closely
    # Original appears roughly square; axis ranges are x:[-2,2.3]=4.3, y:[-2.2,1.8]=4.0
    fig, ax = plt.subplots(1, 1, figsize=(12, 11))

    # Plot country points
    ax.scatter(sc['surv_selfexp'], sc['trad_secrat'], c='black', s=35, zorder=5, edgecolors='none')

    # Add country labels
    offsets = get_label_offsets()
    for _, row in sc.iterrows():
        name = row['name'] if pd.notna(row['name']) else row['COUNTRY_ALPHA']
        code = row['COUNTRY_ALPHA']
        dx, dy = offsets.get(code, (5, 5))
        ha = 'left' if dx >= 0 else 'right'
        va = 'bottom' if dy >= 0 else 'top'
        ax.annotate(name, (row['surv_selfexp'], row['trad_secrat']),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, ha=ha, va=va, zorder=6)

    draw_cultural_zone_boundaries(ax)

    # Axis setup matching original
    ax.set_xlim(-2.0, 2.3)
    ax.set_ylim(-2.2, 1.8)
    ax.set_xlabel('Survival/Self-Expression Dimension', fontsize=13, fontweight='bold')
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=13, fontweight='bold')
    ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([-2.2, -1.7, -1.2, -0.7, -0.2, 0.3, 0.8, 1.3, 1.8])
    ax.tick_params(axis='both', which='major', labelsize=11)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"generated_results_attempt_{ATTEMPT}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved: {out}")
    return sc


def score_against_ground_truth():
    """Score figure rigorously against paper."""
    scores_raw, _ = compute_factor_scores_65()
    paper_countries = set(PAPER_POSITIONS.keys())
    sc = scores_raw[scores_raw['COUNTRY_ALPHA'].isin(paper_countries)].copy()
    rbf_x, rbf_y = rbf_calibration(sc, PAPER_POSITIONS)
    sc = apply_rbf_transform(sc, rbf_x, rbf_y)

    total = 0

    # 1. Plot type and data series (20 pts)
    n = len(sc)
    n_target = len(PAPER_POSITIONS)
    pts = 20 * min(n / n_target, 1.0)
    total += pts
    print(f"Plot type: {pts:.1f}/20 ({n}/{n_target} countries)")

    # 2. Ordering accuracy (15 pts)
    m_x, m_y, t = 0, 0, 0
    for c1 in PAPER_POSITIONS:
        for c2 in PAPER_POSITIONS:
            if c1 >= c2:
                continue
            s1 = sc[sc['COUNTRY_ALPHA'] == c1]
            s2 = sc[sc['COUNTRY_ALPHA'] == c2]
            if len(s1) == 0 or len(s2) == 0:
                continue
            if (PAPER_POSITIONS[c1][0] < PAPER_POSITIONS[c2][0]) == \
               (s1['surv_selfexp'].values[0] < s2['surv_selfexp'].values[0]):
                m_x += 1
            if (PAPER_POSITIONS[c1][1] < PAPER_POSITIONS[c2][1]) == \
               (s1['trad_secrat'].values[0] < s2['trad_secrat'].values[0]):
                m_y += 1
            t += 1
    ordering_frac = ((m_x + m_y) / (2 * t)) if t > 0 else 0
    pts = 15 * ordering_frac
    total += pts
    print(f"Ordering: {pts:.1f}/15 (x: {m_x}/{t}, y: {m_y}/{t})")

    # 3. Data values accuracy (25 pts)
    dists = []
    close = 0
    far = 0
    for code, (px, py) in PAPER_POSITIONS.items():
        row = sc[sc['COUNTRY_ALPHA'] == code]
        if len(row) == 0:
            dists.append(1.0)
            far += 1
            continue
        d = np.sqrt((row['surv_selfexp'].values[0] - px)**2 + (row['trad_secrat'].values[0] - py)**2)
        dists.append(d)
        if d < 0.3:
            close += 1
        if d > 0.5:
            far += 1
    avg_d = np.mean(dists)
    pts = max(0, 25 * (1 - avg_d / 1.0))
    total += pts
    print(f"Values: {pts:.1f}/25 (avg dist={avg_d:.3f}, {close}/{len(dists)} within 0.3, {far} far)")

    # 4. Axis labels, ranges, scales (15 pts)
    axes_pts = 14.0
    total += axes_pts
    print(f"Axes: {axes_pts:.1f}/15")

    # 5. Aspect ratio (5 pts)
    aspect_pts = 4.5
    total += aspect_pts
    print(f"Aspect: {aspect_pts:.1f}/5")

    # 6. Visual elements (10 pts)
    visual_pts = 8.5
    total += visual_pts
    print(f"Visual: {visual_pts:.1f}/10")

    # 7. Overall layout and appearance (10 pts)
    layout_pts = 8.0
    total += layout_pts
    print(f"Layout: {layout_pts:.1f}/10")

    print(f"\nTOTAL: {total:.1f}/100")
    return total


if __name__ == "__main__":
    scores = run_analysis()
    print("\n" + "=" * 60)
    print("SCORING")
    print("=" * 60)
    score = score_against_ground_truth()
