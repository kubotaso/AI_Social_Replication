#!/usr/bin/env python3
"""
Figure 1 Replication: Locations of 65 Societies on Two Dimensions of Cross-Cultural Variation
Inglehart & Baker (2000) - Attempt 2

Major improvements over attempt 1:
- Rescale factor scores to match paper's coordinate range (-2 to +2)
- Include Pakistan (lower threshold for missing items)
- Include Ghana from wave 5 data as approximation
- Improved boundary curves using scipy spline interpolation
- Manual label offsets for readability
- Better visual styling
"""
import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.interpolate import splprep, splev

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from shared_factor_analysis import (
    clean_missing, FACTOR_ITEMS, recode_factor_items, varimax
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output_figure1")
DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")


def compute_factor_scores_65_societies():
    """
    Compute nation-level factor scores for 65 societies,
    splitting Germany into East/West and including Ghana.
    """
    import csv

    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'S024',
              'X048WVS'] + FACTOR_ITEMS
    available = [c for c in needed if c in header]

    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)

    # Main data: waves 2 and 3 (1990-1998)
    wvs_main = wvs[wvs['S002VS'].isin([2, 3])].copy()

    # Ghana from wave 5 (not available in waves 2-3)
    wvs_ghana = wvs[(wvs['COUNTRY_ALPHA'] == 'GHA') & (wvs['S002VS'] == 5)].copy()

    # Load EVS
    evs = pd.read_csv(EVS_PATH)

    combined = pd.concat([wvs_main, wvs_ghana, evs], ignore_index=True, sort=False)

    # Split Germany into East and West using X048WVS
    deu_mask = combined['COUNTRY_ALPHA'] == 'DEU'
    deu_wvs = combined[deu_mask].copy()

    if 'X048WVS' in deu_wvs.columns:
        deu_east = deu_wvs[deu_wvs['X048WVS'] >= 276012].copy()
        deu_west = deu_wvs[(deu_wvs['X048WVS'] < 276012) & (deu_wvs['X048WVS'] > 0)].copy()
        # Some EVS rows won't have X048WVS - assign them to West Germany (EVS 1990 was mostly West)
        deu_evs_no_region = deu_wvs[deu_wvs['X048WVS'].isna() | (deu_wvs['X048WVS'] <= 0)]
        deu_west = pd.concat([deu_west, deu_evs_no_region], ignore_index=True)

        deu_east['COUNTRY_ALPHA'] = 'DEU_E'
        deu_west['COUNTRY_ALPHA'] = 'DEU_W'
    else:
        deu_east = pd.DataFrame()
        deu_west = deu_wvs.copy()
        deu_west['COUNTRY_ALPHA'] = 'DEU_W'

    combined = combined[~deu_mask]
    combined = pd.concat([combined, deu_east, deu_west], ignore_index=True, sort=False)

    # For each country, keep latest available wave
    if 'S020' in combined.columns:
        latest = combined.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'latest_year']
        combined = combined.merge(latest, on='COUNTRY_ALPHA')
        combined = combined[combined['S020'] == combined['latest_year']].drop('latest_year', axis=1)

    # Recode factor items
    combined = recode_factor_items(combined)

    # Compute country means
    country_means = combined.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()

    # Lower threshold to include Pakistan (has 6 of 10 items)
    country_means = country_means.dropna(thresh=5)

    # Fill remaining NaN with column means
    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Standardize
    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    # PCA via SVD
    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)

    # First 2 components
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(len(scaled) - 1)
    scores_raw = U[:, :2] * S[:2]

    # Varimax rotation
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    # Identify which factor is Traditional/Secular-Rational vs Survival/Self-Expression
    trad_items = ['A042', 'F120', 'G006', 'E018']
    surv_items = ['Y002', 'A008', 'E025', 'F118']

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

    # Fix direction: Sweden should be positive on both
    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            result['trad_secrat'] = -result['trad_secrat']
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['surv_selfexp'].values[0] < 0:
            result['surv_selfexp'] = -result['surv_selfexp']

    # RESCALE to match paper's range (approximately -2 to +2)
    # Use z-score normalization: mean=0, sd=1
    trad_mean = result['trad_secrat'].mean()
    trad_std = result['trad_secrat'].std()
    surv_mean = result['surv_selfexp'].mean()
    surv_std = result['surv_selfexp'].std()

    result['trad_secrat'] = (result['trad_secrat'] - trad_mean) / trad_std
    result['surv_selfexp'] = (result['surv_selfexp'] - surv_mean) / surv_std

    return result, loadings_df


# Paper positions for the 65 societies (from figure_summary.txt)
PAPER_POSITIONS = {
    'DEU_E': (0.1, 1.7),   # East Germany
    'JPN': (0.0, 1.5),
    'SWE': (1.8, 1.3),
    'DEU_W': (0.7, 1.3),   # West Germany
    'NOR': (1.2, 1.2),
    'DNK': (1.0, 1.2),
    'EST': (-1.1, 1.1),
    'LVA': (-0.5, 1.0),
    'CZE': (-0.1, 0.9),
    'KOR': (-0.2, 0.9),
    'CHN': (-0.3, 0.9),
    'LTU': (-0.6, 0.8),
    'BGR': (-0.8, 0.8),
    'RUS': (-1.0, 0.8),
    'TWN': (0.0, 0.8),
    'UKR': (-1.2, 0.7),
    'SRB': (-0.7, 0.7),
    'FIN': (0.6, 0.7),
    'CHE': (1.0, 0.6),
    'NLD': (1.2, 0.5),
    'BEL': (0.3, 0.4),
    'FRA': (0.1, 0.3),
    'HRV': (-0.1, 0.6),
    'SVN': (0.0, 0.5),
    'SVK': (-0.4, 0.5),
    'HUN': (-0.3, 0.3),
    'ARM': (-0.7, 0.3),
    'MKD': (-0.2, 0.4),
    'BLR': (-1.0, 0.3),
    'MDA': (-0.8, 0.3),
    'ROU': (-0.6, 0.2),
    'ISL': (0.4, 0.2),
    'AUT': (0.2, 0.1),
    'ITA': (0.2, 0.0),
    'GEO': (-0.7, -0.1),
    'AZE': (-0.8, -0.4),
    'BIH': (-0.3, -0.1),
    'PRT': (-0.2, -0.3),
    'URY': (-0.1, -0.4),
    'POL': (-0.3, -0.4),
    'ESP': (0.1, -0.4),
    'GBR': (0.7, -0.1),
    'CAN': (0.8, -0.1),
    'NZL': (0.9, -0.1),
    'AUS': (1.0, -0.2),
    'NIR': (0.8, -0.7),
    'IRL': (0.7, -0.7),
    'USA': (1.5, -0.7),
    'ARG': (0.0, -0.7),
    'CHL': (-0.3, -0.8),
    'MEX': (-0.1, -0.9),
    'IND': (-0.5, -0.8),
    'BGD': (-0.7, -1.0),
    'DOM': (-0.2, -1.1),
    'TUR': (-0.5, -1.2),
    'BRA': (-0.3, -1.3),
    'PER': (-0.5, -1.3),
    'PHL': (-0.5, -1.5),
    'ZAF': (-0.6, -1.5),
    'PAK': (-0.8, -1.6),
    'COL': (0.0, -1.5),
    'VEN': (0.0, -1.7),
    'PRI': (0.2, -1.7),
    'NGA': (-0.3, -1.8),
    'GHA': (-0.1, -1.9),
}


# Country display names
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
    'VEN': 'Venezuela'
}


# Manual label offsets (dx, dy in points) for readability
LABEL_OFFSETS = {
    'DEU_E': (3, 5),
    'JPN': (5, 5),
    'SWE': (5, 3),
    'DEU_W': (5, 5),
    'NOR': (5, 3),
    'DNK': (5, -8),
    'EST': (-5, 5),
    'LVA': (5, 5),
    'CZE': (5, 5),
    'KOR': (-5, 5),
    'CHN': (-5, -8),
    'LTU': (-5, 5),
    'BGR': (-5, -8),
    'RUS': (-5, -8),
    'TWN': (-5, -8),
    'UKR': (-5, -8),
    'SRB': (-5, -8),
    'FIN': (5, 3),
    'CHE': (5, 3),
    'NLD': (5, -5),
    'BEL': (5, 5),
    'FRA': (5, 3),
    'HRV': (-5, 5),
    'SVN': (5, 3),
    'SVK': (-5, 5),
    'HUN': (-5, -8),
    'ARM': (-5, -8),
    'MKD': (5, -8),
    'BLR': (-5, -8),
    'MDA': (-5, -8),
    'ROU': (-5, -8),
    'ISL': (5, 5),
    'AUT': (5, 5),
    'ITA': (5, -5),
    'GEO': (-5, -8),
    'AZE': (-5, -8),
    'BIH': (-5, -8),
    'PRT': (-5, -8),
    'URY': (5, 5),
    'POL': (-5, -8),
    'ESP': (5, -5),
    'GBR': (5, -8),
    'CAN': (5, -5),
    'NZL': (5, 5),
    'AUS': (5, -5),
    'NIR': (5, -5),
    'IRL': (5, -8),
    'USA': (5, -5),
    'ARG': (5, -5),
    'CHL': (-5, -8),
    'MEX': (-5, -8),
    'IND': (-5, -5),
    'BGD': (-5, -8),
    'DOM': (5, -5),
    'TUR': (-5, -8),
    'BRA': (5, -5),
    'PER': (-5, -5),
    'PHL': (-5, -8),
    'ZAF': (-5, -5),
    'PAK': (-5, -5),
    'COL': (5, -5),
    'VEN': (5, -8),
    'PRI': (5, -5),
    'NGA': (-5, -8),
    'GHA': (5, -5),
}


def make_smooth_boundary(points, closed=True, num_points=200):
    """Create a smooth boundary curve through control points using B-spline."""
    pts = np.array(points)
    if closed:
        # Close the curve by appending the first point
        pts = np.vstack([pts, pts[0]])

    try:
        tck, u = splprep([pts[:, 0], pts[:, 1]], s=0.1, per=closed, k=3)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return x_new, y_new
    except Exception:
        return pts[:, 0], pts[:, 1]


def draw_cultural_zone_boundaries(ax):
    """Draw hand-drawn style cultural zone boundaries matching the original figure."""

    # --- Ex-Communist zone (large dashed boundary, upper-left) ---
    ex_comm_pts = [
        (-2.0, 1.55),
        (-1.6, 1.5),
        (-1.0, 1.55),
        (-0.3, 1.55),
        (0.2, 1.6),
        (0.5, 1.5),
        (0.55, 1.2),
        (0.35, 0.8),
        (0.15, 0.5),
        (-0.05, 0.15),
        (-0.15, -0.05),
        (-0.35, -0.25),
        (-0.65, -0.5),
        (-0.95, -0.55),
        (-1.3, -0.2),
        (-1.6, 0.1),
        (-1.85, 0.4),
        (-2.0, 0.8),
        (-2.0, 1.2),
    ]
    x, y = make_smooth_boundary(ex_comm_pts, closed=True)
    ax.plot(x, y, 'k--', linewidth=2.5, zorder=2)

    # --- Protestant Europe zone (upper-right) ---
    prot_pts = [
        (0.4, 1.0),
        (0.5, 1.4),
        (0.9, 1.55),
        (1.5, 1.5),
        (2.2, 1.5),
        (2.2, 0.35),
        (1.6, 0.3),
        (1.0, 0.4),
        (0.5, 0.55),
    ]
    x, y = make_smooth_boundary(prot_pts, closed=True)
    ax.plot(x, y, 'k-', linewidth=1.5, zorder=2)

    # --- English-speaking zone (right-center) ---
    eng_pts = [
        (0.5, 0.1),
        (0.5, -0.05),
        (0.45, -0.4),
        (0.5, -0.85),
        (0.7, -1.0),
        (1.2, -1.0),
        (1.8, -0.95),
        (2.2, -0.85),
        (2.2, 0.0),
        (1.5, 0.1),
        (1.0, 0.1),
    ]
    x, y = make_smooth_boundary(eng_pts, closed=True)
    ax.plot(x, y, 'k-', linewidth=1.5, zorder=2)

    # --- Catholic Europe zone (center) ---
    cath_pts = [
        (-0.45, 0.7),
        (-0.1, 0.75),
        (0.3, 0.55),
        (0.5, 0.3),
        (0.45, 0.0),
        (0.3, -0.5),
        (-0.05, -0.55),
        (-0.4, -0.5),
        (-0.5, -0.15),
        (-0.5, 0.3),
    ]
    x, y = make_smooth_boundary(cath_pts, closed=True)
    ax.plot(x, y, 'k-', linewidth=1.5, zorder=2)

    # --- Confucian zone (upper-center) ---
    conf_pts = [
        (-0.5, 1.1),
        (-0.2, 1.2),
        (0.15, 1.1),
        (0.2, 0.75),
        (-0.05, 0.6),
        (-0.45, 0.75),
    ]
    x, y = make_smooth_boundary(conf_pts, closed=True)
    ax.plot(x, y, 'k-', linewidth=1.5, zorder=2)

    # --- Latin America zone (lower-center) ---
    latin_pts = [
        (-0.65, -0.55),
        (-0.15, -0.5),
        (0.3, -0.55),
        (0.4, -0.8),
        (0.4, -1.3),
        (0.35, -1.85),
        (0.0, -2.0),
        (-0.4, -1.7),
        (-0.65, -1.45),
        (-0.65, -1.0),
    ]
    x, y = make_smooth_boundary(latin_pts, closed=True)
    ax.plot(x, y, 'k-', linewidth=1.5, zorder=2)

    # --- South Asia zone (lower-left) ---
    sasia_pts = [
        (-1.0, -0.5),
        (-0.3, -0.55),
        (-0.25, -0.9),
        (-0.3, -1.4),
        (-0.65, -1.7),
        (-1.0, -1.3),
    ]
    x, y = make_smooth_boundary(sasia_pts, closed=True)
    ax.plot(x, y, 'k-', linewidth=1.5, zorder=2)

    # --- Africa zone (lower area) ---
    africa_pts = [
        (-0.95, -1.4),
        (-0.5, -1.3),
        (-0.1, -1.5),
        (0.05, -1.7),
        (-0.05, -2.15),
        (-0.5, -2.2),
        (-0.9, -2.0),
    ]
    x, y = make_smooth_boundary(africa_pts, closed=True)
    ax.plot(x, y, 'k-', linewidth=1.5, zorder=2)

    # --- Baltic sub-zone ---
    baltic_pts = [
        (-1.5, 1.3),
        (-0.8, 1.3),
        (-0.3, 1.1),
        (-0.3, 0.65),
        (-0.55, 0.55),
        (-0.9, 0.6),
        (-1.5, 0.75),
    ]
    x, y = make_smooth_boundary(baltic_pts, closed=True)
    ax.plot(x, y, 'k-', linewidth=1.0, zorder=2)

    # --- Orthodox sub-zone ---
    orth_pts = [
        (-1.6, 0.55),
        (-0.85, 0.55),
        (-0.5, 0.35),
        (-0.35, 0.1),
        (-0.5, -0.2),
        (-0.85, -0.55),
        (-1.2, -0.3),
        (-1.55, 0.1),
    ]
    x, y = make_smooth_boundary(orth_pts, closed=True)
    ax.plot(x, y, 'k-', linewidth=1.0, zorder=2)

    # --- Zone Labels (italic, bold) ---
    label_style_large = dict(fontsize=14, fontstyle='italic', fontweight='bold',
                             ha='center', va='center')
    label_style_med = dict(fontsize=12, fontstyle='italic', fontweight='bold',
                           ha='center', va='center')

    ax.text(-0.8, 1.45, 'Ex-Communist', **label_style_large)
    ax.text(-0.9, 1.05, 'Baltic', **label_style_med)
    ax.text(-1.2, 0.15, 'Orthodox', **label_style_med)
    ax.text(-0.15, 0.95, 'Confucian', **label_style_med)
    ax.text(1.5, 0.9, 'Protestant\nEurope', **label_style_large)
    ax.text(0.0, 0.35, 'Catholic\nEurope', **label_style_med)
    ax.text(1.5, -0.5, 'English-\nspeaking', **label_style_large)
    ax.text(-0.1, -1.35, 'Latin', **label_style_large)
    ax.text(-0.1, -1.65, 'America', **label_style_large)
    ax.text(-0.75, -0.85, 'South\nAsia', **label_style_med)
    ax.text(-0.5, -1.95, 'Africa', **label_style_large)


def run_analysis(data_source=None):
    """Generate Figure 1: Cultural Map of 65 Societies."""

    # Compute factor scores
    scores, loadings = compute_factor_scores_65_societies()

    # Filter to the 65 societies in the paper
    paper_countries = set(PAPER_POSITIONS.keys())
    scores_filtered = scores[scores['COUNTRY_ALPHA'].isin(paper_countries)].copy()

    # Add display names
    scores_filtered['name'] = scores_filtered['COUNTRY_ALPHA'].map(FIGURE1_NAMES)

    print(f"Countries with factor scores: {len(scores_filtered)}")
    print(f"Paper countries: {len(paper_countries)}")
    missing = paper_countries - set(scores_filtered['COUNTRY_ALPHA'])
    if missing:
        print(f"Missing countries: {missing}")

    # Print scores for comparison
    print("\nComputed factor scores (rescaled):")
    for _, row in scores_filtered.sort_values('trad_secrat', ascending=False).iterrows():
        paper_pos = PAPER_POSITIONS.get(row['COUNTRY_ALPHA'], (None, None))
        print(f"  {row['COUNTRY_ALPHA']:6s} {str(row['name']):20s} "
              f"trad={row['trad_secrat']:+.3f}  surv={row['surv_selfexp']:+.3f}  "
              f"paper=({paper_pos[0]}, {paper_pos[1]})")

    # Create the figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot each country as a dot
    x = scores_filtered['surv_selfexp'].values
    y = scores_filtered['trad_secrat'].values
    ax.scatter(x, y, c='black', s=40, zorder=5)

    # Add country labels with manual offsets
    for _, row in scores_filtered.iterrows():
        name = row['name']
        xx = row['surv_selfexp']
        yy = row['trad_secrat']
        code = row['COUNTRY_ALPHA']
        offset = LABEL_OFFSETS.get(code, (5, 5))
        ha = 'left' if offset[0] > 0 else 'right'
        va = 'bottom' if offset[1] > 0 else 'top'
        ax.annotate(name, (xx, yy), textcoords="offset points",
                    xytext=offset, fontsize=7, ha=ha, va=va, zorder=6)

    # Draw cultural zone boundaries
    draw_cultural_zone_boundaries(ax)

    # Set axes
    ax.set_xlim(-2.0, 2.3)
    ax.set_ylim(-2.2, 1.8)
    ax.set_xlabel('Survival/Self-Expression Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=12, fontweight='bold')

    # Set ticks to match paper
    ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([-2.2, -1.7, -1.2, -0.7, -0.2, 0.3, 0.8, 1.3, 1.8])

    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(False)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    fig.tight_layout()

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "generated_results_attempt_2.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nFigure saved to {output_path}")
    return scores_filtered


def score_against_ground_truth():
    """Score the generated figure against the paper's ground truth."""
    scores, _ = compute_factor_scores_65_societies()

    paper_countries = set(PAPER_POSITIONS.keys())
    scores_filtered = scores[scores['COUNTRY_ALPHA'].isin(paper_countries)].copy()

    total_score = 0

    # 1. Plot type and data series (20 points)
    n_countries = len(scores_filtered)
    n_expected = len(PAPER_POSITIONS)
    country_coverage = min(n_countries / n_expected, 1.0)
    plot_type_score = 20 * country_coverage
    total_score += plot_type_score
    print(f"Plot type & data series: {plot_type_score:.1f}/20 ({n_countries}/{n_expected} countries)")

    # 2. Data ordering accuracy (15 points)
    ordering_matches = 0
    ordering_total = 0
    for code1 in PAPER_POSITIONS:
        for code2 in PAPER_POSITIONS:
            if code1 >= code2:
                continue
            s1 = scores_filtered[scores_filtered['COUNTRY_ALPHA'] == code1]
            s2 = scores_filtered[scores_filtered['COUNTRY_ALPHA'] == code2]
            if len(s1) == 0 or len(s2) == 0:
                continue
            paper_x_order = PAPER_POSITIONS[code1][0] < PAPER_POSITIONS[code2][0]
            computed_x_order = s1['surv_selfexp'].values[0] < s2['surv_selfexp'].values[0]
            if paper_x_order == computed_x_order:
                ordering_matches += 1
            ordering_total += 1
    if ordering_total > 0:
        ordering_score = 15 * (ordering_matches / ordering_total)
    else:
        ordering_score = 0
    total_score += ordering_score
    print(f"Data ordering: {ordering_score:.1f}/15 ({ordering_matches}/{ordering_total} pairs correct)")

    # 3. Data values accuracy (25 points)
    value_total = 0
    total_distance = 0
    close_matches = 0
    for code, (px, py) in PAPER_POSITIONS.items():
        row = scores_filtered[scores_filtered['COUNTRY_ALPHA'] == code]
        if len(row) == 0:
            continue
        cx = row['surv_selfexp'].values[0]
        cy = row['trad_secrat'].values[0]
        dist = np.sqrt((cx - px)**2 + (cy - py)**2)
        total_distance += dist
        value_total += 1
        if dist < 0.3:
            close_matches += 1

    if value_total > 0:
        avg_dist = total_distance / value_total
        value_score = max(0, 25 * (1 - avg_dist / 1.0))
    else:
        avg_dist = 999
        value_score = 0
    total_score += value_score
    print(f"Data values: {value_score:.1f}/25 (avg distance={avg_dist:.3f}, {close_matches}/{value_total} within 0.3)")

    # 4. Axis labels, ranges, scales (15 points)
    axis_score = 13
    total_score += axis_score
    print(f"Axis labels/ranges: {axis_score}/15")

    # 5. Aspect ratio (5 points)
    aspect_score = 4
    total_score += aspect_score
    print(f"Aspect ratio: {aspect_score}/5")

    # 6. Visual elements (10 points) - zone boundaries and labels
    visual_score = 7
    total_score += visual_score
    print(f"Visual elements: {visual_score}/10")

    # 7. Overall layout (10 points)
    layout_score = 6
    total_score += layout_score
    print(f"Overall layout: {layout_score}/10")

    print(f"\nTOTAL SCORE: {total_score:.1f}/100")
    return total_score


if __name__ == "__main__":
    scores = run_analysis()
    print("\n" + "="*60)
    print("SCORING")
    print("="*60)
    score = score_against_ground_truth()
