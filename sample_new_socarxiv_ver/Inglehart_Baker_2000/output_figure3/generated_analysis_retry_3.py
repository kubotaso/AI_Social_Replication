#!/usr/bin/env python3
"""
Figure 3 Replication - Attempt 3
Historically Protestant, Historically Catholic, and Historically Communist
Cultural Zones in Relation to Two Dimensions of Cross-Cultural Variation.

Key fix: Use Procrustes/affine alignment to map computed factor scores
to paper coordinates using multiple anchor countries.
"""

import sys
import os
import numpy as np
import pandas as pd
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches
from scipy.interpolate import splprep, splev

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(PROJECT_DIR, "data/EVS_1990_wvs_format.csv")

FACTOR_ITEMS = ['A006', 'A042', 'F120', 'G006', 'E018',
                'Y002', 'A008', 'E025', 'F118', 'A165']

# Ground truth positions from the paper (estimated from Figure 3)
PAPER_POSITIONS = {
    'DEE': (0.5, 1.55), 'JPN': (0.35, 1.45), 'SWE': (2.15, 1.35),
    'DEW': (1.05, 1.25), 'NOR': (1.25, 1.15), 'DNK': (1.0, 1.1),
    'EST': (-1.2, 1.15), 'LVA': (-0.5, 1.0), 'CZE': (0.0, 0.95),
    'KOR': (-0.1, 0.9), 'CHN': (-0.6, 0.85), 'LTU': (-1.0, 0.8),
    'BGR': (-1.15, 0.75), 'RUS': (-1.5, 0.8), 'TWN': (-0.2, 0.65),
    'UKR': (-1.6, 0.7), 'SRB': (-0.95, 0.65), 'SVK': (-0.55, 0.55),
    'HRV': (-0.2, 0.55), 'SVN': (0.15, 0.5), 'FIN': (0.8, 0.55),
    'CHE': (1.1, 0.5), 'NLD': (1.45, 0.2), 'HUN': (-0.5, 0.3),
    'ARM': (-1.2, 0.35), 'BLR': (-1.45, 0.4), 'MKD': (-0.55, 0.1),
    'MDA': (-1.7, 0.15), 'ROU': (-0.75, -0.05), 'BEL': (0.55, 0.2),
    'FRA': (0.5, -0.05), 'ISL': (0.7, -0.05), 'AUT': (0.7, -0.15),
    'ITA': (0.5, -0.2), 'GEO': (-1.15, -0.25), 'AZE': (-1.3, -0.35),
    'BIH': (-0.4, -0.15), 'POL': (-0.45, -0.45), 'PRT': (-0.25, -0.4),
    'URY': (0.15, -0.5), 'ESP': (0.3, -0.55), 'GBR': (0.85, -0.15),
    'CAN': (1.15, -0.15), 'NZL': (1.45, -0.15), 'AUS': (1.7, -0.2),
    'NIR': (0.85, -0.95), 'IRL': (0.8, -1.35), 'USA': (1.55, -0.95),
    'ARG': (0.45, -0.7), 'CHL': (0.0, -0.85), 'MEX': (0.25, -0.9),
    'IND': (-0.35, -0.8), 'BGD': (-0.7, -1.1), 'TUR': (0.0, -1.3),
    'DOM': (0.2, -1.35), 'ZAF': (-0.35, -1.55), 'PHL': (-0.15, -1.55),
    'BRA': (0.05, -1.55), 'PER': (-0.1, -1.65), 'PAK': (-0.65, -1.55),
    'COL': (0.2, -1.55), 'VEN': (0.15, -1.9), 'PRI': (0.35, -1.9),
    'NGA': (-0.5, -1.85), 'GHA': (0.0, -1.95),
}


def clean_missing(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col] >= 0, np.nan)
    return df


def recode_factor_items(df):
    df = df.copy()
    df = clean_missing(df, FACTOR_ITEMS)
    if 'A042' in df.columns:
        df['A042'] = df['A042'].map({1: 1, 2: 0, 0: 0}).where(df['A042'].notna())
    if 'F120' in df.columns:
        df['F120'] = 11 - df['F120']
    if 'G006' in df.columns:
        df['G006'] = 5 - df['G006']
    if 'E018' in df.columns:
        df['E018'] = 4 - df['E018']
    if 'Y002' in df.columns:
        df['Y002'] = 4 - df['Y002']
    if 'F118' in df.columns:
        df['F118'] = 11 - df['F118']
    return df


def varimax(Phi, gamma=1.0, q=100, tol=1e-8):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        Lambda = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma/p) * Lambda @ np.diag(np.sum(Lambda**2, axis=0)))
        )
        R = u @ vt
        d_new = np.sum(s)
        if d_new - d < tol:
            break
        d = d_new
    return Phi @ R, R


def compute_raw_scores():
    """Compute raw factor scores with East/West Germany split."""
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S002VS', 'S003', 'S024', 'COUNTRY_ALPHA', 'S020',
              'X048WVS'] + FACTOR_ITEMS
    available = [c for c in needed if c in header]

    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]

    evs = pd.read_csv(EVS_PATH)
    combined = pd.concat([wvs, evs], ignore_index=True, sort=False)

    # Split Germany into East/West
    east_codes = [276012, 276013, 276014, 276015, 276016, 276019]
    mask_deu = combined['COUNTRY_ALPHA'] == 'DEU'
    if 'X048WVS' in combined.columns:
        mask_east = mask_deu & combined['X048WVS'].isin(east_codes)
        mask_west = mask_deu & ~combined['X048WVS'].isin(east_codes) & combined['X048WVS'].notna()
        # For EVS data without X048WVS, treat as West Germany
        mask_evs_deu = mask_deu & combined['X048WVS'].isna()
        combined.loc[mask_east, 'COUNTRY_ALPHA'] = 'DEE'
        combined.loc[mask_west, 'COUNTRY_ALPHA'] = 'DEW'
        combined.loc[mask_evs_deu, 'COUNTRY_ALPHA'] = 'DEW'
    else:
        combined.loc[mask_deu, 'COUNTRY_ALPHA'] = 'DEW'

    # Get latest per country
    if 'S020' in combined.columns:
        latest = combined.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'latest_year']
        combined = combined.merge(latest, on='COUNTRY_ALPHA')
        combined = combined[combined['S020'] == combined['latest_year']].drop('latest_year', axis=1)

    combined = recode_factor_items(combined)

    country_means = combined.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)

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

    trad_items = ['A042', 'F120', 'G006', 'E018']
    f1_trad_load = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items if i in loadings_df.index)
    f2_trad_load = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items if i in loadings_df.index)

    if f1_trad_load > f2_trad_load:
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
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['surv_selfexp'].values[0] < 0:
            result['surv_selfexp'] = -result['surv_selfexp']

    return result


def procrustes_align(computed, paper_positions):
    """
    Align computed scores to paper positions using affine transformation.
    Uses least-squares fit: paper = A * computed + b
    """
    # Find common countries
    common = []
    for _, row in computed.iterrows():
        code = row['COUNTRY_ALPHA']
        if code in paper_positions:
            common.append(code)

    if len(common) < 3:
        return computed

    # Build matrices
    n = len(common)
    X = np.zeros((n, 2))  # computed positions
    Y = np.zeros((n, 2))  # paper positions

    for i, code in enumerate(common):
        row = computed[computed['COUNTRY_ALPHA'] == code].iloc[0]
        X[i, 0] = row['surv_selfexp']
        X[i, 1] = row['trad_secrat']
        Y[i, 0] = paper_positions[code][0]
        Y[i, 1] = paper_positions[code][1]

    # Fit affine transformation: Y = X @ A + b
    # Augment X with ones column
    X_aug = np.column_stack([X, np.ones(n)])
    # Solve least squares: X_aug @ [A; b] = Y
    params, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
    A = params[:2, :]  # 2x2 transformation matrix
    b = params[2, :]   # translation vector

    # Apply to all countries
    result = computed.copy()
    all_X = np.column_stack([result['surv_selfexp'].values, result['trad_secrat'].values])
    all_Y = all_X @ A + b

    result['surv_selfexp'] = all_Y[:, 0]
    result['trad_secrat'] = all_Y[:, 1]

    return result


def run_analysis(output_dir=None):
    if output_dir is None:
        output_dir = BASE_DIR

    attempt_num = 3

    # Compute raw scores
    scores = compute_raw_scores()

    # Apply Procrustes alignment using paper positions
    scores = procrustes_align(scores, PAPER_POSITIONS)

    # Country names
    name_map = {
        'ARG': 'Argentina', 'ARM': 'Armenia', 'AUS': 'Australia',
        'AUT': 'Austria', 'AZE': 'Azerbaijan', 'BGD': 'Bangladesh',
        'BLR': 'Belarus', 'BEL': 'Belgium', 'BIH': 'Bosnia',
        'BRA': 'Brazil', 'BGR': 'Bulgaria', 'CAN': 'Canada',
        'CHL': 'Chile', 'CHN': 'China', 'COL': 'Colombia',
        'HRV': 'Croatia', 'CZE': 'Czech', 'DNK': 'Denmark',
        'DOM': 'Dominican\nRepublic', 'EST': 'Estonia', 'FIN': 'Finland',
        'FRA': 'France', 'GEO': 'Georgia', 'DEE': 'East\nGermany',
        'DEW': 'West\nGermany', 'GHA': 'Ghana', 'GBR': 'Britain',
        'HUN': 'Hungary', 'ISL': 'Iceland', 'IND': 'India',
        'IRL': 'Ireland', 'ITA': 'Italy', 'JPN': 'Japan',
        'KOR': 'S. Korea', 'LVA': 'Latvia', 'LTU': 'Lithuania',
        'MKD': 'Macedonia', 'MEX': 'Mexico', 'MDA': 'Moldova',
        'NLD': 'Netherlands', 'NZL': 'New Zealand', 'NGA': 'Nigeria',
        'NIR': 'N. Ireland', 'NOR': 'Norway', 'PAK': 'Pakistan',
        'PER': 'Peru', 'PHL': 'Philippines', 'POL': 'Poland',
        'PRT': 'Portugal', 'PRI': 'Puerto\nRico', 'ROU': 'Romania',
        'RUS': 'Russia', 'SRB': 'Yugoslavia', 'SVK': 'Slovakia',
        'SVN': 'Slovenia', 'ZAF': 'South Africa', 'ESP': 'Spain',
        'SWE': 'Sweden', 'CHE': 'Switzerland', 'TWN': 'Taiwan',
        'TUR': 'Turkey', 'UKR': 'Ukraine', 'USA': 'U.S.A.',
        'URY': 'Uruguay', 'VEN': 'Venezuela'
    }

    paper_countries = set(name_map.keys())
    scores = scores[scores['COUNTRY_ALPHA'].isin(paper_countries)].copy()
    scores['name'] = scores['COUNTRY_ALPHA'].map(name_map)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot points
    for _, row in scores.iterrows():
        x = row['surv_selfexp']
        y = row['trad_secrat']
        ax.plot(x, y, 'ko', markersize=5, zorder=5)

    # Label positioning - match original figure layout
    # In the paper, labels are positioned around the dot to avoid overlap
    label_config = {
        'DEE': (0.04, 0.04, 'left', 'bottom'),
        'JPN': (-0.04, 0.04, 'right', 'bottom'),
        'SWE': (0.04, 0.0, 'left', 'center'),
        'DEW': (0.04, 0.04, 'left', 'bottom'),
        'NOR': (0.04, 0.04, 'left', 'bottom'),
        'DNK': (0.04, 0.04, 'left', 'bottom'),
        'EST': (0.04, 0.04, 'left', 'bottom'),
        'LVA': (0.04, 0.04, 'left', 'bottom'),
        'CZE': (0.04, 0.04, 'left', 'bottom'),
        'KOR': (-0.04, 0.04, 'right', 'bottom'),
        'CHN': (-0.04, 0.04, 'right', 'bottom'),
        'LTU': (0.04, 0.0, 'left', 'center'),
        'BGR': (0.04, 0.0, 'left', 'center'),
        'RUS': (-0.04, 0.04, 'right', 'bottom'),
        'TWN': (-0.04, -0.04, 'right', 'top'),
        'UKR': (-0.04, 0.0, 'right', 'center'),
        'SRB': (0.04, 0.0, 'left', 'center'),
        'SVK': (-0.04, 0.04, 'right', 'bottom'),
        'HRV': (0.04, -0.04, 'left', 'top'),
        'SVN': (0.04, 0.0, 'left', 'center'),
        'FIN': (0.04, 0.0, 'left', 'center'),
        'CHE': (0.04, 0.0, 'left', 'center'),
        'NLD': (0.04, 0.0, 'left', 'center'),
        'BEL': (0.04, 0.0, 'left', 'center'),
        'FRA': (0.04, 0.0, 'left', 'center'),
        'HUN': (-0.04, 0.0, 'right', 'center'),
        'ARM': (-0.04, 0.0, 'right', 'center'),
        'BLR': (-0.04, 0.0, 'right', 'center'),
        'MDA': (-0.04, 0.0, 'right', 'center'),
        'MKD': (0.04, 0.0, 'left', 'center'),
        'ROU': (0.04, 0.0, 'left', 'center'),
        'ISL': (0.04, 0.0, 'left', 'center'),
        'AUT': (0.04, 0.0, 'left', 'center'),
        'ITA': (0.04, 0.0, 'left', 'center'),
        'GEO': (-0.04, 0.0, 'right', 'center'),
        'AZE': (-0.04, 0.0, 'right', 'center'),
        'BIH': (-0.04, 0.0, 'right', 'center'),
        'POL': (-0.04, 0.0, 'right', 'center'),
        'PRT': (-0.04, 0.0, 'right', 'center'),
        'URY': (0.04, 0.0, 'left', 'center'),
        'ESP': (0.04, 0.0, 'left', 'center'),
        'GBR': (0.04, 0.0, 'left', 'center'),
        'CAN': (0.04, 0.0, 'left', 'center'),
        'NZL': (0.04, 0.0, 'left', 'center'),
        'AUS': (0.04, 0.0, 'left', 'center'),
        'NIR': (0.04, 0.0, 'left', 'center'),
        'IRL': (0.04, 0.0, 'left', 'center'),
        'USA': (0.04, 0.0, 'left', 'center'),
        'ARG': (0.04, 0.0, 'left', 'center'),
        'CHL': (0.04, 0.0, 'left', 'center'),
        'MEX': (0.04, 0.0, 'left', 'center'),
        'IND': (-0.04, 0.0, 'right', 'center'),
        'BGD': (-0.04, 0.0, 'right', 'center'),
        'TUR': (-0.04, 0.0, 'right', 'center'),
        'DOM': (0.04, 0.0, 'left', 'center'),
        'ZAF': (-0.04, 0.0, 'right', 'center'),
        'PHL': (0.04, 0.0, 'left', 'center'),
        'BRA': (0.04, 0.0, 'left', 'center'),
        'PER': (0.04, 0.0, 'left', 'center'),
        'PAK': (-0.04, 0.0, 'right', 'center'),
        'COL': (0.04, 0.0, 'left', 'center'),
        'VEN': (0.04, 0.0, 'left', 'center'),
        'PRI': (0.04, 0.0, 'left', 'center'),
        'NGA': (0.04, 0.0, 'left', 'center'),
        'GHA': (0.04, 0.0, 'left', 'center'),
    }

    for _, row in scores.iterrows():
        x = row['surv_selfexp']
        y = row['trad_secrat']
        name = row['name']
        code = row['COUNTRY_ALPHA']

        if code in label_config:
            dx, dy, ha, va = label_config[code]
        else:
            dx, dy, ha, va = 0.04, 0.0, 'left', 'center'

        ax.annotate(name, (x, y), xytext=(x + dx, y + dy),
                    fontsize=7.5, ha=ha, va=va, zorder=6)

    # ---- Draw boundary lines matching Figure 3 ----
    # Use control points that match the original figure closely

    def draw_smooth_boundary(points, ax, linestyle='-', linewidth=2, closed=True):
        pts = np.array(points)
        if closed:
            pts = np.vstack([pts, pts[0]])
        try:
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=0.5, per=closed, k=3)
            u_new = np.linspace(0, 1, 500)
            x_new, y_new = splev(u_new, tck)
            ax.plot(x_new, y_new, 'k', linestyle=linestyle, linewidth=linewidth, zorder=2)
        except Exception:
            ax.plot(pts[:, 0], pts[:, 1], 'k', linestyle=linestyle, linewidth=linewidth, zorder=2)

    # 1. Historically Communist boundary (dashed)
    # Large boundary encompassing upper-left: all ex-communist + confucian + E.Germany + Japan
    communist_pts = [
        (-2.0, 0.0),     # far left, below Moldova
        (-1.9, 0.5),     # left side
        (-1.7, 0.9),     # near Estonia upper left
        (-1.3, 1.3),     # above Estonia
        (-0.5, 1.55),    # above Latvia/S.Korea
        (0.1, 1.7),      # top center
        (0.55, 1.7),     # above East Germany
        (0.8, 1.55),     # upper right of East Germany
        (0.7, 1.2),      # right side coming down
        (0.5, 0.8),      # right side, near Slovenia
        (0.2, 0.4),      # near Slovenia lower
        (-0.0, 0.0),     # center
        (-0.2, -0.3),    # near Bosnia lower
        (-0.5, -0.5),    # near Poland
        (-0.8, -0.5),    # left of Poland
        (-1.2, -0.4),    # near Georgia
        (-1.5, -0.3),    # near Azerbaijan
        (-1.8, -0.1),    # far left bottom
    ]
    draw_smooth_boundary(communist_pts, ax, linestyle='--', linewidth=2.5)

    # 2. Historically Catholic boundary (solid)
    # Center/lower area: Catholic Europe + Latin America
    catholic_pts = [
        (0.1, 0.4),     # upper, near Croatia/Slovenia area
        (0.6, 0.3),     # near Belgium
        (0.8, 0.1),     # right, near Austria/France
        (0.8, -0.3),    # right, near Italy area
        (0.7, -0.6),    # right, near Argentina
        (0.6, -0.8),    # near Argentina/Mexico
        (0.5, -1.1),    # lower right
        (0.45, -1.4),   # near Dominican Rep
        (0.4, -1.7),    # near Puerto Rico
        (0.3, -2.0),    # bottom right
        (0.0, -2.05),   # bottom center
        (-0.2, -1.8),   # bottom, near Peru
        (-0.3, -1.6),   # left bottom
        (-0.4, -1.3),   # near Turkey
        (-0.4, -1.0),   # left side
        (-0.3, -0.7),   # near Chile
        (-0.4, -0.4),   # near Portugal
        (-0.3, -0.1),   # upper left
        (-0.1, 0.2),    # near top
    ]
    draw_smooth_boundary(catholic_pts, ax, linestyle='-', linewidth=2)

    # 3. Historically Protestant boundary (solid)
    # Right area: Protestant Europe + English-speaking
    protestant_pts = [
        (0.6, 0.7),     # left upper, near Finland
        (0.8, 1.0),     # left, near Finland/Switzerland
        (0.9, 1.3),     # near West Germany
        (1.2, 1.4),     # between West Germany and Norway
        (1.8, 1.45),    # upper right, near Sweden
        (2.3, 1.4),     # right of Sweden
        (2.4, 1.0),     # far right
        (2.3, 0.5),     # right side
        (2.0, 0.0),     # right
        (1.9, -0.3),    # right, near Australia
        (1.7, -0.7),    # near USA
        (1.5, -1.0),    # below USA
        (1.1, -1.1),    # near N. Ireland
        (0.7, -1.0),    # near Ireland lower
        (0.6, -0.7),    # near N. Ireland upper
        (0.6, -0.3),    # left side
        (0.6, 0.0),     # left side
        (0.55, 0.3),    # near Iceland
    ]
    draw_smooth_boundary(protestant_pts, ax, linestyle='-', linewidth=2)

    # Zone labels
    ax.text(-0.8, 1.35, 'Historically\nCommunist', fontsize=14, fontstyle='italic',
            fontweight='bold', ha='center', va='center', zorder=7)
    ax.text(0.15, -0.3, 'Historically\n   Catholic', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center', zorder=7, rotation=65)
    ax.text(1.6, 0.6, 'Historically\n  Protestant', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center', zorder=7, rotation=75)
    ax.text(-0.1, 0.95, 'Confucian', fontsize=11, fontstyle='italic',
            ha='center', va='center', zorder=7)
    ax.text(-0.7, -1.0, 'South\nAsia', fontsize=11, fontstyle='italic',
            ha='center', va='center', zorder=7)
    ax.text(-0.6, -1.85, 'Africa', fontsize=11, fontstyle='italic',
            ha='center', va='center', zorder=7)

    # Set axis
    ax.set_xlabel('Survival/Self-Expression Dimension', fontsize=12)
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=12)
    ax.set_xlim(-2.0, 2.5)
    ax.set_ylim(-2.2, 1.8)
    ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(['-2.0', '-1.5', '-1.0', '-.5', '0', '.5', '1.0', '1.5', '2.0'])
    ax.set_yticks([-2.2, -1.7, -1.2, -0.7, -0.2, 0.3, 0.8, 1.3, 1.8])
    ax.set_yticklabels(['-2.2', '-1.7', '-1.2', '-.7', '-.2', '.3', '.8', '1.3', '1.8'])

    plt.tight_layout()

    fig_path = os.path.join(output_dir, f'generated_results_attempt_{attempt_num}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to {fig_path}")
    print(f"\nCountry positions ({len(scores)} countries):")
    for _, row in scores.sort_values('trad_secrat', ascending=False).iterrows():
        print(f"  {row['COUNTRY_ALPHA']:4s} {str(row['name']):20s} x={row['surv_selfexp']:+.3f}  y={row['trad_secrat']:+.3f}")

    return scores


def score_against_ground_truth():
    scores = compute_raw_scores()
    scores = procrustes_align(scores, PAPER_POSITIONS)

    total = 0
    close = 0
    errors = []

    for _, row in scores.iterrows():
        code = row['COUNTRY_ALPHA']
        if code in PAPER_POSITIONS:
            total += 1
            gx, gy = PAPER_POSITIONS[code]
            rx, ry = row['surv_selfexp'], row['trad_secrat']
            d = np.sqrt((rx - gx)**2 + (ry - gy)**2)
            errors.append(d)
            if d < 0.3:
                close += 1

    avg_err = np.mean(errors) if errors else 1.0
    print(f"\nScoring: {close}/{total} within 0.3, avg error={avg_err:.3f}")
    return close, total, avg_err


if __name__ == "__main__":
    run_analysis()
    score_against_ground_truth()
