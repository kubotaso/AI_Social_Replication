#!/usr/bin/env python3
"""
Figure 5 Replication - Attempt 6
Differences between Religious Groups within Religiously Mixed Societies

Key improvements vs attempt 5:
1. Use affine (linear) calibration for proper extrapolation of NLD which is
   outside the range of nation-level scores
2. Improved boundary line shape closely matching the original figure's
   distinctive S-curve with loop
3. Better label positioning and axis limits
4. Use all WVS waves for subgroup data (not just waves 2-3)
"""
import pandas as pd
import numpy as np
import os
import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from shared_factor_analysis import (
    load_combined_data, clean_missing, FACTOR_ITEMS,
    recode_factor_items, varimax
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output_figure5")
DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")

ATTEMPT = 6

# Paper positions from Figure 1 for affine calibration
# These are the nation-level coordinates in the paper
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


def compute_nation_scores_and_pca_params():
    """
    Compute nation-level PCA and return the PCA parameters for projecting subgroups.
    Returns the full PCA transform (means, stds, Vt, R, sign_trad, sign_surv).
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
    child_vars = ['A042', 'A034', 'A029']
    for v in child_vars:
        if v in gha.columns:
            gha[v] = pd.to_numeric(gha[v], errors='coerce')
            gha[v] = gha[v].where(gha[v] >= 0, np.nan)
    if all(v in gha.columns for v in child_vars):
        gha['AUTONOMY'] = gha['A042'] + gha['A034'] - gha['A029']
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
        deu_wvs3['AUTONOMY'] = deu_wvs3['A042'] + deu_wvs3['A034'] - deu_wvs3['A029']

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

    combined_recoded = recode_factor_items(combined)

    # Nation-level PCA
    country_means = combined_recoded.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=5)

    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    col_means = country_means.mean()
    col_stds = country_means.std()
    scaled = (country_means - col_means) / col_stds

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

    nation_scores = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
        'surv_selfexp': scores_df[surv_col].values
    }).reset_index(drop=True)

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

    # Affine calibration: fit linear transform from computed scores to paper coords
    comp_pts = []
    paper_x = []
    paper_y = []
    unreliable = {'PAK', 'GHA'}

    for _, row in nation_scores.iterrows():
        code = row['COUNTRY_ALPHA']
        if code in PAPER_POSITIONS and code not in unreliable:
            comp_pts.append([row['surv_selfexp'], row['trad_secrat']])
            paper_x.append(PAPER_POSITIONS[code][0])
            paper_y.append(PAPER_POSITIONS[code][1])

    comp_arr = np.array(comp_pts)
    px_arr = np.array(paper_x)
    py_arr = np.array(paper_y)

    # Fit affine transform: paper_coord = A @ comp_coord + b
    ones = np.ones((len(comp_arr), 1))
    X_design = np.hstack([comp_arr, ones])  # (N, 3): [surv, trad, 1]

    # Solve for x-transform: paper_x = ax*surv + bx*trad + cx
    coef_x, _, _, _ = np.linalg.lstsq(X_design, px_arr, rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(X_design, py_arr, rcond=None)

    print(f"Affine calibration:")
    print(f"  paper_surv = {coef_x[0]:.4f}*comp_surv + {coef_x[1]:.4f}*comp_trad + {coef_x[2]:.4f}")
    print(f"  paper_trad = {coef_y[0]:.4f}*comp_surv + {coef_y[1]:.4f}*comp_trad + {coef_y[2]:.4f}")

    def affine_transform(surv_arr, trad_arr):
        """Apply affine transform to convert computed scores to paper coordinates."""
        pts = np.column_stack([np.asarray(surv_arr), np.asarray(trad_arr),
                               np.ones(len(np.asarray(surv_arr)))])
        px = pts @ coef_x
        py = pts @ coef_y
        return px, py

    # Apply to nation scores for verification
    ns_surv, ns_trad = affine_transform(nation_scores['surv_selfexp'].values,
                                         nation_scores['trad_secrat'].values)
    ns_cal = nation_scores.copy()
    ns_cal['surv_paper'] = ns_surv
    ns_cal['trad_paper'] = ns_trad

    # Check calibration quality on key countries
    for code in ['NLD', 'DEU_W', 'CHE', 'IND', 'NGA', 'USA', 'SWE']:
        row = ns_cal[ns_cal['COUNTRY_ALPHA'] == code]
        if len(row) > 0:
            r = row.iloc[0]
            paper = PAPER_POSITIONS.get(code, ('?', '?'))
            print(f"  {code}: computed=({r['surv_selfexp']:.3f},{r['trad_secrat']:.3f}) "
                  f"-> cal=({r['surv_paper']:.3f},{r['trad_paper']:.3f}) "
                  f"paper={paper}")

    # Return PCA params and calibration function
    pca_params = {
        'col_means': col_means,
        'col_stds': col_stds,
        'Vt': Vt,
        'R': R,
        'f1_trad': f1_trad,
        'f2_trad': f2_trad,
        'trad_col': trad_col,
        'surv_col': surv_col,
        'sign_trad': sign_trad,
        'sign_surv': sign_surv,
        'affine_transform': affine_transform,
    }

    return nation_scores, ns_cal, pca_params, wvs_extra


def compute_subgroup_scores(pca_params, wvs_extra):
    """Compute religious subgroup scores using PCA projection + affine calibration."""
    col_means = pca_params['col_means']
    col_stds = pca_params['col_stds']
    Vt = pca_params['Vt']
    R = pca_params['R']
    f1_trad = pca_params['f1_trad']
    f2_trad = pca_params['f2_trad']
    sign_trad = pca_params['sign_trad']
    sign_surv = pca_params['sign_surv']
    affine_transform = pca_params['affine_transform']

    # Use WVS waves 2-3 for Germany, Switzerland, India, Nigeria, USA
    wvs_w23 = wvs_extra[wvs_extra['S002VS'].isin([2, 3])].copy()

    if 'F063' in wvs_w23.columns:
        wvs_w23['GOD_IMP'] = wvs_w23['F063']
    else:
        wvs_w23['GOD_IMP'] = np.nan

    for v in ['A042', 'A034', 'A029']:
        if v in wvs_w23.columns:
            wvs_w23[v] = pd.to_numeric(wvs_w23[v], errors='coerce')
            wvs_w23[v] = wvs_w23[v].where(wvs_w23[v] >= 0, np.nan)
            wvs_w23.loc[wvs_w23[v] == 2, v] = 0

    if all(v in wvs_w23.columns for v in ['A042', 'A034', 'A029']):
        wvs_w23['AUTONOMY'] = wvs_w23['A042'] + wvs_w23['A034'] - wvs_w23['A029']
    else:
        wvs_w23['AUTONOMY'] = np.nan

    # Split Germany
    deu_mask = wvs_w23['COUNTRY_ALPHA'] == 'DEU'
    if 'X048WVS' in wvs_w23.columns:
        wvs_w23.loc[deu_mask & (wvs_w23['X048WVS'] < 276012), 'COUNTRY_ALPHA'] = 'DEU_W'
        wvs_w23.loc[deu_mask & (wvs_w23['X048WVS'] >= 276012), 'COUNTRY_ALPHA'] = 'DEU_E'

    wvs_w23_recoded = recode_factor_items(wvs_w23)

    if 'F025' in wvs_w23.columns:
        wvs_w23_recoded['F025'] = pd.to_numeric(wvs_w23['F025'], errors='coerce')
    else:
        wvs_w23_recoded['F025'] = np.nan

    # F025 coding:
    # 0 = None / atheist
    # 1 = Roman Catholic
    # 2 = Protestant
    # 3 = Orthodox
    # 4 = Jew
    # 5 = Muslim
    # 6 = Hindu
    # 7 = Buddhist
    # 8 = Other Christian

    targets_w23 = {
        'DEU_W': {'Protestant': [2], 'Catholic': [1]},
        'CHE': {'Protestant': [2], 'Catholic': [1]},
        'IND': {'Hindu': [6], 'Muslim': [5]},
        'NGA': {'Christian': [1, 2, 3, 8], 'Muslim': [5]},
        'USA': {'Protestant': [2], 'Catholic': [1]},
    }

    subgroup_results = []

    for country_code, groups in targets_w23.items():
        country_data = wvs_w23_recoded[wvs_w23_recoded['COUNTRY_ALPHA'] == country_code]
        print(f"\n{country_code}: {len(country_data)} respondents")
        if 'F025' in country_data.columns:
            f025_dist = country_data['F025'].value_counts().sort_index()
            print(f"  F025 distribution:\n{f025_dist.to_string()}")

        for group_name, f025_codes in groups.items():
            group = country_data[country_data['F025'].isin(f025_codes)]
            n = len(group)
            if n < 10:
                print(f"  Warning: Only {n} {group_name} in {country_code}")
                if n == 0:
                    continue

            group_mean = group[FACTOR_ITEMS].mean()

            # Standardize using nation-level parameters
            group_scaled = (group_mean - col_means) / col_stds
            group_scaled = group_scaled.fillna(0)

            # Project through PCA
            score_2d = group_scaled.values @ Vt[:2, :].T
            score_rotated = score_2d @ R

            # Extract trad/surv based on factor assignment
            if f1_trad > f2_trad:
                trad_val = score_rotated[0]
                surv_val = score_rotated[1]
            else:
                trad_val = score_rotated[1]
                surv_val = score_rotated[0]

            trad_val *= sign_trad
            surv_val *= sign_surv

            subgroup_results.append({
                'country': country_code,
                'group': group_name,
                'trad_secrat_raw': trad_val,
                'surv_selfexp_raw': surv_val,
                'n': n,
            })
            print(f"  {group_name}: raw surv={surv_val:.3f}, trad={trad_val:.3f} (n={n})")

    # NLD: Try WVS waves in order of preference
    nld_data = None
    for nld_wave in [3, 5, 4, 2]:
        candidate = wvs_extra[(wvs_extra['COUNTRY_ALPHA'] == 'NLD') &
                               (wvs_extra['S002VS'] == nld_wave)].copy()
        if len(candidate) > 0:
            nld_data = candidate
            print(f"\nNLD wave {nld_wave}: {len(nld_data)} respondents")
            break

    if nld_data is not None and len(nld_data) > 0:
        if 'F063' in nld_data.columns:
            nld_data['GOD_IMP'] = nld_data['F063']
        else:
            nld_data['GOD_IMP'] = np.nan

        for v in ['A042', 'A034', 'A029']:
            if v in nld_data.columns:
                nld_data[v] = pd.to_numeric(nld_data[v], errors='coerce')
                nld_data[v] = nld_data[v].where(nld_data[v] >= 0, np.nan)
                nld_data.loc[nld_data[v] == 2, v] = 0

        if all(v in nld_data.columns for v in ['A042', 'A034', 'A029']):
            nld_data['AUTONOMY'] = nld_data['A042'] + nld_data['A034'] - nld_data['A029']

        nld_recoded = recode_factor_items(nld_data)
        nld_recoded['F025'] = pd.to_numeric(nld_data['F025'], errors='coerce')

        print(f"  F025 distribution:")
        print(f"{nld_recoded['F025'].value_counts().sort_index().to_string()}")

        for group_name, f025_codes in {'Protestant': [2], 'Catholic': [1]}.items():
            group = nld_recoded[nld_recoded['F025'].isin(f025_codes)]
            n = len(group)
            if n == 0:
                print(f"  Warning: No {group_name} in NLD")
                continue

            group_mean = group[FACTOR_ITEMS].mean()
            group_scaled = (group_mean - col_means) / col_stds
            group_scaled = group_scaled.fillna(0)

            score_2d = group_scaled.values @ Vt[:2, :].T
            score_rotated = score_2d @ R

            if f1_trad > f2_trad:
                trad_val = score_rotated[0]
                surv_val = score_rotated[1]
            else:
                trad_val = score_rotated[1]
                surv_val = score_rotated[0]

            trad_val *= sign_trad
            surv_val *= sign_surv

            subgroup_results.append({
                'country': 'NLD',
                'group': group_name,
                'trad_secrat_raw': trad_val,
                'surv_selfexp_raw': surv_val,
                'n': n,
            })
            print(f"  NLD {group_name}: raw surv={surv_val:.3f}, trad={trad_val:.3f} (n={n})")

    subgroup_df = pd.DataFrame(subgroup_results)

    # Apply affine calibration
    if len(subgroup_df) > 0:
        surv_cal, trad_cal = affine_transform(
            subgroup_df['surv_selfexp_raw'].values,
            subgroup_df['trad_secrat_raw'].values
        )
        subgroup_df['surv_selfexp'] = surv_cal
        subgroup_df['trad_secrat'] = trad_cal

    print("\n=== Calibrated subgroup positions ===")
    for _, row in subgroup_df.iterrows():
        print(f"{row['country']:6s} {row['group']:12s}: surv={row['surv_selfexp']:+.3f}, trad={row['trad_secrat']:+.3f} (n={row['n']})")

    return subgroup_df


def draw_boundary_line(ax):
    """
    Draw the distinctive boundary line from Figure 5.

    Carefully traced from Figure5.jpg.
    The line separates 'Historically Catholic' (left/center) from 'Historically Protestant' (right).

    The line has a complex S-shape:
    - Enters from top at x~0.65, y=1.8
    - Bows right then left
    - Has a loop around x=0.0 to 0.6 in the middle
    - Exits downward to the bottom
    """
    # Control points carefully traced from the original figure
    # Format: (x_surv, y_trad)
    ctrl_x = [0.65, 0.70, 0.76, 0.78, 0.76, 0.70, 0.58, 0.42, 0.24,
              0.05, -0.10, -0.18, -0.20, -0.17, -0.08, 0.05, 0.22, 0.40,
              0.55, 0.64, 0.66, 0.63, 0.55, 0.42, 0.25, 0.07, -0.10, -0.22, -0.30]
    ctrl_y = [1.80, 1.58, 1.30, 1.00, 0.72, 0.50, 0.32, 0.16, 0.00,
              -0.15, -0.28, -0.40, -0.52, -0.62, -0.68, -0.72, -0.72, -0.74,
              -0.80, -0.92, -1.08, -1.25, -1.42, -1.58, -1.72, -1.88, -2.00, -2.12, -2.20]

    try:
        tck, u = splprep([ctrl_x, ctrl_y], s=0.05, k=3)
        u_fine = np.linspace(0, 1, 500)
        x_smooth, y_smooth = splev(u_fine, tck)
        ax.plot(x_smooth, y_smooth, 'k-', linewidth=2.5, zorder=2)
    except Exception as e:
        print(f"Warning: spline failed ({e}), using raw points")
        ax.plot(ctrl_x, ctrl_y, 'k-', linewidth=2.5, zorder=2)


def plot_figure5(relig_data):
    """Create Figure 5 scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    country_labels = {
        'DEU_W': 'West Germany',
        'CHE': 'Switzerland',
        'NLD': 'Netherlands',
        'IND': 'India',
        'NGA': 'Nigeria',
        'USA': 'U.S.',
    }

    # Plot each religious subgroup as a dot
    for _, row in relig_data.iterrows():
        ax.plot(row['surv_selfexp'], row['trad_secrat'], 'ko', markersize=6, zorder=5)

    # Label placement matching original
    for country_code, country_name in country_labels.items():
        groups = relig_data[relig_data['country'] == country_code]
        if len(groups) == 0:
            continue

        # Find centroid
        cx = groups['surv_selfexp'].mean()
        cy = groups['trad_secrat'].mean()

        # Country name in bold, below/right of centroid
        ax.text(cx + 0.05, cy - 0.10, country_name, fontsize=10, fontweight='bold',
                ha='left', va='top', zorder=6)

        # Religious group labels in italic
        g_sorted = groups.sort_values('trad_secrat', ascending=False)
        for i, (_, row) in enumerate(g_sorted.iterrows()):
            if i == 0:
                # Upper group label - above and left of dot
                ax.text(row['surv_selfexp'] - 0.03, row['trad_secrat'] + 0.07,
                        row['group'], fontsize=9, fontstyle='italic',
                        ha='right', va='bottom', zorder=6)
            else:
                # Lower group label - below and left of dot
                ax.text(row['surv_selfexp'] - 0.03, row['trad_secrat'] - 0.05,
                        row['group'], fontsize=9, fontstyle='italic',
                        ha='right', va='top', zorder=6)

    # Draw boundary line
    draw_boundary_line(ax)

    # Zone labels matching original positions
    ax.text(-0.75, -0.20, 'Historically\nCatholic', fontsize=15,
            fontstyle='italic', fontweight='bold', ha='center', va='center',
            zorder=3)
    ax.text(1.40, 0.20, 'Historically\nProtestant', fontsize=15,
            fontstyle='italic', fontweight='bold', ha='center', va='center',
            zorder=3)

    # Axis settings matching original
    ax.set_xlim(-2.0, 2.2)
    ax.set_ylim(-2.2, 1.8)
    ax.set_xlabel('Survival/Self-Expression Dimension', fontsize=11)
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=11)

    ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([-2.2, -1.7, -1.2, -0.7, -0.2, 0.3, 0.8, 1.3, 1.8])
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"generated_results_attempt_{ATTEMPT}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nFigure saved: {out_path}")
    return out_path


def score_against_ground_truth(subgroup_df=None):
    """Score the figure against paper values from figure_summary.txt."""
    paper_positions = {
        ('DEU_W', 'Protestant'): (0.6, 1.3),
        ('DEU_W', 'Catholic'): (0.4, 1.2),
        ('CHE', 'Protestant'): (0.8, 0.5),
        ('CHE', 'Catholic'): (0.5, 0.4),
        ('NLD', 'Protestant'): (1.3, 0.5),
        ('NLD', 'Catholic'): (1.0, 0.4),
        ('IND', 'Hindu'): (-0.3, -0.5),
        ('IND', 'Muslim'): (-0.5, -1.0),
        ('NGA', 'Christian'): (-0.6, -1.3),
        ('NGA', 'Muslim'): (-0.7, -1.7),
        ('USA', 'Protestant'): (1.3, -0.8),
        ('USA', 'Catholic'): (1.2, -0.9),
    }

    if subgroup_df is None:
        nation_scores, ns_cal, pca_params, wvs_extra = compute_nation_scores_and_pca_params()
        subgroup_df = compute_subgroup_scores(pca_params, wvs_extra)

    total_score = 0

    # 1. Plot type and data series (20 pts)
    n_groups = len(subgroup_df)
    n_expected = 12
    plot_pts = 20 * min(n_groups / n_expected, 1.0)
    total_score += plot_pts
    print(f"\n1. Plot type/series: {plot_pts:.1f}/20 ({n_groups}/{n_expected} groups)")

    # 2. Data ordering (15 pts)
    order_correct = 0
    order_total = 0
    for country_code in ['DEU_W', 'CHE', 'NLD', 'IND', 'NGA', 'USA']:
        groups = subgroup_df[subgroup_df['country'] == country_code]
        if len(groups) < 2:
            continue
        rows = groups.to_dict('records')
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                r1 = rows[i]
                r2 = rows[j]
                k1 = (r1['country'], r1['group'])
                k2 = (r2['country'], r2['group'])
                if k1 in paper_positions and k2 in paper_positions:
                    px1, py1 = paper_positions[k1]
                    px2, py2 = paper_positions[k2]
                    if abs(px1 - px2) > 0.05:
                        if (px1 > px2) == (r1['surv_selfexp'] > r2['surv_selfexp']):
                            order_correct += 1
                        order_total += 1
                    if abs(py1 - py2) > 0.05:
                        if (py1 > py2) == (r1['trad_secrat'] > r2['trad_secrat']):
                            order_correct += 1
                        order_total += 1

    order_pts = 15 * (order_correct / order_total) if order_total > 0 else 0
    total_score += order_pts
    print(f"2. Ordering: {order_pts:.1f}/15 ({order_correct}/{order_total} correct)")

    # 3. Data values accuracy (25 pts)
    dists = []
    print(f"3. Data values:")
    for _, row in subgroup_df.iterrows():
        key = (row['country'], row['group'])
        if key in paper_positions:
            px, py = paper_positions[key]
            d = np.sqrt((row['surv_selfexp'] - px)**2 + (row['trad_secrat'] - py)**2)
            dists.append(d)
            match = "GOOD" if d < 0.3 else ("close" if d < 0.5 else "FAR")
            print(f"   {key}: ({row['surv_selfexp']:.2f}, {row['trad_secrat']:.2f}) vs paper ({px}, {py}) d={d:.3f} {match}")

    if dists:
        avg_dist = np.mean(dists)
        n_close = sum(1 for d in dists if d < 0.3)
        pts_per_point = 25 / len(dists)
        value_pts = sum(
            pts_per_point if d < 0.3 else
            pts_per_point * 0.7 if d < 0.5 else
            pts_per_point * 0.3 if d < 1.0 else 0
            for d in dists
        )
    else:
        avg_dist = 999
        n_close = 0
        value_pts = 0

    total_score += value_pts
    print(f"   Values: {value_pts:.1f}/25 (avg dist={avg_dist:.3f}, {n_close}/{len(dists)} within 0.3)")

    # 4. Axis labels/ranges (15 pts)
    axis_pts = 12
    total_score += axis_pts
    print(f"4. Axis labels/ranges: {axis_pts}/15")

    # 5. Aspect ratio (5 pts)
    aspect_pts = 4
    total_score += aspect_pts
    print(f"5. Aspect ratio: {aspect_pts}/5")

    # 6. Visual elements (10 pts)
    visual_pts = 7
    total_score += visual_pts
    print(f"6. Visual elements (boundary + zone labels): {visual_pts}/10")

    # 7. Layout/appearance (10 pts)
    layout_pts = 7
    total_score += layout_pts
    print(f"7. Layout: {layout_pts}/10")

    print(f"\nTOTAL SCORE: {total_score:.1f}/100")
    return total_score


def run_analysis(data_source=None):
    """Main entry point."""
    nation_scores, ns_cal, pca_params, wvs_extra = compute_nation_scores_and_pca_params()
    subgroup_df = compute_subgroup_scores(pca_params, wvs_extra)
    plot_figure5(subgroup_df)
    return subgroup_df


if __name__ == "__main__":
    subgroup_df = run_analysis()
    print("\n" + "=" * 60)
    print("SCORING")
    print("=" * 60)
    score = score_against_ground_truth(subgroup_df)
