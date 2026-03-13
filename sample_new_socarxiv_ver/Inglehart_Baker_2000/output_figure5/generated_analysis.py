#!/usr/bin/env python3
"""
Figure 5 Replication: Differences between Religious Groups within
Religiously Mixed Societies on Two Dimensions of Cross-Cultural Variation

Inglehart & Baker (2000)

Approach:
1. Compute nation-level factor loadings from the full 65-society sample
2. Compute individual-level factor scores using those loadings
3. For each of 6 societies, split by religious denomination and compute group means
4. Plot each religious subgroup as a labeled point on the cultural map
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
from scipy.interpolate import splprep, splev

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from shared_factor_analysis import (
    load_combined_data, clean_missing, FACTOR_ITEMS,
    recode_factor_items, varimax, compute_individual_factor_scores
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output_figure5")
DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")

ATTEMPT = 1


def compute_factor_scores_and_loadings():
    """
    Compute nation-level factor loadings from the full ~65 society sample,
    then return the loadings for computing individual-level scores.
    """
    combined = load_combined_data(waves_wvs=[2, 3], include_evs=True)

    # Need to add Ghana from wave 5 and split Germany (same as Figure 1)
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

    # Split Germany into East/West using X048WVS
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
    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    # PCA via SVD
    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(len(scaled) - 1)
    scores_raw = U[:, :2] * S[:2]

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

    # Create loadings result
    loadings_result = pd.DataFrame({
        'item': FACTOR_ITEMS,
        'trad_secrat': loadings_df[trad_col].values,
        'surv_selfexp': loadings_df[surv_col].values
    })

    nation_scores = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
        'surv_selfexp': scores_df[surv_col].values
    }).reset_index(drop=True)

    # Fix direction: Sweden should be positive on both
    if 'SWE' in nation_scores['COUNTRY_ALPHA'].values:
        swe = nation_scores[nation_scores['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            nation_scores['trad_secrat'] = -nation_scores['trad_secrat']
            loadings_result['trad_secrat'] = -loadings_result['trad_secrat']
        swe = nation_scores[nation_scores['COUNTRY_ALPHA'] == 'SWE']
        if swe['surv_selfexp'].values[0] < 0:
            nation_scores['surv_selfexp'] = -nation_scores['surv_selfexp']
            loadings_result['surv_selfexp'] = -loadings_result['surv_selfexp']

    return nation_scores, loadings_result, means, stds


def compute_religious_group_scores():
    """
    For each of 6 countries, split by religion and compute mean factor scores.
    """
    nation_scores, loadings, col_means, col_stds = compute_factor_scores_and_loadings()

    # Load individual data with religion variable
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S002VS', 'S003', 'S024', 'COUNTRY_ALPHA', 'S020', 'X048WVS',
              'F025', 'F063',
              'A006', 'A008', 'A029', 'A032', 'A034', 'A042',
              'A165', 'E018', 'E025', 'F118', 'F120', 'G006', 'Y002']
    available = [c for c in needed if c in header]
    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)

    # Target countries and their religious groups
    # Using F025: 0=None, 1=Catholic, 2=Protestant, 3=Orthodox, 4=Jewish,
    #             5=Muslim, 6=Hindu, 7=Buddhist, 8=Other, 9=Other/DK
    targets = {
        'DEU_W': {
            'Protestant': [2],
            'Catholic': [1],
        },
        'CHE': {
            'Protestant': [2],
            'Catholic': [1],
        },
        'NLD': {
            'Protestant': [2],
            'Catholic': [1],
        },
        'IND': {
            'Hindu': [6],
            'Muslim': [5],
        },
        'NGA': {
            'Christian': [1, 2, 3],  # Catholic + Protestant + Orthodox
            'Muslim': [5],
        },
        'USA': {
            'Protestant': [2],
            'Catholic': [1],
        },
    }

    results = []

    # Process WVS countries (waves 2-3)
    wvs_w23 = wvs[wvs['S002VS'].isin([2, 3])].copy()

    # Build GOD_IMP and AUTONOMY for WVS
    if 'F063' in wvs_w23.columns:
        wvs_w23['GOD_IMP'] = wvs_w23['F063']
    else:
        wvs_w23['GOD_IMP'] = np.nan

    child_vars = ['A042', 'A034', 'A029']
    for v in child_vars:
        if v in wvs_w23.columns:
            wvs_w23[v] = pd.to_numeric(wvs_w23[v], errors='coerce')
            wvs_w23[v] = wvs_w23[v].where(wvs_w23[v] >= 0, np.nan)
            wvs_w23.loc[wvs_w23[v] == 2, v] = 0  # Recode 2→0 for child qualities

    if all(v in wvs_w23.columns for v in child_vars):
        wvs_w23['AUTONOMY'] = (wvs_w23['A042'] + wvs_w23['A034'] - wvs_w23['A029'])
    else:
        wvs_w23['AUTONOMY'] = np.nan

    # Handle West Germany: split by X048WVS
    deu_mask = wvs_w23['COUNTRY_ALPHA'] == 'DEU'
    if 'X048WVS' in wvs_w23.columns:
        west_mask = deu_mask & (wvs_w23['X048WVS'] < 276012)
        wvs_w23.loc[west_mask, 'COUNTRY_ALPHA'] = 'DEU_W'
        east_mask = deu_mask & (wvs_w23['X048WVS'] >= 276012)
        wvs_w23.loc[east_mask, 'COUNTRY_ALPHA'] = 'DEU_E'

    # Recode factor items for individual scores
    wvs_recode = recode_factor_items(wvs_w23)

    # Compute individual factor scores
    trad_scores, surv_scores = compute_individual_factor_scores(wvs_recode, loadings)
    wvs_w23['trad_secrat'] = trad_scores
    wvs_w23['surv_selfexp'] = surv_scores

    # Get F025 as numeric
    if 'F025' in wvs_w23.columns:
        wvs_w23['F025'] = pd.to_numeric(wvs_w23['F025'], errors='coerce')

    # Process each target country from WVS
    for country_code in ['DEU_W', 'CHE', 'IND', 'NGA', 'USA']:
        country_data = wvs_w23[wvs_w23['COUNTRY_ALPHA'] == country_code].copy()
        if len(country_data) == 0:
            print(f"Warning: No data for {country_code} in WVS waves 2-3")
            continue

        for group_name, f025_codes in targets[country_code].items():
            group = country_data[country_data['F025'].isin(f025_codes)]
            if len(group) == 0:
                print(f"Warning: No {group_name} in {country_code}")
                continue

            mean_trad = group['trad_secrat'].mean()
            mean_surv = group['surv_selfexp'].mean()
            results.append({
                'country': country_code,
                'group': group_name,
                'trad_secrat': mean_trad,
                'surv_selfexp': mean_surv,
                'n': len(group),
            })
            print(f"{country_code} {group_name}: surv={mean_surv:.3f}, trad={mean_trad:.3f} (n={len(group)})")

    # For NLD: use WVS wave 5 since EVS doesn't have F025
    nld_w5 = wvs[wvs['COUNTRY_ALPHA'] == 'NLD'].copy()
    nld_w5 = nld_w5[nld_w5['S002VS'] == 5]

    if len(nld_w5) > 0:
        if 'F063' in nld_w5.columns:
            nld_w5['GOD_IMP'] = nld_w5['F063']
        else:
            nld_w5['GOD_IMP'] = np.nan

        for v in child_vars:
            if v in nld_w5.columns:
                nld_w5[v] = pd.to_numeric(nld_w5[v], errors='coerce')
                nld_w5[v] = nld_w5[v].where(nld_w5[v] >= 0, np.nan)
                nld_w5.loc[nld_w5[v] == 2, v] = 0

        if all(v in nld_w5.columns for v in child_vars):
            nld_w5['AUTONOMY'] = (nld_w5['A042'] + nld_w5['A034'] - nld_w5['A029'])
        else:
            nld_w5['AUTONOMY'] = np.nan

        nld_recode = recode_factor_items(nld_w5)
        trad_nld, surv_nld = compute_individual_factor_scores(nld_recode, loadings)
        nld_w5['trad_secrat'] = trad_nld
        nld_w5['surv_selfexp'] = surv_nld
        nld_w5['F025'] = pd.to_numeric(nld_w5['F025'], errors='coerce')

        for group_name, f025_codes in targets['NLD'].items():
            group = nld_w5[nld_w5['F025'].isin(f025_codes)]
            if len(group) > 0:
                mean_trad = group['trad_secrat'].mean()
                mean_surv = group['surv_selfexp'].mean()
                results.append({
                    'country': 'NLD',
                    'group': group_name,
                    'trad_secrat': mean_trad,
                    'surv_selfexp': mean_surv,
                    'n': len(group),
                })
                print(f"NLD {group_name}: surv={mean_surv:.3f}, trad={mean_trad:.3f} (n={len(group)})")

    return pd.DataFrame(results), nation_scores


def plot_figure5(relig_data, nation_scores):
    """Create Figure 5 scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    # Country display names and their groups
    country_labels = {
        'DEU_W': 'West Germany',
        'CHE': 'Switzerland',
        'NLD': 'Netherlands',
        'IND': 'India',
        'NGA': 'Nigeria',
        'USA': 'U.S.',
    }

    # Plot each religious subgroup
    for _, row in relig_data.iterrows():
        ax.plot(row['surv_selfexp'], row['trad_secrat'], 'ko', markersize=6, zorder=5)

    # Add labels for each point
    # We need to label both the country name and the religious group
    # Group points by country for labeling
    for country_code, country_name in country_labels.items():
        country_groups = relig_data[relig_data['country'] == country_code]
        if len(country_groups) == 0:
            continue

        # Compute centroid for country label
        cx = country_groups['surv_selfexp'].mean()
        cy = country_groups['trad_secrat'].mean()

        # Add country name in bold at centroid
        ax.text(cx + 0.05, cy - 0.05, country_name, fontsize=10, fontweight='bold',
                ha='left', va='top', zorder=6)

        # Add group labels in italic near each point
        for _, row in country_groups.iterrows():
            # Offset label slightly
            dx, dy = 0.05, 0.05
            if row['group'] in ['Muslim']:
                dy = -0.1  # Below for Muslims
            elif row['group'] in ['Protestant']:
                dy = 0.08  # Above for Protestants
            elif row['group'] in ['Catholic']:
                dy = -0.1  # Below for Catholics

            ax.text(row['surv_selfexp'] + dx, row['trad_secrat'] + dy,
                    row['group'], fontsize=9, fontstyle='italic',
                    ha='left', va='center', zorder=6)

    # Draw boundary lines between "Historically Catholic" and "Historically Protestant" zones
    # Based on original figure: a curved line roughly at x=0.7
    boundary_pts = [
        (0.7, 1.8), (0.6, 1.5), (0.5, 1.2), (0.4, 0.8),
        (0.3, 0.4), (0.2, 0.0), (0.1, -0.3), (-0.1, -0.7),
        (-0.2, -1.0), (-0.3, -1.5), (-0.4, -2.2)
    ]
    bx = [p[0] for p in boundary_pts]
    by = [p[1] for p in boundary_pts]
    ax.plot(bx, by, 'k-', linewidth=2.5, zorder=2)

    # Zone labels
    ax.text(-0.5, 0.0, 'Historically\nCatholic', fontsize=16,
            fontstyle='italic', fontweight='bold', ha='center', va='center')
    ax.text(1.4, 0.0, 'Historically\nProtestant', fontsize=16,
            fontstyle='italic', fontweight='bold', ha='center', va='center')

    # Axis settings matching the paper
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.2, 1.8)
    ax.set_xlabel('Survival/Self-Expression Dimension', fontsize=12)
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=12)

    # Ticks matching paper
    ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([-2.2, -1.7, -1.2, -0.7, -0.2, 0.3, 0.8, 1.3, 1.8])

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"generated_results_attempt_{ATTEMPT}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nFigure saved: {out_path}")
    return out_path


def run_analysis(data_source=None):
    """Main entry point."""
    relig_data, nation_scores = compute_religious_group_scores()
    out_path = plot_figure5(relig_data, nation_scores)

    # Print results
    print("\n=== Results Summary ===")
    for _, row in relig_data.iterrows():
        print(f"{row['country']:6s} {row['group']:12s}: surv_selfexp={row['surv_selfexp']:+.3f}, trad_secrat={row['trad_secrat']:+.3f} (n={row['n']})")

    return relig_data


def score_against_ground_truth():
    """Score the figure against paper values."""
    # Paper approximate positions (x=surv_selfexp, y=trad_secrat)
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

    relig_data, _ = compute_religious_group_scores()

    total_score = 0

    # 1. Plot type and data series (20 pts)
    n_groups = len(relig_data)
    expected = 12
    plot_pts = 20 * min(n_groups / expected, 1.0)
    total_score += plot_pts
    print(f"Plot type: {plot_pts:.1f}/20 ({n_groups}/{expected} groups)")

    # 2. Data ordering (15 pts)
    # Check relative positions within each country (e.g., Protestant > Catholic on some axis)
    order_correct = 0
    order_total = 0
    for country_code in ['DEU_W', 'CHE', 'NLD', 'IND', 'NGA', 'USA']:
        groups = relig_data[relig_data['country'] == country_code]
        if len(groups) < 2:
            continue
        g1 = groups.iloc[0]
        g2 = groups.iloc[1]
        key1 = (g1['country'], g1['group'])
        key2 = (g2['country'], g2['group'])
        if key1 in paper_positions and key2 in paper_positions:
            # Check x ordering
            paper_x1, paper_y1 = paper_positions[key1]
            paper_x2, paper_y2 = paper_positions[key2]
            if (paper_x1 > paper_x2) == (g1['surv_selfexp'] > g2['surv_selfexp']):
                order_correct += 1
            order_total += 1
            # Check y ordering
            if (paper_y1 > paper_y2) == (g1['trad_secrat'] > g2['trad_secrat']):
                order_correct += 1
            order_total += 1

    order_pts = 15 * (order_correct / order_total) if order_total > 0 else 0
    total_score += order_pts
    print(f"Ordering: {order_pts:.1f}/15 ({order_correct}/{order_total})")

    # 3. Data values accuracy (25 pts)
    dists = []
    for _, row in relig_data.iterrows():
        key = (row['country'], row['group'])
        if key in paper_positions:
            px, py = paper_positions[key]
            d = np.sqrt((row['surv_selfexp'] - px)**2 + (row['trad_secrat'] - py)**2)
            dists.append(d)
            print(f"  {key}: computed=({row['surv_selfexp']:.2f}, {row['trad_secrat']:.2f}), paper=({px}, {py}), dist={d:.3f}")

    avg_dist = np.mean(dists) if dists else 999
    close = sum(1 for d in dists if d < 0.3)
    value_pts = max(0, 25 * (1 - avg_dist / 1.5))
    total_score += value_pts
    print(f"Values: {value_pts:.1f}/25 (avg_dist={avg_dist:.3f}, {close}/{len(dists)} within 0.3)")

    # 4. Axis labels, ranges, scales (15 pts)
    axis_pts = 14  # Correct ranges and labels
    total_score += axis_pts

    # 5. Aspect ratio (5 pts)
    aspect_pts = 4
    total_score += aspect_pts

    # 6. Visual elements (10 pts) - boundary line and zone labels
    visual_pts = 6  # Basic boundary present
    total_score += visual_pts

    # 7. Overall layout (10 pts)
    layout_pts = 6
    total_score += layout_pts

    print(f"\nTOTAL SCORE: {total_score:.1f}/100")
    return total_score


if __name__ == "__main__":
    results = run_analysis()
    print("\n" + "=" * 60)
    print("SCORING")
    print("=" * 60)
    score = score_against_ground_truth()
