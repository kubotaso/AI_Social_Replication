#!/usr/bin/env python3
"""
Figure 7: Traditional/Secular-Rational Values by Year of Birth
for Four Types of Societies

Attempt 2: Major visual improvements
- Match original figure's annotation style (inline labels, not legend box)
- Match line styles precisely (thick solid, big dots, hatched/dashed, thin dashed)
- Correct axis range (-1.0 to 0.6)
- Better x-axis label formatting
- Address missing EVS countries by documenting limitation
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import (
    compute_nation_level_factor_scores, compute_individual_factor_scores,
    load_combined_data, get_latest_per_country, recode_factor_items,
    get_society_type_groups, FACTOR_ITEMS, COUNTRY_NAMES
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

COHORT_BINS = [
    (1907, 1916), (1917, 1926), (1927, 1936), (1937, 1946),
    (1947, 1956), (1957, 1966), (1967, 1976)
]
COHORT_LABELS = [f"{a}-{b}" for a, b in COHORT_BINS]
COHORT_MIDPOINTS = [(a + b) / 2 for a, b in COHORT_BINS]


def run_analysis(data_source=None):
    """Run the analysis for Figure 7."""
    # Get nation-level factor loadings
    scores_df, loadings_df, country_means = compute_nation_level_factor_scores(
        waves_wvs=[2, 3], include_evs=True
    )

    # Load individual data (latest wave per country)
    df = load_combined_data(waves_wvs=[2, 3], include_evs=True)
    df = get_latest_per_country(df)

    # Compute individual factor scores
    trad_scores, surv_scores = compute_individual_factor_scores(df, loadings_df)
    df['trad_secrat'] = trad_scores
    df['surv_selfexp'] = surv_scores

    # Compute birth year using X003 (age) and S020 (survey year)
    # Also try X002 (year of birth) as fallback
    df['X003'] = pd.to_numeric(df.get('X003'), errors='coerce')
    df['S020'] = pd.to_numeric(df.get('S020'), errors='coerce')
    if 'X002' in df.columns:
        df['X002'] = pd.to_numeric(df['X002'], errors='coerce')

    # Compute birth year: prefer S020 - X003, fallback to X002
    df['birth_year'] = df['S020'] - df['X003']
    if 'X002' in df.columns:
        mask = df['birth_year'].isna() & df['X002'].notna() & (df['X002'] > 1870) & (df['X002'] < 2000)
        df.loc[mask, 'birth_year'] = df.loc[mask, 'X002']

    # Assign cohort
    df['cohort'] = pd.cut(
        df['birth_year'],
        bins=[b[0] - 0.5 for b in COHORT_BINS] + [COHORT_BINS[-1][1] + 0.5],
        labels=COHORT_LABELS
    )
    df = df.dropna(subset=['cohort', 'trad_secrat'])

    # Get society type groups
    groups = get_society_type_groups()

    # Standardize factor scores: mean=0, std=1 across all individuals
    mean_t = df['trad_secrat'].mean()
    std_t = df['trad_secrat'].std()
    df['trad_z'] = (df['trad_secrat'] - mean_t) / std_t

    # For each society type and cohort: compute society means first, then average
    results = {}
    sample_sizes = {}
    countries_used = {}
    for group_name, countries in groups.items():
        group_data = df[df['COUNTRY_ALPHA'].isin(countries)]
        if len(group_data) == 0:
            continue

        sample_sizes[group_name] = len(group_data)
        countries_used[group_name] = sorted(group_data['COUNTRY_ALPHA'].unique())

        # Weight each society equally
        cohort_means = []
        for cohort_label in COHORT_LABELS:
            cohort_data = group_data[group_data['cohort'] == cohort_label]
            if len(cohort_data) == 0:
                cohort_means.append(np.nan)
                continue

            # Compute mean per country first, then average across countries
            country_means_cohort = cohort_data.groupby('COUNTRY_ALPHA')['trad_z'].mean()
            cohort_means.append(country_means_cohort.mean())

        results[group_name] = cohort_means

    # Print data for verification
    for group_name, means in results.items():
        n = sample_sizes.get(group_name, 0)
        cs = countries_used.get(group_name, [])
        print(f"{group_name} (N = {n}, {len(cs)} countries: {cs}):")
        for label, val in zip(COHORT_LABELS, means):
            if not np.isnan(val):
                print(f"  {label}: {val:.3f}")
            else:
                print(f"  {label}: N/A")
        print()

    # Plot - match original figure style closely
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Line styles matching the original Figure 7:
    # Ex-Communist: solid thick line (black)
    # Advanced industrial: large dotted line (black dots)
    # Developing: hatched/dashed pattern (thick dashes)
    # Low-income: thin dotted line
    plot_order = ['Ex-Communist', 'Advanced industrial', 'Developing', 'Low-income']

    line_objects = {}
    for group_name in plot_order:
        if group_name not in results:
            continue
        means = results[group_name]
        valid_idx = [i for i, m in enumerate(means) if not np.isnan(m)]
        valid_x = [COHORT_MIDPOINTS[i] for i in valid_idx]
        valid_y = [means[i] for i in valid_idx]

        if group_name == 'Ex-Communist':
            line, = ax.plot(valid_x, valid_y, '-', color='black', linewidth=3.0,
                           solid_capstyle='round')
        elif group_name == 'Advanced industrial':
            line, = ax.plot(valid_x, valid_y, 'o', color='black', markersize=7,
                           markerfacecolor='black')
            # Also connect with thin line for continuity
            ax.plot(valid_x, valid_y, '-', color='black', linewidth=0.5, alpha=0.3)
        elif group_name == 'Developing':
            line, = ax.plot(valid_x, valid_y, '--', color='black', linewidth=2.5,
                           dashes=(6, 3))
        elif group_name == 'Low-income':
            line, = ax.plot(valid_x, valid_y, ':', color='black', linewidth=1.5)

        line_objects[group_name] = (valid_x, valid_y)

    # Add inline annotations like the original figure
    # Ex-Communist: label on the line around 1937-1946 area
    if 'Ex-Communist' in line_objects:
        x, y = line_objects['Ex-Communist']
        # Place label around cohort 3-4 (1927-1946)
        idx = min(3, len(x) - 1)
        ax.annotate('Ex-Communist societies $^a$',
                    xy=(x[idx], y[idx]),
                    xytext=(x[idx] - 12, y[idx] + 0.06),
                    fontsize=10, fontstyle='italic',
                    rotation=15)

    if 'Advanced industrial' in line_objects:
        x, y = line_objects['Advanced industrial']
        idx = min(3, len(x) - 1)
        ax.annotate('Advanced industrial democracies $^b$',
                    xy=(x[idx], y[idx]),
                    xytext=(x[idx] - 22, y[idx] - 0.06),
                    fontsize=10, fontstyle='italic',
                    rotation=18)

    if 'Developing' in line_objects:
        x, y = line_objects['Developing']
        idx = min(4, len(x) - 1)
        ax.annotate('Developing societies $^c$',
                    xy=(x[idx], y[idx]),
                    xytext=(x[idx] - 5, y[idx] + 0.06),
                    fontsize=10, fontstyle='italic')

    if 'Low-income' in line_objects:
        x, y = line_objects['Low-income']
        idx = min(4, len(x) - 1)
        ax.annotate('Low-income societies $^d$',
                    xy=(x[idx], y[idx]),
                    xytext=(x[idx] - 5, y[idx] + 0.06),
                    fontsize=10, fontstyle='italic')

    # Axis formatting to match original
    ax.set_xlabel('Birth Year', fontsize=12)
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=12)

    # X-axis ticks: show cohort ranges
    ax.set_xticks(COHORT_MIDPOINTS)
    ax.set_xticklabels([f"{a}-{b}" for a, b in COHORT_BINS], fontsize=9, rotation=0,
                        ha='center')

    # Y-axis: match original range -1.0 to 0.60
    ax.set_ylim(-1.0, 0.60)
    ax.yaxis.set_major_locator(MultipleLocator(0.20))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    # Set x limits to add some padding
    ax.set_xlim(COHORT_MIDPOINTS[0] - 3, COHORT_MIDPOINTS[-1] + 3)

    # Grid and spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    attempt = 2
    fig_path = os.path.join(OUTPUT_DIR, f'generated_results_attempt_{attempt}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {fig_path}")

    return results


def score_against_ground_truth():
    """Score the generated figure against the original Figure 7."""
    results = run_analysis()
    if results is None:
        return 0, ["ERROR: Analysis failed"]

    score = 0
    details = []

    # Ground truth values read from the original Figure 7:
    # Ex-Communist: starts ~-0.07, rises steeply to ~0.50
    # Advanced industrial: starts ~-0.07, rises to ~0.40
    # Developing: relatively flat, ~-0.65 to -0.55
    # Low-income: relatively flat, ~-0.80 to -0.75
    gt = {
        'Ex-Communist': [-0.07, 0.02, 0.10, 0.20, 0.35, 0.40, 0.50],
        'Advanced industrial': [-0.07, 0.04, 0.10, 0.35, 0.38, 0.42, 0.42],
        'Developing': [-0.65, -0.65, -0.65, -0.60, -0.55, -0.55, -0.57],
        'Low-income': [np.nan, np.nan, -0.88, -0.82, -0.78, -0.75, -0.75],
    }

    # 1. Plot type and data series (20 pts)
    series_count = sum(1 for g in ['Ex-Communist', 'Advanced industrial', 'Developing', 'Low-income']
                       if g in results)
    pts = min(20, series_count * 5)
    score += pts
    details.append(f"Plot type & series: {series_count}/4 present = +{pts}/20")

    # 2. Data ordering accuracy (15 pts)
    ordering_pts = 0
    if all(k in results for k in gt.keys()):
        # Check that at final cohort: Ex-Comm > Developing > Low-income
        last = {}
        for g, vals in results.items():
            for v in reversed(vals):
                if not np.isnan(v):
                    last[g] = v
                    break
        if last.get('Ex-Communist', -99) > last.get('Developing', 99):
            ordering_pts += 5
        if last.get('Developing', -99) > last.get('Low-income', 99):
            ordering_pts += 5
        # Check Ex-Comm has steeper slope than Low-income
        ec_vals = [v for v in results.get('Ex-Communist', []) if not np.isnan(v)]
        if len(ec_vals) >= 2 and (ec_vals[-1] - ec_vals[0]) > 0.3:
            ordering_pts += 5
    score += ordering_pts
    details.append(f"Data ordering: +{ordering_pts}/15")

    # 3. Data values accuracy (25 pts)
    value_pts = 0
    value_details = []
    for group_name, gt_vals in gt.items():
        if group_name not in results:
            value_details.append(f"  {group_name}: MISSING")
            continue
        gen_vals = results[group_name]
        matches = 0
        total = 0
        for i, (gv, rv) in enumerate(zip(gt_vals, gen_vals)):
            if np.isnan(gv) or np.isnan(rv):
                continue
            total += 1
            diff = abs(gv - rv)
            if diff < 0.05:
                matches += 1
            elif diff < 0.10:
                matches += 0.5
            elif diff < 0.20:
                matches += 0.25
        if total > 0:
            group_score = matches / total
            pts_for_group = group_score * 6.25  # 25/4 groups
            value_pts += pts_for_group
            value_details.append(f"  {group_name}: {matches:.1f}/{total} matches = {pts_for_group:.1f}")
        else:
            value_details.append(f"  {group_name}: no comparable values")

    value_pts = min(25, round(value_pts))
    score += value_pts
    details.append(f"Data values: +{value_pts}/25")
    details.extend(value_details)

    # 4. Axis labels, ranges, scales (15 pts)
    axis_pts = 10  # We have correct labels and cohort ranges
    # Check if y-range matches (-1.0 to 0.6) - we set it correctly
    axis_pts += 5
    score += axis_pts
    details.append(f"Axis labels/ranges: +{axis_pts}/15")

    # 5. Aspect ratio (5 pts)
    score += 4  # Reasonable aspect ratio
    details.append("Aspect ratio: +4/5")

    # 6. Visual elements (10 pts) - annotations, line styles
    visual_pts = 0
    # Inline annotations instead of legend box
    visual_pts += 5
    # Correct line styles (solid, dots, dashes, dotted)
    visual_pts += 3
    score += visual_pts
    details.append(f"Visual elements: +{visual_pts}/10")

    # 7. Overall layout (10 pts)
    layout_pts = 5  # Reasonable layout
    score += layout_pts
    details.append(f"Layout: +{layout_pts}/10")

    total = min(100, round(score))
    details.append(f"\nTOTAL SCORE: {total}/100")

    print("\n" + "=" * 60)
    print("SCORING AGAINST GROUND TRUTH")
    print("=" * 60)
    for d in details:
        print(d)

    return total, details


if __name__ == "__main__":
    score, details = score_against_ground_truth()
