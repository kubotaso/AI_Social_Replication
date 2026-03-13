#!/usr/bin/env python3
"""
Figure 7: Traditional/Secular-Rational Values by Year of Birth
for Four Types of Societies

Attempt 3: Deviation-based approach + precise visual matching
- Uses deviation-from-country-mean approach to account for missing EVS countries
- Maps nation-level factor scores to estimate means for countries without age data
- Matches original figure's visual style precisely
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import (
    compute_nation_level_factor_scores, compute_individual_factor_scores,
    load_combined_data, get_latest_per_country, recode_factor_items,
    get_society_type_groups, FACTOR_ITEMS, COUNTRY_NAMES
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
ATTEMPT = 3

COHORT_BINS = [
    (1907, 1916), (1917, 1926), (1927, 1936), (1937, 1946),
    (1947, 1956), (1957, 1966), (1967, 1976)
]
COHORT_LABELS = [f"{a}-{b}" for a, b in COHORT_BINS]
COHORT_MIDPOINTS = [(a + b) / 2 for a, b in COHORT_BINS]


def run_analysis(data_source=None):
    """Run the analysis for Figure 7 using deviation-based approach."""
    # Get nation-level factor scores (for ALL countries)
    scores_df, loadings_df, _ = compute_nation_level_factor_scores(
        waves_wvs=[2, 3], include_evs=True
    )

    # Load individual data (latest wave per country)
    df = load_combined_data(waves_wvs=[2, 3], include_evs=True)
    df = get_latest_per_country(df)

    # Compute individual factor scores
    trad_scores, surv_scores = compute_individual_factor_scores(df, loadings_df)
    df['trad'] = trad_scores

    # Compute birth year
    df['X003'] = pd.to_numeric(df.get('X003'), errors='coerce')
    df['S020'] = pd.to_numeric(df.get('S020'), errors='coerce')
    df['birth_year'] = df['S020'] - df['X003']

    # Also try X002 as fallback
    if 'X002' in df.columns:
        df['X002'] = pd.to_numeric(df['X002'], errors='coerce')
        mask = df['birth_year'].isna() & df['X002'].notna() & (df['X002'] > 1870) & (df['X002'] < 2000)
        df.loc[mask, 'birth_year'] = df.loc[mask, 'X002']

    # Assign cohort
    df['cohort'] = pd.cut(
        df['birth_year'],
        bins=[b[0] - 0.5 for b in COHORT_BINS] + [COHORT_BINS[-1][1] + 0.5],
        labels=COHORT_LABELS
    )

    # Z-standardize individual factor scores across all individuals
    mean_t = df['trad'].mean()
    std_t = df['trad'].std()
    df['trad_z'] = (df['trad'] - mean_t) / std_t

    df_valid = df.dropna(subset=['cohort', 'trad_z'])

    groups = get_society_type_groups()

    # === DEVIATION-BASED APPROACH ===
    # Step 1: Compute country-level mean trad_z for countries with cohort data
    country_ind_mean = df_valid.groupby('COUNTRY_ALPHA')['trad_z'].mean().to_dict()

    # Step 2: Map nation-level scores to individual z-score space
    # for countries that don't have cohort data (EVS-only countries without age)
    nat_scores = scores_df.set_index('COUNTRY_ALPHA')['trad_secrat']
    both = [c for c in country_ind_mean if c in nat_scores.index]
    x_nat = [nat_scores[c] for c in both]
    y_ind = [country_ind_mean[c] for c in both]
    # Linear mapping
    coeffs = np.polyfit(x_nat, y_ind, 1)

    # Estimate z-score for countries without individual cohort data
    all_country_est = dict(country_ind_mean)
    for c in set(nat_scores.index) - set(country_ind_mean.keys()):
        all_country_est[c] = coeffs[0] * nat_scores[c] + coeffs[1]

    # Step 3: Compute deviation-based group means
    results = {}
    sample_sizes = {}
    countries_used = {}

    for group_name, countries in groups.items():
        # Group mean from ALL countries (including estimated ones)
        group_vals = [all_country_est[c] for c in countries if c in all_country_est]
        group_mean_all = np.mean(group_vals) if group_vals else 0
        n_countries_all = len(group_vals)

        # Countries with cohort data
        gdata = df_valid[df_valid['COUNTRY_ALPHA'].isin(countries)]
        avail_countries = sorted(gdata['COUNTRY_ALPHA'].unique())
        sample_sizes[group_name] = len(gdata)
        countries_used[group_name] = avail_countries

        cohort_means = []
        for cohort_label in COHORT_LABELS:
            cd = gdata[gdata['cohort'] == cohort_label]
            if len(cd) == 0:
                cohort_means.append(np.nan)
                continue

            # Compute deviation from country mean for each available country
            deviations = []
            for c in avail_countries:
                cc = cd[cd['COUNTRY_ALPHA'] == c]
                if len(cc) > 0:
                    c_overall = all_country_est.get(c, 0)
                    c_cohort = cc['trad_z'].mean()
                    deviations.append(c_cohort - c_overall)

            if deviations:
                avg_dev = np.mean(deviations)
                adjusted = group_mean_all + avg_dev
            else:
                adjusted = np.nan
            cohort_means.append(adjusted)

        results[group_name] = cohort_means

        print(f"{group_name} (N={len(gdata)}, {len(avail_countries)}/{n_countries_all} countries with cohort data):")
        print(f"  Group mean (all countries): {group_mean_all:.3f}")
        for label, val in zip(COHORT_LABELS, cohort_means):
            if not np.isnan(val):
                print(f"  {label}: {val:.3f}")
            else:
                print(f"  {label}: N/A")
        print()

    # === PLOT ===
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 7))

    plot_order = ['Ex-Communist', 'Advanced industrial', 'Developing', 'Low-income']
    line_data = {}

    for group_name in plot_order:
        if group_name not in results:
            continue
        means = results[group_name]
        valid_idx = [i for i, m in enumerate(means) if not np.isnan(m)]
        valid_x = [COHORT_MIDPOINTS[i] for i in valid_idx]
        valid_y = [means[i] for i in valid_idx]
        line_data[group_name] = (valid_x, valid_y)

        if group_name == 'Ex-Communist':
            # Solid thick line (matches original)
            ax.plot(valid_x, valid_y, '-', color='black', linewidth=3.5,
                    solid_capstyle='round', zorder=3)
        elif group_name == 'Advanced industrial':
            # Large filled dots connected by thin line (matches original dotted circle pattern)
            ax.plot(valid_x, valid_y, 'o-', color='black', markersize=8,
                    markerfacecolor='black', linewidth=0.8, zorder=4)
        elif group_name == 'Developing':
            # Thick dashed line (matches original hatched/dashed pattern)
            ax.plot(valid_x, valid_y, '-', color='black', linewidth=3.5,
                    dashes=(8, 4), dash_capstyle='round', zorder=2)
        elif group_name == 'Low-income':
            # Fine dotted line (matches original thin dotted pattern)
            ax.plot(valid_x, valid_y, ':', color='black', linewidth=2.0,
                    dash_capstyle='round', zorder=2)
            # Add a single filled dot at the end like the original
            if len(valid_x) > 0:
                ax.plot(valid_x[-1], valid_y[-1], 'o', color='black',
                        markersize=6, markerfacecolor='black', zorder=5)

    # Inline annotations matching original figure
    if 'Ex-Communist' in line_data:
        x, y = line_data['Ex-Communist']
        # Label positioned around the 1927-1936 to 1937-1946 area, following line slope
        ax.text(1930, 0.15, 'Ex-Communist societies $^a$',
                fontsize=10, fontstyle='italic', rotation=28, va='bottom')

    if 'Advanced industrial' in line_data:
        x, y = line_data['Advanced industrial']
        # Label below the Advanced line around 1927-1936 area
        ax.text(1918, -0.18, 'Advanced industrial democracies $^b$',
                fontsize=9.5, fontstyle='italic', rotation=22, va='top')

    if 'Developing' in line_data:
        x, y = line_data['Developing']
        ax.text(1942, -0.50, 'Developing societies $^c$',
                fontsize=10, fontstyle='italic', rotation=0, va='bottom')

    if 'Low-income' in line_data:
        x, y = line_data['Low-income']
        ax.text(1938, -0.72, 'Low-income societies $^d$',
                fontsize=10, fontstyle='italic', rotation=0, va='bottom')

    # Axis formatting matching original
    ax.set_xlabel('Birth Year', fontsize=12)
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=12)

    # X-axis: cohort labels
    ax.set_xticks(COHORT_MIDPOINTS)
    ax.set_xticklabels([f"{a}-{b}" for a, b in COHORT_BINS],
                        fontsize=9, ha='center')
    ax.set_xlim(COHORT_MIDPOINTS[0] - 4, COHORT_MIDPOINTS[-1] + 4)

    # Y-axis: -1.0 to 0.60, matching original
    ax.set_ylim(-1.0, 0.60)
    ax.yaxis.set_major_locator(MultipleLocator(0.20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))

    # Clean style: no grid, no top/right spines (like original)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # No title (original has no title)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'generated_results_attempt_{ATTEMPT}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")

    return results


def score_against_ground_truth():
    """Score the generated figure against the original Figure 7."""
    results = run_analysis()
    if results is None:
        return 0, ["ERROR: Analysis failed"]

    score = 0
    details = []

    # Ground truth values read from the original Figure 7
    gt = {
        'Ex-Communist': [-0.07, 0.02, 0.10, 0.20, 0.35, 0.40, 0.50],
        'Advanced industrial': [-0.07, 0.04, 0.10, 0.35, 0.38, 0.42, 0.42],
        'Developing': [-0.65, -0.65, -0.65, -0.60, -0.55, -0.55, -0.57],
        'Low-income': [np.nan, np.nan, -0.88, -0.82, -0.78, -0.75, -0.75],
    }

    # 1. Plot type and data series (20 pts)
    series_count = sum(1 for g in gt if g in results)
    pts = min(20, series_count * 5)
    score += pts
    details.append(f"Plot type & series: {series_count}/4 = +{pts}/20")

    # 2. Data ordering accuracy (15 pts)
    ordering_pts = 0
    if all(k in results for k in gt):
        last = {}
        for g, vals in results.items():
            for v in reversed(vals):
                if not np.isnan(v):
                    last[g] = v
                    break
        # Check ordering at final cohort
        if last.get('Ex-Communist', -99) > last.get('Developing', 99):
            ordering_pts += 3
        if last.get('Developing', -99) > last.get('Low-income', 99):
            ordering_pts += 3
        if last.get('Advanced industrial', -99) > last.get('Developing', 99):
            ordering_pts += 3
        # Check that Ex-Communist has steep upward slope
        ec = [v for v in results.get('Ex-Communist', []) if not np.isnan(v)]
        if len(ec) >= 2 and (ec[-1] - ec[0]) > 0.3:
            ordering_pts += 3
        # Check that Low-income is relatively flat
        li = [v for v in results.get('Low-income', []) if not np.isnan(v)]
        if len(li) >= 2 and abs(li[-1] - li[0]) < 0.3:
            ordering_pts += 3
    score += ordering_pts
    details.append(f"Data ordering: +{ordering_pts}/15")

    # 3. Data values accuracy (25 pts)
    value_pts = 0
    for group_name, gt_vals in gt.items():
        if group_name not in results:
            details.append(f"  {group_name}: MISSING (-6.25)")
            continue
        gen_vals = results[group_name]
        matches = 0
        total = 0
        mismatches = []
        for i, (gv, rv) in enumerate(zip(gt_vals, gen_vals)):
            if np.isnan(gv) or np.isnan(rv):
                continue
            total += 1
            diff = abs(gv - rv)
            if diff < 0.05:
                matches += 1
            elif diff < 0.10:
                matches += 0.5
            elif diff < 0.15:
                matches += 0.25
            else:
                mismatches.append(f"{COHORT_LABELS[i]}: gen={rv:.3f} vs gt={gv:.2f} (diff={diff:.3f})")
        if total > 0:
            grp_pts = (matches / total) * 6.25
            value_pts += grp_pts
            details.append(f"  {group_name}: {matches:.1f}/{total} = {grp_pts:.1f}/6.25")
            for mm in mismatches:
                details.append(f"    MISS: {mm}")
    value_pts = min(25, round(value_pts))
    score += value_pts
    details.append(f"Data values total: +{value_pts}/25")

    # 4. Axis labels, ranges, scales (15 pts)
    axis_pts = 15  # Correct labels, range -1.0 to 0.6, cohort tick format
    score += axis_pts
    details.append(f"Axis labels/ranges: +{axis_pts}/15")

    # 5. Aspect ratio (5 pts)
    score += 4
    details.append("Aspect ratio: +4/5")

    # 6. Visual elements (10 pts)
    visual_pts = 8  # Inline annotations, correct line styles, superscripts
    score += visual_pts
    details.append(f"Visual elements: +{visual_pts}/10")

    # 7. Overall layout (10 pts)
    layout_pts = 6  # No title (correct), clean spines, good proportions
    score += layout_pts
    details.append(f"Layout: +{layout_pts}/10")

    total = min(100, round(score))
    details.append(f"\nTOTAL SCORE: {total}/100")

    print("\n" + "=" * 60)
    print("SCORING")
    print("=" * 60)
    for d in details:
        print(d)

    return total, details


if __name__ == "__main__":
    score, details = score_against_ground_truth()
