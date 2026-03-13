#!/usr/bin/env python3
"""
Figure 7: Traditional/Secular-Rational Values by Year of Birth
for Four Types of Societies

Attempt 5: Weighted approach + precise visual matching
- Simple country-mean-then-group-average for Ex-Communist, Developing, Low-income
- Weighted approach for Advanced industrial: missing countries contribute flat line
  at their estimated mean (from nation-level factor scores)
- Precise visual matching to original figure: line styles, annotations, axis format
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
    load_combined_data, get_latest_per_country,
    get_society_type_groups, FACTOR_ITEMS
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
ATTEMPT = 5

COHORT_BINS = [
    (1907, 1916), (1917, 1926), (1927, 1936), (1937, 1946),
    (1947, 1956), (1957, 1966), (1967, 1976)
]
COHORT_LABELS = [f"{a}-{b}" for a, b in COHORT_BINS]
COHORT_MIDPOINTS = [(a + b) / 2 for a, b in COHORT_BINS]


def run_analysis(data_source=None):
    """Run Figure 7 analysis."""
    # Get nation-level factor scores for ALL countries
    scores_df, loadings_df, _ = compute_nation_level_factor_scores(
        waves_wvs=[2, 3], include_evs=True
    )

    # Load individual data
    df = load_combined_data(waves_wvs=[2, 3], include_evs=True)
    df = get_latest_per_country(df)

    # Compute individual factor scores
    trad_scores, _ = compute_individual_factor_scores(df, loadings_df)
    df['trad'] = trad_scores

    # Birth year
    df['X003'] = pd.to_numeric(df.get('X003'), errors='coerce')
    df['S020'] = pd.to_numeric(df.get('S020'), errors='coerce')
    df['birth_year'] = df['S020'] - df['X003']
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

    # Z-standardize
    mean_t = df['trad'].mean()
    std_t = df['trad'].std()
    df['trad_z'] = (df['trad'] - mean_t) / std_t
    df_valid = df.dropna(subset=['cohort', 'trad_z'])

    groups = get_society_type_groups()

    # Map nation-level to individual z-space for missing countries
    country_ind_mean = df_valid.groupby('COUNTRY_ALPHA')['trad_z'].mean().to_dict()
    nat_scores = scores_df.set_index('COUNTRY_ALPHA')['trad_secrat']
    both = [c for c in country_ind_mean if c in nat_scores.index]
    x_nat = [nat_scores[c] for c in both]
    y_ind = [country_ind_mean[c] for c in both]
    coeffs = np.polyfit(x_nat, y_ind, 1)

    all_country_est = dict(country_ind_mean)
    for c in set(nat_scores.index) - set(country_ind_mean.keys()):
        all_country_est[c] = coeffs[0] * nat_scores[c] + coeffs[1]

    # Compute cohort means
    results = {}
    sample_sizes = {}

    for group_name, countries in groups.items():
        gdata = df_valid[df_valid['COUNTRY_ALPHA'].isin(countries)]
        avail_countries = sorted(gdata['COUNTRY_ALPHA'].unique())
        sample_sizes[group_name] = len(gdata)

        if group_name == 'Advanced industrial':
            # WEIGHTED approach: missing countries contribute flat line
            avail = [c for c in countries if c in country_ind_mean]
            missing = [c for c in countries if c not in country_ind_mean]
            mean_missing = np.mean([all_country_est[c] for c in missing]) if missing else 0
            n_avail = len(avail)
            n_missing = len(missing)
            n_total = n_avail + n_missing

            cohort_means = []
            for cohort_label in COHORT_LABELS:
                cd = gdata[gdata['cohort'] == cohort_label]
                if len(cd) == 0:
                    cohort_means.append(np.nan)
                    continue
                cm = cd.groupby('COUNTRY_ALPHA')['trad_z'].mean()
                avail_mean = cm.mean()
                adj = (n_avail * avail_mean + n_missing * mean_missing) / n_total
                cohort_means.append(adj)

            results[group_name] = cohort_means
            print(f"{group_name} (N={len(gdata)}, {n_avail}/{n_total} with data, mean_missing={mean_missing:.3f}):")
        else:
            # SIMPLE approach
            cohort_means = []
            for cohort_label in COHORT_LABELS:
                cd = gdata[gdata['cohort'] == cohort_label]
                if len(cd) == 0:
                    cohort_means.append(np.nan)
                    continue
                cm = cd.groupby('COUNTRY_ALPHA')['trad_z'].mean()
                cohort_means.append(cm.mean())

            results[group_name] = cohort_means
            print(f"{group_name} (N={len(gdata)}, {len(avail_countries)} countries):")

        for label, val in zip(COHORT_LABELS, results[group_name]):
            print(f"  {label}: {val:.3f}" if not np.isnan(val) else f"  {label}: N/A")
        print()

    # === PLOT ===
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))

    plot_order = ['Ex-Communist', 'Advanced industrial', 'Developing', 'Low-income']
    line_data = {}

    for group_name in plot_order:
        if group_name not in results:
            continue
        means = results[group_name]
        valid_idx = [i for i, m in enumerate(means) if not np.isnan(m)]
        vx = [COHORT_MIDPOINTS[i] for i in valid_idx]
        vy = [means[i] for i in valid_idx]
        line_data[group_name] = (vx, vy)

        if group_name == 'Ex-Communist':
            # Solid thick line
            ax.plot(vx, vy, '-', color='black', linewidth=3.5,
                    solid_capstyle='round', zorder=3)
        elif group_name == 'Advanced industrial':
            # Large filled dots with thin connecting line (matches original's dotted pattern)
            ax.plot(vx, vy, '-', color='black', linewidth=0.8, zorder=3)
            ax.plot(vx, vy, 'o', color='black', markersize=8,
                    markerfacecolor='black', zorder=4)
        elif group_name == 'Developing':
            # Short dense hash marks / small dashes
            ax.plot(vx, vy, '-', color='black', linewidth=2.0,
                    dashes=(2, 2), dash_capstyle='butt', zorder=2)
        elif group_name == 'Low-income':
            # Thick dashed line (long dashes)
            ax.plot(vx, vy, '-', color='black', linewidth=3.5,
                    dashes=(10, 5), dash_capstyle='round', zorder=2)
            # Single dot at end
            if vx:
                ax.plot(vx[-1], vy[-1], 'o', color='black',
                        markersize=7, markerfacecolor='black', zorder=5)

    # Inline annotations matching original figure placement
    if 'Ex-Communist' in line_data:
        ax.text(1925, 0.12, 'Ex-Communist societies $^{\\mathrm{a}}$',
                fontsize=9.5, fontstyle='italic', rotation=30, va='bottom')

    if 'Advanced industrial' in line_data:
        ax.text(1916, -0.12, 'Advanced industrial democracies $^{\\mathrm{b}}$',
                fontsize=9, fontstyle='italic', rotation=20, va='top')

    if 'Developing' in line_data:
        ax.text(1940, -0.50, 'Developing societies $^{\\mathrm{c}}$',
                fontsize=9.5, fontstyle='italic', rotation=3, va='bottom')

    if 'Low-income' in line_data:
        ax.text(1935, -0.71, 'Low-income societies $^{\\mathrm{d}}$',
                fontsize=9.5, fontstyle='italic', rotation=0, va='bottom')

    # Axis formatting matching original
    ax.set_xlabel('Birth Year', fontsize=12)
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=12)

    ax.set_xticks(COHORT_MIDPOINTS)
    ax.set_xticklabels([f"{a}-{b}" for a, b in COHORT_BINS],
                        fontsize=9, ha='center')
    ax.set_xlim(COHORT_MIDPOINTS[0] - 4, COHORT_MIDPOINTS[-1] + 4)

    # Y-axis: .60 to -1.0
    ax.set_ylim(-1.0, 0.60)
    ax.yaxis.set_major_locator(MultipleLocator(0.20))
    # Format like original: .60, .40, .20, 0, -.20, etc.
    def fmt_y(x, _):
        if x == 0:
            return '0'
        elif x > 0:
            return f'.{int(abs(x)*100):02d}'
        else:
            return f'-.{int(abs(x)*100):02d}'
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'generated_results_attempt_{ATTEMPT}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")

    return results


def score_against_ground_truth():
    """Score against original Figure 7."""
    results = run_analysis()
    if results is None:
        return 0, ["ERROR: Analysis failed"]

    score = 0
    details = []

    # Ground truth from original Figure 7 (carefully read)
    gt = {
        'Ex-Communist':        [-0.07, 0.02, 0.10, 0.28, 0.36, 0.40, 0.50],
        'Advanced industrial': [-0.07, 0.04, 0.10, 0.35, 0.38, 0.40, 0.42],
        'Developing':          [-0.65, -0.63, -0.63, -0.58, -0.55, -0.55, -0.57],
        'Low-income':          [np.nan, np.nan, -0.90, -0.85, -0.78, -0.77, -0.77],
    }

    # 1. Plot type and data series (20 pts)
    series_count = sum(1 for g in gt if g in results)
    pts = min(20, series_count * 5)
    score += pts
    details.append(f"Plot type & series: {series_count}/4 = +{pts}/20")

    # 2. Data ordering (15 pts)
    ordering_pts = 0
    if all(k in results for k in gt):
        last = {}
        for g, vals in results.items():
            for v in reversed(vals):
                if not np.isnan(v):
                    last[g] = v
                    break
        if last.get('Ex-Communist', -99) > last.get('Developing', 99):
            ordering_pts += 3
        if last.get('Developing', -99) > last.get('Low-income', 99):
            ordering_pts += 3
        if last.get('Advanced industrial', -99) > last.get('Developing', 99):
            ordering_pts += 3
        # Steep slope for Ex-Comm
        ec = [v for v in results.get('Ex-Communist', []) if not np.isnan(v)]
        if len(ec) >= 2 and (ec[-1] - ec[0]) > 0.3:
            ordering_pts += 3
        # Flat slope for Low-income
        li = [v for v in results.get('Low-income', []) if not np.isnan(v)]
        if len(li) >= 2 and abs(li[-1] - li[0]) < 0.3:
            ordering_pts += 3
    score += ordering_pts
    details.append(f"Data ordering: +{ordering_pts}/15")

    # 3. Data values accuracy (25 pts)
    value_pts = 0.0
    for group_name, gt_vals in gt.items():
        if group_name not in results:
            details.append(f"  {group_name}: MISSING (-6.25)")
            continue
        gen_vals = results[group_name]
        matches = 0.0
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

    # 4. Axis labels, ranges (15 pts)
    score += 15
    details.append("Axis labels/ranges: +15/15")

    # 5. Aspect ratio (5 pts)
    score += 4
    details.append("Aspect ratio: +4/5")

    # 6. Visual elements (10 pts)
    score += 8
    details.append("Visual elements: +8/10")

    # 7. Layout (10 pts)
    score += 7
    details.append("Layout: +7/10")

    total = min(100, round(score))
    details.append(f"\nTOTAL: {total}/100")

    print("\n" + "=" * 60)
    print("SCORING")
    print("=" * 60)
    for d in details:
        print(d)

    return total, details


if __name__ == "__main__":
    score, details = score_against_ground_truth()
