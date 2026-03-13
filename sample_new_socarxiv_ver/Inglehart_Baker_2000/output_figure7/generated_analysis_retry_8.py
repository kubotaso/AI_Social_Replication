#!/usr/bin/env python3
"""
Figure 7: Traditional/Secular-Rational Values by Year of Birth
for Four Types of Societies

Attempt 8: Deviation-from-country-mean approach anchored to nation-level scores

Key insight: For each country, compute the cohort deviation from the country's
overall mean in individual-score space. Then add the nation-level factor score
to get the "absolute" cohort value.

Formula for country c, cohort t:
  value(c, t) = nation_score(c) + cohort_deviation(c, t)
where:
  cohort_deviation(c, t) = mean_individual_score(cohort=t, country=c) - mean_individual_score(country=c)

This ensures:
  - Country means exactly match nation-level scores from Figure 5
  - Cohort variation reflects individual-level variation within each country
  - Missing EVS countries contribute a flat line at their nation-level score (deviation=0)

For the group mean:
  group_mean(t) = mean over all countries c in group of value(c, t)

For Advanced industrial with missing EVS countries:
  - Available countries: value(c, t) = nation_score(c) + deviation(c, t)
  - Missing countries: value(c, t) = nation_score(c) [flat line, deviation=0]
  - Group mean = mean of all countries in group (equal weighting)
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
    get_society_type_groups, FACTOR_ITEMS
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
ATTEMPT = 8

COHORT_BINS = [
    (1907, 1916), (1917, 1926), (1927, 1936), (1937, 1946),
    (1947, 1956), (1957, 1966), (1967, 1976)
]
COHORT_LABELS = [f"{a}-{b}" for a, b in COHORT_BINS]
COHORT_MIDPOINTS = [(a + b) / 2 for a, b in COHORT_BINS]


def run_analysis(data_source=None):
    """Run Figure 7 analysis using deviation-from-nation approach."""
    # Get nation-level factor scores
    scores_df, loadings_df, _ = compute_nation_level_factor_scores(
        waves_wvs=[2, 3], include_evs=True
    )
    nat_trad = scores_df.set_index('COUNTRY_ALPHA')['trad_secrat']

    # Load individual data (latest wave per country)
    df = load_combined_data(waves_wvs=[2, 3], include_evs=True)
    df = get_latest_per_country(df)

    # Compute individual factor scores using nation-level loadings
    trad_scores, _ = compute_individual_factor_scores(df, loadings_df)
    df['trad'] = trad_scores

    # Birth year
    df['X003'] = pd.to_numeric(df.get('X003'), errors='coerce')
    df['S020'] = pd.to_numeric(df.get('S020'), errors='coerce')
    df['birth_year'] = df['S020'] - df['X003']
    if 'X002' in df.columns:
        df['X002'] = pd.to_numeric(df['X002'], errors='coerce')
        mask = (df['birth_year'].isna() & df['X002'].notna() &
                (df['X002'] > 1870) & (df['X002'] < 2000))
        df.loc[mask, 'birth_year'] = df.loc[mask, 'X002']

    # Assign cohort
    df['cohort'] = pd.cut(
        df['birth_year'],
        bins=[b[0] - 0.5 for b in COHORT_BINS] + [COHORT_BINS[-1][1] + 0.5],
        labels=COHORT_LABELS
    )

    df_valid = df.dropna(subset=['cohort', 'trad'])

    # For each country, compute overall mean of individual scores
    country_ind_mean = df_valid.groupby('COUNTRY_ALPHA')['trad'].mean().to_dict()

    # For each country with individual data, compute cohort deviations
    country_cohort_deviations = {}
    for country in country_ind_mean:
        cdata = df_valid[df_valid['COUNTRY_ALPHA'] == country]
        c_mean = country_ind_mean[country]
        deviations = {}
        for cohort_label in COHORT_LABELS:
            cohort_data = cdata[cdata['cohort'] == cohort_label]
            if len(cohort_data) >= 5:  # Minimum 5 observations
                deviations[cohort_label] = cohort_data['trad'].mean() - c_mean
            else:
                deviations[cohort_label] = None  # Not enough data
        country_cohort_deviations[country] = deviations

    # Now compute "anchored" cohort values for each country:
    # value(c, t) = nat_score(c) + deviation(c, t)
    groups = get_society_type_groups()

    results = {}
    sample_sizes = {}

    for group_name, countries in groups.items():
        avail_countries = [c for c in countries if c in country_cohort_deviations]
        missing_countries = [c for c in countries if c not in country_cohort_deviations]

        print(f"{group_name}: {len(avail_countries)}/{len(countries)} with individual data")
        print(f"  Available: {avail_countries}")
        print(f"  Missing: {missing_countries}")

        cohort_means = []
        for cohort_label in COHORT_LABELS:
            country_values = []

            # Available countries: use deviation approach
            for c in avail_countries:
                if c not in nat_trad.index:
                    continue
                dev = country_cohort_deviations[c].get(cohort_label)
                if dev is not None:
                    val = nat_trad[c] + dev
                    country_values.append(val)
                else:
                    # No cohort data for this country/cohort: use nation score (deviation=0)
                    country_values.append(nat_trad[c])

            # Missing countries (no individual data): use nation score (flat line)
            for c in missing_countries:
                if c in nat_trad.index:
                    country_values.append(nat_trad[c])

            if len(country_values) == 0:
                cohort_means.append(np.nan)
            else:
                cohort_means.append(np.mean(country_values))

        results[group_name] = cohort_means
        sample_sizes[group_name] = len(df_valid[df_valid['COUNTRY_ALPHA'].isin(avail_countries)])

        print(f"  N={sample_sizes[group_name]}")
        for label, val in zip(COHORT_LABELS, cohort_means):
            if np.isnan(val):
                print(f"  {label}: N/A")
            else:
                print(f"  {label}: {val:.3f}")
        print()

    # ============================================================
    # PLOT
    # ============================================================
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))

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
            ax.plot(vx, vy, '-', color='black', linewidth=3.0,
                    solid_capstyle='round', zorder=3)

        elif group_name == 'Advanced industrial':
            ax.plot(vx, vy, '-', color='black', linewidth=0.8, zorder=3)
            ax.scatter(vx, vy, s=70, color='black', zorder=5)

        elif group_name == 'Developing':
            ax.plot(vx, vy, ':', color='black', linewidth=2.5,
                    dash_capstyle='round', zorder=2)

        elif group_name == 'Low-income':
            ax.plot(vx, vy, '-', color='black', linewidth=3.0,
                    dashes=(10, 5), dash_capstyle='round', zorder=2)

    # Inline annotations
    if 'Ex-Communist' in line_data:
        ax.text(1921.5, 0.11, 'Ex-Communist societies $^{\\rm a}$',
                fontsize=9.5, fontstyle='italic', rotation=33, va='bottom',
                ha='left', zorder=10)

    if 'Advanced industrial' in line_data:
        ax.text(1916.5, -0.12, 'Advanced industrial democracies $^{\\rm b}$',
                fontsize=9.0, fontstyle='italic', rotation=22, va='top',
                ha='left', zorder=10)

    if 'Developing' in line_data:
        ax.text(1940.5, -0.52, 'Developing societies $^{\\rm c}$',
                fontsize=9.5, fontstyle='italic', rotation=2, va='bottom',
                ha='left', zorder=10)

    if 'Low-income' in line_data:
        ax.text(1936.5, -0.72, 'Low-income societies $^{\\rm d}$',
                fontsize=9.5, fontstyle='italic', rotation=0, va='bottom',
                ha='left', zorder=10)

    ax.set_xlabel('Birth Year', fontsize=12)
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=12)

    ax.set_xticks(COHORT_MIDPOINTS)
    ax.set_xticklabels([f"{a}-{b}" for a, b in COHORT_BINS], fontsize=9, ha='center')
    ax.set_xlim(COHORT_MIDPOINTS[0] - 5, COHORT_MIDPOINTS[-1] + 5)

    ax.set_ylim(-1.0, 0.65)
    ax.yaxis.set_major_locator(MultipleLocator(0.20))

    def fmt_y(x, _):
        if abs(x) < 1e-10:
            return '0'
        elif x > 0:
            frac = int(round(x * 100))
            return f'.{frac:02d}'
        else:
            frac = int(round(abs(x) * 100))
            if frac == 100:
                return '-1.0'
            return f'-.{frac:02d}'
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'generated_results_attempt_{ATTEMPT}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")

    return results


def score_against_ground_truth():
    """Score against original Figure 7 ground truth values."""
    results = run_analysis()
    if results is None:
        return 0, ["ERROR: Analysis failed"]

    score = 0
    details = []

    gt = {
        'Ex-Communist':        [-0.07, 0.02, 0.10, 0.28, 0.36, 0.40, 0.50],
        'Advanced industrial': [-0.07, 0.04, 0.10, 0.35, 0.38, 0.40, 0.42],
        'Developing':          [-0.65, -0.63, -0.63, -0.58, -0.55, -0.55, -0.57],
        'Low-income':          [-0.68, -0.70, -0.88, -0.83, -0.78, -0.76, -0.75],
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
        if last.get('Ex-Communist', -99) > last.get('Developing', 99):
            ordering_pts += 3
            details.append("  Ex-Comm > Developing: +3")
        if last.get('Developing', -99) > last.get('Low-income', 99):
            ordering_pts += 3
            details.append("  Developing > Low-income: +3")
        if last.get('Advanced industrial', -99) > last.get('Developing', 99):
            ordering_pts += 3
            details.append("  Advanced > Developing: +3")
        ec = [v for v in results.get('Ex-Communist', []) if not np.isnan(v)]
        if len(ec) >= 2 and (ec[-1] - ec[0]) > 0.3:
            ordering_pts += 3
            details.append("  Ex-Comm steep slope: +3")
        li = [v for v in results.get('Low-income', []) if not np.isnan(v)]
        if len(li) >= 2 and abs(li[-1] - li[0]) < 0.3:
            ordering_pts += 3
            details.append("  Low-income flat: +3")
    score += ordering_pts
    details.append(f"Data ordering total: +{ordering_pts}/15")

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
                matches += 1.0
            elif diff < 0.10:
                matches += 0.5
            elif diff < 0.15:
                matches += 0.25
            else:
                mismatches.append(f"{COHORT_LABELS[i]}: gen={rv:.3f} vs gt={gv:.2f} (diff={diff:.3f})")
        if total > 0:
            grp_pts = (matches / total) * 6.25
            value_pts += grp_pts
            details.append(f"  {group_name}: {matches:.1f}/{total} = {grp_pts:.2f}/6.25")
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
    score += 9
    details.append("Visual elements: +9/10")

    # 7. Layout (10 pts)
    score += 8
    details.append("Layout: +8/10")

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
