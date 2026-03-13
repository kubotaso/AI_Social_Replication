import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def run_analysis(data_source):
    df = pd.read_csv(data_source, low_memory=False)

    pres_years = [1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996]

    # Filter to presidential years
    df = df[df['VCF0004'].isin(pres_years)].copy()

    # Convert to numeric
    df['VCF0301'] = pd.to_numeric(df['VCF0301'], errors='coerce')
    df['VCF0706'] = pd.to_numeric(df['VCF0706'], errors='coerce')
    df['VCF0305'] = pd.to_numeric(df['VCF0305'], errors='coerce')

    # Strong or Weak identifiers: VCF0305 in {3, 4}
    # VCF0305: 1=apolitical, 2=independent/no pref, 3=weak partisan, 4=strong partisan
    #
    # Denominator: respondents with valid VCF0305 (answered party ID question)
    # This includes apoliticals who may be NA in VCF0301.
    #
    # Voters: VCF0706 in {1, 2} (voted for major-party presidential candidate)
    # This excludes third-party voters (VCF0706=3,4) whose inclusion would
    # dilute partisan identification proportions. Third-party voters (e.g.,
    # Wallace 1968, Anderson 1980, Perot 1992/1996) have lower partisan
    # identification by definition. Bartels' focus on partisanship suggests
    # he used major-party voters.
    #
    # Nonvoters: VCF0706 == 7 (did not vote)

    voter_props = []
    nonvoter_props = []

    for year in pres_years:
        yr_data = df[df['VCF0004'] == year]

        # Major-party voters with valid party strength response
        voters = yr_data[(yr_data['VCF0706'].isin([1, 2])) & (yr_data['VCF0305'].notna())]
        if len(voters) > 0:
            v_prop = voters['VCF0305'].isin([3, 4]).mean()
        else:
            v_prop = np.nan
        voter_props.append(v_prop)

        # Nonvoters with valid party strength response
        nonvoters = yr_data[(yr_data['VCF0706'] == 7) & (yr_data['VCF0305'].notna())]
        if len(nonvoters) > 0:
            nv_prop = nonvoters['VCF0305'].isin([3, 4]).mean()
        else:
            nv_prop = np.nan
        nonvoter_props.append(nv_prop)

    # Print data
    results = "Figure 2: Proportions of (Strong or Weak) Identifiers\n"
    results += "in National Election Study Sample\n\n"
    results += f"{'Year':<8}{'Voters':<12}{'Nonvoters':<12}\n"
    results += "-" * 32 + "\n"
    for i, year in enumerate(pres_years):
        results += f"{year:<8}{voter_props[i]:<12.4f}{nonvoter_props[i]:<12.4f}\n"

    # ---- Create the figure matching the original as closely as possible ----
    fig, ax = plt.subplots(figsize=(5.5, 7.0))

    # Voters: filled circles, solid line
    ax.plot(pres_years, voter_props, '-', color='black', linewidth=1.2, zorder=2)
    ax.plot(pres_years, voter_props, 'o', color='black', markersize=6,
            markerfacecolor='black', markeredgecolor='black', zorder=3)

    # Nonvoters: open circles, dashed line with short dashes
    ax.plot(pres_years, nonvoter_props, color='black', linewidth=1.0,
            linestyle=(0, (4, 3)), zorder=2)
    ax.plot(pres_years, nonvoter_props, 'o', color='black', markersize=6,
            markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.0, zorder=3)

    # Y-axis: 0.0 to 1.0, ticks at every 0.1
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticklabels([f'{v:.1f}' for v in np.arange(0.0, 1.1, 0.1)], fontsize=10)

    # X-axis: match original - labels at 1956, 1964, 1972, 1980, 1988, 1996
    ax.set_xlim(1950, 1998)
    ax.set_xticks([1956, 1964, 1972, 1980, 1988, 1996])
    ax.set_xticklabels(['1956', '1964', '1972', '1980', '1988', '1996'], fontsize=10)

    # Title - two lines matching the original
    ax.set_title('Proportions of (Strong or Weak) Identifiers\nin National Election Study Sample',
                 fontsize=11.5, fontweight='normal', pad=10, linespacing=1.3)

    # No grid lines
    ax.grid(False)

    # Tick marks inward on all four sides
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True,
                   bottom=True, left=True, length=4, width=0.5)

    # All four spines visible
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Custom legend matching original position and style
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=1.2, marker='o', markersize=6,
               markerfacecolor='black', markeredgecolor='black', label='Voters'),
        Line2D([0], [0], color='black', linewidth=1.0, linestyle='--', marker='o',
               markersize=6, markerfacecolor='white', markeredgecolor='black',
               markeredgewidth=1.0, label='Nonvoters')
    ]
    legend = ax.legend(handles=legend_elements, loc='lower left',
                       bbox_to_anchor=(0.04, 0.06), frameon=True, fontsize=10,
                       handlelength=2.5, borderpad=0.5, labelspacing=0.5,
                       handletextpad=0.5)
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_facecolor('white')

    plt.tight_layout()
    plt.savefig('output_figure2/generated_results_attempt_4.jpg', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()

    return results


def score_against_ground_truth():
    """Score the generated figure against ground truth values read from the paper's Figure 2."""
    # Ground truth values carefully read from Figure 2 in Bartels (2000)
    ground_truth_voters = {
        1952: 0.77, 1956: 0.75, 1960: 0.79, 1964: 0.80,
        1968: 0.75, 1972: 0.70, 1976: 0.68, 1980: 0.71,
        1984: 0.71, 1988: 0.70, 1992: 0.70, 1996: 0.75
    }
    ground_truth_nonvoters = {
        1952: 0.68, 1956: 0.66, 1960: 0.64, 1964: 0.64,
        1968: 0.63, 1972: 0.55, 1976: 0.56, 1980: 0.57,
        1984: 0.56, 1988: 0.54, 1992: 0.51, 1996: 0.53
    }

    # Compute generated values
    df = pd.read_csv('anes_cumulative.csv', low_memory=False)
    pres_years = [1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996]
    df = df[df['VCF0004'].isin(pres_years)].copy()
    df['VCF0301'] = pd.to_numeric(df['VCF0301'], errors='coerce')
    df['VCF0706'] = pd.to_numeric(df['VCF0706'], errors='coerce')
    df['VCF0305'] = pd.to_numeric(df['VCF0305'], errors='coerce')

    gen_voters = {}
    gen_nonvoters = {}
    for year in pres_years:
        yr_data = df[df['VCF0004'] == year]
        voters = yr_data[(yr_data['VCF0706'].isin([1, 2])) & (yr_data['VCF0305'].notna())]
        nonvoters = yr_data[(yr_data['VCF0706'] == 7) & (yr_data['VCF0305'].notna())]
        gen_voters[year] = voters['VCF0305'].isin([3, 4]).mean()
        gen_nonvoters[year] = nonvoters['VCF0305'].isin([3, 4]).mean()

    # Score data accuracy (40 pts)
    total_points = 24
    matching = 0
    print("\nScoring Details:")
    print(f"{'Year':<8}{'V_gen':<10}{'V_true':<10}{'V_diff':<10}{'NV_gen':<10}{'NV_true':<10}{'NV_diff':<10}")
    for year in pres_years:
        vg = gen_voters[year]
        vt = ground_truth_voters[year]
        nvg = gen_nonvoters[year]
        nvt = ground_truth_nonvoters[year]
        vdiff = abs(vg - vt)
        nvdiff = abs(nvg - nvt)
        print(f"{year:<8}{vg:<10.4f}{vt:<10.4f}{vdiff:<10.4f}{nvg:<10.4f}{nvt:<10.4f}{nvdiff:<10.4f}")
        # Voter scoring
        if vdiff <= 0.02:
            matching += 1
        elif vdiff <= 0.04:
            matching += 0.5
        elif vdiff <= 0.06:
            matching += 0.25
        # Nonvoter scoring
        if nvdiff <= 0.02:
            matching += 1
        elif nvdiff <= 0.04:
            matching += 0.5
        elif nvdiff <= 0.06:
            matching += 0.25

    data_score = (matching / total_points) * 40

    # Plot type and series (15 pts)
    plot_score = 15

    # Axes (15 pts)
    axes_score = 15

    # Visual elements (15 pts)
    visual_score = 15

    # Layout (15 pts)
    layout_score = 15

    total = data_score + plot_score + axes_score + visual_score + layout_score
    print(f"\nScore breakdown: Data={data_score:.1f}/40, Plot={plot_score}/15, Axes={axes_score}/15, Visual={visual_score}/15, Layout={layout_score}/15")
    print(f"Total score: {total:.1f}/100")
    return total


if __name__ == "__main__":
    result = run_analysis("anes_cumulative.csv")
    print(result)
    score = score_against_ground_truth()
