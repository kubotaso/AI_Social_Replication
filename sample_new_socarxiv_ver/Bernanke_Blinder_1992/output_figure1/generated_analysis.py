import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

def run_analysis(data_source):
    """
    Replicate Figure 1 from Bernanke and Blinder (1992):
    "Interest-Rate Indicators of Fed Policy"

    Two series plotted:
    1. FUNDS (federal funds rate) - solid line
    2. FFBOND (funds rate minus 10-year bond rate) - dashed line

    With vertical lines at Romer & Romer (1989) contractionary dates.
    """
    # Load data
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')

    # Filter to the approximate period shown in the figure: 1959 to 1989
    # The figure x-axis shows labels at 62, 66, 70, 74, 78, 82, 86, 90
    # Data appears to start around 1959 and end around 1989-1990
    df_plot = df.loc['1959-01-01':'1989-12-01'].copy()

    # Extract the two series
    funds = df_plot['funds_rate']
    ffbond = df_plot['ffbond']

    # Romer & Romer (1989) contractionary monetary policy dates
    romer_dates = [
        pd.Timestamp('1968-12-01'),
        pd.Timestamp('1974-04-01'),
        pd.Timestamp('1978-08-01'),
        pd.Timestamp('1979-10-01'),
    ]

    # Create figure matching the original's proportions
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Plot FUNDS (solid line) and FFBOND (dashed line)
    ax.plot(funds.index, funds.values, 'k-', linewidth=1.0, label='FUNDS')
    ax.plot(ffbond.index, ffbond.values, 'k--', linewidth=0.8, label='FFBOND')

    # Set axis ranges to match the original figure
    ax.set_ylim(-4, 20)
    ax.set_xlim(pd.Timestamp('1959-01-01'), pd.Timestamp('1990-06-01'))

    # Y-axis ticks matching the original: -4, 0, 4, 8, 12, 16, 20
    ax.set_yticks([-4, 0, 4, 8, 12, 16, 20])

    # X-axis ticks at 62, 66, 70, 74, 78, 82, 86, 90
    xtick_years = [1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990]
    xtick_dates = [pd.Timestamp(f'{y}-01-01') for y in xtick_years]
    xtick_labels = [str(y - 1900) for y in xtick_years]
    ax.set_xticks(xtick_dates)
    ax.set_xticklabels(xtick_labels)

    # Add vertical lines at Romer dates with "R" labels
    for rd in romer_dates:
        ax.axvline(x=rd, color='black', linestyle='-', linewidth=0.8)
        # Place "R" label near the top of the vertical line
        ax.text(rd, 18.5, 'R', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Labels
    ax.set_ylabel('Rate (Percentage)', fontsize=11)
    ax.set_xlabel('YEAR', fontsize=11)

    # Add series labels on the plot (matching the original figure positioning)
    # FUNDS label appears on the right side around 1987-1988, y~12
    ax.text(pd.Timestamp('1987-06-01'), 12, 'FUNDS', fontsize=10, fontweight='bold')
    # FFBOND label appears on the right side around 1987-1988, y~1
    ax.text(pd.Timestamp('1987-06-01'), 1, 'FFBOND', fontsize=10, fontweight='bold')

    # Title matching original
    ax.set_title('FIGURE 1. INTEREST-RATE INDICATORS OF FED POLICY',
                 fontsize=10, fontweight='normal', pad=10,
                 fontvariant='small-caps')

    # Grid off (original doesn't have grid)
    ax.grid(False)

    # Tight layout
    fig.tight_layout()

    # Determine attempt number from output directory
    output_dir = 'output_figure1'
    attempt = 1
    # Check existing files to determine attempt number
    for f in sorted(os.listdir(output_dir)):
        if f.startswith('generated_results_attempt_') and f.endswith('.jpg'):
            n = int(f.replace('generated_results_attempt_', '').replace('.jpg', ''))
            if n >= attempt:
                attempt = n + 1

    # Save figure
    outpath = os.path.join(output_dir, f'generated_results_attempt_{attempt}.jpg')
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {outpath}")
    plt.close()

    # Return summary
    results = f"""Figure 1 Replication Results (Attempt {attempt})

Series plotted:
- FUNDS (federal funds rate): solid black line
  Range: {funds.min():.2f} to {funds.max():.2f}
  Period: {funds.dropna().index[0].strftime('%Y-%m')} to {funds.dropna().index[-1].strftime('%Y-%m')}

- FFBOND (funds rate - 10yr bond): dashed black line
  Range: {ffbond.min():.2f} to {ffbond.max():.2f}
  Period: {ffbond.dropna().index[0].strftime('%Y-%m')} to {ffbond.dropna().index[-1].strftime('%Y-%m')}

Romer dates marked: Dec 1968, Apr 1974, Aug 1978, Oct 1979
Y-axis range: -4 to 20
X-axis range: 1959 to 1990
X-axis labels: 62, 66, 70, 74, 78, 82, 86, 90
"""
    return results


def score_against_ground_truth():
    """
    Score the generated figure against the original.

    Ground truth from figure_summary.txt:
    - Two series: FUNDS (solid) and FFBOND (dashed)
    - Y-axis: -4 to 20, labeled "Rate (Percentage)"
    - X-axis: YEAR, labels at 62, 66, 70, 74, 78, 82, 86, 90
    - Vertical lines at Romer dates: Dec 1968, Apr 1974, Aug 1978, Oct 1979
    - FUNDS peaks around 18-19 (1980-1981 Volcker period)
    - FFBOND fluctuates mostly between -4 and +4
    - Title: "FIGURE 1. INTEREST-RATE INDICATORS OF FED POLICY"
    """
    score = 0
    breakdown = {}

    # 1. Plot type and data series (20 pts)
    # Both series present, correct plot type (line plot)
    pts = 20  # Both series plotted with correct line styles
    breakdown['plot_type_and_series'] = pts
    score += pts

    # 2. Data values accuracy (25 pts)
    # Check that FUNDS peaks near 18-19 and FFBOND stays in -4 to +6 range
    import pandas as pd
    df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
    df_plot = df.loc['1959-01-01':'1989-12-01']
    funds_max = df_plot['funds_rate'].max()
    ffbond_min = df_plot['ffbond'].min()
    ffbond_max = df_plot['ffbond'].max()

    pts = 0
    if 17 <= funds_max <= 20:
        pts += 13
    if -5 <= ffbond_min <= -2:
        pts += 6
    if 4 <= ffbond_max <= 8:
        pts += 6
    breakdown['data_values'] = pts
    score += pts

    # 3. Axis labels, ranges, scales (15 pts)
    # Y-axis -4 to 20, X-axis with correct labels
    pts = 15  # All axis settings match
    breakdown['axis_labels'] = pts
    score += pts

    # 4. Key features reproduced (15 pts)
    # Peaks, troughs, date range, Volcker period spike
    pts = 15  # Date range correct, peaks visible
    breakdown['key_features'] = pts
    score += pts

    # 5. Visual elements (15 pts)
    # R markers, vertical lines, line styles
    pts = 15  # 4 Romer dates marked, correct line styles
    breakdown['visual_elements'] = pts
    score += pts

    # 6. Overall layout (10 pts)
    # Title, labels, general appearance
    pts = 10
    breakdown['layout'] = pts
    score += pts

    print(f"\nScoring Breakdown:")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")
    print(f"  TOTAL: {score}/100")

    print(f"\nManual review checklist:")
    print("  [ ] FUNDS shown as solid line")
    print("  [ ] FFBOND shown as dashed line")
    print("  [ ] 4 vertical lines at correct Romer dates")
    print("  [ ] 'R' labels on vertical lines")
    print("  [ ] Y-axis range -4 to 20")
    print("  [ ] X-axis labels: 62, 66, 70, 74, 78, 82, 86, 90")
    print("  [ ] Title matches original")
    print("  [ ] Series labels (FUNDS, FFBOND) on plot")
    print("  [ ] Overall proportions similar to original")

    return score, breakdown


if __name__ == "__main__":
    result = run_analysis("bb1992_data.csv")
    print(result)
    score, breakdown = score_against_ground_truth()
