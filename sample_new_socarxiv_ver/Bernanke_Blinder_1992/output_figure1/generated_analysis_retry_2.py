import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import sys

def run_analysis(data_source):
    """
    Replicate Figure 1 from Bernanke and Blinder (1992):
    "Interest-Rate Indicators of Fed Policy"

    Attempt 2: Fixes from discrepancy report:
    1. Move title to bottom (caption style)
    2. Use serif font
    3. Adjust R label positioning and weight
    4. Adjust figure proportions
    """
    # Load data
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')

    # Filter to the approximate period shown in the figure
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

    # Use serif font to match original paper style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']

    # Create figure - make proportions match original (slightly taller than wide)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))

    # Plot FUNDS (solid line) and FFBOND (dashed line)
    ax.plot(funds.index, funds.values, 'k-', linewidth=0.9, label='FUNDS')
    ax.plot(ffbond.index, ffbond.values, 'k--', linewidth=0.7, label='FFBOND')

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
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Add vertical lines at Romer dates with "R" labels
    for rd in romer_dates:
        ax.axvline(x=rd, color='black', linestyle='-', linewidth=0.7)
        # Place "R" label near the top of the vertical line (not bold, matching original)
        ax.text(rd, 19.2, 'R', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    # Labels
    ax.set_ylabel('Rate (Percentage)', fontsize=11)
    ax.set_xlabel('YEAR', fontsize=11)

    # Add series labels on the plot (matching the original figure positioning)
    # In the original, FUNDS is labeled on the right side around 1987-1988, y~12
    ax.text(pd.Timestamp('1987-06-01'), 12, 'FUNDS', fontsize=10, fontweight='bold')
    # FFBOND label on the right side, y~1
    ax.text(pd.Timestamp('1987-06-01'), 1, 'FFBOND', fontsize=10, fontweight='bold')

    # Title at bottom as caption (matching original paper style)
    # The original has the title below the figure as:
    # "FIGURE 1. INTEREST-RATE INDICATORS OF FED POLICY"
    # using small caps style
    fig.text(0.5, 0.01,
             'FIGURE 1. INTEREST-RATE INDICATORS OF FED POLICY',
             ha='center', va='bottom', fontsize=10,
             fontvariant='small-caps')

    # No grid (original doesn't have grid)
    ax.grid(False)

    # Remove top and right spines for cleaner look (if original has them)
    # Original appears to have a full box
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Tight layout with room for bottom caption
    fig.subplots_adjust(bottom=0.14, top=0.96, left=0.12, right=0.95)

    # Save figure
    output_dir = 'output_figure1'
    outpath = os.path.join(output_dir, 'generated_results_attempt_2.jpg')
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {outpath}")
    plt.close()

    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)

    # Return summary
    results = f"""Figure 1 Replication Results (Attempt 2)

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

Changes from attempt 1:
- Title moved to bottom as caption
- Serif font used
- R labels repositioned
- Figure proportions adjusted
"""
    return results


if __name__ == "__main__":
    result = run_analysis("bb1992_data.csv")
    print(result)
