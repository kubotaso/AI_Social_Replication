import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def run_analysis(data_source):
    """
    Replicate Figure 1 from Bernanke and Blinder (1992):
    "Interest-Rate Indicators of Fed Policy"

    Attempt 3: Fixes from discrepancy report:
    1. R labels at varying heights matching original
    2. Series labels (FUNDS, FFBOND) not bold
    3. Fine-tuned proportions and styling
    """
    # Load data
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')

    # Filter to the period shown in the figure
    df_plot = df.loc['1959-01-01':'1989-12-01'].copy()

    # Extract the two series
    funds = df_plot['funds_rate']
    ffbond = df_plot['ffbond']

    # Romer & Romer (1989) contractionary monetary policy dates
    # with R label y-positions matching the original figure
    romer_info = [
        (pd.Timestamp('1968-12-01'), 7.5),    # R at y~7-8
        (pd.Timestamp('1974-04-01'), 12.0),   # R at y~12
        (pd.Timestamp('1978-08-01'), 8.0),    # R at y~8
        (pd.Timestamp('1979-10-01'), 18.5),   # R at y~18-19 (near top)
    ]

    # Use serif font to match original paper style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']

    # Create figure matching original proportions
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.5))

    # Plot FUNDS (solid line) and FFBOND (dashed line)
    ax.plot(funds.index, funds.values, 'k-', linewidth=0.9)
    ax.plot(ffbond.index, ffbond.values, 'k--', linewidth=0.7)

    # Set axis ranges to match the original figure
    ax.set_ylim(-4, 20)
    ax.set_xlim(pd.Timestamp('1959-01-01'), pd.Timestamp('1990-06-01'))

    # Y-axis ticks: -4, 0, 4, 8, 12, 16, 20
    ax.set_yticks([-4, 0, 4, 8, 12, 16, 20])

    # X-axis ticks at 62, 66, 70, 74, 78, 82, 86, 90
    xtick_years = [1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990]
    xtick_dates = [pd.Timestamp(f'{y}-01-01') for y in xtick_years]
    xtick_labels = [str(y - 1900) for y in xtick_years]
    ax.set_xticks(xtick_dates)
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Add vertical lines at Romer dates with "R" labels at varying heights
    for rd, r_y in romer_info:
        ax.axvline(x=rd, color='black', linestyle='-', linewidth=0.7)
        ax.text(rd, r_y, 'R', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    # Labels
    ax.set_ylabel('Rate (Percentage)', fontsize=11)
    ax.set_xlabel('YEAR', fontsize=11)

    # Add series labels on the plot (regular weight, matching original)
    ax.text(pd.Timestamp('1987-01-01'), 12.5, 'FUNDS', fontsize=10)
    ax.text(pd.Timestamp('1987-01-01'), 1.0, 'FFBOND', fontsize=10)

    # Title at bottom as caption
    fig.text(0.5, 0.01,
             'FIGURE 1. INTEREST-RATE INDICATORS OF FED POLICY',
             ha='center', va='bottom', fontsize=10,
             fontvariant='small-caps')

    # No grid
    ax.grid(False)

    # Keep full box (all spines visible)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    # Layout
    fig.subplots_adjust(bottom=0.15, top=0.96, left=0.12, right=0.95)

    # Save figure
    output_dir = 'output_figure1'
    outpath = os.path.join(output_dir, 'generated_results_attempt_3.jpg')
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {outpath}")
    plt.close()

    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)

    # Return summary
    results = f"""Figure 1 Replication Results (Attempt 3)

Series plotted:
- FUNDS (federal funds rate): solid black line
  Range: {funds.min():.2f} to {funds.max():.2f}

- FFBOND (funds rate - 10yr bond): dashed black line
  Range: {ffbond.min():.2f} to {ffbond.max():.2f}

Romer dates with R positions:
  Dec 1968 (y=7.5), Apr 1974 (y=12), Aug 1978 (y=8), Oct 1979 (y=18.5)

Changes from attempt 2:
- R labels at varying heights matching original
- Series labels in regular weight (not bold)
- Minor proportion adjustments
"""
    return results


if __name__ == "__main__":
    result = run_analysis("bb1992_data.csv")
    print(result)
