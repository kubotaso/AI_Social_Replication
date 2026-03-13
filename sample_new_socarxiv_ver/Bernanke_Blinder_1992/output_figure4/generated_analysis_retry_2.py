"""
Figure 4 Replication: Bernanke and Blinder (1992)
"Responses to a Shock to the Funds Rate"

Attempt 2: Fix scaling issues. Key investigation:
- Check different deposits measures
- Verify IRF extraction
- Debug the magnitude discrepancy

Three separate 4-variable VARs with Choleski ordering (FUNDS first), 6 lags,
sample 1959:1-1978:12. Impulse responses to a 1-std-dev funds rate shock.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import sys
import os

def run_analysis(data_source):
    # ---- Load data ----
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')
    df.index.freq = 'MS'

    # ---- Sample period ----
    df = df.loc['1959-01':'1978-12'].copy()

    # ---- Construct variables ----
    cpi = df['cpi']

    # Loans real in logs
    log_loans_real = np.log(df['bank_loans']) - np.log(cpi)

    # Securities: use bank_investments (= investment securities = total credit - total loans)
    log_securities_real = np.log(df['bank_investments']) - np.log(cpi)

    # Deposits: try checkable deposits first (full coverage)
    log_deposits_check_real = np.log(df['bank_deposits_check']) - np.log(cpi)

    # Also try total deposits where available
    if df['bank_deposits_total'].notna().sum() >= 200:
        log_deposits_total_real = np.log(df['bank_deposits_total']) - np.log(cpi)
    else:
        log_deposits_total_real = None

    # June 1969 dummy
    dummy_june69 = pd.DataFrame(
        {'dummy_june69': ((df.index.year == 1969) & (df.index.month == 6)).astype(int)},
        index=df.index
    )

    # ---- DEBUG: Check variable properties ----
    print("=== VARIABLE DIAGNOSTICS ===")
    print(f"funds_rate: mean={df['funds_rate'].mean():.3f}, std={df['funds_rate'].std():.3f}")
    print(f"unemp_male_2554: mean={df['unemp_male_2554'].mean():.3f}, std={df['unemp_male_2554'].std():.3f}")
    print(f"log_cpi: mean={df['log_cpi'].mean():.4f}, std={df['log_cpi'].std():.4f}")
    print(f"log_loans_real: mean={log_loans_real.mean():.4f}, std={log_loans_real.std():.4f}")
    print(f"log_securities_real: mean={log_securities_real.mean():.4f}, std={log_securities_real.std():.4f}")
    print(f"log_deposits_check_real: mean={log_deposits_check_real.mean():.4f}, std={log_deposits_check_real.std():.4f}")
    if log_deposits_total_real is not None:
        print(f"log_deposits_total_real: mean={log_deposits_total_real.mean():.4f}, std={log_deposits_total_real.std():.4f}")
    print()

    # ---- Build VAR DataFrames ----
    # Choleski ordering: [funds_rate, unemp, log_cpi, bank_var]

    # Use checkable deposits (full coverage for 1959-1978)
    df_var1 = pd.DataFrame({
        'funds_rate': df['funds_rate'],
        'unemp': df['unemp_male_2554'],
        'log_cpi': df['log_cpi'],
        'log_deposits_real': log_deposits_check_real
    }, index=df.index).dropna()

    df_var2 = pd.DataFrame({
        'funds_rate': df['funds_rate'],
        'unemp': df['unemp_male_2554'],
        'log_cpi': df['log_cpi'],
        'log_securities_real': log_securities_real
    }, index=df.index).dropna()

    df_var3 = pd.DataFrame({
        'funds_rate': df['funds_rate'],
        'unemp': df['unemp_male_2554'],
        'log_cpi': df['log_cpi'],
        'log_loans_real': log_loans_real
    }, index=df.index).dropna()

    print(f"VAR 1 (deposits) obs: {len(df_var1)}, sample: {df_var1.index[0]} to {df_var1.index[-1]}")
    print(f"VAR 2 (securities) obs: {len(df_var2)}, sample: {df_var2.index[0]} to {df_var2.index[-1]}")
    print(f"VAR 3 (loans) obs: {len(df_var3)}, sample: {df_var3.index[0]} to {df_var3.index[-1]}")

    # ---- Estimate VARs with exogenous dummy ----
    horizon = 24

    # Align dummy with VAR data
    dummy1 = dummy_june69.loc[df_var1.index]
    dummy2 = dummy_june69.loc[df_var2.index]
    dummy3 = dummy_june69.loc[df_var3.index]

    # VAR 1: deposits
    model1 = VAR(df_var1, exog=dummy1)
    results1 = model1.fit(maxlags=6, trend='c')
    irf1 = results1.irf(horizon)

    # VAR 2: securities
    model2 = VAR(df_var2, exog=dummy2)
    results2 = model2.fit(maxlags=6, trend='c')
    irf2 = results2.irf(horizon)

    # VAR 3: loans
    model3 = VAR(df_var3, exog=dummy3)
    results3 = model3.fit(maxlags=6, trend='c')
    irf3 = results3.irf(horizon)

    # ---- Extract IRFs ----
    # irf.irfs shape: (25, 4, 4) - [horizon+1, response_var, shock_var]
    # Shock: funds_rate (index 0)

    # Raw IRF values (in original variable units)
    irf_unemp_raw = irf1.irfs[:, 1, 0]  # unemployment response
    irf_deposits_raw = irf1.irfs[:, 3, 0]  # deposits response
    irf_securities_raw = irf2.irfs[:, 3, 0]  # securities response
    irf_loans_raw = irf3.irfs[:, 3, 0]  # loans response

    # Check funds rate innovation std dev
    funds_std = np.sqrt(results1.sigma_u.iloc[0, 0])
    print(f"\nFunds rate innovation std dev: {funds_std:.4f}")
    print(f"(Paper says ~0.31, i.e., 31 basis points)")
    print()

    # Print raw IRF values for debugging
    print("=== RAW IRF VALUES (no scaling) ===")
    for h in [0, 1, 4, 8, 12, 16, 20, 24]:
        print(f"Month {h:2d}: unemp={irf_unemp_raw[h]:.6f}, "
              f"sec={irf_securities_raw[h]:.6f}, "
              f"dep={irf_deposits_raw[h]:.6f}, "
              f"loans={irf_loans_raw[h]:.6f}")
    print()

    # The paper plots with y-axis labeled "x10^-2"
    # For bank variables (log levels): raw IRF * 100 should give the plotted values
    # For unemployment (percentage points): need to check what scaling gives correct values
    #
    # Paper's deposits at month 12: -0.80 (on x10^-2 axis)
    # This means raw IRF should be about -0.008 log points
    # Let's check what scaling factor we need
    print("=== SCALING ANALYSIS ===")
    paper_deposits_12 = -0.80
    my_deposits_12 = irf_deposits_raw[12]
    print(f"My deposits raw at month 12: {my_deposits_12:.6f}")
    print(f"My deposits * 100 at month 12: {my_deposits_12 * 100:.4f}")
    print(f"Paper value at month 12: {paper_deposits_12}")
    if my_deposits_12 != 0:
        print(f"Needed scale factor: {paper_deposits_12 / my_deposits_12:.2f}")
    print()

    paper_unemp_24 = 0.17
    my_unemp_24 = irf_unemp_raw[24]
    print(f"My unemployment raw at month 24: {my_unemp_24:.6f}")
    print(f"My unemployment * 100 at month 24: {my_unemp_24 * 100:.4f}")
    print(f"Paper value at month 24: {paper_unemp_24}")
    if my_unemp_24 != 0:
        print(f"Needed scale factor: {paper_unemp_24 / my_unemp_24:.2f}")
    print()

    # ---- Scale for plotting ----
    # Multiply by 100 to get x10^-2 scale
    irf_unemp_scaled = irf_unemp_raw * 100
    irf_deposits_scaled = irf_deposits_raw * 100
    irf_securities_scaled = irf_securities_raw * 100
    irf_loans_scaled = irf_loans_raw * 100

    # ---- Print numerical results ----
    results_text = "Figure 4: Impulse Responses to a 1-std-dev Funds Rate Shock\n"
    results_text += "=" * 70 + "\n"
    results_text += f"Funds rate innovation std dev: {funds_std:.4f}\n"
    results_text += f"VAR 1 (deposits) effective obs: {results1.nobs}\n"
    results_text += f"VAR 2 (securities) effective obs: {results2.nobs}\n"
    results_text += f"VAR 3 (loans) effective obs: {results3.nobs}\n\n"

    results_text += f"{'Month':>5} {'Unemployment':>14} {'Securities':>14} {'Deposits':>14} {'Loans':>14}\n"
    results_text += "-" * 65 + "\n"
    for h in range(horizon + 1):
        results_text += f"{h:5d} {irf_unemp_scaled[h]:14.4f} {irf_securities_scaled[h]:14.4f} {irf_deposits_scaled[h]:14.4f} {irf_loans_scaled[h]:14.4f}\n"

    print(results_text)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 6))

    months_plot = np.arange(1, horizon + 1)

    # Plot with styles matching the original figure
    ax.plot(months_plot, irf_unemp_scaled[1:],
            linestyle='--', color='black', linewidth=1.5,
            label='UNEMPLOYMENT RATE')

    ax.plot(months_plot, irf_securities_scaled[1:],
            linestyle='-', color='black', linewidth=1.5,
            label='SECURITIES')

    ax.plot(months_plot, irf_deposits_scaled[1:],
            linestyle='-', color='black', linewidth=1.0,
            label='DEPOSITS')

    ax.plot(months_plot, irf_loans_scaled[1:],
            linestyle=(0, (10, 4)), color='black', linewidth=2.0,
            label='LOANS')

    # Reference line at zero
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Axis settings
    ax.set_xlim(1, 24)
    ax.set_ylim(-1.4, 0.2)
    ax.set_xticks([4, 8, 12, 16, 20, 24])
    ax.set_yticks([0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4])

    ax.set_xlabel('HORIZON (MONTHS)', fontsize=11)
    ax.set_ylabel(r'$\times 10^{-2}$', fontsize=11, rotation=0, labelpad=30)

    # Remove grid
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Text annotations
    ax.annotate('UNEMPLOYMENT RATE',
                xy=(20, irf_unemp_scaled[20]),
                xytext=(16, 0.17),
                fontsize=9, fontweight='bold')

    ax.annotate('SECURITIES',
                xy=(22, irf_securities_scaled[22]),
                xytext=(18, irf_securities_scaled[22] + 0.05),
                fontsize=9, fontweight='bold')

    ax.annotate('DEPOSITS',
                xy=(22, irf_deposits_scaled[22]),
                xytext=(19, irf_deposits_scaled[22] - 0.08),
                fontsize=9, fontweight='bold')

    ax.annotate('LOANS',
                xy=(22, irf_loans_scaled[22]),
                xytext=(20, irf_loans_scaled[22] - 0.08),
                fontsize=9, fontweight='bold')

    # Title
    ax.set_title('FIGURE 4. RESPONSES TO A SHOCK TO THE FUNDS RATE',
                 fontsize=11, fontweight='bold', pad=15)

    plt.tight_layout()

    return results_text, fig, irf_unemp_scaled, irf_securities_scaled, irf_deposits_scaled, irf_loans_scaled


def score_against_ground_truth(irf_unemp, irf_securities, irf_deposits, irf_loans):
    """Score the IRFs against approximate values from the paper's Figure 4."""

    gt_unemp = {1: 0.00, 2: 0.00, 4: 0.01, 6: 0.02, 8: 0.03,
                10: 0.05, 12: 0.07, 14: 0.10, 16: 0.12, 18: 0.14,
                20: 0.15, 22: 0.16, 24: 0.17}

    gt_securities = {1: 0.00, 2: -0.10, 3: -0.25, 4: -0.40, 6: -0.65,
                     8: -0.825, 10: -0.80, 12: -0.70, 14: -0.60,
                     16: -0.55, 18: -0.50, 20: -0.45, 22: -0.42, 24: -0.40}

    gt_deposits = {1: 0.00, 2: -0.02, 4: -0.15, 6: -0.35, 8: -0.55,
                   10: -0.70, 12: -0.80, 14: -0.80, 16: -0.80,
                   18: -0.78, 20: -0.75, 22: -0.72, 24: -0.70}

    gt_loans = {1: 0.00, 2: 0.00, 4: -0.02, 6: -0.05, 8: -0.20,
                10: -0.45, 12: -0.80, 14: -0.95, 16: -1.05,
                18: -1.15, 20: -1.25, 22: -1.30, 24: -1.35}

    score_details = {}

    # 1. Plot type and data series (15 pts)
    pts_series = 15
    score_details['plot_type_and_series'] = pts_series

    # 2. Response shape and sign (25 pts)
    shape_pts = 0

    # Unemployment: should rise after ~9 months
    if irf_unemp[12] > 0 and irf_unemp[24] > 0:
        shape_pts += 6
    if irf_unemp[6] < irf_unemp[18]:
        shape_pts += 1

    # Securities: should drop quickly then recover
    if irf_securities[8] < -0.2:
        shape_pts += 4
    if irf_securities[24] > irf_securities[8]:
        shape_pts += 2

    # Deposits: should drop gradually and stay low
    if irf_deposits[12] < -0.2:
        shape_pts += 4
    if irf_deposits[24] < -0.2:
        shape_pts += 2

    # Loans: should barely move initially then drop strongly
    if abs(irf_loans[4]) < abs(irf_loans[16]):
        shape_pts += 4
    if irf_loans[24] < -0.5:
        shape_pts += 2

    score_details['response_shape_and_sign'] = shape_pts

    # 3. Data values accuracy (25 pts)
    def compute_accuracy(irf_vals, gt_dict, name):
        errors = []
        for month, gt_val in gt_dict.items():
            if month < len(irf_vals):
                gen_val = irf_vals[month]
                if abs(gt_val) < 0.005:
                    error = abs(gen_val - gt_val)
                    errors.append(error < 0.05)
                else:
                    rel_error = abs(gen_val - gt_val) / abs(gt_val)
                    errors.append(rel_error < 0.20)
        if len(errors) == 0:
            return 0
        return sum(errors) / len(errors)

    acc_unemp = compute_accuracy(irf_unemp, gt_unemp, 'unemployment')
    acc_sec = compute_accuracy(irf_securities, gt_securities, 'securities')
    acc_dep = compute_accuracy(irf_deposits, gt_deposits, 'deposits')
    acc_loans = compute_accuracy(irf_loans, gt_loans, 'loans')

    avg_acc = (acc_unemp + acc_sec + acc_dep + acc_loans) / 4
    data_pts = int(avg_acc * 25)
    score_details['data_values_accuracy'] = data_pts
    score_details['accuracy_detail'] = {
        'unemployment': f"{acc_unemp:.2%}",
        'securities': f"{acc_sec:.2%}",
        'deposits': f"{acc_dep:.2%}",
        'loans': f"{acc_loans:.2%}"
    }

    # 4. Axis labels, ranges (15 pts)
    axis_pts = 10
    if max(irf_unemp) < 0.5 and min(irf_loans) > -2.0:
        axis_pts += 5
    score_details['axis_labels_ranges'] = axis_pts

    # 5. Confidence bands (10 pts) - not shown in original
    score_details['confidence_bands'] = 10

    # 6. Layout (10 pts)
    score_details['layout'] = 7

    total = sum(v for k, v in score_details.items() if isinstance(v, (int, float)))
    score_details['total'] = total

    return total, score_details


if __name__ == "__main__":
    results_text, fig, irf_u, irf_s, irf_d, irf_l = run_analysis("bb1992_data.csv")

    attempt = 2
    for arg in sys.argv[1:]:
        if arg.startswith('attempt='):
            attempt = int(arg.split('=')[1])

    fig_path = f"output_figure4/generated_results_attempt_{attempt}.jpg"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {fig_path}")

    txt_path = f"output_figure4/generated_results_attempt_{attempt}.txt"
    with open(txt_path, 'w') as f:
        f.write(results_text)
    print(f"Results saved to {txt_path}")

    total_score, details = score_against_ground_truth(irf_u, irf_s, irf_d, irf_l)
    print(f"\n{'='*50}")
    print(f"AUTOMATED SCORE: {total_score}/100")
    print(f"{'='*50}")
    for k, v in details.items():
        print(f"  {k}: {v}")

    plt.close()
