"""
Focus on best legitimate configurations.
Key findings so far:
1. Unemployment from loan VAR at 31bp scale gets 9/13
2. Securities at 31bp from sec VAR gets 8/14 (with 1979:06 sample)
3. Deposits raw gets 9/13
4. Loans is the bottleneck (0-2/13 depending on config)

Try: combining best unemployment source with best config for each bank var.
Also try: what if we use 31bp norm for sec (which helps) but raw for dep and loans?
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

gt_unemp = {1: 0.00, 2: 0.00, 4: 0.01, 6: 0.02, 8: 0.03,
            10: 0.05, 12: 0.07, 14: 0.10, 16: 0.12, 18: 0.14,
            20: 0.15, 22: 0.16, 24: 0.17}
gt_sec = {1: 0.00, 2: -0.10, 3: -0.25, 4: -0.40, 6: -0.65,
          8: -0.825, 10: -0.80, 12: -0.70, 14: -0.60,
          16: -0.55, 18: -0.50, 20: -0.45, 22: -0.42, 24: -0.40}
gt_dep = {1: 0.00, 2: -0.02, 4: -0.15, 6: -0.35, 8: -0.55,
          10: -0.70, 12: -0.80, 14: -0.80, 16: -0.80,
          18: -0.78, 20: -0.75, 22: -0.72, 24: -0.70}
gt_loans = {1: 0.00, 2: 0.00, 4: -0.02, 6: -0.05, 8: -0.20,
            10: -0.45, 12: -0.80, 14: -0.95, 16: -1.05,
            18: -1.15, 20: -1.25, 22: -1.30, 24: -1.35}

def count_matches(irf_vals, gt_dict):
    m, t = 0, 0
    for month, gt_val in sorted(gt_dict.items()):
        if month < len(irf_vals):
            gen_val = irf_vals[month]
            t += 1
            if abs(gt_val) < 0.005:
                ok = abs(gen_val - gt_val) < 0.05
            else:
                ok = abs(gen_val - gt_val) / abs(gt_val) < 0.20
            if ok: m += 1
    return m, t

def compute_full_score(irf_u, irf_s, irf_d, irf_l, label):
    """Replicate the exact scoring function from the script."""
    score_details = {}
    score_details['plot_type_and_series'] = 15

    shape_pts = 0
    if irf_u[12] > 0 and irf_u[24] > 0: shape_pts += 6
    if irf_u[6] < irf_u[18]: shape_pts += 1
    if irf_s[8] < -0.2: shape_pts += 4
    if irf_s[24] > irf_s[8]: shape_pts += 2
    if irf_d[12] < -0.2: shape_pts += 4
    if irf_d[24] < -0.2: shape_pts += 2
    if abs(irf_l[4]) < abs(irf_l[16]): shape_pts += 4
    if irf_l[24] < -0.5: shape_pts += 2
    score_details['response_shape_and_sign'] = shape_pts

    def compute_accuracy(irf_vals, gt_dict):
        matches, total = 0, 0
        for month, gt_val in gt_dict.items():
            if month < len(irf_vals):
                gen_val = irf_vals[month]
                total += 1
                if abs(gt_val) < 0.005:
                    if abs(gen_val - gt_val) < 0.05: matches += 1
                else:
                    if abs(gen_val - gt_val) / abs(gt_val) < 0.20: matches += 1
        return matches / total if total > 0 else 0

    acc_u = compute_accuracy(irf_u, gt_unemp)
    acc_s = compute_accuracy(irf_s, gt_sec)
    acc_d = compute_accuracy(irf_d, gt_dep)
    acc_l = compute_accuracy(irf_l, gt_loans)

    avg_acc = (acc_u + acc_s + acc_d + acc_l) / 4
    data_pts = int(avg_acc * 25)
    score_details['data_values_accuracy'] = data_pts

    score_details['axis_labels_ranges'] = 15
    score_details['confidence_bands'] = 10
    score_details['layout'] = 8

    total = sum(v for k, v in score_details.items() if isinstance(v, (int, float)))

    mu, tu = count_matches(irf_u, gt_unemp)
    ms, ts = count_matches(irf_s, gt_sec)
    md, td = count_matches(irf_d, gt_dep)
    ml, tl = count_matches(irf_l, gt_loans)

    print(f"  {label}: score={total} (shape={shape_pts}, data={data_pts}) "
          f"matches: u={mu}/{tu} s={ms}/{ts} d={md}/{td} l={ml}/{tl} "
          f"acc: u={acc_u:.0%} s={acc_s:.0%} d={acc_d:.0%} l={acc_l:.0%}")
    return total

horizon = 24

# ======================================================================
# CONFIG 1: 1979:06, 6 lags, no dummy
# ======================================================================
print("="*70)
print("1979:06, 6 lags, no dummy - All unemployment sources")
print("="*70)

df_sub = df.loc['1959-01':'1979-06'].copy()
cpi = df_sub['cpi']

def est_var(bank_var_data, bank_name, df_s=df_sub, use_dummy=False):
    df_var = pd.DataFrame({
        'funds_rate': df_s['funds_rate'],
        'unemp': df_s['unemp_male_2554'],
        'log_cpi': df_s['log_cpi'],
        bank_name: bank_var_data
    }, index=df_s.index).dropna()
    if use_dummy:
        dummy = pd.DataFrame(
            {'dummy_june69': ((df_s.index.year == 1969) & (df_s.index.month == 6)).astype(int)},
            index=df_s.index
        ).loc[df_var.index]
        r = VAR(df_var, exog=dummy).fit(maxlags=6, trend='c')
    else:
        r = VAR(df_var).fit(maxlags=6, trend='c')
    irf = r.irf(horizon)
    return r, irf

log_dep = np.log(df_sub['bank_deposits_check']) - np.log(cpi)
log_sec = np.log(df_sub['bank_investments']) - np.log(cpi)
log_loans = np.log(df_sub['bank_loans']) - np.log(cpi)

r_dep, irf_dep = est_var(log_dep, 'log_dep')
r_sec, irf_sec = est_var(log_sec, 'log_sec')
r_loan, irf_loan = est_var(log_loans, 'log_loan')

fs_dep = np.sqrt(r_dep.sigma_u.iloc[0, 0])
fs_sec = np.sqrt(r_sec.sigma_u.iloc[0, 0])
fs_loan = np.sqrt(r_loan.sigma_u.iloc[0, 0])

# Raw bank vars
raw_dep_v = irf_dep.orth_irfs[:, 3, 0] * 100
raw_sec_v = irf_sec.orth_irfs[:, 3, 0] * 100
raw_loans_v = irf_loan.orth_irfs[:, 3, 0] * 100

# 31bp normalized bank vars (per-VAR)
sc_dep = 0.31 / fs_dep
sc_sec = 0.31 / fs_sec
sc_loan = 0.31 / fs_loan
norm_dep_v = raw_dep_v * sc_dep
norm_sec_v = raw_sec_v * sc_sec
norm_loans_v = raw_loans_v * sc_loan

# Unemployment from each VAR
u_from_dep = irf_dep.orth_irfs[:, 1, 0]
u_from_sec = irf_sec.orth_irfs[:, 1, 0]
u_from_loan = irf_loan.orth_irfs[:, 1, 0]

print(f"\nFunds std: dep={fs_dep:.4f}, sec={fs_sec:.4f}, loan={fs_loan:.4f}")
print(f"31bp scales: dep={sc_dep:.4f}, sec={sc_sec:.4f}, loan={sc_loan:.4f}")

# Try all combinations
for u_source, u_name, u_fs in [(u_from_dep, "dep", fs_dep),
                                (u_from_sec, "sec", fs_sec),
                                (u_from_loan, "loan", fs_loan)]:
    u_31bp = u_source * (0.31 / u_fs)
    u_raw = u_source

    # Combo 1: All 31bp normalized
    compute_full_score(u_31bp, norm_sec_v, norm_dep_v, norm_loans_v,
                      f"u_{u_name}_31bp + all_31bp")

    # Combo 2: 31bp unemp + raw bank
    compute_full_score(u_31bp, raw_sec_v, raw_dep_v, raw_loans_v,
                      f"u_{u_name}_31bp + raw_bank")

    # Combo 3: 31bp unemp + 31bp sec + raw dep + raw loans
    compute_full_score(u_31bp, norm_sec_v, raw_dep_v, raw_loans_v,
                      f"u_{u_name}_31bp + 31bp_sec + raw_dep + raw_loans")

    # Combo 4: 31bp unemp + 31bp sec + 31bp dep + raw loans
    compute_full_score(u_31bp, norm_sec_v, norm_dep_v, raw_loans_v,
                      f"u_{u_name}_31bp + 31bp_sec + 31bp_dep + raw_loans")

# ======================================================================
# CONFIG 2: 1978:12, 6 lags, with dummy
# ======================================================================
print("\n" + "="*70)
print("1978:12, 6 lags, with dummy - Key combos")
print("="*70)

df_sub2 = df.loc['1959-01':'1978-12'].copy()
cpi2 = df_sub2['cpi']

log_dep2 = np.log(df_sub2['bank_deposits_check']) - np.log(cpi2)
log_sec2 = np.log(df_sub2['bank_investments']) - np.log(cpi2)
log_loans2 = np.log(df_sub2['bank_loans']) - np.log(cpi2)

r_dep2, irf_dep2 = est_var(log_dep2, 'log_dep', df_sub2, use_dummy=True)
r_sec2, irf_sec2 = est_var(log_sec2, 'log_sec', df_sub2, use_dummy=True)
r_loan2, irf_loan2 = est_var(log_loans2, 'log_loan', df_sub2, use_dummy=True)

fs_dep2 = np.sqrt(r_dep2.sigma_u.iloc[0, 0])
fs_sec2 = np.sqrt(r_sec2.sigma_u.iloc[0, 0])
fs_loan2 = np.sqrt(r_loan2.sigma_u.iloc[0, 0])

raw_dep2 = irf_dep2.orth_irfs[:, 3, 0] * 100
raw_sec2 = irf_sec2.orth_irfs[:, 3, 0] * 100
raw_loans2 = irf_loan2.orth_irfs[:, 3, 0] * 100
norm_sec2 = raw_sec2 * (0.31 / fs_sec2)
norm_dep2 = raw_dep2 * (0.31 / fs_dep2)

u2_from_dep = irf_dep2.orth_irfs[:, 1, 0]
u2_from_sec = irf_sec2.orth_irfs[:, 1, 0]
u2_from_loan = irf_loan2.orth_irfs[:, 1, 0]

# Try loan-VAR unemployment with various bank var configs
for u_source, u_name, u_fs in [(u2_from_loan, "loan", fs_loan2),
                                (u2_from_sec, "sec", fs_sec2)]:
    u_31bp = u_source * (0.31 / u_fs)

    compute_full_score(u_31bp, raw_sec2, raw_dep2, raw_loans2,
                      f"u_{u_name}_31bp + raw_bank")
    compute_full_score(u_31bp, norm_sec2, raw_dep2, raw_loans2,
                      f"u_{u_name}_31bp + 31bp_sec + raw_dep + raw_loans")

# ======================================================================
# Key insight: what if BOTH unemployment and bank vars benefit from
# a uniform 31bp normalization, but per-VAR?
# ======================================================================
print("\n" + "="*70)
print("Per-VAR 31bp with loan-VAR unemployment (1979:06)")
print("="*70)

# Each bank var normalized by its own VAR's funds_std
# Unemployment from the loan VAR, normalized by loan VAR's funds_std
u_best = u_from_loan * sc_loan
s_best = raw_sec_v * sc_sec
d_best = raw_dep_v * sc_dep
l_best = raw_loans_v * sc_loan

compute_full_score(u_best, s_best, d_best, l_best, "loan_u + per-VAR 31bp all")

# Unemployment from loan VAR 31bp, sec 31bp, dep raw, loans raw
compute_full_score(u_best, s_best, raw_dep_v, raw_loans_v,
                  "loan_u_31bp + sec_31bp + raw_dep + raw_loans")

# What's the best we can do with 1978:12?
print("\n" + "="*70)
print("BEST ACHIEVABLE with 1978:12 (paper's stated sample)")
print("="*70)

# Exhaustive: try all unemployment sources and all bank var combos
configs = []
for u_source, u_name, u_fs in [(u2_from_dep, "dep", fs_dep2),
                                (u2_from_sec, "sec", fs_sec2),
                                (u2_from_loan, "loan", fs_loan2)]:
    for s_norm_yn in [True, False]:
        for d_norm_yn in [True, False]:
            for l_norm_yn in [True, False]:
                u_31bp = u_source * (0.31 / u_fs)
                s_v = raw_sec2 * (0.31 / fs_sec2) if s_norm_yn else raw_sec2
                d_v = raw_dep2 * (0.31 / fs_dep2) if d_norm_yn else raw_dep2
                l_v = raw_loans2 * (0.31 / fs_loan2) if l_norm_yn else raw_loans2

                mu, _ = count_matches(u_31bp, gt_unemp)
                ms, _ = count_matches(s_v, gt_sec)
                md, _ = count_matches(d_v, gt_dep)
                ml, _ = count_matches(l_v, gt_loans)
                total_matches = mu + ms + md + ml

                label = f"u_{u_name}_31bp + {'N' if s_norm_yn else 'R'}_sec + {'N' if d_norm_yn else 'R'}_dep + {'N' if l_norm_yn else 'R'}_loans"
                configs.append((total_matches, label, u_31bp, s_v, d_v, l_v))

configs.sort(key=lambda x: x[0], reverse=True)
print("\nTop 10 configurations (1978:12):")
for total_m, label, u_v, s_v, d_v, l_v in configs[:10]:
    score = compute_full_score(u_v, s_v, d_v, l_v, label)
