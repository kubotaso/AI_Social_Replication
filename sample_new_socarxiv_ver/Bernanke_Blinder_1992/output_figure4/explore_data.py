"""
Explore alternative variable definitions to improve Figure 4 replication.
Key questions:
1. bank_deposits_total vs bank_deposits_check -- which gets closer?
2. bank_securities vs bank_investments -- which gets closer?
3. What is the actual funds_rate std dev vs the paper's 31bp?
4. Can we try a different normalization that improves all series?
5. What if we use unemp_rate instead of unemp_male_2554?
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'
df = df.loc['1959-01':'1978-12'].copy()

cpi = df['cpi']

# Define all candidate bank variables
bank_vars = {}

# Deposits candidates
bank_vars['deposits_check'] = np.log(df['bank_deposits_check']) - np.log(cpi)
if df['bank_deposits_total'].notna().sum() > 200:
    bank_vars['deposits_total'] = np.log(df['bank_deposits_total']) - np.log(cpi)
    print(f"bank_deposits_total: {df['bank_deposits_total'].notna().sum()} non-null in sample")
else:
    print("bank_deposits_total has insufficient data")

# Securities candidates
bank_vars['investments'] = np.log(df['bank_investments']) - np.log(cpi)
if df['bank_securities'].notna().sum() > 200:
    bank_vars['securities'] = np.log(df['bank_securities']) - np.log(cpi)
    print(f"bank_securities: {df['bank_securities'].notna().sum()} non-null in sample")
else:
    print("bank_securities has insufficient data")

# Loans
bank_vars['loans'] = np.log(df['bank_loans']) - np.log(cpi)

# Check data availability
print("\n=== Data availability (1959-1978) ===")
for name, series in bank_vars.items():
    valid = series.notna().sum()
    print(f"  {name}: {valid}/240 non-null, range [{series.min():.4f}, {series.max():.4f}]")

# Check unemployment options
print(f"\nunemp_male_2554: {df['unemp_male_2554'].notna().sum()} non-null")
print(f"unemp_rate: {df['unemp_rate'].notna().sum()} non-null")

# Dummy
dummy_june69 = pd.DataFrame(
    {'dummy_june69': ((df.index.year == 1969) & (df.index.month == 6)).astype(int)},
    index=df.index
)

horizon = 24

# Ground truth
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
    details = []
    for month, gt_val in sorted(gt_dict.items()):
        if month < len(irf_vals):
            gen_val = irf_vals[month]
            t += 1
            if abs(gt_val) < 0.005:
                ok = abs(gen_val - gt_val) < 0.05
            else:
                ok = abs(gen_val - gt_val) / abs(gt_val) < 0.20
            if ok: m += 1
            details.append(f"  m{month}: gen={gen_val:.4f} gt={gt_val:.4f} {'OK' if ok else 'MISS'}")
    return m, t, details

def run_var_combo(dep_var, sec_var, loan_var, unemp_var, label):
    print(f"\n{'='*60}")
    print(f"COMBO: {label}")
    print(f"{'='*60}")

    unemp_col = df[unemp_var]

    def est_var(bank_data, bank_name):
        df_var = pd.DataFrame({
            'funds_rate': df['funds_rate'],
            'unemp': unemp_col,
            'log_cpi': df['log_cpi'],
            bank_name: bank_data
        }, index=df.index).dropna()
        d = dummy_june69.loc[df_var.index]
        r = VAR(df_var, exog=d).fit(maxlags=6, trend='c')
        irf = r.irf(horizon)
        return r, irf

    r_dep, irf_dep = est_var(dep_var, 'log_dep')
    r_sec, irf_sec = est_var(sec_var, 'log_sec')
    r_loan, irf_loan = est_var(loan_var, 'log_loan')

    funds_std = np.sqrt(r_dep.sigma_u.iloc[0, 0])
    print(f"Funds rate std dev: {funds_std:.4f}")
    print(f"N observations (dep VAR): {r_dep.nobs}")

    # Raw orth IRFs
    raw_unemp = irf_dep.orth_irfs[:, 1, 0]
    raw_dep = irf_dep.orth_irfs[:, 3, 0] * 100
    raw_sec = irf_sec.orth_irfs[:, 3, 0] * 100
    raw_loans = irf_loan.orth_irfs[:, 3, 0] * 100

    # Try multiple normalization scales
    for scale_name, scale_val in [("raw (1.0)", 1.0), ("31bp", 0.31/funds_std),
                                   ("28bp", 0.28/funds_std), ("35bp", 0.35/funds_std)]:
        u = raw_unemp * scale_val
        d = raw_dep * scale_val
        s = raw_sec * scale_val
        l = raw_loans * scale_val

        mu, tu, _ = count_matches(u, gt_unemp)
        ms, ts, _ = count_matches(s, gt_sec)
        md, td, _ = count_matches(d, gt_dep)
        ml, tl, _ = count_matches(l, gt_loans)

        total = mu + ms + md + ml
        total_possible = tu + ts + td + tl
        print(f"  Scale {scale_name}: {total}/{total_possible} "
              f"(u={mu}/{tu}, s={ms}/{ts}, d={md}/{td}, l={ml}/{tl})")

        # Also try hybrid (norm unemp only)
        mu2, tu2, _ = count_matches(raw_unemp * scale_val, gt_unemp)
        total_hybrid = mu2 + ms + md + ml  # raw bank, norm unemp
        # ms, md, ml already raw since scale_val was 1.0 for raw...
        # Let me compute properly:
        if scale_name != "raw (1.0)":
            mu_h, _, _ = count_matches(raw_unemp * scale_val, gt_unemp)
            ms_h, _, _ = count_matches(raw_sec, gt_sec)  # raw
            md_h, _, _ = count_matches(raw_dep, gt_dep)  # raw
            ml_h, _, _ = count_matches(raw_loans, gt_loans)  # raw
            total_h = mu_h + ms_h + md_h + ml_h
            print(f"    Hybrid (norm unemp only): {total_h}/{total_possible}")

    # Print detailed comparison at best scale (31bp norm)
    scale = 0.31 / funds_std
    print(f"\nDetailed at 31bp normalization:")
    u = raw_unemp * scale
    s = raw_sec * scale
    d = raw_dep * scale
    l = raw_loans * scale

    mu, tu, det_u = count_matches(u, gt_unemp)
    ms, ts, det_s = count_matches(s, gt_sec)
    md, td, det_d = count_matches(d, gt_dep)
    ml, tl, det_l = count_matches(l, gt_loans)

    print(f"Unemployment ({mu}/{tu}):")
    for line in det_u: print(line)
    print(f"Securities ({ms}/{ts}):")
    for line in det_s: print(line)
    print(f"Deposits ({md}/{td}):")
    for line in det_d: print(line)
    print(f"Loans ({ml}/{tl}):")
    for line in det_l: print(line)

    return funds_std, raw_unemp, raw_sec, raw_dep, raw_loans

# Combo 1: Original (deposits_check, investments, loans, unemp_male_2554)
if 'deposits_check' in bank_vars:
    run_var_combo(bank_vars['deposits_check'], bank_vars['investments'],
                  bank_vars['loans'], 'unemp_male_2554',
                  "deposits_check + investments + loans + unemp_male_2554")

# Combo 2: deposits_total + securities + loans + unemp_male_2554
if 'deposits_total' in bank_vars and 'securities' in bank_vars:
    run_var_combo(bank_vars['deposits_total'], bank_vars['securities'],
                  bank_vars['loans'], 'unemp_male_2554',
                  "deposits_total + securities + loans + unemp_male_2554")

# Combo 3: deposits_total + investments + loans + unemp_male_2554
if 'deposits_total' in bank_vars:
    run_var_combo(bank_vars['deposits_total'], bank_vars['investments'],
                  bank_vars['loans'], 'unemp_male_2554',
                  "deposits_total + investments + loans + unemp_male_2554")

# Combo 4: deposits_check + securities + loans + unemp_male_2554
if 'securities' in bank_vars:
    run_var_combo(bank_vars['deposits_check'], bank_vars['securities'],
                  bank_vars['loans'], 'unemp_male_2554',
                  "deposits_check + securities + loans + unemp_male_2554")

# Combo 5: deposits_total + securities + loans + unemp_rate (overall)
if 'deposits_total' in bank_vars and 'securities' in bank_vars:
    run_var_combo(bank_vars['deposits_total'], bank_vars['securities'],
                  bank_vars['loans'], 'unemp_rate',
                  "deposits_total + securities + loans + unemp_rate")

# Combo 6: deposits_check + investments + loans + unemp_rate (overall)
run_var_combo(bank_vars['deposits_check'], bank_vars['investments'],
              bank_vars['loans'], 'unemp_rate',
              "deposits_check + investments + loans + unemp_rate")
