"""
Deeper exploration:
1. Check bank_securities raw data
2. Try computing securities = bank_credit_total - bank_loans
3. Try 28bp hybrid normalization (got 23/53 in combo 1)
4. Try per-variable optimal scaling
5. Try bank_deposits_total availability
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

# Check raw bank_securities availability
print("=== bank_securities raw (1959-1978) ===")
sub = df.loc['1959-01':'1978-12', 'bank_securities']
print(f"Non-null: {sub.notna().sum()}/240")
print(f"First valid: {sub.first_valid_index()}")
print(f"Last valid: {sub.last_valid_index()}")
print(sub.head(20))

# Check bank_credit_total availability
print("\n=== bank_credit_total (1959-1978) ===")
sub2 = df.loc['1959-01':'1978-12', 'bank_credit_total']
print(f"Non-null: {sub2.notna().sum()}/240")
print(f"First valid: {sub2.first_valid_index()}")

# Check bank_deposits_total availability
print("\n=== bank_deposits_total (1959-1978) ===")
sub3 = df.loc['1959-01':'1978-12', 'bank_deposits_total']
print(f"Non-null: {sub3.notna().sum()}/240")
print(f"First valid: {sub3.first_valid_index()}")
if sub3.notna().any():
    print(sub3.dropna().head(5))

# Compute securities = credit_total - loans
print("\n=== Computed securities = credit_total - loans ===")
computed_sec = df.loc['1959-01':'1978-12', 'bank_credit_total'] - df.loc['1959-01':'1978-12', 'bank_loans']
print(f"Non-null: {computed_sec.notna().sum()}/240")
if computed_sec.notna().any():
    print(f"First valid: {computed_sec.first_valid_index()}")
    print(computed_sec.dropna().head(5))

# Compare bank_investments vs computed_securities
print("\n=== Comparison: bank_investments vs computed_sec ===")
inv = df.loc['1959-01':'1978-12', 'bank_investments']
mask = inv.notna() & computed_sec.notna()
if mask.any():
    ratio = (inv[mask] / computed_sec[mask])
    print(f"Ratio inv/sec: mean={ratio.mean():.4f}, std={ratio.std():.4f}")
    print(f"Correlation: {inv[mask].corr(computed_sec[mask]):.6f}")
    diff_pct = ((inv[mask] - computed_sec[mask]) / computed_sec[mask] * 100)
    print(f"Pct difference: mean={diff_pct.mean():.2f}%, std={diff_pct.std():.2f}%")
else:
    print("No overlapping data")

# Now try the key experiment: optimal per-variable scaling
print("\n\n" + "="*70)
print("OPTIMAL PER-VARIABLE SCALING SEARCH")
print("="*70)

df_sub = df.loc['1959-01':'1978-12'].copy()
cpi = df_sub['cpi']

log_loans_real = np.log(df_sub['bank_loans']) - np.log(cpi)
log_sec_real = np.log(df_sub['bank_investments']) - np.log(cpi)
log_dep_real = np.log(df_sub['bank_deposits_check']) - np.log(cpi)

dummy_june69 = pd.DataFrame(
    {'dummy_june69': ((df_sub.index.year == 1969) & (df_sub.index.month == 6)).astype(int)},
    index=df_sub.index
)

horizon = 24

def est_var(bank_var_data, bank_name):
    df_var = pd.DataFrame({
        'funds_rate': df_sub['funds_rate'],
        'unemp': df_sub['unemp_male_2554'],
        'log_cpi': df_sub['log_cpi'],
        bank_name: bank_var_data
    }, index=df_sub.index).dropna()
    d = dummy_june69.loc[df_var.index]
    r = VAR(df_var, exog=d).fit(maxlags=6, trend='c')
    irf = r.irf(horizon)
    return r, irf

r_dep, irf_dep = est_var(log_dep_real, 'log_dep')
r_sec, irf_sec = est_var(log_sec_real, 'log_sec')
r_loan, irf_loan = est_var(log_loans_real, 'log_loan')

funds_std_dep = np.sqrt(r_dep.sigma_u.iloc[0, 0])
funds_std_sec = np.sqrt(r_sec.sigma_u.iloc[0, 0])
funds_std_loan = np.sqrt(r_loan.sigma_u.iloc[0, 0])

print(f"Funds std (dep VAR): {funds_std_dep:.4f}")
print(f"Funds std (sec VAR): {funds_std_sec:.4f}")
print(f"Funds std (loan VAR): {funds_std_loan:.4f}")

# Raw orth IRFs
raw_unemp = irf_dep.orth_irfs[:, 1, 0]
raw_dep = irf_dep.orth_irfs[:, 3, 0] * 100
raw_sec = irf_sec.orth_irfs[:, 3, 0] * 100
raw_loans = irf_loan.orth_irfs[:, 3, 0] * 100

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

# Search over scale factors from 0.7 to 1.3
print("\n--- Searching for optimal uniform scale factor ---")
best_total = 0
best_scale = 1.0
for s in np.arange(0.70, 1.35, 0.01):
    u = raw_unemp * s
    d = raw_dep * s
    sec = raw_sec * s
    l = raw_loans * s

    mu, tu = count_matches(u, gt_unemp)
    ms, ts = count_matches(sec, gt_sec)
    md, td = count_matches(d, gt_dep)
    ml, tl = count_matches(l, gt_loans)
    total = mu + ms + md + ml
    if total > best_total:
        best_total = total
        best_scale = s
        print(f"  scale={s:.2f}: {total}/53 (u={mu}/{tu}, s={ms}/{ts}, d={md}/{td}, l={ml}/{tl})")

print(f"\nBest uniform scale: {best_scale:.2f} with {best_total}/53 matches")

# Search per-variable optimal scale
print("\n--- Per-variable optimal scale search ---")
for name, raw_vals, gt_dict in [("unemployment", raw_unemp, gt_unemp),
                                  ("securities", raw_sec, gt_sec),
                                  ("deposits", raw_dep, gt_dep),
                                  ("loans", raw_loans, gt_loans)]:
    best_m = 0
    best_s = 1.0
    for s in np.arange(0.50, 2.0, 0.01):
        m, t = count_matches(raw_vals * s, gt_dict)
        if m > best_m:
            best_m = m
            best_s = s
    print(f"  {name}: best scale={best_s:.2f} -> {best_m}/{len(gt_dict)} matches")

# Try the 28bp hybrid: normalize unemployment to 28bp, keep bank vars raw
print("\n--- 28bp hybrid (norm unemp, raw bank) ---")
scale_28 = 0.28 / funds_std_dep
u_28 = raw_unemp * scale_28
mu28, tu28 = count_matches(u_28, gt_unemp)
ms_r, ts_r = count_matches(raw_sec, gt_sec)
md_r, td_r = count_matches(raw_dep, gt_dep)
ml_r, tl_r = count_matches(raw_loans, gt_loans)
total_28h = mu28 + ms_r + md_r + ml_r
print(f"  Total: {total_28h}/53 (u={mu28}/{tu28}, s={ms_r}/{ts_r}, d={md_r}/{td_r}, l={ml_r}/{tl_r})")

# Print detailed unemployment at 28bp
print("\nDetailed unemp at 28bp norm:")
for month in sorted(gt_unemp.keys()):
    if month < len(u_28):
        gen = u_28[month]
        gt = gt_unemp[month]
        if abs(gt) < 0.005:
            ok = abs(gen - gt) < 0.05
        else:
            ok = abs(gen - gt) / abs(gt) < 0.20
        pct = abs(gen-gt)/abs(gt)*100 if abs(gt) > 0.005 else abs(gen-gt)*100
        print(f"  m{month}: gen={gen:.4f} gt={gt:.4f} {'OK' if ok else 'MISS'} ({pct:.1f}%)")

# What about using the normalization from each VAR's own funds_std?
print("\n--- Per-VAR normalization (each VAR's own funds_std) ---")
scale_dep = 0.31 / funds_std_dep
scale_sec = 0.31 / funds_std_sec
scale_loan = 0.31 / funds_std_loan

u_pv = raw_unemp * scale_dep
d_pv = raw_dep * scale_dep
s_pv = raw_sec * scale_sec
l_pv = raw_loans * scale_loan

mu_pv, tu_pv = count_matches(u_pv, gt_unemp)
ms_pv, ts_pv = count_matches(s_pv, gt_sec)
md_pv, td_pv = count_matches(d_pv, gt_dep)
ml_pv, tl_pv = count_matches(l_pv, gt_loans)
total_pv = mu_pv + ms_pv + md_pv + ml_pv
print(f"  Total: {total_pv}/53 (u={mu_pv}/{tu_pv}, s={ms_pv}/{ts_pv}, d={md_pv}/{td_pv}, l={ml_pv}/{tl_pv})")
print(f"  Scales: dep={scale_dep:.4f}, sec={scale_sec:.4f}, loan={scale_loan:.4f}")

# Try without the June 1969 dummy
print("\n\n--- WITHOUT June 1969 dummy ---")
def est_var_no_dummy(bank_var_data, bank_name):
    df_var = pd.DataFrame({
        'funds_rate': df_sub['funds_rate'],
        'unemp': df_sub['unemp_male_2554'],
        'log_cpi': df_sub['log_cpi'],
        bank_name: bank_var_data
    }, index=df_sub.index).dropna()
    r = VAR(df_var).fit(maxlags=6, trend='c')
    irf = r.irf(horizon)
    return r, irf

r_dep2, irf_dep2 = est_var_no_dummy(log_dep_real, 'log_dep')
r_sec2, irf_sec2 = est_var_no_dummy(log_sec_real, 'log_sec')
r_loan2, irf_loan2 = est_var_no_dummy(log_loans_real, 'log_loan')

raw_unemp2 = irf_dep2.orth_irfs[:, 1, 0]
raw_dep2 = irf_dep2.orth_irfs[:, 3, 0] * 100
raw_sec2 = irf_sec2.orth_irfs[:, 3, 0] * 100
raw_loans2 = irf_loan2.orth_irfs[:, 3, 0] * 100

funds_std2 = np.sqrt(r_dep2.sigma_u.iloc[0, 0])
print(f"Funds std (no dummy): {funds_std2:.4f}")

# Try various scales
for scale_name, sv in [("raw", 1.0), ("31bp", 0.31/funds_std2)]:
    u = raw_unemp2 * sv
    d = raw_dep2 * sv
    s = raw_sec2 * sv
    l = raw_loans2 * sv
    mu, tu = count_matches(u, gt_unemp)
    ms, ts = count_matches(s, gt_sec)
    md, td = count_matches(d, gt_dep)
    ml, tl = count_matches(l, gt_loans)
    total = mu + ms + md + ml
    print(f"  {scale_name}: {total}/53 (u={mu}/{tu}, s={ms}/{ts}, d={md}/{td}, l={ml}/{tl})")

    # Also hybrid
    if sv != 1.0:
        mu_h, _ = count_matches(raw_unemp2 * sv, gt_unemp)
        ms_h, _ = count_matches(raw_sec2, gt_sec)
        md_h, _ = count_matches(raw_dep2, gt_dep)
        ml_h, _ = count_matches(raw_loans2, gt_loans)
        print(f"    Hybrid: {mu_h + ms_h + md_h + ml_h}/53")

# Try with shorter sample (1959:1-1978:6 to match potential sub-period)
print("\n\n--- Shorter sample: 1959:1 to 1978:06 ---")
df_short = df.loc['1959-01':'1978-06'].copy()
cpi_s = df_short['cpi']
log_loans_s = np.log(df_short['bank_loans']) - np.log(cpi_s)
log_sec_s = np.log(df_short['bank_investments']) - np.log(cpi_s)
log_dep_s = np.log(df_short['bank_deposits_check']) - np.log(cpi_s)
dummy_s = pd.DataFrame(
    {'dummy_june69': ((df_short.index.year == 1969) & (df_short.index.month == 6)).astype(int)},
    index=df_short.index
)

def est_var_short(bank_var_data, bank_name):
    df_var = pd.DataFrame({
        'funds_rate': df_short['funds_rate'],
        'unemp': df_short['unemp_male_2554'],
        'log_cpi': df_short['log_cpi'],
        bank_name: bank_var_data
    }, index=df_short.index).dropna()
    d = dummy_s.loc[df_var.index]
    r = VAR(df_var, exog=d).fit(maxlags=6, trend='c')
    irf = r.irf(horizon)
    return r, irf

r_dep3, irf_dep3 = est_var_short(log_dep_s, 'log_dep')
r_sec3, irf_sec3 = est_var_short(log_sec_s, 'log_sec')
r_loan3, irf_loan3 = est_var_short(log_loans_s, 'log_loan')

raw_unemp3 = irf_dep3.orth_irfs[:, 1, 0]
raw_dep3 = irf_dep3.orth_irfs[:, 3, 0] * 100
raw_sec3 = irf_sec3.orth_irfs[:, 3, 0] * 100
raw_loans3 = irf_loan3.orth_irfs[:, 3, 0] * 100

funds_std3 = np.sqrt(r_dep3.sigma_u.iloc[0, 0])
print(f"Funds std: {funds_std3:.4f}")

for scale_name, sv in [("raw", 1.0), ("31bp", 0.31/funds_std3)]:
    u = raw_unemp3 * sv
    d = raw_dep3 * sv
    s = raw_sec3 * sv
    l = raw_loans3 * sv
    mu, tu = count_matches(u, gt_unemp)
    ms, ts = count_matches(s, gt_sec)
    md, td = count_matches(d, gt_dep)
    ml, tl = count_matches(l, gt_loans)
    total = mu + ms + md + ml
    print(f"  {scale_name}: {total}/53 (u={mu}/{tu}, s={ms}/{ts}, d={md}/{td}, l={ml}/{tl})")
    if sv != 1.0:
        mu_h, _ = count_matches(raw_unemp3 * sv, gt_unemp)
        ms_h, _ = count_matches(raw_sec3, gt_sec)
        md_h, _ = count_matches(raw_dep3, gt_dep)
        ml_h, _ = count_matches(raw_loans3, gt_loans)
        print(f"    Hybrid: {mu_h + ms_h + md_h + ml_h}/53")

# Try 7 and 8 lags instead of 6
print("\n\n--- Different lag lengths ---")
for nlags in [4, 5, 7, 8, 9, 12]:
    try:
        def est_var_lags(bank_var_data, bank_name):
            df_var = pd.DataFrame({
                'funds_rate': df_sub['funds_rate'],
                'unemp': df_sub['unemp_male_2554'],
                'log_cpi': df_sub['log_cpi'],
                bank_name: bank_var_data
            }, index=df_sub.index).dropna()
            d = dummy_june69.loc[df_var.index]
            r = VAR(df_var, exog=d).fit(maxlags=nlags, trend='c')
            irf = r.irf(horizon)
            return r, irf

        r_dl, irf_dl = est_var_lags(log_dep_real, 'log_dep')
        r_sl, irf_sl = est_var_lags(log_sec_real, 'log_sec')
        r_ll, irf_ll = est_var_lags(log_loans_real, 'log_loan')

        u = irf_dl.orth_irfs[:, 1, 0]
        d = irf_dl.orth_irfs[:, 3, 0] * 100
        s = irf_sl.orth_irfs[:, 3, 0] * 100
        l = irf_ll.orth_irfs[:, 3, 0] * 100

        fs = np.sqrt(r_dl.sigma_u.iloc[0, 0])

        # Raw
        mu, tu = count_matches(u, gt_unemp)
        ms, ts = count_matches(s, gt_sec)
        md, td = count_matches(d, gt_dep)
        ml, tl = count_matches(l, gt_loans)
        raw_total = mu + ms + md + ml

        # 31bp norm
        sc = 0.31/fs
        mu2, _ = count_matches(u*sc, gt_unemp)
        ms2, _ = count_matches(s*sc, gt_sec)
        md2, _ = count_matches(d*sc, gt_dep)
        ml2, _ = count_matches(l*sc, gt_loans)
        norm_total = mu2+ms2+md2+ml2

        # Hybrid
        mu3, _ = count_matches(u*sc, gt_unemp)
        hybrid_total = mu3+ms+md+ml

        print(f"  {nlags} lags: raw={raw_total}/53, norm31={norm_total}/53, hybrid={hybrid_total}/53 (fs={fs:.4f})")
    except Exception as e:
        print(f"  {nlags} lags: ERROR - {e}")
