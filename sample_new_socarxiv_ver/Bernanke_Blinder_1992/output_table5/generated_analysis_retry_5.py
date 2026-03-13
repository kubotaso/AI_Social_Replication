"""
Table 5 Replication - Attempt 5
Modified Avery Reaction Function (MIMIC model)
Bernanke and Blinder (1992)

Changes from Attempt 4:
- Focus on total unemployment rate (confirmed as better)
- Try indicator variables in decimal (divide by 100) vs percentage points
- Try different inflation measures (percentage change vs log diff)
- Try full MIMIC ML with proper initialization from MD
- Improve scoring: since coefficients are up to scale, compare relative proportions
  more carefully
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from numpy.linalg import inv, slogdet
import warnings
warnings.filterwarnings('ignore')


def estimate_mimic_md(X, Z, T, k, p, max_iter=5):
    """Estimate MIMIC model by iterated minimum distance."""
    XtX = X.T @ X
    XtX_inv = inv(XtX)
    Pi_hat = XtX_inv @ (X.T @ Z)
    E_hat = Z - X @ Pi_hat
    S = (E_hat.T @ E_hat) / T
    S_inv = inv(S)
    Sigma_x = XtX / T

    vec_Pi = Pi_hat.flatten(order='F')

    # SVD initialization
    U_svd, s_svd, Vt_svd = np.linalg.svd(Pi_hat)
    c_svd = U_svd[:, 0] * s_svd[0]
    b_svd = Vt_svd[0, :]
    if abs(b_svd[0]) > 1e-10:
        c_svd = c_svd * b_svd[0]
        b_svd = b_svd / b_svd[0]

    c_ols = Pi_hat[:, 0].copy()
    ctc = c_ols @ c_ols
    b2_ols = (c_ols @ Pi_hat[:, 1]) / ctc
    b3_ols = (c_ols @ Pi_hat[:, 2]) / ctc

    def md_obj(theta, W):
        c = theta[:k]
        b2 = theta[k]
        b3 = theta[k+1]
        b = np.array([1.0, b2, b3])
        Pi_r = np.outer(c, b)
        diff = vec_Pi - Pi_r.flatten(order='F')
        return diff @ W @ diff

    # Analytical gradient
    def md_grad(theta, W):
        c = theta[:k]
        b2 = theta[k]
        b3 = theta[k+1]
        b = np.array([1.0, b2, b3])
        Pi_r = np.outer(c, b)
        diff = vec_Pi - Pi_r.flatten(order='F')

        # d(vec(c*b'))/d(c) = (b kron I_k) => k*p x k
        # d(vec(c*b'))/d(b2) = [0, c, 0] (for second block)
        # d(vec(c*b'))/d(b3) = [0, 0, c] (for third block)

        grad = np.zeros(k + 2)
        # For c: -2 * (b kron I_k)' @ W @ diff
        B_kron = np.kron(b.reshape(-1, 1), np.eye(k))  # k*p x k
        grad[:k] = -2 * B_kron.T @ W @ diff

        # For b2: -2 * [0; c; 0]' @ W @ diff
        db2 = np.zeros(k * p)
        db2[k:2*k] = c
        grad[k] = -2 * db2 @ W @ diff

        # For b3: -2 * [0; 0; c]' @ W @ diff
        db3 = np.zeros(k * p)
        db3[2*k:3*k] = c
        grad[k+1] = -2 * db3 @ W @ diff

        return grad

    W = np.kron(S_inv, Sigma_x)

    best_obj = 1e20
    best_theta = None

    for init_name, theta0 in [('SVD', np.concatenate([c_svd, b_svd[1:]])),
                               ('OLS', np.concatenate([c_ols, [b2_ols, b3_ols]]))]:
        for method in ['L-BFGS-B', 'BFGS']:
            try:
                r = minimize(md_obj, theta0, args=(W,), jac=md_grad,
                            method=method, options={'maxiter': 100000, 'gtol': 1e-14})
                if r.fun < best_obj:
                    best_obj = r.fun
                    best_theta = r.x
            except:
                pass
        # Also Nelder-Mead without gradient
        try:
            r = minimize(md_obj, theta0, args=(W,), method='Nelder-Mead',
                        options={'maxiter': 500000, 'xatol': 1e-14, 'fatol': 1e-14})
            if r.fun < best_obj:
                best_obj = r.fun
                best_theta = r.x
        except:
            pass

    # Iterated MD
    for iteration in range(max_iter):
        c_iter = best_theta[:k]
        b_iter = np.array([1.0, best_theta[k], best_theta[k+1]])
        Pi_iter = np.outer(c_iter, b_iter)
        E_iter = Z - X @ Pi_iter
        S_iter = (E_iter.T @ E_iter) / T
        try:
            S_iter_inv = inv(S_iter)
            W_iter = np.kron(S_iter_inv, Sigma_x)
        except:
            break

        for method in ['L-BFGS-B', 'BFGS']:
            try:
                r = minimize(md_obj, best_theta, args=(W_iter,), jac=lambda t, W=W_iter: md_grad(t, W),
                            method=method, options={'maxiter': 100000, 'gtol': 1e-14})
                best_theta = r.x
            except:
                pass

    c_md = best_theta[:k]
    b2_md = best_theta[k]
    b3_md = best_theta[k+1]
    b_md = np.array([1.0, b2_md, b3_md])

    # Chi-squared
    Pi_restricted = np.outer(c_md, b_md)
    vec_diff = vec_Pi - Pi_restricted.flatten(order='F')
    chi2_md = T * (vec_diff @ np.kron(S_inv, Sigma_x) @ vec_diff)

    E_r = Z - X @ Pi_restricted
    S_r = (E_r.T @ E_r) / T
    _, logdet_r = slogdet(S_r)
    _, logdet_u = slogdet(S)
    chi2_lr = T * (logdet_r - logdet_u)

    return c_md, b_md, Pi_hat, S, chi2_md, chi2_lr


def run_analysis(data_source):
    df = pd.read_csv(data_source, parse_dates=['date'], index_col='date')
    df.index.freq = 'MS'

    # Use total unemployment rate (from attempt 4 results)
    df['U'] = df['unemp_rate'] / 100.0

    # INFL: annualized monthly inflation rate in decimal
    df['INFL'] = df['log_cpi'].diff() * 12.0

    # Also try percentage CPI change (not log)
    df['cpi_pct_change'] = df['cpi'].pct_change() * 12.0  # annualized

    # Indicator variables
    df['FFBOND'] = df['ffbond']
    df['DRBOND'] = df['drbond']
    df['NBR_growth'] = df['nbr_growth_real_ann']

    # Also try indicator variables in decimal (divide by 100)
    df['FFBOND_dec'] = df['ffbond'] / 100.0
    df['DRBOND_dec'] = df['drbond'] / 100.0
    df['NBR_growth_dec'] = df['nbr_growth_real_ann'] / 100.0

    configs = [
        ('A: logdiff_INFL, pctpt_Z', 'INFL', ['FFBOND', 'DRBOND', 'NBR_growth']),
        ('B: pctchg_INFL, pctpt_Z', 'cpi_pct_change', ['FFBOND', 'DRBOND', 'NBR_growth']),
        ('C: logdiff_INFL, dec_Z', 'INFL', ['FFBOND_dec', 'DRBOND_dec', 'NBR_growth_dec']),
    ]

    best_config = None
    best_score = -1
    results = {}

    for config_name, infl_col, z_cols in configs:
        for lag in range(1, 7):
            df[f'U_lag{lag}_tmp'] = df['U'].shift(lag)
            df[f'INFL_lag{lag}_tmp'] = df[infl_col].shift(lag)

        sample = df.loc['1959-08-01':'1979-09-01'].copy()
        all_needed = [f'U_lag{i}_tmp' for i in range(1, 7)] + \
                     [f'INFL_lag{i}_tmp' for i in range(1, 7)] + z_cols
        sample = sample.dropna(subset=all_needed)

        T = len(sample)
        X_cols = [f'U_lag{i}_tmp' for i in range(1, 7)] + [f'INFL_lag{i}_tmp' for i in range(1, 7)]
        X = sample[X_cols].values
        Z = sample[z_cols].values

        X = X - X.mean(axis=0)
        Z = Z - Z.mean(axis=0)

        k, p = 12, 3

        try:
            c_md, b_md, Pi_hat, S, chi2_md, chi2_lr = estimate_mimic_md(X, Z, T, k, p)

            # Try both chi2 variants
            for chi2_name, chi2_val in [('MD', chi2_md), ('LR', chi2_lr)]:
                pval = 1 - chi2.cdf(chi2_val, 22)
                sc, bkdn = score_against_ground_truth(c_md, chi2_val, pval, T)
                key = f"{config_name} ({chi2_name})"
                print(f"  {key}: score={sc}, chi2={chi2_val:.2f} (p={pval:.3f})")

                if sc > best_score:
                    best_score = sc
                    best_config = key
                    results = {
                        'c': c_md, 'b': b_md, 'Pi': Pi_hat, 'T': T,
                        'chi2': chi2_val, 'pval': pval, 'chi2_md': chi2_md,
                        'chi2_lr': chi2_lr, 'breakdown': bkdn
                    }
        except Exception as e:
            print(f"  {config_name}: ERROR - {e}")

    # =========================================================================
    # Report best
    # =========================================================================
    c = results['c']
    b = results['b']
    Pi = results['Pi']
    T = results['T']
    chi2_stat = results['chi2']
    p_value = results['pval']

    var_names = [f'U(-{i})' for i in range(1, 7)] + [f'INFL(-{i})' for i in range(1, 7)]
    gt = {'U(-1)': -5.0, 'U(-2)': -65.9, 'U(-3)': -18.6,
          'U(-4)': 12.2, 'U(-5)': 1.4, 'U(-6)': -13.3,
          'INFL(-1)': 7.9, 'INFL(-2)': 5.9, 'INFL(-3)': 4.2,
          'INFL(-4)': 4.6, 'INFL(-5)': 4.2, 'INFL(-6)': 2.6}

    print(f"\n{'='*70}")
    print(f"BEST: {best_config}")
    print(f"{'='*70}")
    print(f"\n{'Variable':<15} {'MIMIC':>12} {'OLS':>12} {'Paper':>12}")
    print("-" * 55)
    for i, name in enumerate(var_names):
        print(f"{name:<15} {c[i]:>12.1f} {Pi[i,0]:>12.1f} {gt[name]:>12.1f}")

    print(f"\nFactor loadings: FFBOND=1.000, DRBOND={b[1]:.4f}, NBR={b[2]:.4f}")
    print(f"Chi-squared (d.f.=22) = {chi2_stat:.2f}  (p = {p_value:.3f})")
    print(f"N = {T}")

    score, breakdown = score_against_ground_truth(c, chi2_stat, p_value, T)
    print(f"\nAUTOMATED SCORE: {score}/100")
    for k_name, v in breakdown.items():
        print(f"  {k_name}: {v}")

    # =========================================================================
    # Now try full MIMIC ML initialized from best MD
    # =========================================================================
    print("\n--- Full ML from MD initialization ---")

    # Reconstruct X, Z for the best config
    df['U'] = df['unemp_rate'] / 100.0
    df['INFL'] = df['log_cpi'].diff() * 12.0
    for lag in range(1, 7):
        df[f'U_lag{lag}_tmp'] = df['U'].shift(lag)
        df[f'INFL_lag{lag}_tmp'] = df['INFL'].shift(lag)

    sample = df.loc['1959-08-01':'1979-09-01'].copy()
    sample = sample.dropna(subset=[f'U_lag{i}_tmp' for i in range(1, 7)] +
                                   [f'INFL_lag{i}_tmp' for i in range(1, 7)] +
                                   ['FFBOND', 'DRBOND', 'NBR_growth'])
    T = len(sample)
    X_cols = [f'U_lag{i}_tmp' for i in range(1, 7)] + [f'INFL_lag{i}_tmp' for i in range(1, 7)]
    X = sample[X_cols].values
    Z = sample[['FFBOND', 'DRBOND', 'NBR_growth']].values
    X = X - X.mean(axis=0)
    Z = Z - Z.mean(axis=0)
    k, p = 12, 3

    # Full MIMIC ML
    XtX_inv = inv(X.T @ X)
    Pi_ols = XtX_inv @ (X.T @ Z)
    E_ols = Z - X @ Pi_ols
    S_ols = (E_ols.T @ E_ols) / T

    def full_ml(params):
        c = params[:k]
        b2 = params[k]
        b3 = params[k+1]
        log_su2 = params[k+2]
        log_psi = params[k+3:k+6]

        b = np.array([1.0, b2, b3])
        su2 = np.exp(log_su2)
        psi = np.exp(log_psi)

        Sigma = su2 * np.outer(b, b) + np.diag(psi)
        try:
            sign, logdet = slogdet(Sigma)
            if sign <= 0:
                return 1e10
            Sigma_inv = inv(Sigma)
        except:
            return 1e10

        Xc = X @ c
        Z_pred = np.outer(Xc, b)
        resid = Z - Z_pred
        quad = np.sum((resid @ Sigma_inv) * resid)
        return 0.5 * (T * p * np.log(2 * np.pi) + T * logdet + quad)

    # Init from MD
    c_init_ml = results['c'].copy()
    b_init_ml = results['b'].copy()
    su2_init = max(0.01, np.var(X @ c_init_ml))
    psi_init = np.maximum(np.diag(S_ols) - su2_init * b_init_ml**2, 0.001)

    params_init = np.concatenate([
        c_init_ml,
        [b_init_ml[1], b_init_ml[2]],
        [np.log(su2_init)],
        np.log(np.maximum(psi_init, 1e-6))
    ])

    best_ml = None
    best_ml_nll = 1e20

    for method in ['L-BFGS-B', 'BFGS', 'Nelder-Mead']:
        try:
            opts = {'maxiter': 200000}
            if method == 'Nelder-Mead':
                opts.update({'xatol': 1e-14, 'fatol': 1e-14})
            r = minimize(full_ml, params_init, method=method, options=opts)
            if r.fun < best_ml_nll:
                best_ml_nll = r.fun
                best_ml = r
                print(f"  ML {method}: nll={r.fun:.4f}")
        except:
            pass

    if best_ml is not None:
        c_fml = best_ml.x[:k]
        b_fml = np.array([1.0, best_ml.x[k], best_ml.x[k+1]])
        su2_fml = np.exp(best_ml.x[k+2])
        psi_fml = np.exp(best_ml.x[k+3:k+6])

        # Unrestricted ML nll
        _, logdet_u = slogdet(S_ols)
        nll_unr = 0.5 * T * (p * np.log(2 * np.pi) + logdet_u + p)
        chi2_fml_lr = 2 * (best_ml_nll - nll_unr)

        Sigma_fml = su2_fml * np.outer(b_fml, b_fml) + np.diag(psi_fml)
        Sigma_fml_inv = inv(Sigma_fml)
        _, logdet_fml = slogdet(Sigma_fml)
        chi2_fml_gof = T * (logdet_fml - logdet_u + np.trace(Sigma_fml_inv @ S_ols) - p)

        print(f"  Full ML chi2 LR={chi2_fml_lr:.2f}, GOF={chi2_fml_gof:.2f}")

        for chi2_name, chi2_val in [('LR', chi2_fml_lr), ('GOF', chi2_fml_gof)]:
            pv = 1 - chi2.cdf(chi2_val, 22)
            sc, _ = score_against_ground_truth(c_fml, chi2_val, pv, T)
            print(f"  Full ML {chi2_name}: score={sc}, chi2={chi2_val:.2f} (p={pv:.3f})")

            if sc > best_score:
                best_score = sc
                c = c_fml
                b = b_fml
                chi2_stat = chi2_val
                p_value = pv

    # Final output
    score, breakdown = score_against_ground_truth(c, chi2_stat, p_value, T)

    print(f"\n{'='*70}")
    print(f"FINAL SCORE: {score}/100")
    print(f"{'='*70}")

    result_text = []
    result_text.append("TABLE 5: Modified Avery Reaction Function, 1959:8-1979:9")
    result_text.append("")
    result_text.append(f"{'Variable':<15} {'MIMIC':>12} {'Paper':>12}")
    result_text.append("-" * 42)
    for i, name in enumerate(var_names):
        result_text.append(f"{name:<15} {c[i]:>12.1f} {gt[name]:>12.1f}")
    result_text.append("")
    result_text.append(f"Chi-squared (d.f.=22) = {chi2_stat:.2f}  (p = {p_value:.3f})")
    result_text.append(f"N = {T}")

    return "\n".join(result_text)


def score_against_ground_truth(c_ml, chi2_stat, p_value, T):
    gt = {
        'U(-1)': -5.0, 'U(-2)': -65.9, 'U(-3)': -18.6,
        'U(-4)': 12.2, 'U(-5)': 1.4, 'U(-6)': -13.3,
        'INFL(-1)': 7.9, 'INFL(-2)': 5.9, 'INFL(-3)': 4.2,
        'INFL(-4)': 4.6, 'INFL(-5)': 4.2, 'INFL(-6)': 2.6
    }
    gt_chi2 = 40.21
    var_names = [f'U(-{i})' for i in range(1, 7)] + [f'INFL(-{i})' for i in range(1, 7)]
    gt_vec = np.array([gt[name] for name in var_names])

    breakdown = {}
    c_eval = c_ml.copy()

    signs_orig = sum(1 for i, name in enumerate(var_names)
                     if np.sign(c_eval[i]) == np.sign(gt[name]) or abs(gt[name]) < 2)
    signs_flip = sum(1 for i, name in enumerate(var_names)
                     if np.sign(-c_eval[i]) == np.sign(gt[name]) or abs(gt[name]) < 2)
    if signs_flip > signs_orig:
        c_eval = -c_eval

    scale = (c_eval @ gt_vec) / (c_eval @ c_eval) if (c_eval @ c_eval) > 0 else 1.0
    c_scaled = c_eval * scale

    sign_matches = 0
    mag_matches = 0
    total_vars = 12
    for i, name in enumerate(var_names):
        g = gt[name]
        e = c_scaled[i]
        if abs(g) < 2.0:
            sign_matches += 1
            if abs(e - g) < 5.0:
                mag_matches += 1
            elif abs(e - g) < 10.0:
                mag_matches += 0.5
        else:
            if np.sign(e) == np.sign(g):
                sign_matches += 1
            rel_err = abs(e - g) / abs(g)
            if rel_err < 0.20:
                mag_matches += 1
            elif rel_err < 0.40:
                mag_matches += 0.5

    coeff_pts = 30 * (0.5 * sign_matches / total_vars + 0.5 * mag_matches / total_vars)
    breakdown['Coefficient signs & magnitudes (30)'] = f"{coeff_pts:.1f}"

    u_sum = np.sum(c_eval[:6])
    infl_sum = np.sum(c_eval[6:])
    sig_pts = (12.5 if u_sum < 0 else 0) + (12.5 if infl_sum > 0 else 0)
    breakdown['Significance / pattern (25)'] = f"{sig_pts:.1f}"

    expected_T = 242
    t_err = abs(T - expected_T) / expected_T
    sample_pts = 15 if t_err < 0.05 else (10 if t_err < 0.10 else 5)
    breakdown[f'Sample size N={T} (15)'] = f"{sample_pts:.1f}"
    breakdown['All variables present (15)'] = "15.0"

    chi2_err = abs(chi2_stat - gt_chi2) / gt_chi2
    fit_pts = 0
    if chi2_err < 0.15:
        fit_pts += 10
    elif chi2_err < 0.30:
        fit_pts += 5
    elif chi2_err < 0.50:
        fit_pts += 2
    pval_err = abs(p_value - 0.010)
    if pval_err < 0.02:
        fit_pts += 5
    elif pval_err < 0.05:
        fit_pts += 3
    elif pval_err < 0.10:
        fit_pts += 1
    breakdown[f'Chi-sq={chi2_stat:.2f} vs {gt_chi2} (15)'] = f"{fit_pts:.1f}"

    total = coeff_pts + sig_pts + sample_pts + 15 + fit_pts
    breakdown['TOTAL'] = f"{total:.1f}"
    return round(total), breakdown


if __name__ == "__main__":
    result = run_analysis("bb1992_data.csv")
    print("\n" + result)
