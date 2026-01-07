def run_analysis(data_source, sep=None, na_values=None):
    import math
    import numpy as np
    import pandas as pd

    # ---------------------------
    # Helpers
    # ---------------------------
    def read_data(path, sep_opt, na_opt):
        # Default to comma if sep is None to avoid pandas engine warnings
        use_sep = sep_opt if sep_opt is not None else ','
        return pd.read_csv(path, sep=use_sep, na_values=na_opt, low_memory=False)

    def indicator_from_condition(base_series, cond_bool):
        # Return 1.0 for True, 0.0 for False, NaN where base_series is NaN
        out = pd.Series(np.nan, index=base_series.index, dtype=float)
        notna = base_series.notna()
        out.loc[notna & cond_bool] = 1.0
        out.loc[notna & (~cond_bool)] = 0.0
        return out

    def construct_dv(df):
        # 18 genre items
        genres = [
            'bigband', 'blugrass', 'country', 'blues', 'musicals', 'classicl', 'folk',
            'gospel', 'jazz', 'latin', 'moodeasy', 'newage', 'opera', 'rap', 'reggae',
            'conrock', 'oldies', 'hvymetal'
        ]
        # Require completeness on all 18 items
        complete_mask = df[genres].notna().all(axis=1)
        # Dislike if response in {4,5}
        dislike = df[genres].isin([4, 5]).astype(int)
        # Sum dislikes; set NaN if any missing among 18 items
        dv = dislike.sum(axis=1).astype(float)
        dv.loc[~complete_mask] = np.nan
        return dv

    def construct_income_pc(df):
        # Income per capita: realinc / hompop, hompop >= 1
        realinc = pd.to_numeric(df['realinc'], errors='coerce')
        hompop = pd.to_numeric(df['hompop'], errors='coerce')
        out = realinc / hompop
        out[(realinc.isna()) | (hompop.isna()) | (hompop < 1)] = np.nan
        return out

    def construct_prestige(df):
        return pd.to_numeric(df['prestg80'], errors='coerce')

    def construct_education(df):
        return pd.to_numeric(df['educ'], errors='coerce')

    def construct_age(df):
        age = pd.to_numeric(df['age'], errors='coerce')
        # Cap 89+ at 89
        return age.clip(upper=89)

    def construct_female(df):
        sex = pd.to_numeric(df['sex'], errors='coerce')
        return indicator_from_condition(sex, sex == 2)

    def construct_southern(df):
        region = pd.to_numeric(df['region'], errors='coerce')
        return indicator_from_condition(region, region == 3)

    def construct_hispanic_black_otherrace(df):
        # Hispanic mapping using ethnic codes
        # Spain/Other Spanish: 25, 38
        # Mexico, Puerto Rico: 17, 22
        # Central America (exclude 601 Belize): 602, 603, 604, 605, 606, 607, 699
        # South America (exclude 503, 507, 508, 511): 501, 502, 504, 505, 506, 509, 510, 512, 513
        # Caribbean Spanish: 801 (Cuba), 803 (Dominican Republic)
        hisp_codes = {
            25, 38, 17, 22, 602, 603, 604, 605, 606, 607, 699,
            501, 502, 504, 505, 506, 509, 510, 512, 513,
            801, 803
        }
        ethnic = pd.to_numeric(df['ethnic'], errors='coerce').astype('Int64')
        hisp_mask = ethnic.isin(hisp_codes).fillna(False)

        race = pd.to_numeric(df['race'], errors='coerce')
        # Black: race==2 & not Hispanic
        black = indicator_from_condition(race, (race == 2) & (~hisp_mask))
        # Other race: race==3 & not Hispanic
        other_race = indicator_from_condition(race, (race == 3) & (~hisp_mask))
        # Hispanic indicator (1 if Hispanic, else 0; missing if ethnic missing)
        hisp_ind = pd.Series(np.nan, index=df.index, dtype=float)
        hisp_ind.loc[ethnic.notna()] = hisp_mask.loc[ethnic.notna()].astype(float)

        return hisp_ind, black, other_race

    def construct_religion_dummies(df):
        # Conservative Protestant: relig==1 and denom in {1,6,7}
        relig = pd.to_numeric(df['relig'], errors='coerce')
        denom = pd.to_numeric(df['denom'], errors='coerce')
        cons_mask = (relig == 1) & (denom.isin([1, 6, 7]))
        cons_prot = indicator_from_condition(relig, cons_mask)

        # No religion: relig==4
        no_relig = indicator_from_condition(relig, relig == 4)

        return cons_prot, no_relig

    def construct_political_intolerance(df):
        # Create 15 intolerant item indicators; 1=intolerant, 0 otherwise, NaN if missing
        items = {
            'spkath': 2, 'colath': 5, 'libath': 1,
            'spkrac': 2, 'colrac': 5, 'librac': 1,
            'spkcom': 2, 'colcom': 4, 'libcom': 1,
            'spkmil': 2, 'colmil': 5, 'libmil': 1,
            'spkhomo': 2, 'colhomo': 5, 'libhomo': 1
        }
        scores = []
        for col, val in items.items():
            s = pd.to_numeric(df[col], errors='coerce')
            score = pd.Series(np.nan, index=df.index, dtype=float)
            score.loc[s.notna()] = (s.loc[s.notna()] == val).astype(float)
            scores.append(score)
        # Require complete cases on all 15 items
        mat = np.column_stack([s.values for s in scores])
        row_has_nan = np.any(np.isnan(mat), axis=1)
        total = np.nansum(mat, axis=1)
        total[row_has_nan] = np.nan
        return pd.Series(total, index=df.index, dtype=float)

    def ols_fit(y, X):
        # Add intercept
        Xmat = np.column_stack([np.ones(len(y)), X])
        # Drop any rows with NaN in X or y for safety (should be already filtered)
        mask = np.isfinite(y) & np.all(np.isfinite(Xmat), axis=1)
        Xmat = Xmat[mask, :]
        yv = y[mask]
        n, p1 = Xmat.shape  # p1 includes intercept
        k = p1 - 1

        # Solve OLS
        XtX = Xmat.T @ Xmat
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ (Xmat.T @ yv)  # includes intercept
        y_hat = Xmat @ beta
        resid = yv - y_hat

        # Metrics
        sse = float(resid.T @ resid)
        y_mean = float(np.mean(yv))
        tss = float(((yv - y_mean) ** 2).sum())
        r2 = 1.0 - (sse / tss) if tss > 0 else np.nan
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p1) if n > p1 else np.nan

        # Var-cov and t-stats for slopes (intercept included too)
        df_resid = n - p1
        if df_resid > 0:
            sigma2 = sse / df_resid
            vcov = sigma2 * XtX_inv
            se = np.sqrt(np.diag(vcov))
            t_stats = beta / se
        else:
            se = np.full_like(beta, np.nan, dtype=float)
            t_stats = np.full_like(beta, np.nan, dtype=float)

        return {
            'n': n,
            'k': k,
            'beta_unstd': beta,      # [intercept, slopes...]
            'se_unstd': se,          # std errors for [intercept, slopes...]
            't_unstd': t_stats,      # t-stats for [intercept, slopes...]
            'r2': r2,
            'adj_r2': adj_r2,
            'resid_df': df_resid,
            'X_used_mask': mask
        }

    def standardized_betas(y, X, fit_res):
        # Compute standardized betas for slopes only (not intercept)
        # beta_std_j = beta_unstd_j * (SD(X_j) / SD(Y))
        mask = fit_res['X_used_mask']
        yv = y[mask]
        Xv = X[mask, :]
        # standard deviations with ddof=1
        sd_y = np.std(yv, ddof=1)
        sd_x = np.std(Xv, axis=0, ddof=1)
        slopes_unstd = fit_res['beta_unstd'][1:]  # exclude intercept
        with np.errstate(invalid='ignore', divide='ignore'):
            betas = slopes_unstd * (sd_x / sd_y)
        return betas

    def stars_from_t(t_vals, df):
        # Use normal approximation to assign stars (robust without SciPy)
        # Critical z values: 1.96 (*), 2.576 (**), 3.291 (***)
        # This approximation is acceptable for moderate/large df.
        stars = []
        for t in t_vals:
            if not np.isfinite(t):
                stars.append('')
                continue
            at = abs(float(t))
            if at >= 3.291:
                stars.append('***')
            elif at >= 2.576:
                stars.append('**')
            elif at >= 1.960:
                stars.append('*')
            else:
                stars.append('')
        return stars

    def summarize_model(title, y, X, x_labels):
        fit = ols_fit(y, X)
        beta_std = standardized_betas(y, X, fit)
        # t-stats for slopes are same as unstandardized slopes (up to scale), use unstd t for stars
        t_slopes = fit['t_unstd'][1:]  # exclude intercept
        star_list = stars_from_t(t_slopes, fit['resid_df'])

        n = fit['n']
        r2 = fit['r2']
        adj_r2 = fit['adj_r2']
        intercept = fit['beta_unstd'][0]

        lines = []
        lines.append(f"{title} (N={n}; R2={r2:.3f}; Adj R2={adj_r2:.3f}; Constant={intercept:.3f})")
        for lab, b, s in zip(x_labels, beta_std, star_list):
            # keep three decimals for coefficients
            if pd.isna(b):
                coef_str = "NA"
            else:
                coef_str = f"{b:.3f}"
            if s:
                lines.append(f"  - {lab}: {coef_str}{s}")
            else:
                lines.append(f"  - {lab}: {coef_str}")
        return "\n".join(lines), {
            'title': title,
            'N': n,
            'R2': r2,
            'AdjR2': adj_r2,
            'Intercept': intercept,
            'betas': dict(zip(x_labels, beta_std)),
            'stars': dict(zip(x_labels, star_list))
        }

    # ---------------------------
    # Load and prepare data
    # ---------------------------
    df = read_data(data_source, sep, na_values)
    # Restrict to 1993
    if 'year' in df.columns:
        df = df.loc[pd.to_numeric(df['year'], errors='coerce') == 1993].copy()
    else:
        # If year not present, proceed (assumes file already filtered)
        df = df.copy()

    # Dependent variable
    df['_dv'] = construct_dv(df)

    # Core predictors
    df['_educ'] = construct_education(df)
    df['_inc_pc'] = construct_income_pc(df)
    df['_prestige'] = construct_prestige(df)

    # Demographic predictors
    df['_female'] = construct_female(df)
    df['_age'] = construct_age(df)
    hisp, black, other_race = construct_hispanic_black_otherrace(df)
    df['_hisp'] = hisp
    df['_black'] = black
    df['_other'] = other_race
    cons_prot, no_relig = construct_religion_dummies(df)
    df['_cons_prot'] = cons_prot
    df['_no_relig'] = no_relig
    df['_south'] = construct_southern(df)

    # Political intolerance index (0-15)
    df['_intol'] = construct_political_intolerance(df)

    # ---------------------------
    # Model specifications
    # ---------------------------
    # Model 1 vars
    m1_vars = ['Education', 'Household income per capita', 'Occupational prestige']
    m1_X_cols = ['_educ', '_inc_pc', '_prestige']

    # Model 2 vars (adds demographics)
    m2_vars = [
        'Education', 'Household income per capita', 'Occupational prestige',
        'Female', 'Age', 'Black', 'Hispanic', 'Other race',
        'Conservative Protestant', 'No religion', 'Southern'
    ]
    m2_X_cols = ['_educ', '_inc_pc', '_prestige', '_female', '_age', '_black', '_hisp', '_other',
                 '_cons_prot', '_no_relig', '_south']

    # Model 3 vars (adds political intolerance)
    m3_vars = [
        'Education', 'Household income per capita', 'Occupational prestige',
        'Female', 'Age', 'Black', 'Hispanic', 'Other race',
        'Conservative Protestant', 'No religion', 'Southern', 'Political intolerance'
    ]
    m3_X_cols = ['_educ', '_inc_pc', '_prestige', '_female', '_age', '_black', '_hisp', '_other',
                 '_cons_prot', '_no_relig', '_south', '_intol']

    # ---------------------------
    # Build matrices with listwise deletion per model
    # ---------------------------
    def build_xy(df_in, xcols):
        cols = ['_dv'] + xcols
        sub = df_in[cols].copy()
        mask = sub.notna().all(axis=1)
        sub = sub.loc[mask]
        y = sub['_dv'].to_numpy(dtype=float)
        X = sub[xcols].to_numpy(dtype=float)
        return y, X

    out_text_lines = []

    # Model 1
    y1, X1 = build_xy(df, m1_X_cols)
    txt1, res1 = summarize_model("1) SES Model", y1, X1, m1_vars)
    out_text_lines.append(txt1)

    # Model 2
    y2, X2 = build_xy(df, m2_X_cols)
    txt2, res2 = summarize_model("2) Demographic Model (adds group-identity controls)", y2, X2, m2_vars)
    out_text_lines.append(txt2)

    # Model 3
    y3, X3 = build_xy(df, m3_X_cols)
    txt3, res3 = summarize_model("3) Political Intolerance Model (adds intolerance scale)", y3, X3, m3_vars)
    out_text_lines.append(txt3)

    # Footer note
    out_text_lines.append("")
    out_text_lines.append("Notes:")
    out_text_lines.append("- Dependent variable: Number of music genres disliked.")
    out_text_lines.append("- Coefficients shown are standardized betas; constants (intercepts) are from unstandardized OLS.")
    out_text_lines.append("- Standard errors are not reported; significance indicated by stars: * p<.05, ** p<.01, *** p<.001 (normal-approximation).")
    out_text = "\n".join(out_text_lines)

    return out_text