def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def gss_missing_to_nan(s):
        """
        Conservative GSS-like missing handling:
        - Negative values are usually missing/inapplicable
        - Large category placeholders like 8/9/98/99 and 0 sometimes mean missing depending on item
        Because codes vary by item, we apply:
          * always: negatives -> NaN
          * optionally per-variable: specific invalid codes filtered with valid ranges/sets later
        """
        s = to_num(s)
        s = s.where(~(s < 0), np.nan)
        return s

    def star(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def ensure_finite(df_):
        d = df_.copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        return d

    def standardized_betas_from_unstd(res, X, y):
        """
        Compute standardized betas from an unstandardized OLS:
          beta_j = b_j * sd(X_j) / sd(y)
        Uses the estimation sample (already listwise deleted).
        """
        sd_y = float(np.nanstd(y, ddof=0))
        betas = {}
        for col in X.columns:
            sd_x = float(np.nanstd(X[col], ddof=0))
            b = float(res.params.get(col, np.nan))
            if sd_y == 0 or np.isnan(sd_y) or np.isnan(sd_x) or sd_x == 0:
                betas[col] = np.nan
            else:
                betas[col] = b * (sd_x / sd_y)
        return betas

    def ols_table(df_model, dv, x_cols, model_label):
        """
        Model-specific listwise deletion only on dv + x_cols.
        - Fit unstandardized OLS for intercept, fit stats, and p-values
        - Convert slopes to standardized betas via sd ratio (keeps intercept unstandardized)
        """
        needed = [dv] + x_cols
        d = df_model[needed].copy()
        d = ensure_finite(d)
        # numeric coercion
        for c in needed:
            d[c] = gss_missing_to_nan(d[c])

        nonmissing_before = d.notna().sum().sort_index()

        d = d.dropna(axis=0, how="any").copy()
        n = int(len(d))
        if n == 0:
            out = pd.DataFrame(
                [{"term": "Constant", "beta": np.nan, "p": np.nan, "sig": ""}]
                + [{"term": c, "beta": np.nan, "p": np.nan, "sig": ""} for c in x_cols]
            )
            fit = {"model": model_label, "n": 0, "r2": np.nan, "adj_r2": np.nan}
            return out, fit, nonmissing_before, d

        y = d[dv].astype(float)
        X = d[x_cols].astype(float)

        # Drop any constant predictors (avoid singularities / NaN pvalues)
        kept = [c for c in x_cols if X[c].nunique(dropna=True) > 1]
        dropped = [c for c in x_cols if c not in kept]
        Xk = X[kept].copy()

        Xk_const = sm.add_constant(Xk, has_constant="add")
        res = sm.OLS(y, Xk_const).fit()

        betas = standardized_betas_from_unstd(res, Xk, y)

        rows = []
        rows.append(
            {
                "term": "Constant",
                "beta": float(res.params.get("const", np.nan)),
                "p": float(res.pvalues.get("const", np.nan)),
                "sig": star(res.pvalues.get("const", np.nan)),
            }
        )

        for c in x_cols:
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({"term": c, "beta": float(betas.get(c, np.nan)), "p": p, "sig": star(p)})
            else:
                rows.append({"term": c, "beta": np.nan, "p": np.nan, "sig": ""})

        out = pd.DataFrame(rows)
        fit = {"model": model_label, "n": int(res.nobs), "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj)}

        # write per-model report
        path = f"./output/table1_{model_label.lower().replace(' ', '_')}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{model_label}\n")
            f.write("=" * len(model_label) + "\n\n")
            f.write("Non-missing counts BEFORE listwise deletion (DV + model predictors):\n")
            f.write(nonmissing_before.to_string())
            f.write("\n\n")
            if dropped:
                f.write("Dropped predictors due to zero variance AFTER listwise deletion:\n")
                f.write(", ".join(dropped) + "\n\n")
            f.write("Fit statistics:\n")
            f.write(f"N = {fit['n']}\n")
            f.write(f"R^2 = {fit['r2']:.6f}\n")
            f.write(f"Adj R^2 = {fit['adj_r2']:.6f}\n\n")
            f.write("Coefficients:\n")
            f.write("- Constant is unstandardized (raw DV units)\n")
            f.write("- Predictors are standardized coefficients (beta), computed from unstandardized OLS via sd ratio\n")
            f.write("- Stars from two-tailed p-values: * <.05, ** <.01, *** <.001\n\n")
            disp = out.copy()
            disp["beta"] = pd.to_numeric(disp["beta"], errors="coerce").round(6)
            disp["p"] = pd.to_numeric(disp["p"], errors="coerce").round(6)
            f.write(disp.to_string(index=False))
            f.write("\n")

        return out, fit, nonmissing_before, d

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993 (per instruction)
    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # Dependent variable: # music genres disliked (0-18), listwise across 18 items
    # disliked = 4 or 5 on 1..5 scale; DK/NA/etc -> missing; require complete 18 items
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal",
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items: {missing_music}")

    music = df[music_items].apply(gss_missing_to_nan)

    # enforce valid 1..5 only (everything else -> missing)
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)

    # listwise requirement across all 18 items for DV construction
    dv = disliked.sum(axis=1)
    dv = dv.where(~disliked.isna().any(axis=1), np.nan)
    df["num_genres_disliked"] = dv

    # DV descriptives
    with open("./output/table1_dv_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("DV: Number of music genres disliked (0-18)\n")
        f.write("Construction: count of 18 genre items rated 4 ('dislike') or 5 ('dislike very much').\n")
        f.write("Non-1..5 responses treated as missing; DV requires complete responses on all 18 items.\n\n")
        f.write(df["num_genres_disliked"].describe().to_string())
        f.write("\n")

    # ----------------------------
    # Predictors
    # ----------------------------
    # Education, prestige
    df["educ_yrs"] = gss_missing_to_nan(df.get("educ", np.nan))
    df["prestg80"] = gss_missing_to_nan(df.get("prestg80", np.nan))

    # Per-capita income: REALINC / HOMPOP (positive hompop)
    df["realinc"] = gss_missing_to_nan(df.get("realinc", np.nan))
    df["hompop"] = gss_missing_to_nan(df.get("hompop", np.nan))
    df.loc[df["hompop"] <= 0, "hompop"] = np.nan
    df.loc[df["realinc"] < 0, "realinc"] = np.nan
    df["inc_pc"] = df["realinc"] / df["hompop"]
    df["inc_pc"] = df["inc_pc"].replace([np.inf, -np.inf], np.nan)

    # Female dummy: SEX 1=male, 2=female
    sex = gss_missing_to_nan(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    # Age
    df["age"] = gss_missing_to_nan(df.get("age", np.nan))
    # Keep 89 topcode as 89; already numeric.

    # Race dummies: RACE 1=white, 2=black, 3=other
    race = gss_missing_to_nan(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic dummy: use ETHNIC if present; common coding is 1=not hispanic, 2=hispanic
    eth = gss_missing_to_nan(df.get("ethnic", np.nan))
    eth = eth.where(eth.isin([1, 2]), np.nan)
    df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))

    # Religion: RELIG 1=Protestant, 2=Catholic, 3=Jewish, 4=None, 5=Other (common)
    relig = gss_missing_to_nan(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: approximate using DENOM among Protestants.
    # Keep broad, non-missing for all with RELIG observed:
    # cons_prot = 1 if Protestant and denom in {Baptist (1), Other fundamentalist/sectarian proxy (6)}
    denom = gss_missing_to_nan(df.get("denom", np.nan))
    denom = denom.where(denom.isin(list(range(0, 15))), np.nan)  # conservative valid set
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])

    # If relig is observed but denom missing, set cons_prot=0 for non-protestants, and missing for protestants w/unknown denom
    cons = np.full(len(df), np.nan, dtype=float)
    # non-protestants with known relig: 0
    cons[(relig.notna()) & (~is_prot)] = 0.0
    # protestants with known denom: based on denom_cons
    cons[(is_prot) & (denom.notna())] = denom_cons[(is_prot) & (denom.notna())].astype(float)
    df["cons_prot"] = cons

    # South dummy: REGION==3
    region = gss_missing_to_nan(df.get("region", np.nan))
    region = region.where(region.isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance: 15 items, intolerant codes as specified
    tol_items = {
        "spkath": 2, "colath": 5, "libath": 1,
        "spkrac": 2, "colrac": 5, "librac": 1,
        "spkcom": 2, "colcom": 4, "libcom": 1,
        "spkmil": 2, "colmil": 5, "libmil": 1,
        "spkhomo": 2, "colhomo": 5, "libhomo": 1,
    }
    for v in tol_items:
        if v not in df.columns:
            df[v] = np.nan
        df[v] = gss_missing_to_nan(df[v])

    tol = df[list(tol_items.keys())].copy()
    intoler = pd.DataFrame(index=df.index)
    for v, bad_code in tol_items.items():
        s = tol[v]
        # treat any non-missing value not equal to bad_code as tolerant (0)
        intoler[v] = np.where(s.isna(), np.nan, (s == bad_code).astype(float))

    pol_intol = intoler.sum(axis=1)
    pol_intol = pol_intol.where(~intoler.isna().any(axis=1), np.nan)
    df["pol_intol"] = pol_intol

    with open("./output/table1_pol_intol_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("Political intolerance scale (0-15)\n")
        f.write("Construction: sum of 15 'intolerant' responses across 5 groups x 3 contexts.\n")
        f.write("Listwise across the 15 items.\n\n")
        f.write(df["pol_intol"].describe().to_string())
        f.write("\n")

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    dv_name = "num_genres_disliked"

    m1_x = ["educ_yrs", "inc_pc", "prestg80"]
    m2_x = m1_x + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    t1, fit1, nm1, frame1 = ols_table(df, dv_name, m1_x, "Model 1 (SES)")
    t2, fit2, nm2, frame2 = ols_table(df, dv_name, m2_x, "Model 2 (Demographic)")
    t3, fit3, nm3, frame3 = ols_table(df, dv_name, m3_x, "Model 3 (Political intolerance)")

    fit_stats = pd.DataFrame([fit1, fit2, fit3])[["model", "n", "r2", "adj_r2"]]

    # Save a combined human-readable summary
    with open("./output/table1_summary.txt", "w", encoding="utf-8") as f:
        f.write("Table 1 replication summary (computed from raw data)\n")
        f.write("=================================================\n\n")
        f.write("Fit statistics:\n")
        f.write(fit_stats.to_string(index=False))
        f.write("\n\n")
        f.write("Notes:\n")
        f.write("- DV: number of music genres disliked (0-18), listwise across 18 music items.\n")
        f.write("- Predictors: standardized coefficients (beta) computed from unstandardized OLS slopes via sd ratio.\n")
        f.write("- Constant: unstandardized OLS intercept.\n")
        f.write("- Listwise deletion is applied per model on DV + that model's predictors.\n")

        f.write("\n\nModel 1 coefficients:\n")
        f.write(t1.to_string(index=False))
        f.write("\n\nModel 2 coefficients:\n")
        f.write(t2.to_string(index=False))
        f.write("\n\nModel 3 coefficients:\n")
        f.write(t3.to_string(index=False))
        f.write("\n")

    # Diagnostics to help catch n-collapse causes
    diag_rows = []
    for name, cols, frame in [
        ("Model 1 (SES)", [dv_name] + m1_x, frame1),
        ("Model 2 (Demographic)", [dv_name] + m2_x, frame2),
        ("Model 3 (Political intolerance)", [dv_name] + m3_x, frame3),
    ]:
        diag_rows.append(
            {
                "model": name,
                "n_after_listwise": len(frame),
                "dv_mean": float(frame[dv_name].mean()) if len(frame) else np.nan,
                "dv_sd": float(frame[dv_name].std(ddof=0)) if len(frame) else np.nan,
            }
        )
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv("./output/table1_diagnostics.csv", index=False)

    return {
        "fit_stats": fit_stats,
        "tables": {
            "Model 1 (SES)": t1,
            "Model 2 (Demographic)": t2,
            "Model 3 (Political intolerance)": t3,
        },
        "diagnostics": diagnostics,
    }