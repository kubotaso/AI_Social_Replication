def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def is_gss_missing(s):
        """
        Conservative GSS-style missing handling:
        - negative values (often -1,-2,-3,-4) treated as missing
        - very large sentinel codes (>= 90) often mean DK/NA in many categorical vars
        We still apply variable-specific valid ranges below to avoid over-dropping.
        """
        s = to_num(s)
        return (s < 0) | (s >= 90)

    def coerce_valid_range(s, valid_min=None, valid_max=None, valid_set=None):
        s = to_num(s)
        s = s.mask(is_gss_missing(s), np.nan)
        if valid_set is not None:
            s = s.where(s.isin(valid_set), np.nan)
        if valid_min is not None:
            s = s.where(s >= valid_min, np.nan)
        if valid_max is not None:
            s = s.where(s <= valid_max, np.nan)
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

    def standardize_betas_from_unstd(res, df_model, y_col, x_cols):
        """
        Compute standardized betas from an unstandardized OLS result:
            beta_j = b_j * sd(x_j) / sd(y)
        using sample SD (ddof=1) on the estimation sample.
        """
        y = df_model[y_col].astype(float)
        sd_y = y.std(ddof=1)
        betas = {}
        if sd_y is None or np.isnan(sd_y) or sd_y == 0:
            for c in x_cols:
                betas[c] = np.nan
            return betas
        for c in x_cols:
            x = df_model[c].astype(float)
            sd_x = x.std(ddof=1)
            if sd_x is None or np.isnan(sd_x) or sd_x == 0:
                betas[c] = np.nan
            else:
                betas[c] = float(res.params[c]) * float(sd_x) / float(sd_y)
        return betas

    def fit_table1_model(df, y_col, x_cols, model_name):
        needed = [y_col] + x_cols
        d = df[needed].copy()

        # Listwise deletion per model
        before = d.notna().sum()

        d = d.dropna(axis=0, how="any").copy()

        # Drop zero-variance predictors (should not happen with good coding, but guard)
        kept = []
        dropped = []
        for c in x_cols:
            if d[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(d) == 0 or len(kept) == 0:
            rows = [{"term": "Constant", "value": np.nan}]
            for c in x_cols:
                rows.append({"term": c, "value": np.nan})
            out = pd.DataFrame(rows)
            fit = {"model": model_name, "n": int(len(d)), "r2": np.nan, "adj_r2": np.nan}
            return out, fit, before, dropped, d

        y = d[y_col].astype(float)
        X = d[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        # Standardized betas for predictors (intercept stays unstandardized)
        betas = standardize_betas_from_unstd(res, d, y_col, kept)

        rows = []
        # intercept (unstandardized)
        rows.append({"term": "Constant", "value": float(res.params["const"]), "p": float(res.pvalues["const"])})
        for c in x_cols:
            if c in kept:
                p = float(res.pvalues[c])
                rows.append({"term": c, "value": float(betas[c]), "p": p})
            else:
                rows.append({"term": c, "value": np.nan, "p": np.nan})

        out = pd.DataFrame(rows)
        out["sig"] = out["p"].apply(star)
        fit = {"model": model_name, "n": int(res.nobs), "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj)}
        return out, fit, before, dropped, d

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993
    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: Number of music genres disliked (0-18), listwise across 18 items
    # dislike/dislike very much = 4 or 5; valid 1..5; else missing
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items: {missing_music}")

    music = pd.DataFrame({c: coerce_valid_range(df[c], valid_set=[1, 2, 3, 4, 5]) for c in music_items})
    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)

    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[music.isna().any(axis=1), "num_genres_disliked"] = np.nan  # strict listwise across 18

    # Save DV descriptives
    dv = df["num_genres_disliked"]
    with open("./output/table1_dv_descriptives.txt", "w", encoding="utf-8") as f:
        f.write("DV: Number of music genres disliked (0-18)\n")
        f.write("Constructed as count of 18 genre ratings coded 4/5 (dislike/dislike very much).\n")
        f.write("Strict listwise across all 18 items; non-1..5 and DK/NA codes treated as missing.\n\n")
        f.write(dv.describe().to_string())
        f.write("\n")

    # ----------------------------
    # Predictors: recoding with explicit valid ranges
    # ----------------------------
    # SES
    df["educ_yrs"] = coerce_valid_range(df.get("educ", np.nan), valid_min=0, valid_max=30)  # years schooling
    df["prestg80"] = coerce_valid_range(df.get("prestg80", np.nan), valid_min=0, valid_max=100)

    realinc = coerce_valid_range(df.get("realinc", np.nan), valid_min=0, valid_max=1e7)
    hompop = coerce_valid_range(df.get("hompop", np.nan), valid_min=1, valid_max=50)
    inc_pc = realinc / hompop
    inc_pc = inc_pc.replace([np.inf, -np.inf], np.nan)
    df["inc_pc"] = inc_pc

    # Demographics / group identity
    # female: 1=male, 2=female
    sex = coerce_valid_range(df.get("sex", np.nan), valid_set=[1, 2])
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age"] = coerce_valid_range(df.get("age", np.nan), valid_min=18, valid_max=89)  # 89 includes topcode

    # race: 1 white, 2 black, 3 other
    race = coerce_valid_range(df.get("race", np.nan), valid_set=[1, 2, 3])
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC if present.
    # In many GSS extracts, ETHNIC is a Hispanic-origin indicator; accept 1/2 codes.
    ethnic = coerce_valid_range(df.get("ethnic", np.nan), valid_set=[1, 2])
    df["hispanic"] = np.where(ethnic.isna(), np.nan, (ethnic == 2).astype(float))

    # Religion:
    # RELIG codes vary by extract; for our selected file, treat 1..13 as valid.
    relig = coerce_valid_range(df.get("relig", np.nan), valid_set=list(range(1, 14)))
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))  # "none"

    # Conservative Protestant:
    # Use RELIG==1 (Protestant) and DENOM categories; keep non-Protestants as 0 when religion known.
    # This avoids creating NA for most cases and prevents catastrophic listwise loss.
    denom = coerce_valid_range(df.get("denom", np.nan), valid_set=list(range(0, 15)))
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])  # approximation: Baptist or Other Protestant
    cons_prot = np.where(relig.isna(), np.nan, 0.0)  # default 0 when relig known
    # if Protestant and denom known, set per denom; if Protestant but denom missing, set missing
    cons_prot = np.where(is_prot & denom.notna(), denom_cons.astype(float), cons_prot)
    cons_prot = np.where(is_prot & denom.isna(), np.nan, cons_prot)
    df["cons_prot"] = cons_prot.astype(float)

    # South: REGION==3 (South), valid 1..9
    region = coerce_valid_range(df.get("region", np.nan), valid_set=list(range(1, 10)))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0-15): sum of 15 intolerant responses
    tol_spec = {
        "spkath": ("spkath", 2, [1, 2]),
        "colath": ("colath", 5, [4, 5]),  # allow/deny teaching (extract coding uses 4/5)
        "libath": ("libath", 1, [1, 2]),
        "spkrac": ("spkrac", 2, [1, 2]),
        "colrac": ("colrac", 5, [4, 5]),
        "librac": ("librac", 1, [1, 2]),
        "spkcom": ("spkcom", 2, [1, 2]),
        "colcom": ("colcom", 4, [4, 5]),  # fired? (extract coding includes 4/5)
        "libcom": ("libcom", 1, [1, 2]),
        "spkmil": ("spkmil", 2, [1, 2]),
        "colmil": ("colmil", 5, [4, 5]),
        "libmil": ("libmil", 1, [1, 2]),
        "spkhomo": ("spkhomo", 2, [1, 2]),
        "colhomo": ("colhomo", 5, [4, 5]),
        "libhomo": ("libhomo", 1, [1, 2]),
    }

    intoler_items = []
    intoler_df = pd.DataFrame(index=df.index)
    for key, (vname, bad_code, valid_set) in tol_spec.items():
        s = coerce_valid_range(df.get(vname, np.nan), valid_set=valid_set)
        intoler_df[key] = np.where(s.isna(), np.nan, (s == bad_code).astype(float))
        intoler_items.append(key)

    df["pol_intol"] = intoler_df.sum(axis=1)
    df.loc[intoler_df.isna().any(axis=1), "pol_intol"] = np.nan  # strict listwise across 15

    # ----------------------------
    # Models (Table 1)
    # ----------------------------
    y = "num_genres_disliked"
    m1_x = ["educ_yrs", "inc_pc", "prestg80"]
    m2_x = m1_x + ["female", "age", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    t1_m1, fit1, before1, dropped1, frame1 = fit_table1_model(df, y, m1_x, "Model 1 (SES)")
    t1_m2, fit2, before2, dropped2, frame2 = fit_table1_model(df, y, m2_x, "Model 2 (Demographic)")
    t1_m3, fit3, before3, dropped3, frame3 = fit_table1_model(df, y, m3_x, "Model 3 (Political intolerance)")

    # ----------------------------
    # Output: human-readable text files
    # ----------------------------
    def write_model_output(path, model_name, fit, before_counts, dropped, table):
        with open(path, "w", encoding="utf-8") as f:
            f.write(model_name + "\n")
            f.write("=" * len(model_name) + "\n\n")
            f.write("Non-missing counts BEFORE model-wise listwise deletion (DV + predictors):\n")
            f.write(before_counts.to_string())
            f.write("\n\n")
            if dropped:
                f.write("Dropped predictors due to zero variance after deletion:\n")
                f.write(", ".join(dropped) + "\n\n")

            f.write("Fit statistics:\n")
            f.write(f"N = {fit['n']}\n")
            f.write(f"R^2 = {fit['r2']:.6f}\n")
            f.write(f"Adj R^2 = {fit['adj_r2']:.6f}\n\n")

            f.write("Coefficients (Table 1 style):\n")
            f.write("- Constant is unstandardized (raw DV units)\n")
            f.write("- Predictors are standardized betas computed from unstandardized OLS (beta = b*SDx/SDy)\n")
            f.write("- Stars from two-tailed p-values: * <.05, ** <.01, *** <.001\n\n")

            out = table.copy()
            out["value"] = pd.to_numeric(out["value"], errors="coerce")
            out["value"] = out["value"].round(6)
            out["p"] = pd.to_numeric(out["p"], errors="coerce").round(6)
            f.write(out[["term", "value", "sig"]].to_string(index=False))
            f.write("\n")

    write_model_output("./output/table1_model1_ses.txt", "Model 1 (SES)", fit1, before1, dropped1, t1_m1)
    write_model_output("./output/table1_model2_demographic.txt", "Model 2 (Demographic)", fit2, before2, dropped2, t1_m2)
    write_model_output("./output/table1_model3_political_intolerance.txt", "Model 3 (Political intolerance)", fit3, before3, dropped3, t1_m3)

    # Also save a combined summary
    fit_stats = pd.DataFrame([fit1, fit2, fit3])[["model", "n", "r2", "adj_r2"]]
    with open("./output/table1_fit_stats.txt", "w", encoding="utf-8") as f:
        f.write("Table 1 fit statistics (computed)\n")
        f.write(fit_stats.to_string(index=False))
        f.write("\n")

    # Combined coefficient tables to dict return
    tables = {
        "Model 1 (SES)": t1_m1[["term", "value", "sig"]].copy(),
        "Model 2 (Demographic)": t1_m2[["term", "value", "sig"]].copy(),
        "Model 3 (Political intolerance)": t1_m3[["term", "value", "sig"]].copy(),
    }

    # Save combined coefficients
    with open("./output/table1_coefficients_all_models.txt", "w", encoding="utf-8") as f:
        for name, tab in tables.items():
            f.write(name + "\n")
            f.write("-" * len(name) + "\n")
            f.write(tab.to_string(index=False))
            f.write("\n\n")

    # Diagnostics: nonmissing and value counts for key dummies to prevent N collapse
    diag_vars = ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south", "pol_intol"]
    diag = {}
    for v in diag_vars:
        if v in df.columns:
            s = df[v]
            diag[v] = {
                "nonmissing": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
                "nunique": int(s.nunique(dropna=True)),
            }
    diag_df = pd.DataFrame(diag).T
    diag_df.index.name = "variable"
    with open("./output/table1_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write("Diagnostics (post-recode, before model-wise deletion)\n")
        f.write("Counts/means help detect coding that collapses N under listwise deletion.\n\n")
        f.write(diag_df.to_string())
        f.write("\n")

    return {"fit_stats": fit_stats, "tables": tables, "model_frames": {"Model 1 (SES)": frame1, "Model 2 (Demographic)": frame2, "Model 3 (Political intolerance)": frame3}}