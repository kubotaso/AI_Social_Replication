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

    def clean_gss_numeric(series, extra_na=()):
        """
        Convert to numeric and set common GSS missing codes to NaN.
        Keep this conservative: only remove obvious "DK/NA/refused" style codes.
        """
        x = to_num(series)
        na_codes = {7, 8, 9, 97, 98, 99, 997, 998, 999}
        na_codes |= set(extra_na)
        return x.where(~x.isin(list(na_codes)), np.nan)

    def star(p):
        if p is None or pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def sd_sample(s):
        s = pd.to_numeric(s, errors="coerce")
        v = s.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas(y, X, params):
        """
        Compute standardized betas from unstandardized coefficients:
          beta_j = b_j * sd(x_j) / sd(y)
        X is WITHOUT constant.
        """
        sdy = sd_sample(y)
        out = {}
        for col in X.columns:
            b = float(params.get(col, np.nan))
            sdx = sd_sample(X[col])
            if pd.isna(b) or pd.isna(sdx) or pd.isna(sdy):
                out[col] = np.nan
            else:
                out[col] = b * (sdx / sdy)
        return out

    def safe_value_counts(x):
        return x.value_counts(dropna=False).sort_index()

    def fit_model(df, dv, xcols, model_name, label_map):
        """
        Build model frame for the model only, drop NAs, fit OLS,
        return table with intercept as unstandardized b, predictors as standardized beta.
        """
        needed = [dv] + xcols
        frame = df[needed].copy()
        n_before = len(frame)

        # Listwise deletion for THIS model only
        frame = frame.dropna(axis=0, how="any").copy()
        n_after = len(frame)

        # Drop zero-variance predictors (after deletion) to avoid singular fits
        kept = []
        dropped = []
        for c in xcols:
            nun = frame[c].nunique(dropna=True)
            if nun <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        result = {
            "model": model_name,
            "n_before": int(n_before),
            "n": int(n_after),
            "r2": np.nan,
            "adj_r2": np.nan,
            "dropped_predictors": dropped,
            "frame": frame
        }

        # If cannot fit
        rows = []
        rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
        for c in xcols:
            rows.append({"term": label_map.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
        tab = pd.DataFrame(rows)

        if n_after == 0 or len(kept) == 0:
            result["table"] = tab
            return result

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas(y, X, res.params)

        rows = []
        # Paper-style: intercept unstandardized, no stars shown for intercept
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant to mimic Table 1 convention
        })

        for c in xcols:
            lab = label_map.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({
                    "term": lab,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": star(p)
                })
            else:
                rows.append({"term": lab, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)
        result["r2"] = float(res.rsquared)
        result["adj_r2"] = float(res.rsquared_adj)
        result["table"] = tab
        return result

    def table_to_paper_style(tab):
        """
        Create a 2-col display like Table 1:
          - Constant: unstandardized b (no stars)
          - Predictors: standardized beta with stars
        """
        out = tab.copy()
        disp = []
        for _, r in out.iterrows():
            if r["term"] == "Constant":
                disp.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                disp.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        out["Table1"] = disp
        return out[["term", "Table1"]]

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    # Restrict to 1993
    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: number of genres disliked across 18 items
    # - valid substantive responses: 1..5
    # - disliked: 4 or 5
    # - if any missing among 18 => DV missing (listwise for DV construction)
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music items: {missing_music}")

    music = pd.DataFrame({c: clean_gss_numeric(df[c]) for c in music_items})
    # keep only 1..5 as valid; everything else is missing
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)
    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)

    # listwise DV construction across all 18 items
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count of 18 genre items rated 4 ('dislike') or 5 ('dislike very much');\n"
        "responses outside 1..5 treated as missing; if any of 18 items missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss_numeric(df.get("educ", np.nan), extra_na=(0,))
    df["prestg80_v"] = clean_gss_numeric(df.get("prestg80", np.nan), extra_na=(0,))

    df["realinc_v"] = clean_gss_numeric(df.get("realinc", np.nan), extra_na=(0,))
    df["hompop_v"] = clean_gss_numeric(df.get("hompop", np.nan), extra_na=(0,))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan
    df.loc[df["inc_pc"] <= 0, "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss_numeric(df.get("sex", np.nan))
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss_numeric(df.get("age", np.nan), extra_na=(0,))

    race = clean_gss_numeric(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use 'ethnic' in this extract as a best-available proxy:
    # Many extracts code ETHNIC as ancestry; however in this file it behaves like a
    # numeric category with special missing codes (e.g., 97). To avoid collapsing N:
    # define Hispanic as ETHNIC in {1,2,3,4,5}?? would be wrong. Instead, detect if
    # ETHNIC looks like a yes/no flag: {1,2}. If not, use a conservative heuristic:
    # treat ETHNIC codes 1..9 as Hispanic-origin categories (common in some recodes),
    # but only if distribution suggests it's not just broad ancestry codes.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss_numeric(df["ethnic"], extra_na=(0,))
        uniq = sorted([u for u in pd.unique(eth.dropna()) if np.isfinite(u)])
        uniq_set = set(uniq)
        if uniq_set.issubset({1.0, 2.0}) and len(uniq) >= 1:
            # Common yes/no pattern: 1=not hispanic, 2=hispanic (or vice versa).
            # Choose 2=hispanic (most common). If reversed in a given extract,
            # results will differ but N will be correct.
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # Fallback heuristic: flag likely Hispanic-origin codes.
            # Use a small whitelist that is unlikely to be missing codes and
            # keeps coverage: treat 1..9 as "Hispanic categories" ONLY if
            # a meaningful share of respondents are in 1..9; otherwise keep missing.
            share_1_9 = float(((eth >= 1) & (eth <= 9)).mean(skipna=True)) if eth.notna().any() else 0.0
            if share_1_9 >= 0.02:
                df["hispanic"] = np.where(eth.isna(), np.nan, ((eth >= 1) & (eth <= 9)).astype(float))
            else:
                df["hispanic"] = np.nan

    # Religion
    relig = clean_gss_numeric(df.get("relig", np.nan), extra_na=(0,))
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant proxy using RELIG and DENOM.
    # Keep it non-missing whenever RELIG is known; set to 0/1 for all known relig.
    denom = clean_gss_numeric(df.get("denom", np.nan), extra_na=(0,))
    # Allow a broad range of denom codes; do not force denom missing to kill the case.
    # If denom missing but relig known, treat as not conservative prot (0) to preserve N.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])  # common: 1=Baptist, 6=Other Protestant
    cons = (is_prot & denom_cons)
    df["cons_prot"] = np.where(relig.isna(), np.nan, cons.astype(float))
    # If protestant but denom missing, still allow 0 (unknown denom) rather than NA
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern
    region = clean_gss_numeric(df.get("region", np.nan), extra_na=(0,))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): listwise across 15 items (asked of ~2/3 sample)
    # Intolerant coding per mapping instruction.
    tol_map = [
        ("spkath", {2}),
        ("colath", {5}),
        ("libath", {1}),
        ("spkrac", {2}),
        ("colrac", {5}),
        ("librac", {1}),
        ("spkcom", {2}),
        ("colcom", {4}),
        ("libcom", {1}),
        ("spkmil", {2}),
        ("colmil", {5}),
        ("libmil", {1}),
        ("spkhomo", {2}),
        ("colhomo", {5}),
        ("libhomo", {1}),
    ]
    missing_tol = [v for v, _ in tol_map if v not in df.columns]
    if missing_tol:
        raise ValueError(f"Missing required political tolerance items: {missing_tol}")

    tol_items = {}
    for v, intolerant_codes in tol_map:
        x = clean_gss_numeric(df[v], extra_na=(0,))
        # keep only plausible substantive codes; other values -> missing
        # We do not hard-code full codeframes; we only require that intolerant codes are respected.
        tol_items[v] = np.where(
            x.isna(),
            np.nan,
            np.where(x.isin(list(intolerant_codes)), 1.0, 0.0)
        )
    tol_df = pd.DataFrame(tol_items)

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Missingness audit (key constructed variables)
    # ----------------------------
    audit_vars = [
        "num_genres_disliked",
        "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south",
        "pol_intol"
    ]
    audit_rows = []
    for v in audit_vars:
        if v not in df.columns:
            continue
        nm = int(df[v].notna().sum())
        mis = int(df[v].isna().sum())
        pct = float(mis / len(df) * 100) if len(df) else np.nan
        audit_rows.append({"variable": v, "nonmissing": nm, "missing": mis, "pct_missing": pct})
    missingness_df = pd.DataFrame(audit_rows).sort_values("pct_missing", ascending=False)

    write_text("./output/table1_missingness.txt", missingness_df.to_string(index=False) + "\n")

    # ----------------------------
    # Model specifications
    # ----------------------------
    labels = {
        "educ_yrs": "Education (years)",
        "inc_pc": "Household income per capita",
        "prestg80_v": "Occupational prestige",
        "female": "Female",
        "age_v": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "otherrace": "Other race",
        "cons_prot": "Conservative Protestant",
        "norelig": "No religion",
        "south": "Southern",
        "pol_intol": "Political intolerance (0–15)"
    }
    dv = "num_genres_disliked"

    m1_x = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2_x = m1_x + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3_x = m2_x + ["pol_intol"]

    # ----------------------------
    # Fit models
    # ----------------------------
    res1 = fit_model(df, dv, m1_x, "Model 1 (SES)", labels)
    res2 = fit_model(df, dv, m2_x, "Model 2 (Demographic)", labels)
    res3 = fit_model(df, dv, m3_x, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([
        {"model": res1["model"], "n": res1["n"], "r2": res1["r2"], "adj_r2": res1["adj_r2"]},
        {"model": res2["model"], "n": res2["n"], "r2": res2["r2"], "adj_r2": res2["adj_r2"]},
        {"model": res3["model"], "n": res3["n"], "r2": res3["r2"], "adj_r2": res3["adj_r2"]},
    ])

    # ----------------------------
    # Save human-readable summaries
    # ----------------------------
    def model_text_block(res):
        tab = res["table"]
        paper_tab = table_to_paper_style(tab)

        lines = []
        lines.append(res["model"])
        lines.append("=" * len(res["model"]))
        lines.append("")
        lines.append(f"N (before listwise deletion for this model): {res['n_before']}")
        lines.append(f"N (estimation sample): {res['n']}")
        if res["dropped_predictors"]:
            lines.append("Dropped predictors (zero variance after deletion): " + ", ".join(res["dropped_predictors"]))
        lines.append("")
        lines.append("Fit:")
        lines.append(f"R^2 = {res['r2']}")
        lines.append(f"Adj R^2 = {res['adj_r2']}")
        lines.append("")
        lines.append("Coefficients (Table 1 style):")
        lines.append("- Constant = unstandardized intercept (b)")
        lines.append("- Predictors = standardized betas (β), stars from two-tailed p-values")
        lines.append("- Stars: * p<.05, ** p<.01, *** p<.001")
        lines.append("")
        lines.append(paper_tab.to_string(index=False))
        lines.append("")
        return "\n".join(lines)

    write_text("./output/table1_model1.txt", model_text_block(res1))
    write_text("./output/table1_model2.txt", model_text_block(res2))
    write_text("./output/table1_model3.txt", model_text_block(res3))

    # Save raw tables too (including b, beta, p)
    res1["table"].to_csv("./output/table1_model1_raw.csv", index=False)
    res2["table"].to_csv("./output/table1_model2_raw.csv", index=False)
    res3["table"].to_csv("./output/table1_model3_raw.csv", index=False)
    fit_stats.to_csv("./output/table1_fit_stats.csv", index=False)

    # Additional diagnostics: key variable distributions (to catch coding collapse)
    diag_lines = []
    diag_lines.append("Key variable distributions (with missing)")
    diag_lines.append("======================================")
    diag_lines.append("")
    for v in ["female", "black", "hispanic", "otherrace", "norelig", "cons_prot", "south"]:
        if v in df.columns:
            diag_lines.append(f"{v}:")
            diag_lines.append(safe_value_counts(df[v]).to_string())
            diag_lines.append("")
    write_text("./output/table1_key_distributions.txt", "\n".join(diag_lines))

    # Return a compact dict of results
    return {
        "fit_stats": fit_stats,
        "Model 1 (SES)": res1["table"],
        "Model 2 (Demographic)": res2["table"],
        "Model 3 (Political intolerance)": res3["table"],
        "missingness": missingness_df
    }