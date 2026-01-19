def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Common GSS-style missing codes. We keep this conservative and treat these as missing everywhere.
    GSS_NA_CODES = {0, 7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss(s, extra_na=()):
        x = to_num(s)
        na = set(GSS_NA_CODES) | set(extra_na)
        return x.where(~x.isin(list(na)), np.nan)

    def sig_star(p):
        if p is None or pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def sample_sd(x):
        x = pd.to_numeric(x, errors="coerce")
        v = x.var(ddof=1)
        if pd.isna(v) or v <= 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas(y, X, params):
        # beta_j = b_j * SD(x_j) / SD(y) computed on estimation sample
        sdy = sample_sd(y)
        out = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sample_sd(X[c])
            out[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy)) else b * (sdx / sdy)
        return out

    def format_table1(tab):
        # Constant shown as unstandardized b, predictors shown as standardized beta with stars
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                val = "" if pd.isna(r["b"]) else f"{float(r['b']):.3f}"
            else:
                val = "" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}"
            out.append(val)
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def listwise_report(df, cols):
        # Report how many would be dropped due to each variable missing within the model's candidate sample
        # and overall complete-case N.
        frame = df[cols].copy()
        n_total = len(frame)
        n_complete = int(frame.dropna().shape[0])
        miss = frame.isna().sum().sort_values(ascending=False)
        lines = []
        lines.append(f"Total rows considered: {n_total}")
        lines.append(f"Complete-case rows (listwise): {n_complete}")
        lines.append("")
        lines.append("Missing counts by variable (within rows considered):")
        lines.append(miss.to_string())
        return "\n".join(lines) + "\n"

    def fit_model(df, dv, xcols, model_name, labels):
        # model-specific listwise deletion on dv + xcols
        need = [dv] + xcols
        frame = df[need].copy()

        # drop rows missing any model variable
        frame_cc = frame.dropna(axis=0, how="any").copy()

        # drop any zero-variance predictors in this analytic sample
        kept, dropped = [], []
        for c in xcols:
            if frame_cc[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        meta = {
            "model": model_name,
            "n": int(len(frame_cc)),
            "r2": np.nan,
            "adj_r2": np.nan,
            "dropped_predictors": ",".join(dropped) if dropped else ""
        }

        # If no data/predictors, return empty shells
        rows = []
        if len(frame_cc) == 0 or len(kept) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            tab = pd.DataFrame(rows)
            return meta, tab, frame_cc, frame

        y = frame_cc[dv].astype(float)
        X = frame_cc[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        res = sm.OLS(y, Xc).fit()
        betas = standardized_betas(y, X, res.params)

        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star constant
        })

        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rows.append({
                    "term": term,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": sig_star(p)
                })
            else:
                rows.append({"term": term, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)
        return meta, tab, frame_cc, frame

    # ----------------------------
    # Read + year restriction
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")

    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # Dependent variable: number of music genres disliked (0–18)
    # Rule: 4/5 => disliked; 1/2/3 => not disliked; DK/NA => missing
    # DV missing if ANY of 18 items missing.
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music_cols = [c for c in music_items if c not in df.columns]
    if missing_music_cols:
        raise ValueError(f"Missing required music columns: {missing_music_cols}")

    music = pd.DataFrame(index=df.index)
    for c in music_items:
        x = clean_gss(df[c])
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    dv = disliked.sum(axis=1)
    dv.loc[disliked.isna().any(axis=1)] = np.nan
    df["num_genres_disliked"] = dv

    # DV descriptives
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Rule: count of 18 genre items rated 4 or 5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    # ----------------------------
    # Predictors (Table 1)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    # Income per capita:
    # Use REALINC (constant dollars) / HOMPOP.
    # Keep strictly positive values; missing if either component missing.
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["realinc_v"] <= 0, "realinc_v"] = np.nan
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan
    df.loc[df["inc_pc"] <= 0, "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other

    # Hispanic origin: ETHNIC is available. To avoid prior sign/coverage issues:
    # Use a strict binary coding when possible: 1=not hispanic, 2=hispanic.
    # If not binary, best-effort: treat 1 as not Hispanic; codes >=2 as Hispanic-origin.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth >= 2).astype(float))

    # Mutually exclusive race/ethnicity dummies with reference = non-Hispanic White
    # black: non-Hispanic Black
    # otherrace: non-Hispanic Other race
    # hispanic: Hispanic (any race), and for mutual exclusivity set black/otherrace=0 when Hispanic==1
    # If hispanic missing, these will remain based on race only; Model listwise deletion will handle.
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Enforce mutual exclusivity if Hispanic is observed:
    his_obs = df["hispanic"].notna()
    df.loc[his_obs & (df["hispanic"] == 1), ["black", "otherrace"]] = 0.0

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant:
    # Use RELIG==1 and DENOM in {1,2,3,4,5,6} are common; without full codebook, keep a conservative proxy:
    # Baptist (1) and "other Protestant" (6) as conservative, matching earlier best-effort.
    denom = clean_gss(df.get("denom", np.nan))
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant but denom missing, treat as not conservative to avoid dropping many cases.
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern: REGION==3 (per mapping instruction)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): sum intolerant responses across 15 items.
    # Missingness rule (IMPORTANT FIX): allow partial completion and prorate to 0–15,
    # requiring at least 12 of 15 items observed (80%) to reduce over-deletion.
    tol_items = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1}),
    ]
    missing_tol_cols = [c for c, _ in tol_items if c not in df.columns]
    if missing_tol_cols:
        raise ValueError(f"Missing required political tolerance columns: {missing_tol_cols}")

    tol_mat = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        # keep plausible substantive codes
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_mat[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    n_obs = tol_mat.notna().sum(axis=1).astype(float)
    sum_intol = tol_mat.sum(axis=1, skipna=True).astype(float)

    # prorate to 15 if enough items observed; else missing
    min_items = 12.0
    pol_intol = (sum_intol / n_obs) * 15.0
    pol_intol = pol_intol.where(n_obs >= min_items, np.nan)

    # Keep as 0-15 count-like scale: round to nearest integer (proration can create non-integers)
    # This preserves the intended 0–15 metric while allowing partial completion.
    df["pol_intol"] = np.round(pol_intol, 0)

    # ----------------------------
    # Diagnostics: missingness (overall)
    # ----------------------------
    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol"
    ]
    miss_rows = []
    for v in diag_vars:
        if v not in df.columns:
            continue
        nonmiss = int(df[v].notna().sum())
        miss = int(df[v].isna().sum())
        miss_rows.append({
            "variable": v,
            "nonmissing": nonmiss,
            "missing": miss,
            "pct_missing": (miss / (nonmiss + miss) * 100.0) if (nonmiss + miss) else np.nan
        })
    missingness = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    write_text("./output/table1_missingness_overall.txt", missingness.to_string(index=False) + "\n")

    # Quick sanity checks for dummies
    sanity = {}
    for v in ["female", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]:
        if v in df.columns:
            s = df[v]
            sanity[v] = {
                "nonmissing": int(s.notna().sum()),
                "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
                "sd": float(s.std(ddof=1, skipna=True)) if s.notna().sum() > 1 else np.nan,
                "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
                "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
            }
    sanity_df = pd.DataFrame(sanity).T
    write_text("./output/table1_dummy_sanity.txt", sanity_df.to_string() + "\n")

    # ----------------------------
    # Models (Table 1)
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
        "pol_intol": "Political intolerance (0–15)",
    }

    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1_cc, frame1_raw = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels)
    meta2, tab2, frame2_cc, frame2_raw = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3_cc, frame3_raw = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # ----------------------------
    # Save human-readable outputs (text)
    # ----------------------------
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text(
        "./output/model1_listwise_missingness.txt",
        listwise_report(df, ["num_genres_disliked"] + m1)
    )
    write_text(
        "./output/model2_listwise_missingness.txt",
        listwise_report(df, ["num_genres_disliked"] + m2)
    )
    write_text(
        "./output/model3_listwise_missingness.txt",
        listwise_report(df, ["num_genres_disliked"] + m3)
    )

    # Full coefficient tables (b, beta, p, stars)
    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    # Table 1 style (constant unstd, predictors standardized betas with stars)
    t1 = format_table1(tab1)
    t2 = format_table1(tab2)
    t3 = format_table1(tab3)
    write_text("./output/model1_table1style.txt", t1.to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", t2.to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", t3.to_string(index=False) + "\n")

    # Combine into a single Table 1 panel (wide)
    def wide_panel(t, colname):
        out = t[["term", "Table1"]].copy()
        out = out.rename(columns={"Table1": colname})
        return out

    panel = wide_panel(t1, "Model 1 (SES)")
    panel = panel.merge(wide_panel(t2, "Model 2 (Demographic)"), on="term", how="outer")
    panel = panel.merge(wide_panel(t3, "Model 3 (Political intolerance)"), on="term", how="outer")

    # Order terms to match table
    term_order = [
        "Constant",
        "Education (years)",
        "Household income per capita",
        "Occupational prestige",
        "Female",
        "Age",
        "Black",
        "Hispanic",
        "Other race",
        "Conservative Protestant",
        "No religion",
        "Southern",
        "Political intolerance (0–15)",
    ]
    panel["__order"] = panel["term"].map({t: i for i, t in enumerate(term_order)})
    panel = panel.sort_values(["__order", "term"]).drop(columns="__order")
    write_text("./output/table1_panel.txt", panel.to_string(index=False) + "\n")

    # Add model sample descriptive snapshots (means/sds) to help debug discrepancies
    def sample_profile(frame_cc, xcols, name):
        cols = ["num_genres_disliked"] + xcols
        prof = {}
        for c in cols:
            s = frame_cc[c]
            prof[c] = {
                "n": int(s.notna().sum()),
                "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
                "sd": float(s.std(ddof=1, skipna=True)) if s.notna().sum() > 1 else np.nan,
                "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
                "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
            }
        prof_df = pd.DataFrame(prof).T
        write_text(f"./output/{name}_sample_profile.txt", prof_df.to_string() + "\n")

    sample_profile(frame1_cc, m1, "model1")
    sample_profile(frame2_cc, m2, "model2")
    sample_profile(frame3_cc, m3, "model3")

    # Return key results
    return {
        "fit_stats": fit_stats,
        "table1_panel": panel,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "missingness_overall": missingness,
    }