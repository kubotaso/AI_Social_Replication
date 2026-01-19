def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Conservative set of common GSS missing codes; treat only if present in a given variable
    COMMON_GSS_NA = {0, 7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_gss(s, valid=None, extra_na=()):
        """Convert to numeric; drop common NA codes; optionally enforce valid set/range."""
        x = to_num(s)
        na = set(COMMON_GSS_NA) | set(extra_na)
        x = x.where(~x.isin(list(na)), np.nan)
        if valid is not None:
            if isinstance(valid, (set, list, tuple)):
                x = x.where(x.isin(list(valid)), np.nan)
            elif isinstance(valid, tuple) and len(valid) == 2:
                lo, hi = valid
                x = x.where((x >= lo) & (x <= hi), np.nan)
        return x

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

    def wmean(x, w):
        m = (~pd.isna(x)) & (~pd.isna(w)) & (w > 0)
        if m.sum() == 0:
            return np.nan
        xv = x[m].astype(float).to_numpy()
        wv = w[m].astype(float).to_numpy()
        return float(np.sum(wv * xv) / np.sum(wv))

    def wvar(x, w):
        # Weighted population variance with reliability correction not applied
        m = (~pd.isna(x)) & (~pd.isna(w)) & (w > 0)
        if m.sum() == 0:
            return np.nan
        xv = x[m].astype(float).to_numpy()
        wv = w[m].astype(float).to_numpy()
        mu = np.sum(wv * xv) / np.sum(wv)
        return float(np.sum(wv * (xv - mu) ** 2) / np.sum(wv))

    def wsd(x, w):
        v = wvar(x, w)
        return np.nan if (pd.isna(v) or v < 0) else float(np.sqrt(v))

    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    def format_num(x, nd=3):
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"

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
    # Rule: 1 if item in {4,5}, 0 if in {1,2,3}, missing otherwise.
    # DV missing if ANY of 18 items missing.
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music columns: {missing_music}")

    music = pd.DataFrame(index=df.index)
    for c in music_items:
        x = clean_gss(df[c])
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss(df.get("sex", np.nan), valid={1, 2})
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    race = clean_gss(df.get("race", np.nan), valid={1, 2, 3})  # 1=white,2=black,3=other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic (use ETHNIC, best-effort but NOT reversed; keep missing low by treating any non-1 as Hispanic when codes are multi-category)
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # If binary {1,2}: 1=not hispanic, 2=hispanic
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth == 2).astype(float))
        else:
            # Many GSS ETHNIC codings: 1=not hispanic; 2..n = specific Hispanic origins
            # Treat 1 as non-Hispanic, any other positive code as Hispanic
            df["hispanic"] = np.where(eth.isna(), np.nan, (eth != 1).astype(float))

    # Religion
    relig = clean_gss(df.get("relig", np.nan), valid={1, 2, 3, 4, 5})
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant proxy (kept simple): Protestant + (Baptist or Other Protestant)
    is_prot = (relig == 1)
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom.isin([1, 6])).astype(float))
    # If Protestant but denom missing, set to 0 (avoid dropping Protestants due to denom missingness)
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # South: mapping instruction says REGION==3 is South
    region = clean_gss(df.get("region", np.nan))
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # ----------------------------
    # Political intolerance (0–15)
    # IMPORTANT FIX: do NOT require all 15 items non-missing.
    # Construct as sum over non-missing items, but require at least a minimum number answered.
    # Given ~2/3 asked, this preserves the asked subsample and avoids over-dropping.
    # ----------------------------
    tol_items = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1}),
    ]
    missing_tol = [c for c, _ in tol_items if c not in df.columns]
    if missing_tol:
        raise ValueError(f"Missing required political tolerance columns: {missing_tol}")

    tol_mat = pd.DataFrame(index=df.index)
    for c, intolerant in tol_items:
        x = clean_gss(df[c])
        # keep plausible codes across these items; anything else is missing
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_mat[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant)).astype(float))

    df["pol_intol_sum"] = tol_mat.sum(axis=1, skipna=True)
    df["pol_intol_n"] = tol_mat.notna().sum(axis=1)

    # Require at least 12 of 15 answered (keeps "asked" cases but allows limited item nonresponse)
    MIN_ANSWERED = 12
    df["pol_intol"] = np.where(df["pol_intol_n"] >= MIN_ANSWERED, df["pol_intol_sum"], np.nan)

    # ----------------------------
    # Weights (if present): use if available; otherwise unweighted
    # ----------------------------
    weight_col = None
    for cand in ["wtssall", "wtss", "weight", "wtsall", "wtssnr"]:
        if cand in df.columns:
            weight_col = cand
            break

    if weight_col is not None:
        w = clean_gss(df[weight_col])
        w = w.where(np.isfinite(w) & (w > 0), np.nan)
    else:
        w = pd.Series(1.0, index=df.index)

    df["_w_"] = w

    # ----------------------------
    # Fitting (weighted OLS if weights present) + standardized betas
    # Betas computed using weighted SDs on the estimation sample.
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

    def fit_one(model_name, dv, xcols):
        cols = [dv] + xcols + ["_w_"]
        frame = df[cols].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        # Drop any zero-variance predictors (after listwise for this model)
        kept, dropped = [], []
        for c in xcols:
            if frame[c].nunique(dropna=True) <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        if len(frame) == 0 or len(kept) == 0:
            meta = {"model": model_name, "n": int(len(frame)), "r2": np.nan, "adj_r2": np.nan, "weight": weight_col or "none", "dropped": ",".join(dropped)}
            out = []
            out.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                out.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(out), frame

        y = frame[dv].astype(float)
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")
        ww = frame["_w_"].astype(float)

        # WLS with weights if provided; unweighted reduces to OLS with w=1
        res = sm.WLS(y, Xc, weights=ww).fit()

        # Weighted SDs for beta computation on estimation sample
        sdy = wsd(y, ww)
        betas = {}
        for c in kept:
            b = safe_float(res.params.get(c, np.nan))
            sdx = wsd(X[c], ww)
            betas[c] = np.nan if (pd.isna(b) or pd.isna(sdx) or pd.isna(sdy) or sdy == 0) else b * (sdx / sdy)

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj),
            "weight": weight_col or "none",
            "dropped": ",".join(dropped) if dropped else ""
        }

        rows = []
        rows.append({
            "term": "Constant",
            "b": safe_float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": safe_float(res.pvalues.get("const", np.nan)),
            "sig": ""
        })

        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                p = safe_float(res.pvalues.get(c, np.nan))
                rows.append({
                    "term": term,
                    "b": safe_float(res.params.get(c, np.nan)),
                    "beta": safe_float(betas.get(c, np.nan)),
                    "p": p,
                    "sig": sig_star(p)
                })
            else:
                rows.append({"term": term, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        return meta, pd.DataFrame(rows), frame

    def table1_style(tab):
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out.append(format_num(r["b"], 3))
            else:
                out.append((format_num(r["beta"], 3) + (r["sig"] or "")).strip())
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

    # ----------------------------
    # Models
    # ----------------------------
    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, f1 = fit_one("Model 1 (SES)", "num_genres_disliked", m1)
    meta2, tab2, f2 = fit_one("Model 2 (Demographic)", "num_genres_disliked", m2)
    meta3, tab3, f3 = fit_one("Model 3 (Political intolerance)", "num_genres_disliked", m3)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # ----------------------------
    # Diagnostics / sanity checks written to disk
    # ----------------------------
    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # DV descriptives (unweighted and weighted)
    dv = df["num_genres_disliked"]
    dv_w = df["_w_"]
    dv_desc = dv.describe()
    dv_w_mean = wmean(dv, dv_w)
    dv_w_sd = wsd(dv, dv_w)

    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count across 18 genres where response in {4,5}; DK/NA treated missing; if any genre missing => DV missing.\n\n"
        "Unweighted describe():\n"
        f"{dv_desc.to_string()}\n\n"
        f"Weighted mean (if weights available; else same as unweighted): {dv_w_mean:.4f}\n"
        f"Weighted SD   (if weights available; else same as unweighted): {dv_w_sd:.4f}\n"
    )

    # Missingness summary (overall, not model-specific)
    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol", "pol_intol_n"
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
    write_text("./output/table1_missingness.txt", missingness.to_string(index=False) + "\n")

    # Key frequency checks for dummies in each model estimation sample
    def freq_in_frame(frame, col):
        if col not in frame.columns:
            return "NA"
        s = frame[col]
        return f"n={len(s)}, nonmiss={int(s.notna().sum())}, mean={float(s.mean()):.4f}, sd={float(s.std(ddof=1)) if s.notna().sum()>1 else np.nan}"

    freq_report = []
    freq_report.append("Estimation-sample checks (means for dummies should not be 0/1 constant unless truly absent)\n")
    for name, frame in [("Model 1", f1), ("Model 2", f2), ("Model 3", f3)]:
        freq_report.append(f"\n[{name} sample]\n")
        for c in ["black", "hispanic", "otherrace", "female", "south", "cons_prot", "norelig"]:
            if c in frame.columns:
                freq_report.append(f"{c}: {freq_in_frame(frame, c)}\n")
        if "pol_intol" in frame.columns:
            freq_report.append(f"pol_intol: {freq_in_frame(frame, 'pol_intol')}\n")
            freq_report.append(f"pol_intol_n: {freq_in_frame(frame, 'pol_intol_n')}\n")
    write_text("./output/table1_sample_checks.txt", "".join(freq_report))

    # ----------------------------
    # Write regression outputs (human-readable)
    # ----------------------------
    def panel_text(meta, tab, tab_style):
        lines = []
        lines.append(f"{meta['model']}\n")
        lines.append(f"n = {meta['n']}\n")
        lines.append(f"R^2 = {meta['r2']:.3f}   Adjusted R^2 = {meta['adj_r2']:.3f}\n")
        lines.append(f"Weight = {meta['weight']}\n")
        if meta.get("dropped"):
            lines.append(f"Dropped predictors (zero variance): {meta['dropped']}\n")
        lines.append("\nFull (b, beta, p, stars):\n")
        lines.append(tab.to_string(index=False))
        lines.append("\n\nTable 1 style (predictors standardized beta; constant unstandardized):\n")
        lines.append(tab_style.to_string(index=False))
        lines.append("\n")
        return "".join(lines)

    tab1_style = table1_style(tab1)
    tab2_style = table1_style(tab2)
    tab3_style = table1_style(tab3)

    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")
    write_text("./output/table1_model1.txt", panel_text(meta1, tab1, tab1_style))
    write_text("./output/table1_model2.txt", panel_text(meta2, tab2, tab2_style))
    write_text("./output/table1_model3.txt", panel_text(meta3, tab3, tab3_style))

    # Combined Table1-style panel
    def merge_panels(p1, p2, p3):
        out = p1.rename(columns={"Table1": "Model 1"}).copy()
        out = out.merge(p2.rename(columns={"Table1": "Model 2"}), on="term", how="outer")
        out = out.merge(p3.rename(columns={"Table1": "Model 3"}), on="term", how="outer")
        return out

    combined = merge_panels(tab1_style, tab2_style, tab3_style)
    write_text("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Return results for programmatic use
    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": tab1_style,
        "model2_table1style": tab2_style,
        "model3_table1style": tab3_style,
        "combined_table1style": combined,
        "missingness": missingness
    }