def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Conservative set of common GSS missing codes; keep it simple and consistent.
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

    def wmean(x, w):
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return np.nan
        return float(np.sum(w[m] * x[m]) / np.sum(w[m]))

    def wvar(x, w):
        # unbiased-ish weighted variance; fine for standardization purposes
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() <= 1:
            return np.nan
        ww = w[m]
        xx = x[m]
        mu = np.sum(ww * xx) / np.sum(ww)
        v_num = np.sum(ww * (xx - mu) ** 2)
        # Use effective df adjustment (Kish) to avoid extreme bias
        w_sum = np.sum(ww)
        w_sum2 = np.sum(ww ** 2)
        denom = w_sum - (w_sum2 / w_sum) if w_sum > 0 else np.nan
        if denom is None or not np.isfinite(denom) or denom <= 0:
            return float(v_num / w_sum) if w_sum > 0 else np.nan
        return float(v_num / denom)

    def wsd(x, w):
        v = wvar(x, w)
        return float(np.sqrt(v)) if (v is not None and np.isfinite(v) and v >= 0) else np.nan

    def standardize_beta_from_b(y, X, b, w):
        # beta_j = b_j * SD_w(x_j) / SD_w(y)
        sdy = wsd(y.values.astype(float), w.values.astype(float))
        out = {}
        if sdy is None or not np.isfinite(sdy) or sdy == 0:
            for c in X.columns:
                out[c] = np.nan
            return out
        for c in X.columns:
            sdx = wsd(X[c].values.astype(float), w.values.astype(float))
            bj = float(b.get(c, np.nan))
            out[c] = np.nan if (not np.isfinite(bj) or sdx is None or not np.isfinite(sdx) or sdx == 0) else bj * (sdx / sdy)
        return out

    def fit_wls_model(df, dv, xcols, weight_col, model_name, labels, allow_drop_lowvar=True):
        cols = [dv] + xcols + ([weight_col] if weight_col else [])
        frame = df[cols].copy()

        # Model-specific listwise deletion only on model vars (and weight if present).
        frame = frame.dropna(axis=0, how="any").copy()
        if len(frame) == 0:
            meta = {"model": model_name, "n": 0, "r2": np.nan, "adj_r2": np.nan, "dropped": ""}
            tab = pd.DataFrame([{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}] +
                               [{"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""} for c in xcols])
            return meta, tab, frame

        # weights: default 1.0
        if weight_col:
            w = frame[weight_col].astype(float).copy()
            w = w.where(np.isfinite(w) & (w > 0), np.nan)
            frame = frame.loc[w.notna()].copy()
            w = frame[weight_col].astype(float).copy()
        else:
            w = pd.Series(1.0, index=frame.index)

        y = frame[dv].astype(float)
        Xraw = frame[xcols].astype(float)

        kept, dropped = [], []
        if allow_drop_lowvar:
            for c in xcols:
                # check variance with weights; if essentially zero, drop
                s = wsd(Xraw[c].values, w.values)
                if s is None or not np.isfinite(s) or s == 0:
                    dropped.append(c)
                else:
                    kept.append(c)
        else:
            kept = list(xcols)

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": np.nan,
            "adj_r2": np.nan,
            "dropped": ",".join(dropped) if dropped else ""
        }

        if len(kept) == 0:
            tab = pd.DataFrame([{"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}] +
                               [{"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""} for c in xcols])
            return meta, tab, frame

        X = Xraw[kept]
        Xc = sm.add_constant(X, has_constant="add")

        # WLS to approximate typical GSS weighting practice; if no weight variable exists, w=1 -> OLS.
        res = sm.WLS(y, Xc, weights=w).fit()

        # Weighted R2
        yhat = res.fittedvalues
        mu = wmean(y.values, w.values)
        sse = np.sum(w.values * (y.values - yhat.values) ** 2)
        sst = np.sum(w.values * (y.values - mu) ** 2)
        r2w = np.nan if (sst <= 0 or not np.isfinite(sst)) else float(1.0 - (sse / sst))
        n = int(len(frame))
        p = int(len(kept))
        adj = np.nan
        if np.isfinite(r2w) and n > p + 1:
            adj = float(1.0 - (1.0 - r2w) * (n - 1) / (n - p - 1))

        meta["r2"] = r2w
        meta["adj_r2"] = adj

        betas = standardize_beta_from_b(y, X, res.params, w)

        rows = []
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # do not star intercept
        })

        # keep original xcols order in output; fill dropped with NaNs
        for c in xcols:
            term = labels.get(c, c)
            if c in kept:
                pval = float(res.pvalues.get(c, np.nan))
                rows.append({
                    "term": term,
                    "b": float(res.params.get(c, np.nan)),
                    "beta": float(betas.get(c, np.nan)),
                    "p": pval,
                    "sig": sig_star(pval)
                })
            else:
                rows.append({"term": term, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})

        tab = pd.DataFrame(rows)
        return meta, tab, frame

    def table1_style(tab):
        # Paper style: predictors shown as standardized beta with stars; intercept unstandardized; no SE printed
        disp = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                disp.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                disp.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": disp})

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ----------------------------
    # Read + restrict to 1993
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected 'year' column in dataset.")
    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # Weights (best-effort)
    # ----------------------------
    weight_col = None
    for candidate in ["wtssall", "wtss", "wtsall", "weight", "wt"]:
        if candidate in df.columns:
            weight_col = candidate
            break
    if weight_col is None:
        df["_w_"] = 1.0
        weight_col = "_w_"
    else:
        df[weight_col] = clean_gss(df[weight_col])
        df.loc[~np.isfinite(df[weight_col]) | (df[weight_col] <= 0), weight_col] = np.nan

    # ----------------------------
    # DV: number of genres disliked (0–18)
    # As described: dislike/dislike very much => 1; else 0; DK/NA missing;
    # listwise across the 18 items for DV construction.
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    miss_music = [c for c in music_items if c not in df.columns]
    if miss_music:
        raise ValueError(f"Missing required music columns: {miss_music}")

    music = pd.DataFrame(index=df.index)
    for c in music_items:
        x = clean_gss(df[c])
        x = x.where(x.isin([1, 2, 3, 4, 5]), np.nan)
        music[c] = x

    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)
    dv = disliked.sum(axis=1)
    dv.loc[disliked.isna().any(axis=1)] = np.nan
    df["num_genres_disliked"] = dv

    # ----------------------------
    # Predictors
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss(df.get("educ", np.nan))
    df["prestg80_v"] = clean_gss(df.get("prestg80", np.nan))

    # income per capita: REALINC / HOMPOP (as instructed)
    df["realinc_v"] = clean_gss(df.get("realinc", np.nan))
    df["hompop_v"] = clean_gss(df.get("hompop", np.nan))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race/ethnicity: make mutually exclusive categories with White non-Hispanic as reference
    # Use RACE for race and ETHNIC for Hispanic origin (available in provided vars).
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other
    eth = clean_gss(df.get("ethnic", np.nan))
    # In GSS, ETHNIC commonly: 1=not Hispanic, 2=Hispanic; treat only these as valid binary.
    hisp = eth.where(eth.isin([1, 2]), np.nan)

    # Construct mutually exclusive:
    # Hispanic overrides race if hisp==2; else race distinguishes white/black/other.
    df["hispanic"] = np.where(hisp.isna(), np.nan, (hisp == 2).astype(float))

    df["black"] = np.nan
    df["otherrace"] = np.nan

    # For those known Hispanic==1 (not Hispanic), use race to define black/other; white is omitted ref.
    m_not_hisp = (df["hispanic"] == 0)
    df.loc[m_not_hisp & race.notna(), "black"] = (race.loc[m_not_hisp & race.notna()] == 2).astype(float)
    df.loc[m_not_hisp & race.notna(), "otherrace"] = (race.loc[m_not_hisp & race.notna()] == 3).astype(float)

    # For those Hispanic==1, set race dummies to 0 (so Hispanic is its own category)
    m_hisp = (df["hispanic"] == 1)
    df.loc[m_hisp, "black"] = 0.0
    df.loc[m_hisp, "otherrace"] = 0.0

    # If Hispanic missing, keep race dummies missing (to avoid forcing sample drops unpredictably)
    # This should keep Model 2 sample closer to paper if ETHNIC is well-populated.

    # Religion: No religion from RELIG
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: best-effort using RELIG==1 and DENOM in conservative-coded buckets.
    # (This is approximate but avoids inducing missingness.)
    denom = clean_gss(df.get("denom", np.nan))
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 2, 3, 6])  # broad net: baptist + other prot-ish categories
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Region: south dummy, using mapping instruction REGION==3
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15): allow partial completion to reduce excess listwise loss.
    tol_items = [
        ("spkath", {2}), ("colath", {5}), ("libath", {1}),
        ("spkrac", {2}), ("colrac", {5}), ("librac", {1}),
        ("spkcom", {2}), ("colcom", {4}), ("libcom", {1}),
        ("spkmil", {2}), ("colmil", {5}), ("libmil", {1}),
        ("spkhomo", {2}), ("colhomo", {5}), ("libhomo", {1}),
    ]
    miss_tol = [c for c, _ in tol_items if c not in df.columns]
    if miss_tol:
        raise ValueError(f"Missing required political tolerance columns: {miss_tol}")

    tol = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        x = x.where(x.isin([1, 2, 3, 4, 5, 6]), np.nan)
        tol[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    tol_sum = tol.sum(axis=1, skipna=True)
    tol_n = tol.notna().sum(axis=1)

    # Partial scoring rule (to better match paper N): require at least 12 of 15 answered,
    # then rescale to 15-item metric and round to nearest integer; else missing.
    # This preserves 0–15 interpretation while reducing missingness vs strict listwise.
    df["pol_intol"] = np.nan
    m_enough = tol_n >= 12
    df.loc[m_enough, "pol_intol"] = np.rint((tol_sum.loc[m_enough] / tol_n.loc[m_enough]) * 15.0)
    df.loc[m_enough, "pol_intol"] = df.loc[m_enough, "pol_intol"].clip(0, 15)

    # ----------------------------
    # Diagnostics (DV descriptives, missingness)
    # ----------------------------
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Construction: count across 18 genres of responses 4/5; DK/NA -> missing; if ANY genre missing -> DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

    diag_vars = [
        "num_genres_disliked", "educ_yrs", "inc_pc", "prestg80_v",
        "female", "age_v", "black", "hispanic", "otherrace",
        "cons_prot", "norelig", "south", "pol_intol", weight_col
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
        "pol_intol": "Political intolerance (0–15)"
    }

    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1 = fit_wls_model(df, "num_genres_disliked", m1, weight_col, "Model 1 (SES)", labels)
    meta2, tab2, frame2 = fit_wls_model(df, "num_genres_disliked", m2, weight_col, "Model 2 (Demographic)", labels)
    meta3, tab3, frame3 = fit_wls_model(df, "num_genres_disliked", m3, weight_col, "Model 3 (Political intolerance)", labels)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # ----------------------------
    # Save human-readable outputs
    # ----------------------------
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")

    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")

    write_text("./output/model1_table1style.txt", table1_style(tab1).to_string(index=False) + "\n")
    write_text("./output/model2_table1style.txt", table1_style(tab2).to_string(index=False) + "\n")
    write_text("./output/model3_table1style.txt", table1_style(tab3).to_string(index=False) + "\n")

    # Combined table1-style
    t1 = table1_style(tab1).rename(columns={"Table1": "Model 1 (SES)"})
    t2 = table1_style(tab2).rename(columns={"Table1": "Model 2 (Demographic)"})
    t3 = table1_style(tab3).rename(columns={"Table1": "Model 3 (Political intolerance)"})
    combined = t1.merge(t2, on="term", how="outer").merge(t3, on="term", how="outer")
    write_text("./output/table1_combined_table1style.txt", combined.to_string(index=False) + "\n")

    # Short summary
    summary_lines = []
    summary_lines.append("Table 1 replication (computed from data)\n")
    summary_lines.append("Notes:")
    summary_lines.append("- Predictors shown as standardized coefficients (beta) with stars from WLS p-values.")
    summary_lines.append("- Intercepts shown as unstandardized constants.")
    summary_lines.append(f"- Weight column used: {weight_col if weight_col in df.columns else 'None'}")
    summary_lines.append("")
    summary_lines.append("Fit statistics:")
    summary_lines.append(fit_stats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Model 1 (Table1-style):")
    summary_lines.append(table1_style(tab1).to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Model 2 (Table1-style):")
    summary_lines.append(table1_style(tab2).to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Model 3 (Table1-style):")
    summary_lines.append(table1_style(tab3).to_string(index=False))
    summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    write_text("./output/summary.txt", summary_text)

    return {
        "fit_stats": fit_stats,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "model1_table1style": table1_style(tab1),
        "model2_table1style": table1_style(tab2),
        "model3_table1style": table1_style(tab3),
        "table1_combined_table1style": combined,
        "missingness": missingness
    }