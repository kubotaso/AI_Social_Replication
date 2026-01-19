def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    # Common GSS-style special codes (dataset-specific; safe to treat as missing when present)
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

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def wmean(x, w):
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return np.nan
        return float(np.sum(w[m] * x[m]) / np.sum(w[m]))

    def wvar(x, w):
        # Weighted variance (population-style). Adequate for beta standardization replication.
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() <= 1:
            return np.nan
        xm = wmean(x[m], w[m])
        ww = w[m]
        xv = x[m] - xm
        denom = np.sum(ww)
        if denom <= 0:
            return np.nan
        return float(np.sum(ww * (xv ** 2)) / denom)

    def wsd(x, w):
        v = wvar(np.asarray(x, float), np.asarray(w, float))
        if v is None or (not np.isfinite(v)) or v < 0:
            return np.nan
        return float(np.sqrt(v))

    def standardized_betas(y, X, params, w=None):
        # beta_j = b_j * SD(x_j)/SD(y) on estimation sample.
        # If weights provided, use weighted SD; else unweighted sample SD (ddof=1).
        out = {}
        if w is None:
            sdy = float(pd.Series(y).astype(float).std(ddof=1))
            for c in X.columns:
                b = float(params.get(c, np.nan))
                sdx = float(pd.Series(X[c]).astype(float).std(ddof=1))
                out[c] = np.nan if (not np.isfinite(b) or not np.isfinite(sdx) or not np.isfinite(sdy) or sdy == 0) else b * (sdx / sdy)
            return out
        else:
            sdy = wsd(y, w)
            for c in X.columns:
                b = float(params.get(c, np.nan))
                sdx = wsd(X[c].values, w)
                out[c] = np.nan if (not np.isfinite(b) or not np.isfinite(sdx) or not np.isfinite(sdy) or sdy == 0) else b * (sdx / sdy)
            return out

    def fit_model(df, dv, xcols, model_name, labels, weight_col=None):
        cols = [dv] + xcols + ([weight_col] if weight_col else [])
        frame = df[cols].copy()

        # Model-specific listwise deletion ONLY on dv + xcols (+ weight if used)
        frame = frame.dropna(axis=0, how="any").copy()

        # Prepare weights
        w = None
        if weight_col:
            w = frame[weight_col].astype(float).values
            # drop nonpositive weights
            m = np.isfinite(w) & (w > 0)
            frame = frame.loc[m].copy()
            w = frame[weight_col].astype(float).values

        # Drop any zero-variance predictors in this analytic sample
        kept, dropped = [], []
        for c in xcols:
            nun = frame[c].nunique(dropna=True)
            if nun <= 1:
                dropped.append(c)
            else:
                kept.append(c)

        meta = {
            "model": model_name,
            "n": int(len(frame)),
            "r2": np.nan,
            "adj_r2": np.nan,
            "dropped": ",".join(dropped) if dropped else "",
            "weight_used": (weight_col if weight_col else "")
        }

        rows = []
        if len(frame) == 0 or len(kept) == 0:
            rows.append({"term": "Constant", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            for c in xcols:
                rows.append({"term": labels.get(c, c), "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            return meta, pd.DataFrame(rows), frame

        y = frame[dv].astype(float).values
        X = frame[kept].astype(float)
        Xc = sm.add_constant(X, has_constant="add")

        if weight_col:
            res = sm.WLS(y, Xc, weights=w).fit()
        else:
            res = sm.OLS(y, Xc).fit()

        # Fit stats
        meta["r2"] = float(res.rsquared)
        meta["adj_r2"] = float(res.rsquared_adj)

        betas = standardized_betas(y, X, res.params, w=w if weight_col else None)

        # constant unstandardized; predictors include b, beta, p, stars
        rows.append({
            "term": "Constant",
            "b": float(res.params.get("const", np.nan)),
            "beta": np.nan,
            "p": float(res.pvalues.get("const", np.nan)),
            "sig": ""  # no stars for constant in table output
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

        return meta, pd.DataFrame(rows), frame

    def table1_style(tab):
        # Constant: show unstandardized b; predictors: show standardized beta + stars
        out = []
        for _, r in tab.iterrows():
            if r["term"] == "Constant":
                out.append("" if pd.isna(r["b"]) else f"{float(r['b']):.3f}")
            else:
                out.append("" if pd.isna(r["beta"]) else f"{float(r['beta']):.3f}{r['sig']}")
        return pd.DataFrame({"term": tab["term"].values, "Table1": out})

    # ----------------------------
    # Load data + year restriction
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" not in df.columns:
        raise ValueError("Expected column 'year' in dataset.")
    df["year_v"] = clean_gss(df["year"])
    df = df.loc[df["year_v"] == 1993].copy()

    # ----------------------------
    # Dependent variable: count of 18 genres disliked (4/5)
    # - DK/NA -> missing
    # - If ANY of the 18 items missing => DV missing (listwise within DV construction)
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

    # Demographics: sex, age
    sex = clean_gss(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)  # 1=male,2=female
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss(df.get("age", np.nan))
    df.loc[df["age_v"] <= 0, "age_v"] = np.nan

    # Race/ethnicity (mutually exclusive, White omitted reference)
    # Use 'race' for Black/Other; use 'ethnic' for Hispanic.
    race = clean_gss(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1 white, 2 black, 3 other

    eth = None
    if "ethnic" in df.columns:
        eth = clean_gss(df["ethnic"])
        # Keep positive numeric codes as valid categories; missing codes already NA
        eth = eth.where(np.isfinite(eth) & (eth >= 1) & (eth <= 99), np.nan)

    # Hispanic coding:
    # If ETHNIC is binary 1/2, treat 2 as Hispanic.
    # Else treat 1 as "not hispanic", and any 2..99 as Hispanic-origin (best effort).
    df["hispanic"] = np.nan
    if eth is not None:
        uniq = set(pd.unique(eth.dropna()))
        if uniq and uniq.issubset({1.0, 2.0}):
            hisp = (eth == 2)
        else:
            hisp = (eth >= 2) & (eth <= 99)
        df["hispanic"] = np.where(eth.isna(), np.nan, hisp.astype(float))

    # Build mutually exclusive groups:
    # - If hispanic==1 => Hispanic=1 and Black/Other=0
    # - Else if race==2 => Black=1
    # - Else if race==3 => Other race=1
    # - Else (race==1 and not Hispanic) => White reference (all 0)
    # If either component missing such that classification is ambiguous => set all three to NA
    black = np.full(len(df), np.nan, dtype=float)
    oth = np.full(len(df), np.nan, dtype=float)
    hisp = df["hispanic"].values if "hispanic" in df.columns else np.full(len(df), np.nan)

    race_v = race.values if isinstance(race, pd.Series) else np.full(len(df), np.nan)

    # Determine who can be classified (need hispanic known AND race known)
    can_class = np.isfinite(hisp) & np.isfinite(race_v)

    # default zeros for classifiable
    black[can_class] = 0.0
    oth[can_class] = 0.0

    # apply mutually exclusive assignment
    is_hisp = can_class & (hisp == 1)
    black[is_hisp] = 0.0
    oth[is_hisp] = 0.0

    not_hisp = can_class & (hisp == 0)
    black[not_hisp & (race_v == 2)] = 1.0
    oth[not_hisp & (race_v == 3)] = 1.0

    df["black"] = black
    df["otherrace"] = oth
    # keep hispanic as-is but enforce mutual exclusivity by setting hispanic=0 for black/other when not_hisp
    # (already handled by original hisp variable; we keep df["hispanic"] itself)
    df.loc[~can_class, ["black", "otherrace", "hispanic"]] = np.nan

    # Religion
    relig = clean_gss(df.get("relig", np.nan))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5]), np.nan)  # 1 prot,2 cath,3 jew,4 none,5 other
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    denom = clean_gss(df.get("denom", np.nan))
    # Conservative Protestant (approximation based on available vars):
    # Protestant + DENOM in (Baptist=1) OR (Other Protestant=6) OR (No denom=7 sometimes treated as evangelical-ish).
    # Keep conservative_prot=0 for non-Protestants; missing only if RELIG missing.
    is_prot = (relig == 1)
    denom_cons = denom.isin([1, 6, 7])
    cons = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant but denom missing, set 0 (avoid unnecessary case loss)
    cons = np.where(is_prot & denom.isna() & relig.notna(), 0.0, cons)
    df["cons_prot"] = cons

    # South (REGION==3 per instruction)
    region = clean_gss(df.get("region", np.nan))
    region = region.where(region.isin(list(range(1, 10))), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0–15)
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

    tol_df = pd.DataFrame(index=df.index)
    for c, intolerant_codes in tol_items:
        x = clean_gss(df[c])
        # keep plausible codes; treat others as missing
        x = x.where(np.isfinite(x) & (x >= 1) & (x <= 6), np.nan)
        tol_df[c] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Weight handling (optional; dataset doesn't include weights in provided vars)
    # ----------------------------
    weight_col = None
    for cand in ["wtssall", "wtssnr", "wtss", "weight", "wtsall"]:
        if cand in df.columns:
            w = clean_gss(df[cand])
            # treat nonpositive as missing
            w = w.where(w > 0, np.nan)
            df[cand] = w
            if df[cand].notna().sum() > 0:
                weight_col = cand
                break

    # ----------------------------
    # Diagnostics: distributions + missingness
    # ----------------------------
    dv_desc = df["num_genres_disliked"].describe()
    write_text(
        "./output/table1_dv_descriptives.txt",
        "DV: Number of music genres disliked (0–18)\n"
        "Rule: count of 18 genre items rated 4/5; DK/NA treated as missing; if any genre item missing => DV missing.\n\n"
        + dv_desc.to_string()
        + "\n"
    )

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
        "pol_intol": "Political intolerance (0–15)",
    }

    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    meta1, tab1, frame1 = fit_model(df, "num_genres_disliked", m1, "Model 1 (SES)", labels, weight_col=weight_col)
    meta2, tab2, frame2 = fit_model(df, "num_genres_disliked", m2, "Model 2 (Demographic)", labels, weight_col=weight_col)
    meta3, tab3, frame3 = fit_model(df, "num_genres_disliked", m3, "Model 3 (Political intolerance)", labels, weight_col=weight_col)

    fit_stats = pd.DataFrame([meta1, meta2, meta3])

    # Table 1-style columns
    t1 = table1_style(tab1)
    t2 = table1_style(tab2)
    t3 = table1_style(tab3)

    # Merge into a single Table 1-like view
    table1 = t1.merge(t2, on="term", how="outer", suffixes=(" (M1)", " (M2)"))
    table1 = table1.merge(t3, on="term", how="outer")
    table1 = table1.rename(columns={"Table1": "Table1 (M3)"})
    table1 = table1.rename(columns={"Table1 (M1)": "Model 1", "Table1 (M2)": "Model 2", "Table1 (M3)": "Model 3"})

    # Ensure consistent ordering
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
    table1["__ord"] = table1["term"].map({t: i for i, t in enumerate(term_order)}).fillna(9999).astype(int)
    table1 = table1.sort_values(["__ord", "term"]).drop(columns="__ord").reset_index(drop=True)

    # Save full outputs
    write_text("./output/table1_fit_stats.txt", fit_stats.to_string(index=False) + "\n")
    write_text("./output/model1_full.txt", tab1.to_string(index=False) + "\n")
    write_text("./output/model2_full.txt", tab2.to_string(index=False) + "\n")
    write_text("./output/model3_full.txt", tab3.to_string(index=False) + "\n")
    write_text("./output/table1_replication.txt", table1.to_string(index=False) + "\n")

    # Extra QA: check race dummy variation within each model analytic sample
    def race_qa(frame, name):
        if frame is None or len(frame) == 0:
            return f"{name}: empty estimation sample\n"
        lines = [f"{name}: race dummy means (estimation sample)\n"]
        for v in ["black", "hispanic", "otherrace"]:
            if v in frame.columns:
                lines.append(f"  {v}: mean={frame[v].mean():.4f}, min={frame[v].min():.0f}, max={frame[v].max():.0f}, n={frame[v].notna().sum()}\n")
        # mutual exclusivity check
        if all(v in frame.columns for v in ["black", "hispanic", "otherrace"]):
            s = frame[["black", "hispanic", "otherrace"]].sum(axis=1)
            lines.append(f"  sum(black+hispanic+otherrace): min={s.min():.0f}, max={s.max():.0f}, pct_eq1={(s==1).mean()*100:.2f}%, pct_eq0={(s==0).mean()*100:.2f}%\n")
        return "".join(lines)

    qa_text = ""
    qa_text += race_qa(frame1, "Model 1 (SES) sample")
    qa_text += "\n" + race_qa(frame2, "Model 2 (Demographic) sample")
    qa_text += "\n" + race_qa(frame3, "Model 3 (Political intolerance) sample")
    write_text("./output/table1_race_qa.txt", qa_text)

    # Final human-readable summary
    summary = []
    summary.append("Table 1 replication (computed from data; not copied from paper)\n")
    summary.append(f"Data source: {data_source}\n")
    summary.append("Year restricted to 1993.\n")
    summary.append(f"Weight used: {weight_col if weight_col else 'None found'}\n\n")
    summary.append("Fit statistics:\n")
    summary.append(fit_stats.to_string(index=False))
    summary.append("\n\nTable 1-style coefficients (predictors: standardized beta w/ stars; constant: unstandardized):\n")
    summary.append(table1.to_string(index=False))
    summary.append("\n")
    write_text("./output/summary.txt", "".join(summary))

    return {
        "fit_stats": fit_stats,
        "table1": table1,
        "model1_full": tab1,
        "model2_full": tab2,
        "model3_full": tab3,
        "missingness": missingness,
        "weight_used": pd.DataFrame({"weight_col": [weight_col if weight_col else ""]})
    }