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

    def clean_gss_numeric(series, extra_na=()):
        """
        Convert to numeric and set common GSS-style missing codes to NaN.
        Keep conservative; don't wipe valid substantive values.
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
        beta_j = b_j * sd(x_j) / sd(y), using the estimation sample.
        X must NOT include constant.
        """
        sdy = sd_sample(y)
        betas = {}
        for c in X.columns:
            b = float(params.get(c, np.nan))
            sdx = sd_sample(X[c])
            if pd.isna(b) or pd.isna(sdx) or pd.isna(sdy) or sdy == 0:
                betas[c] = np.nan
            else:
                betas[c] = b * (sdx / sdy)
        return betas

    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def fit_table1_model(df, dv, xcols, model_label, label_map):
        """
        Per-model listwise deletion on exactly dv + xcols.
        Fit OLS with intercept (unstandardized variables), compute standardized betas.
        Return:
          - fit stats (n, r2, adj_r2)
          - Table 1 style rows (Constant shown as unstandardized b; predictors shown as beta+stars)
          - Full coefficient table (b, beta, p, stars) for debugging
        """
        needed = [dv] + xcols
        frame = df[needed].copy()
        frame = frame.dropna(axis=0, how="any").copy()

        out = {
            "model": model_label,
            "n": int(len(frame)),
            "r2": np.nan,
            "adj_r2": np.nan,
            "table1": None,
            "full": None,
        }

        if len(frame) == 0:
            # empty shell tables
            rows1 = [{"Variable": "Constant", "Value": ""}]
            rowsf = [{"term": "const", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in xcols:
                rows1.append({"Variable": label_map.get(c, c), "Value": ""})
                rowsf.append({"term": c, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            out["table1"] = pd.DataFrame(rows1)
            out["full"] = pd.DataFrame(rowsf)
            return out

        y = frame[dv].astype(float)
        X = frame[xcols].astype(float)

        # drop any zero-variance predictors in THIS model's estimation sample
        kept = []
        for c in xcols:
            if X[c].nunique(dropna=True) > 1:
                kept.append(c)

        if len(kept) == 0:
            rows1 = [{"Variable": "Constant", "Value": ""}]
            rowsf = [{"term": "const", "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""}]
            for c in xcols:
                rows1.append({"Variable": label_map.get(c, c), "Value": ""})
                rowsf.append({"term": c, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
            out["table1"] = pd.DataFrame(rows1)
            out["full"] = pd.DataFrame(rowsf)
            return out

        Xk = X[kept]
        Xc = sm.add_constant(Xk, has_constant="add")
        res = sm.OLS(y, Xc).fit()

        betas = standardized_betas(y, Xk, res.params)

        out["r2"] = float(res.rsquared)
        out["adj_r2"] = float(res.rsquared_adj)

        # Table 1 display: Constant (unstandardized b), predictors (standardized beta + stars)
        rows1 = []
        const_b = float(res.params.get("const", np.nan))
        rows1.append({"Variable": "Constant", "Value": "" if pd.isna(const_b) else f"{const_b:.3f}"})
        for c in xcols:
            lab = label_map.get(c, c)
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                bta = float(betas.get(c, np.nan))
                val = "" if pd.isna(bta) else f"{bta:.3f}{star(p)}"
            else:
                val = ""
            rows1.append({"Variable": lab, "Value": val})
        out["table1"] = pd.DataFrame(rows1)

        # Full table for debugging/transparency
        rowsf = []
        rowsf.append(
            {
                "term": "const",
                "b": float(res.params.get("const", np.nan)),
                "beta": np.nan,
                "p": float(res.pvalues.get("const", np.nan)),
                "sig": "",
            }
        )
        for c in xcols:
            if c in kept:
                p = float(res.pvalues.get(c, np.nan))
                rowsf.append(
                    {
                        "term": c,
                        "b": float(res.params.get(c, np.nan)),
                        "beta": float(betas.get(c, np.nan)),
                        "p": p,
                        "sig": star(p),
                    }
                )
            else:
                rowsf.append({"term": c, "b": np.nan, "beta": np.nan, "p": np.nan, "sig": ""})
        out["full"] = pd.DataFrame(rowsf)

        return out

    # ----------------------------
    # Read and restrict to 1993
    # ----------------------------
    df = pd.read_csv(data_source)
    df.columns = [c.lower() for c in df.columns]

    if "year" in df.columns:
        df = df.loc[to_num(df["year"]) == 1993].copy()

    # ----------------------------
    # DV: number of genres disliked (0-18)
    # rule: dislike = 4 or 5; valid substantive = 1..5; any missing among 18 => DV missing
    # ----------------------------
    music_items = [
        "bigband", "blugrass", "country", "blues", "musicals", "classicl",
        "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera",
        "rap", "reggae", "conrock", "oldies", "hvymetal"
    ]
    missing_music = [c for c in music_items if c not in df.columns]
    if missing_music:
        raise ValueError(f"Missing required music item columns: {missing_music}")

    music = pd.DataFrame({c: clean_gss_numeric(df[c]) for c in music_items})
    music = music.where(music.isin([1, 2, 3, 4, 5]), np.nan)
    disliked = music.isin([4, 5]).astype(float).where(music.notna(), np.nan)

    df["num_genres_disliked"] = disliked.sum(axis=1)
    df.loc[disliked.isna().any(axis=1), "num_genres_disliked"] = np.nan

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
    # Predictors (coded to minimize unintended missingness)
    # ----------------------------
    # SES
    df["educ_yrs"] = clean_gss_numeric(df.get("educ", np.nan), extra_na=(0,))
    df["prestg80_v"] = clean_gss_numeric(df.get("prestg80", np.nan), extra_na=(0,))

    df["realinc_v"] = clean_gss_numeric(df.get("realinc", np.nan), extra_na=(0,))
    df["hompop_v"] = clean_gss_numeric(df.get("hompop", np.nan), extra_na=(0,))
    df.loc[df["hompop_v"] <= 0, "hompop_v"] = np.nan
    df["inc_pc"] = df["realinc_v"] / df["hompop_v"]
    df.loc[~np.isfinite(df["inc_pc"]), "inc_pc"] = np.nan
    # Do not drop nonpositive inc_pc unless clearly invalid; keep to avoid excessive missingness.
    # (realinc/hompop should be >=0; negative values likely indicate miscoding.)
    df.loc[df["inc_pc"] < 0, "inc_pc"] = np.nan

    # Demographics
    sex = clean_gss_numeric(df.get("sex", np.nan))
    sex = sex.where(sex.isin([1, 2]), np.nan)
    df["female"] = np.where(sex.isna(), np.nan, (sex == 2).astype(float))

    df["age_v"] = clean_gss_numeric(df.get("age", np.nan), extra_na=(0,))

    race = clean_gss_numeric(df.get("race", np.nan))
    race = race.where(race.isin([1, 2, 3]), np.nan)  # 1=white, 2=black, 3=other
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["otherrace"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: use ETHNIC in this extract; construct a 0/1 with broad coverage.
    # Assumption (common GSS recode): 1=not Hispanic, 2=Hispanic. If not binary, treat 1 as "not hispanic"
    # and 2 as "hispanic" when present; otherwise set missing.
    df["hispanic"] = np.nan
    if "ethnic" in df.columns:
        eth = clean_gss_numeric(df["ethnic"], extra_na=(0,))
        # If codes include 1 and 2, use them. Keep others as missing (avoid inventing).
        df["hispanic"] = np.where(eth.isna(), np.nan, np.nan)
        df.loc[eth == 1, "hispanic"] = 0.0
        df.loc[eth == 2, "hispanic"] = 1.0
        # if only one of {1,2} appears, still ok; if neither appears, will stay missing.

    # Religion
    relig = clean_gss_numeric(df.get("relig", np.nan), extra_na=(0,))
    relig = relig.where(relig.isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.nan)
    df["norelig"] = np.where(relig.isna(), np.nan, (relig == 4).astype(float))

    # Conservative Protestant: derived from RELIG and DENOM; keep non-missing when RELIG known.
    denom = clean_gss_numeric(df.get("denom", np.nan), extra_na=(0,))
    is_prot = relig == 1
    # Conservative Protestant proxy: Baptist and "other Protestant" (common conservative buckets in GSS denom recode)
    denom_cons = denom.isin([1, 6])
    df["cons_prot"] = np.where(relig.isna(), np.nan, (is_prot & denom_cons).astype(float))
    # If Protestant but denom missing: treat as not conservative (0) rather than missing, to avoid shrinking N.
    df.loc[is_prot & denom.isna() & relig.notna(), "cons_prot"] = 0.0

    # Southern
    region = clean_gss_numeric(df.get("region", np.nan), extra_na=(0,))
    # Keep common region codes 1..9; south is 3 in provided mapping
    region = region.where(region.isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.nan)
    df["south"] = np.where(region.isna(), np.nan, (region == 3).astype(float))

    # Political intolerance (0-15): sum of 15 intolerant responses, listwise on all 15 items.
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
        raise ValueError(f"Missing required political tolerance columns: {missing_tol}")

    tol_df = pd.DataFrame(index=df.index)
    for v, intolerant_codes in tol_map:
        x = clean_gss_numeric(df[v], extra_na=(0,))
        # keep only plausible response codes; otherwise missing
        # SPK*: typically 1/2; COL*: 4/5; LIB*: 1/2/3 depending. We'll keep values 1..6 to be safe.
        x = x.where((x >= 1) & (x <= 6), np.nan)
        tol_df[v] = np.where(x.isna(), np.nan, x.isin(list(intolerant_codes)).astype(float))

    df["pol_intol"] = tol_df.sum(axis=1)
    df.loc[tol_df.isna().any(axis=1), "pol_intol"] = np.nan

    # ----------------------------
    # Model specs (Table 1)
    # ----------------------------
    dv = "num_genres_disliked"

    m1 = ["educ_yrs", "inc_pc", "prestg80_v"]
    m2 = m1 + ["female", "age_v", "black", "hispanic", "otherrace", "cons_prot", "norelig", "south"]
    m3 = m2 + ["pol_intol"]

    label_map = {
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

    # ----------------------------
    # Fit models with per-model listwise deletion
    # ----------------------------
    res1 = fit_table1_model(df, dv, m1, "Model 1 (SES)", label_map)
    res2 = fit_table1_model(df, dv, m2, "Model 2 (Demographic)", label_map)
    res3 = fit_table1_model(df, dv, m3, "Model 3 (Political intolerance)", label_map)

    fit_stats = pd.DataFrame(
        [
            {"model": res1["model"], "n": res1["n"], "r2": res1["r2"], "adj_r2": res1["adj_r2"]},
            {"model": res2["model"], "n": res2["n"], "r2": res2["r2"], "adj_r2": res2["adj_r2"]},
            {"model": res3["model"], "n": res3["n"], "r2": res3["r2"], "adj_r2": res3["adj_r2"]},
        ]
    )

    # ----------------------------
    # Save human-readable outputs
    # ----------------------------
    def df_to_text(df_in, title):
        return title + "\n" + df_in.to_string(index=False) + "\n"

    txt = []
    txt.append("Table 1 replication (computed from provided GSS 1993 extract)\n")
    txt.append("Note: Table 1 reports standardized OLS coefficients (β) for predictors and an unstandardized constant.\n")
    txt.append("\nFIT STATS\n")
    txt.append(fit_stats.to_string(index=False))
    txt.append("\n\nMODEL 1 (Table 1-style)\n")
    txt.append(res1["table1"].to_string(index=False))
    txt.append("\n\nMODEL 2 (Table 1-style)\n")
    txt.append(res2["table1"].to_string(index=False))
    txt.append("\n\nMODEL 3 (Table 1-style)\n")
    txt.append(res3["table1"].to_string(index=False))
    txt.append("\n")
    write_text("./output/table1_summary.txt", "\n".join(txt))

    # Debug/full tables
    write_text("./output/table1_model1_full.txt", res1["full"].to_string(index=False) + "\n")
    write_text("./output/table1_model2_full.txt", res2["full"].to_string(index=False) + "\n")
    write_text("./output/table1_model3_full.txt", res3["full"].to_string(index=False) + "\n")

    # Missingness overview (in 1993 sample)
    vars_for_missing = [dv] + sorted(set(m3))
    miss = []
    for v in vars_for_missing:
        s = df[v]
        miss.append(
            {
                "variable": v,
                "nonmissing": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "pct_missing": float(100.0 * s.isna().mean()),
            }
        )
    miss_df = pd.DataFrame(miss).sort_values(["pct_missing", "variable"], ascending=[False, True])
    write_text("./output/table1_missingness.txt", miss_df.to_string(index=False) + "\n")

    return {
        "fit_stats": fit_stats,
        "model1_table1": res1["table1"],
        "model2_table1": res2["table1"],
        "model3_table1": res3["table1"],
        "model1_full": res1["full"],
        "model2_full": res2["full"],
        "model3_full": res3["full"],
        "missingness": miss_df,
    }