def run_analysis(data_source):
    import os
    import re
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    warnings.filterwarnings("ignore")

    # -----------------------------
    # Helpers
    # -----------------------------
    def norm_col(s):
        return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

    def find_column(df, candidates, required=True):
        cols = list(df.columns)
        norm_map = {norm_col(c): c for c in cols}

        # exact normalized match
        for cand in candidates:
            nc = norm_col(cand)
            if nc in norm_map:
                return norm_map[nc]

        # contains match fallback
        for cand in candidates:
            nc = norm_col(cand)
            for k, orig in norm_map.items():
                if nc and (nc in k or k in nc):
                    return orig

        if required:
            raise KeyError(
                f"Could not find required column among candidates={candidates}. "
                f"Available columns={list(df.columns)}"
            )
        return None

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def dislike_indicator_from_gss_item(series):
        """
        Expected numeric coding 1..5:
          1 Like very much
          2 Like it
          3 Mixed feelings
          4 Dislike
          5 Dislike very much
        DK/NA/out-of-range -> NaN
        Returns float series: {0.0, 1.0, NaN}
        """
        x = to_num(series)
        out = pd.Series(np.nan, index=x.index, dtype=float)
        valid = x.between(1, 5)
        out.loc[valid] = (x.loc[valid] >= 4).astype(float)
        return out

    def like_audience_mask(series):
        """Audience = Like very much (1) or Like it (2). Others (incl DK/NA) -> False."""
        x = to_num(series)
        return x.isin([1, 2])

    def fit_logit_irls(X, y, max_iter=100, tol=1e-10):
        """
        Logistic regression via IRLS / Newton-Raphson.
        X includes intercept column.
        y in {0,1}.
        Returns beta vector.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n, k = X.shape
        beta = np.zeros(k, dtype=float)

        for _ in range(max_iter):
            eta = X @ beta
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
            w = p * (1.0 - p)
            w = np.clip(w, 1e-9, None)

            z = eta + (y - p) / w

            WX = X * w[:, None]
            XtWX = X.T @ WX
            XtWz = X.T @ (w * z)

            try:
                beta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.pinv(XtWX) @ XtWz

            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        return beta

    # -----------------------------
    # Read data
    # -----------------------------
    df = pd.read_csv(data_source)

    year_col = find_column(df, ["YEAR", "year"])
    educ_col = find_column(df, ["EDUC", "educ", "education", "years_education", "yrs_educ"])

    # Canonical GSS 1993 music battery (with robust fallbacks)
    genre_col_candidates = {
        "Latin/Salsa": ["LATIN", "MUSICLAT", "MUSICLATIN", "MUSIC_SALSA", "latin", "salsa"],
        "Jazz": ["JAZZ", "MUSICJAZ", "MUSICJAZZ", "jazz"],
        "Blues/R&B": ["BLUES", "MUSICBLU", "MUSICBLUES", "blues", "rnb", "rhythmandblues", "rhythmblues"],
        "Show Tunes": ["MUSICALS", "MUSICMUS", "MUSICMUSICALS", "musicals", "showtunes", "showtune"],
        "Oldies": ["OLDIES", "MUSICOLD", "MUSICOLDIES", "oldies"],
        "Classical": ["CLASSICL", "MUSICCLA", "MUSICCLASSICAL", "classicl", "classical"],
        "Reggae": ["REGGAE", "MUSICREG", "MUSICREGGAE", "reggae"],
        "Swing": ["BIGBAND", "MUSICBIG", "MUSICBIGBAND", "bigband", "swing"],
        "New Age/Space": ["NEWAGE", "MUSICNEW", "MUSICNEWAGE", "newage", "newagespace", "space"],
        "Opera": ["OPERA", "MUSICOPR", "MUSICOPERA", "opera"],
        "Bluegrass": ["BLUGRASS", "MUSICBLG", "MUSICBLUEGRASS", "blugrass", "bluegrass"],
        "Folk": ["FOLK", "MUSICFOL", "MUSICFOLK", "folk"],
        "Easy Listening": ["MOODEASY", "MUSICEZL", "musiceasy", "moodeasy", "easylistening", "mood"],
        "Pop/Rock": ["CONROCK", "MUSICPOP", "MUSICROK", "MUSICROCK", "conrock", "poprock", "contemporaryrock"],
        "Rap": ["RAP", "MUSICRAP", "rap"],
        "Heavy Metal": ["HVYMETAL", "MUSICMET", "hvymetal", "heavymetal", "metal"],
        "Country": ["COUNTRY", "MUSICCNT", "MUSICCOUNTRY", "country", "countrywestern"],
        "Gospel": ["GOSPEL", "MUSICGOS", "MUSICGOSPEL", "gospel"],
    }
    genre_cols = {g: find_column(df, cands, required=True) for g, cands in genre_col_candidates.items()}

    # -----------------------------
    # Filter to 1993 + clean education
    # -----------------------------
    df = df.copy()
    df[year_col] = to_num(df[year_col])
    df = df.loc[df[year_col] == 1993].copy()

    df[educ_col] = to_num(df[educ_col])
    df.loc[~df[educ_col].between(0, 25), educ_col] = np.nan

    # Sample mean education for reference line (computed from 1993 subset)
    sample_mean_educ = float(df[educ_col].mean(skipna=True)) if df[educ_col].notna().any() else np.nan

    # -----------------------------
    # Build dislike indicators for all genres
    # -----------------------------
    dislike = {g: dislike_indicator_from_gss_item(df[col]) for g, col in genre_cols.items()}

    # -----------------------------
    # Run 18 logits + audience education means
    # -----------------------------
    genre_list = list(genre_cols.keys())
    rows = []

    for g in genre_list:
        # T_-g: number of OTHER genres not disliked, requiring observed info for all 17 others
        others = [h for h in genre_list if h != g]
        other_dislike = pd.DataFrame({h: dislike[h] for h in others})

        # 1 if not disliked, 0 if disliked, NaN if missing
        not_disliked = (other_dislike == 0).astype(float).where(other_dislike.notna(), np.nan)

        # tolerance depends on observed info for other 17 genres
        complete_other = other_dislike.notna().all(axis=1)
        T_minus_g = not_disliked.sum(axis=1).where(complete_other, np.nan)

        y = dislike[g]
        educ = df[educ_col]

        # CRITICAL: drop NaNs before fitting
        model_df = pd.DataFrame({"y": y, "T": T_minus_g, "educ": educ}).dropna()
        model_df = model_df.loc[model_df["y"].isin([0.0, 1.0])].copy()

        beta_T = np.nan
        if (
            model_df.shape[0] >= 50
            and model_df["y"].nunique() == 2
            and model_df["T"].nunique() >= 2
            and model_df["educ"].nunique() >= 2
        ):
            X = np.column_stack(
                [
                    np.ones(model_df.shape[0], dtype=float),
                    model_df["T"].to_numpy(dtype=float),
                    model_df["educ"].to_numpy(dtype=float),
                ]
            )
            yv = model_df["y"].to_numpy(dtype=float)
            beta = fit_logit_irls(X, yv)
            beta_T = float(beta[1])

        like_mask = like_audience_mask(df[genre_cols[g]])
        mean_edu = float(df.loc[like_mask, educ_col].mean(skipna=True)) if like_mask.any() else np.nan

        rows.append({"genre": g, "coef_tolerance": beta_T, "mean_edu": mean_edu})

    res = pd.DataFrame(rows)

    # -----------------------------
    # Force the exact x-axis order in the article Figure 1 (per feedback)
    # -----------------------------
    genre_order = [
        "Latin/Salsa",
        "Jazz",
        "Blues/R&B",
        "Show Tunes",
        "Oldies",
        "Classical",
        "Reggae",
        "Swing",
        "New Age/Space",
        "Opera",
        "Bluegrass",
        "Folk",
        "Easy Listening",
        "Pop/Rock",
        "Rap",
        "Heavy Metal",
        "Country",
        "Gospel",
    ]
    res = res.set_index("genre").reindex(genre_order).reset_index()

    # Ensure 1:1 genre mapping
    res = res.drop_duplicates(subset=["genre"])

    # -----------------------------
    # Plot (matplotlib only): no title; two y-axes; in-plot labels
    # -----------------------------
    out_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_scripts/output_run_all/20260130_222138/Figure1/generated_results.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(len(res))

    fig, ax = plt.subplots(figsize=(10.5, 4.3), dpi=200)

    # LEFT axis: tolerance coefficient (SOLID)
    ax.plot(
        x,
        res["coef_tolerance"].to_numpy(dtype=float),
        color="black",
        linewidth=2.3,
        linestyle="-",
        zorder=3,
    )

    # RIGHT axis: mean education (DASH-DOT)
    ax2 = ax.twinx()
    edu_line = ax2.plot(
        x,
        res["mean_edu"].to_numpy(dtype=float),
        color="black",
        linewidth=2.3,
        linestyle="-.",
        zorder=2,
    )[0]
    # make dash-dot spacing more print-like (long dash, gap, dot, gap)
    edu_line.set_dashes([8, 3, 2, 3])

    # Sample mean education dotted line on RIGHT axis (computed)
    if np.isfinite(sample_mean_educ):
        ax2.axhline(sample_mean_educ, color="black", linewidth=1.2, linestyle=(0, (1, 3)), zorder=1)

    # X axis ticks + labels
    ax.set_xticks(x)
    ax.set_xticklabels(res["genre"].tolist(), rotation=55, ha="right")
    ax.set_xlabel("Type of Music", fontweight="bold")

    # Left y-axis formatting
    ax.set_ylim(-0.5, -0.1)
    ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, pos: f"{v:.1f}".replace("-0.", "-.").replace("0.", "."))
    )
    ax.set_ylabel(
        "Coefficients for Musical Tolerance in Affecting Odds (Probability)\nof Disliking Each Individual Genre",
        fontweight="bold",
    )

    # Right y-axis formatting
    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])
    ax2.set_ylabel(
        "Mean Educational Level of Respondents Who\nReported Liking Each Audience",
        fontweight="bold",
        rotation=270,
        labelpad=33,
    )

    # Styling: remove top spines; heavier axes/ticks (print look)
    ax.grid(False)
    ax2.grid(False)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    for a in (ax, ax2):
        a.tick_params(axis="both", which="major", width=1.6, length=9, color="black")

    for s in ax.spines.values():
        s.set_linewidth(1.6)
    for s in ax2.spines.values():
        s.set_linewidth(1.6)

    # -----------------------------
    # In-plot annotations (point to correct series)
    # -----------------------------
    # Use an early point for callouts (index 2 = Blues/R&B)
    i = 2

    if i < len(res) and np.isfinite(res.loc[i, "mean_edu"]):
        ax2.annotate(
            "Mean Education of Genre Audience",
            xy=(i, float(res.loc[i, "mean_edu"])),
            xycoords=("data", "data"),
            xytext=(0.7, 14.45),
            textcoords=("data", "data"),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.4),
            fontsize=12,
            fontweight="bold",
        )

    if i < len(res) and np.isfinite(res.loc[i, "coef_tolerance"]):
        ax.annotate(
            "Coefficient for Musical Tolerance",
            xy=(i, float(res.loc[i, "coef_tolerance"])),
            xycoords=("data", "data"),
            xytext=(1.2, -0.475),
            textcoords=("data", "data"),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.4),
            fontsize=12,
            fontweight="bold",
        )

    if np.isfinite(sample_mean_educ):
        ax2.annotate(
            "Sample Mean Education",
            xy=(0.38, sample_mean_educ),
            xycoords=("axes fraction", "data"),
            xytext=(0.38, sample_mean_educ - 0.55),
            textcoords=("axes fraction", "data"),
            ha="center",
            va="top",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4),
            fontsize=12,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, format="jpg")
    plt.close(fig)

    return os.path.abspath(out_path)