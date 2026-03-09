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
            if not nc:
                continue
            for k, orig in norm_map.items():
                if nc in k or k in nc:
                    return orig

        if required:
            raise KeyError(
                f"Could not find required column among candidates={candidates}. "
                f"Available columns={list(df.columns)}"
            )
        return None

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def recode_music_item(series):
        """
        Map GSS-style music item to numeric 1..5 where possible.
        Accepts numeric or string labels.
        Returns numeric series with NaN for DK/NA/invalid.
        """
        s = series.copy()

        # First try numeric coercion
        x = to_num(s)
        if x.notna().any():
            x_valid = x.where(x.between(1, 5), np.nan)
            # If many are valid, use it as-is
            if x_valid.notna().sum() >= max(5, int(0.2 * len(x_valid))):
                return x_valid

        # Otherwise, map string labels
        s2 = s.astype(str).str.strip().str.lower()

        # Normalize common variants
        s2 = (
            s2.str.replace(r"\s+", " ", regex=True)
            .str.replace("dont", "don't", regex=False)
            .str.replace("do not", "don't", regex=False)
        )

        mapping = {
            "like very much": 1,
            "like it": 2,
            "mixed feelings": 3,
            "dislike": 4,
            "dislike it": 4,
            "dislike very much": 5,
            "dont know much about it": np.nan,
            "don't know much about it": np.nan,
            "dk": np.nan,
            "dont know": np.nan,
            "don't know": np.nan,
            "no answer": np.nan,
            "na": np.nan,
            "n/a": np.nan,
            "refused": np.nan,
        }

        x2 = s2.map(mapping)
        x2 = to_num(x2).where(to_num(x2).between(1, 5), np.nan)
        return x2

    def dislike_indicator(series_1to5):
        """
        D_g = 1 if response is 4 or 5; 0 if 1..3; NaN otherwise.
        """
        x = series_1to5
        out = pd.Series(np.nan, index=x.index, dtype=float)
        valid = x.between(1, 5)
        out.loc[valid] = (x.loc[valid] >= 4).astype(float)
        return out

    def like_audience_mask(series_1to5):
        """
        Audience = Like very much (1) or Like it (2).
        """
        x = series_1to5
        return x.isin([1, 2])

    def fit_logit_irls(X, y, max_iter=100, tol=1e-10):
        """
        Logistic regression via IRLS / Newton-Raphson.
        X includes intercept column.
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
        "Latin/Salsa": ["LATIN", "MUSICLAT", "MUSICLATIN", "musiclat", "musiclatin"],
        "Jazz": ["JAZZ", "MUSICJAZ", "MUSICJAZZ", "musicjaz", "musicjazz"],
        "Blues/R&B": ["BLUES", "MUSICBLU", "MUSICBLUES", "musicblu", "musicblues"],
        "Show Tunes": ["MUSICALS", "MUSICMUS", "MUSICMUSICALS", "musicmus", "musicmusicals", "showtunes", "musicals"],
        "Oldies": ["OLDIES", "MUSICOLD", "MUSICOLDIES", "musicold", "musicoldies"],
        "Classical": ["CLASSICL", "MUSICCLA", "MUSICCLASSICAL", "classicl", "musiccla", "musicclassical"],
        "Swing": ["BIGBAND", "MUSICBIG", "MUSICBIGBAND", "bigband", "musicbig", "musicbigband"],
        "New Age/Space": ["NEWAGE", "MUSICNEW", "MUSICNEWAGE", "newage", "musicnew", "musicnewage"],
        "Opera": ["OPERA", "MUSICOPR", "MUSICOPERA", "musicopr", "musicopera"],
        "Bluegrass": ["BLUGRASS", "MUSICBLG", "MUSICBLUEGRASS", "blugrass", "musicblg", "musicbluegrass"],
        "Folk": ["FOLK", "MUSICFOL", "MUSICFOLK", "musicfol", "musicfolk"],
        "Reggae": ["REGGAE", "MUSICREG", "MUSICREGGAE", "musicreg", "musicreggae"],
        "Easy Listening": ["MOODEASY", "MUSICEZL", "musicezl", "moodeasy", "easylistening", "mood"],
        "Pop/Rock": ["CONROCK", "MUSICPOP", "MUSICROK", "MUSICROCK", "conrock", "musicpop", "musicrok", "musicrock"],
        "Rap": ["RAP", "MUSICRAP", "musicrap"],
        "Heavy Metal": ["HVYMETAL", "MUSICMET", "hvymetal", "musicmet"],
        "Country": ["COUNTRY", "MUSICCNT", "MUSICCOUNTRY", "countrywestern", "musiccnt", "musiccountry"],
        "Gospel": ["GOSPEL", "MUSICGOS", "MUSICGOSPEL", "musicgos", "musicgospel"],
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

    # Compute sample mean education (1993 subset)
    sample_mean_educ = float(df[educ_col].mean(skipna=True)) if df[educ_col].notna().any() else np.nan

    # -----------------------------
    # Recode all music items to 1..5 and build dislike indicators
    # -----------------------------
    music_1to5 = {}
    dislike = {}
    for g, col in genre_cols.items():
        music_1to5[g] = recode_music_item(df[col])
        dislike[g] = dislike_indicator(music_1to5[g])

    # -----------------------------
    # Run 18 logits + audience education means
    # -----------------------------
    genre_list = list(genre_cols.keys())
    rows = []

    for g in genre_list:
        others = [h for h in genre_list if h != g]
        other_dislike = pd.DataFrame({h: dislike[h] for h in others})

        # not_disliked: 1 if other genre is not disliked; 0 if disliked; NaN if missing
        not_disliked = (other_dislike == 0.0).astype(float).where(other_dislike.notna(), np.nan)

        # tolerance requires observed info for all 17 other genres
        complete_other = other_dislike.notna().all(axis=1)
        T_minus_g = not_disliked.sum(axis=1).where(complete_other, np.nan)

        y = dislike[g]
        educ = df[educ_col]

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

        like_mask = like_audience_mask(music_1to5[g])
        mean_edu = float(df.loc[like_mask, educ_col].mean(skipna=True)) if like_mask.any() else np.nan

        rows.append({"genre": g, "coef_tolerance": beta_T, "mean_edu": mean_edu})

    res = pd.DataFrame(rows)

    # -----------------------------
    # Force exact x-axis order to match Figure 1 (scan)
    # -----------------------------
    genre_order = [
        "Latin/Salsa",
        "Jazz",
        "Blues/R&B",
        "Show Tunes",
        "Oldies",
        "Classical",
        "Swing/Big Band",      # placeholder to be replaced if needed
        "New Age/Space",       # placeholder to be replaced if needed
        "Opera",               # placeholder to be replaced if needed
        "Bluegrass",           # placeholder to be replaced if needed
        "Folk",                # placeholder to be replaced if needed
        "Reggae",              # placeholder to be replaced if needed
        "Easy Listening",
        "Pop/Rock",
        "Rap",
        "Heavy Metal",
        "Country",
        "Gospel",
    ]

    # The analysis summary's shown order includes Swing and New Age and Opera/Bluegrass placement.
    # Our canonical keys use "Swing" and "Reggae" separately; reconcile by direct mapping:
    # Final required order (per summary): Latin/Salsa, Jazz, Blues/R&B, Show Tunes, Oldies,
    # Classical/Chamber, Swing/Big Band, New Age/Space, Opera, Bluegrass, Folk, Reggae,
    # Easy Listening, Pop/Contemporary Rock, Rap, Heavy Metal, Country/Western, Gospel
    required_order = [
        "Latin/Salsa",
        "Jazz",
        "Blues/R&B",
        "Show Tunes",
        "Oldies",
        "Classical",
        "Swing",
        "New Age/Space",
        "Opera",
        "Bluegrass",
        "Folk",
        "Reggae",
        "Easy Listening",
        "Pop/Rock",
        "Rap",
        "Heavy Metal",
        "Country",
        "Gospel",
    ]

    res = res.drop_duplicates(subset=["genre"]).set_index("genre").reindex(required_order).reset_index()
    res["genre"] = pd.Categorical(res["genre"], categories=required_order, ordered=True)
    res = res.sort_values("genre").reset_index(drop=True)

    # -----------------------------
    # Plot (matplotlib only): two y-axes; correct mapping & styles; in-plot labels
    # -----------------------------
    out_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_scripts/output_run_all/20260130_222138/Figure1/generated_results.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(len(res), dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 4.3), dpi=200)

    # Left axis: tolerance coefficient (SOLID)
    ax.plot(
        x,
        res["coef_tolerance"].to_numpy(dtype=float),
        color="black",
        linewidth=2.5,
        linestyle="-",
        marker=None,
        zorder=3,
    )

    # Right axis: mean education (DASH-DOT)
    ax2 = ax.twinx()
    ax2.plot(
        x,
        res["mean_edu"].to_numpy(dtype=float),
        color="black",
        linewidth=2.5,
        linestyle="-.",
        marker=None,
        zorder=2,
    )

    # Sample mean education dotted reference line on RIGHT axis
    if np.isfinite(sample_mean_educ):
        ax2.axhline(sample_mean_educ, color="black", linewidth=1.2, linestyle=(0, (2, 2)), zorder=1)

    # X axis ticks + labels
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in res["genre"].tolist()], rotation=55, ha="right")
    ax.set_xlabel("Type of Music", fontweight="bold")

    # Left y-axis formatting + label
    ax.set_ylim(-0.5, -0.1)
    ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])

    def tick_fmt(v, pos):
        s = f"{v:.1f}"
        s = s.replace("-0.", "-.").replace("0.", ".")
        return s

    ax.yaxis.set_major_formatter(FuncFormatter(tick_fmt))
    ax.set_ylabel(
        "Coefficients for Musical Tolerance in Affecting Odds (Probability) of Disliking Each Individual Genre",
        fontweight="bold",
    )

    # Right y-axis formatting + label
    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])
    ax2.set_ylabel(
        "Mean Educational Level of Respondents Who Reported Liking Each Audience",
        fontweight="bold",
        rotation=270,
        labelpad=32,
    )

    # Styling: remove top spines
    ax.grid(False)
    ax2.grid(False)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    for a in (ax, ax2):
        a.tick_params(axis="both", which="major", width=2.0, length=10, color="black")

    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax2.spines["right"].set_linewidth(2.0)

    # -----------------------------
    # In-plot annotations positioned to match GT geometry more closely
    # -----------------------------
    # Mean education label near early genres (Jazz/Blues region), pointing to ax2 series
    idx_blues = int(required_order.index("Blues/R&B"))
    if idx_blues < len(res) and np.isfinite(res.loc[idx_blues, "mean_edu"]):
        ax2.annotate(
            "Mean Education of Genre Audience",
            xy=(idx_blues, float(res.loc[idx_blues, "mean_edu"])),
            xycoords=("data", "data"),
            xytext=(idx_blues - 1.2, 14.35),
            textcoords=("data", "data"),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="center",
        )

    # Coefficient label low-left, pointing to ax series near Jazz
    idx_jazz = int(required_order.index("Jazz"))
    if idx_jazz < len(res) and np.isfinite(res.loc[idx_jazz, "coef_tolerance"]):
        ax.annotate(
            "Coefficient for Musical Tolerance",
            xy=(idx_jazz, float(res.loc[idx_jazz, "coef_tolerance"])),
            xycoords=("data", "data"),
            xytext=(idx_jazz + 0.8, -0.475),
            textcoords=("data", "data"),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="center",
        )

    # Sample mean education label+arrow near Reggae/Swing region (place over Swing)
    if np.isfinite(sample_mean_educ):
        idx_swing = int(required_order.index("Swing"))
        ax2.annotate(
            "Sample Mean Education",
            xy=(idx_swing, sample_mean_educ),
            xycoords=("data", "data"),
            xytext=(idx_swing, sample_mean_educ - 0.45),
            textcoords=("data", "data"),
            ha="center",
            va="top",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
            fontsize=13,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, format="jpg")
    plt.close(fig)

    return os.path.abspath(out_path)