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
        Return numeric 1..5 where possible; DK/NA/invalid -> NaN.
        Accepts numeric or string labels.
        """
        # numeric path
        x = to_num(series)
        if x.notna().sum() > 0:
            x = x.where(x.between(1, 5), np.nan)
            return x

        # string path
        s = series.astype(str).str.strip().str.lower()
        s = s.str.replace(r"\s+", " ", regex=True)
        s = s.str.replace("dont", "don't", regex=False).str.replace("do not", "don't", regex=False)

        mapping = {
            "like very much": 1,
            "like it": 2,
            "mixed feelings": 3,
            "dislike": 4,
            "dislike it": 4,
            "dislike very much": 5,
            "don't know much about it": np.nan,
            "dont know much about it": np.nan,
            "dk": np.nan,
            "don't know": np.nan,
            "dont know": np.nan,
            "no answer": np.nan,
            "na": np.nan,
            "n/a": np.nan,
            "refused": np.nan,
        }
        x2 = s.map(mapping)
        x2 = to_num(x2).where(to_num(x2).between(1, 5), np.nan)
        return x2

    def dislike_indicator(x_1to5):
        """
        D_g = 1 if 4 or 5; 0 if 1..3; missing otherwise.
        """
        x = x_1to5
        out = pd.Series(np.nan, index=x.index, dtype=float)
        valid = x.between(1, 5)
        out.loc[valid] = (x.loc[valid] >= 4).astype(float)
        return out

    def like_audience_mask(x_1to5):
        """
        Audience = Like very much (1) or Like it (2).
        """
        return x_1to5.isin([1, 2])

    def fit_logit_irls(X, y, max_iter=100, tol=1e-10):
        """
        Logistic regression via IRLS / Newton-Raphson.
        X includes intercept column.
        Returns beta vector.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        beta = np.zeros(X.shape[1], dtype=float)

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

    year_col = find_column(df, ["YEAR", "year"], required=True)
    educ_col = find_column(df, ["EDUC", "educ", "education", "years_education", "yrs_educ"], required=True)

    # Canonical GSS 1993 music battery
    genre_col_candidates = {
        "Latin/Salsa": ["LATIN", "MUSICLAT", "MUSICLATIN", "musiclat", "musiclatin", "latin"],
        "Jazz": ["JAZZ", "MUSICJAZ", "MUSICJAZZ", "musicjaz", "musicjazz", "jazz"],
        "Blues/R&B": ["BLUES", "MUSICBLU", "MUSICBLUES", "musicblu", "musicblues", "blues"],
        "Show Tunes": ["MUSICALS", "MUSICMUS", "MUSICMUSICALS", "musicmus", "musicmusicals", "musicals", "showtunes"],
        "Oldies": ["OLDIES", "MUSICOLD", "MUSICOLDIES", "musicold", "musicoldies", "oldies"],
        "Classical": ["CLASSICL", "MUSICCLA", "MUSICCLASSICAL", "classicl", "musiccla", "musicclassical", "classical"],
        "Swing": ["BIGBAND", "MUSICBIG", "MUSICBIGBAND", "bigband", "musicbig", "musicbigband", "swing"],
        "New Age/Space": ["NEWAGE", "MUSICNEW", "MUSICNEWAGE", "newage", "musicnew", "musicnewage", "space"],
        "Opera": ["OPERA", "MUSICOPR", "MUSICOPERA", "musicopr", "musicopera", "opera"],
        "Bluegrass": ["BLUGRASS", "MUSICBLG", "MUSICBLUEGRASS", "blugrass", "musicblg", "musicbluegrass", "bluegrass"],
        "Folk": ["FOLK", "MUSICFOL", "MUSICFOLK", "musicfol", "musicfolk", "folk"],
        "Reggae": ["REGGAE", "MUSICREG", "MUSICREGGAE", "musicreg", "musicreggae", "reggae"],
        "Easy Listening": ["MOODEASY", "MUSICEZL", "musicezl", "moodeasy", "easylistening", "mood"],
        "Pop/Rock": ["CONROCK", "MUSICPOP", "MUSICROK", "MUSICROCK", "conrock", "musicpop", "musicrok", "musicrock", "poprock"],
        "Rap": ["RAP", "MUSICRAP", "musicrap", "rap"],
        "Heavy Metal": ["HVYMETAL", "MUSICMET", "hvymetal", "musicmet", "heavymetal", "metal"],
        "Country": ["COUNTRY", "MUSICCNT", "MUSICCOUNTRY", "countrywestern", "musiccnt", "musiccountry", "country"],
        "Gospel": ["GOSPEL", "MUSICGOS", "MUSICGOSPEL", "musicgos", "musicgospel", "gospel"],
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

    sample_mean_educ = float(df[educ_col].mean(skipna=True)) if df[educ_col].notna().any() else np.nan

    # -----------------------------
    # Recode music items to 1..5 and build dislike indicators
    # -----------------------------
    music_1to5 = {}
    dislike = {}
    for g, col in genre_cols.items():
        music_1to5[g] = recode_music_item(df[col])
        dislike[g] = dislike_indicator(music_1to5[g])

    # -----------------------------
    # Compute per-genre logit tolerance coefficient + mean education of audience
    # -----------------------------
    genre_list = list(genre_cols.keys())
    rows = []

    for g in genre_list:
        others = [h for h in genre_list if h != g]
        other_dislike = pd.DataFrame({h: dislike[h] for h in others})

        # tolerance: count of other genres NOT disliked (1..17), requires complete info on other 17
        complete_other = other_dislike.notna().all(axis=1)
        not_disliked = (other_dislike == 0.0).astype(float).where(other_dislike.notna(), np.nan)
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

        like_mask = like_audience_mask(music_1to5[g])
        mean_edu = float(df.loc[like_mask, educ_col].mean(skipna=True)) if like_mask.any() else np.nan

        rows.append({"genre": g, "coef_tolerance": beta_T, "mean_edu": mean_edu})

    res = pd.DataFrame(rows)

    # -----------------------------
    # Enforce EXACT x-axis order (GT)
    # -----------------------------
    required_order = [
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

    res = res.drop_duplicates(subset=["genre"]).set_index("genre").reindex(required_order).reset_index()
    res["genre"] = pd.Categorical(res["genre"], categories=required_order, ordered=True)
    res = res.sort_values("genre").reset_index(drop=True)

    # -----------------------------
    # Plot (matplotlib only)
    # -----------------------------
    out_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_scripts/output_run_all/20260130_222138/Figure1/generated_results.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(len(required_order), dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 4.3), dpi=200)
    ax2 = ax.twinx()

    # LEFT axis: tolerance coefficient (SOLID)
    ax.plot(
        x,
        res["coef_tolerance"].to_numpy(dtype=float),
        color="black",
        linewidth=2.5,
        linestyle="-",
        marker=None,
        zorder=3,
    )

    # RIGHT axis: mean education (DASH-DOT)
    edu_line = ax2.plot(
        x,
        res["mean_edu"].to_numpy(dtype=float),
        color="black",
        linewidth=2.5,
        linestyle="-.",
        marker=None,
        zorder=2,
    )[0]

    # Ensure a clear dash-dot appearance
    edu_line.set_dashes([6, 3, 1.5, 3])

    # Sample mean education dotted reference line on RIGHT axis (computed, not hard-coded)
    if np.isfinite(sample_mean_educ):
        ax2.axhline(sample_mean_educ, color="black", linewidth=1.2, linestyle=(0, (2, 2)), zorder=1)

    # X axis ticks + labels
    ax.set_xticks(x)
    ax.set_xticklabels(required_order, rotation=55, ha="right")
    ax.set_xlabel("Type of Music", fontweight="bold")

    # Left y-axis formatting + label (verbatim from summary)
    ax.set_ylim(-0.5, -0.1)
    ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])

    def tick_fmt(v, pos):
        s = f"{v:.1f}"
        return s.replace("-0.", "-.").replace("0.", ".")

    ax.yaxis.set_major_formatter(FuncFormatter(tick_fmt))
    ax.set_ylabel(
        "Coefficients for Musical Tolerance in Affecting Odds (Probability) of Disliking Each Individual Genre",
        fontweight="bold",
    )

    # Right y-axis formatting + label (verbatim from summary)
    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])
    ax2.set_ylabel(
        "Mean Educational Level of Respondents Who Reported Liking Each Audience",
        fontweight="bold",
        rotation=270,
        labelpad=32,
    )

    # Styling to match monochrome figure
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
    # In-plot annotations (point to correct series/axes)
    # -----------------------------
    idx_blues = int(required_order.index("Blues/R&B"))
    if idx_blues < len(res) and np.isfinite(res.loc[idx_blues, "mean_edu"]):
        ax2.annotate(
            "Mean Education of Genre Audience",
            xy=(idx_blues, float(res.loc[idx_blues, "mean_edu"])),
            xycoords=("data", "data"),
            xytext=(idx_blues - 1.3, 14.35),
            textcoords=("data", "data"),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="center",
        )

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

    # Sample mean education label+arrow near Reggae/Swing region (use Reggae position)
    if np.isfinite(sample_mean_educ):
        idx_reggae = int(required_order.index("Reggae"))
        ax2.annotate(
            "Sample Mean Education",
            xy=(idx_reggae, sample_mean_educ),
            xycoords=("data", "data"),
            xytext=(idx_reggae, sample_mean_educ - 0.45),
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