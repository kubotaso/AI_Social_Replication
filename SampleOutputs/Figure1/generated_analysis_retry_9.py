def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.ticker import FuncFormatter
    import statsmodels.api as sm

    # ----------------------------
    # Helpers
    # ----------------------------
    def find_col(df, preferred_names, contains_any=None):
        cols = list(df.columns)
        lower_map = {c.lower(): c for c in cols}

        for name in preferred_names:
            if name is None:
                continue
            key = str(name).lower()
            if key in lower_map:
                return lower_map[key]

        if contains_any:
            tokens = [str(t).lower() for t in contains_any if t is not None and str(t).strip() != ""]
            hits = []
            for c in cols:
                cl = c.lower()
                if any(tok in cl for tok in tokens):
                    hits.append(c)
            if len(hits) == 1:
                return hits[0]

        for name in preferred_names:
            if name is None:
                continue
            n = str(name).lower()
            matches = [c for c in cols if n in c.lower()]
            if len(matches) == 1:
                return matches[0]

        return None

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def recode_dislike(series):
        # 1 if 4/5, 0 if 1/2/3, NaN otherwise
        x = to_num(series)
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out[(x >= 1) & (x <= 3)] = 0.0
        out[(x == 4) | (x == 5)] = 1.0
        return out

    def recode_like(series):
        # 1 if 1/2, 0 if 3/4/5, NaN otherwise
        x = to_num(series)
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out[(x == 1) | (x == 2)] = 1.0
        out[(x >= 3) & (x <= 5)] = 0.0
        return out

    def no_leading_zero(x, pos):
        s = f"{x:.1f}"
        s = s.replace("-0.", "-.").replace("0.", ".")
        return s

    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(data_source)

    # YEAR filter
    year_col = find_col(df, ["YEAR", "year"])
    if year_col is not None:
        df[year_col] = to_num(df[year_col])
        df = df.loc[df[year_col] == 1993].copy()

    # EDUC
    educ_col = find_col(df, ["EDUC", "educ", "education", "years_education", "schooling"])
    if educ_col is None:
        raise ValueError("Could not find education column (EDUC/educ).")
    df[educ_col] = to_num(df[educ_col])

    # CRITICAL: drop missing education before anything else
    df = df.dropna(subset=[educ_col]).copy()

    # ----------------------------
    # Exact Figure-1 genre order and labels (GT)
    # ----------------------------
    order = [
        "Latin/Salsa", "Jazz", "Blues/R&B", "Show Tunes", "Oldies", "Classical",
        "Reggae", "Swing", "New Age/Space", "Opera", "Bluegrass", "Folk",
        "Easy Listening", "Pop/Rock", "Rap", "Heavy Metal", "Country", "Gospel"
    ]

    # Map labels -> dataset columns (robust)
    genre_candidates = {
        "Latin/Salsa": ["LATIN", "latin"],
        "Jazz": ["JAZZ", "jazz"],
        "Blues/R&B": ["BLUES", "blues"],
        "Show Tunes": ["MUSICALS", "musicals", "SHOWTUNES", "showtunes"],
        "Oldies": ["OLDIES", "oldies"],
        "Classical": ["CLASSICL", "classicl", "CLASSICAL", "classical"],
        "Reggae": ["REGGAE", "reggae"],
        "Swing": ["BIGBAND", "bigband", "SWING", "swing"],
        "New Age/Space": ["NEWAGE", "newage", "NEW_AGE", "new_age"],
        "Opera": ["OPERA", "opera"],
        "Bluegrass": ["BLUGRASS", "blugrass", "BLUEGRASS", "bluegrass"],
        "Folk": ["FOLK", "folk"],
        "Easy Listening": ["MOODEASY", "moodeasy", "EASYLIST", "easylist", "EASY_LISTENING", "easy_listening"],
        "Pop/Rock": ["CONROCK", "conrock", "POPROCK", "poprock", "POP_ROCK", "pop_rock", "ROCK", "rock"],
        "Rap": ["RAP", "rap", "HIPHOP", "hiphop", "HIP_HOP", "hip_hop"],
        "Heavy Metal": ["HVYMETAL", "hvymetal", "HEAVYMETAL", "heavymetal", "HEAVY_METAL", "heavy_metal"],
        "Country": ["COUNTRY", "country", "COUNTRYWESTERN", "countrywestern", "COUNTRY_WESTERN", "country_western"],
        "Gospel": ["GOSPEL", "gospel"],
    }

    resolved_cols = {}
    for label in order:
        col = find_col(df, genre_candidates.get(label, []))
        if col is None:
            tokens = label.lower().replace("/", " ").replace("&", " ").replace("-", " ").split()
            col = find_col(df, [], contains_any=tokens)
        if col is None:
            raise ValueError(f"Could not find column for genre '{label}'.")
        resolved_cols[label] = col

    # ----------------------------
    # Recode dislikes/likes
    # ----------------------------
    dislike = pd.DataFrame(index=df.index)
    like = pd.DataFrame(index=df.index)

    for label in order:
        col = resolved_cols[label]
        dislike[label] = recode_dislike(df[col])
        like[label] = recode_like(df[col])

    # Sample mean education (respondent-level; education already non-missing)
    sample_mean_edu = float(df[educ_col].mean()) if len(df) else np.nan

    # ----------------------------
    # Fit per-genre logistic models; compute audience mean education
    # CRITICAL: drop NaN BEFORE fitting each model
    # ----------------------------
    rows = []
    for g in order:
        y = dislike[g]
        educ = df[educ_col]

        others = [h for h in order if h != g]
        others_dislike = dislike[others]

        # Tolerance_-g = count of "not disliked" across other 17 genres
        not_disliked = 1.0 - others_dislike

        # Require all other 17 genres present to compute tolerance_-g
        tol_minus_g = not_disliked.sum(axis=1, min_count=len(others))

        model_df = pd.DataFrame({"y": y, "tolerance": tol_minus_g, "educ": educ}).dropna()

        coef_tol = np.nan
        pval_tol = np.nan
        if len(model_df) > 0 and model_df["y"].nunique() >= 2 and model_df["tolerance"].nunique() >= 2:
            X = sm.add_constant(model_df[["tolerance", "educ"]], has_constant="add")
            try:
                fit = sm.Logit(model_df["y"], X).fit(disp=False, maxiter=300)
                coef_tol = float(fit.params["tolerance"])
                pval_tol = float(fit.pvalues["tolerance"])
            except Exception:
                # Separation fallback (still returns coefficients)
                try:
                    fit = sm.Logit(model_df["y"], X).fit_regularized(disp=False, maxiter=2000)
                    coef_tol = float(fit.params["tolerance"])
                    pval_tol = np.nan
                except Exception:
                    coef_tol = np.nan
                    pval_tol = np.nan

        # Mean education among "likers" (1/2 only)
        aud_df = pd.DataFrame({"like": like[g], "educ": educ}).dropna()
        if len(aud_df) and (aud_df["like"] == 1.0).any():
            mean_edu = float(aud_df.loc[aud_df["like"] == 1.0, "educ"].mean())
        else:
            mean_edu = np.nan

        rows.append(
            {
                "genre": g,
                "coef_tolerance": coef_tol,
                "pval": pval_tol,
                "mean_edu": mean_edu,
            }
        )

    res = pd.DataFrame(rows)

    # Enforce EXACT x-order (do not sort by values)
    res = res.set_index("genre").reindex(order).reset_index()

    # ----------------------------
    # Plot (match Figure 1 mapping/styles)
    # ----------------------------
    out_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_scripts/output_run_all/20260119_074740/Figure1/generated_results.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(len(order), dtype=int)

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=150)
    ax2 = ax.twinx()

    coef_vals = res["coef_tolerance"].to_numpy(dtype="float64")
    edu_vals = res["mean_edu"].to_numpy(dtype="float64")

    # LEFT axis: coefficient (solid)
    ax.plot(x, coef_vals, color="black", lw=1.6, ls="-", zorder=3)

    # RIGHT axis: mean education (dash-dot)
    edu_line = ax2.plot(x, edu_vals, color="black", lw=1.6, ls="-.", zorder=2)[0]
    # Printed-figure-like dash cadence
    edu_line.set_dashes([8, 3, 2, 3])  # dash, gap, dot, gap

    # RIGHT axis: sample mean education (dotted)
    if np.isfinite(sample_mean_edu):
        ax2.axhline(sample_mean_edu, color="black", lw=1.0, ls=":", zorder=1)

    # Axis labels (comparable to article)
    ax.set_xlabel("Type of Music", fontweight="bold")
    ax.set_ylabel(
        "Coefficients for Musical Tolerance as It Affects\n"
        "One’s Probability of Disliking Each Music Genre"
    )
    ax2.set_ylabel(
        "Mean Educational Level of Respondents Who\n"
        "Reported Liking Each Music Genre",
        rotation=270,
        labelpad=28
    )

    # X ticks
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")

    # Y axes: ranges/ticks/format
    ax.set_ylim(-0.5, -0.1)
    ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])
    ax.yaxis.set_major_formatter(FuncFormatter(no_leading_zero))

    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

    # No gridlines (match GT)
    ax.grid(False)
    ax2.grid(False)

    # Classic B/W look
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1)
    ax2.tick_params(direction="out", length=6, width=1)

    # ----------------------------
    # Callouts with leader lines + arrow for sample mean education
    # ----------------------------
    def first_finite_idx(arr):
        for i, v in enumerate(arr):
            if np.isfinite(v):
                return i
        return None

    # Prefer anchoring around early genres (like the printed figure), fallback to first finite
    anchor_genre = "Blues/R&B"
    anchor_idx = order.index(anchor_genre) if anchor_genre in order else (first_finite_idx(edu_vals) or 2)

    if np.isfinite(edu_vals[anchor_idx]):
        ax2.annotate(
            "Mean Education of Genre Audience",
            xy=(anchor_idx, float(edu_vals[anchor_idx])),
            xycoords=("data", "data"),
            xytext=(max(0.5, anchor_idx - 1.0), 14.2),
            textcoords=("data", "data"),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="center",
        )

    if np.isfinite(coef_vals[anchor_idx]):
        ax.annotate(
            "Coefficient for Musical Tolerance",
            xy=(anchor_idx, float(coef_vals[anchor_idx])),
            xycoords=("data", "data"),
            xytext=(max(0.5, anchor_idx + 0.4), -0.47),
            textcoords=("data", "data"),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="center",
        )

    if np.isfinite(sample_mean_edu):
        mean_x = order.index("Reggae") if "Reggae" in order else int(min(6, len(order) - 1))
        ax2.annotate(
            "Sample Mean Education",
            xy=(mean_x, sample_mean_edu),
            xycoords=("data", "data"),
            xytext=(mean_x, 12.7),
            textcoords=("data", "data"),
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0),
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, format="jpg")
    plt.close(fig)

    return os.path.abspath(out_path)