def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    # ---------- Helpers ----------
    def find_col(df_cols, preferred):
        cols = list(df_cols)
        lower = {c.lower(): c for c in cols}
        for name in preferred:
            if name.lower() in lower:
                return lower[name.lower()]
        # fallback: partial contains
        for name in preferred:
            n = name.lower()
            matches = [c for c in cols if n in c.lower()]
            if len(matches) == 1:
                return matches[0]
        return None

    def coerce_numeric(s):
        return pd.to_numeric(s, errors="coerce")

    def compute_dislike(series):
        # Dislike = 1 if response is 4 or 5; else 0 if 1-3; missing otherwise
        x = coerce_numeric(series)
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out[(x >= 1) & (x <= 3)] = 0.0
        out[(x == 4) | (x == 5)] = 1.0
        return out

    def compute_like(series):
        # Like = 1 if response is 1 or 2; else 0 if 3-5; missing otherwise
        x = coerce_numeric(series)
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out[(x == 1) | (x == 2)] = 1.0
        out[(x >= 3) & (x <= 5)] = 0.0
        return out

    # ---------- Load ----------
    df = pd.read_csv(data_source)

    # normalize columns (keep original too)
    df_cols = df.columns

    # YEAR filter (case-insensitive)
    year_col = find_col(df_cols, ["YEAR", "year"])
    if year_col is not None:
        df[year_col] = coerce_numeric(df[year_col])
        df = df.loc[df[year_col] == 1993].copy()

    educ_col = find_col(df_cols, ["EDUC", "educ", "education"])
    if educ_col is None:
        raise ValueError("Could not find EDUC column.")

    df[educ_col] = coerce_numeric(df[educ_col])

    # Genre mapping (preferred canonical -> possible alternatives)
    genre_specs = [
        ("Latin/Salsa", ["LATIN", "latin", "latin_salsa", "salsa", "mariachi"]),
        ("Jazz", ["JAZZ", "jazz"]),
        ("Blues/R&B", ["BLUES", "blues", "r&b", "rnb"]),
        ("Show Tunes", ["MUSICALS", "musicals", "showtunes", "show_tunes"]),
        ("Oldies", ["OLDIES", "oldies"]),
        ("Classical/Chamber", ["CLASSICL", "classicl", "classical", "classical_chamber"]),
        ("Swing/Big Band", ["BIGBAND", "bigband", "big_band", "swing"]),
        ("New Age/Space", ["NEWAGE", "newage", "new_age", "space"]),
        ("Opera", ["OPERA", "opera"]),
        ("Bluegrass", ["BLUGRASS", "blugrass", "bluegrass"]),
        ("Folk", ["FOLK", "folk"]),
        ("Easy Listening", ["MOODEASY", "moodeasy", "easylist", "easy_listening", "easy"]),
        ("Pop/Contemporary Rock", ["CONROCK", "conrock", "poprock", "pop_rock", "contemporary_rock", "rock"]),
        ("Reggae", ["REGGAE", "reggae"]),
        ("Rap", ["RAP", "rap", "hiphop", "hip_hop"]),
        ("Heavy Metal", ["HVYMETAL", "hvymetal", "heavymetal", "heavy_metal"]),
        ("Country/Western", ["COUNTRY", "country", "countrywestern", "country_western"]),
        ("Gospel", ["GOSPEL", "gospel"]),
    ]

    # Resolve genre columns
    genres = []
    for label, candidates in genre_specs:
        col = find_col(df_cols, candidates)
        if col is None:
            raise ValueError(f"Could not find column for genre: {label}")
        genres.append((label, col))

    # ---------- Build dislike matrix ----------
    dislike_df = pd.DataFrame(index=df.index)
    like_df = pd.DataFrame(index=df.index)
    for label, col in genres:
        dislike_df[label] = compute_dislike(df[col])
        like_df[label] = compute_like(df[col])

    # ---------- Fit per-genre logits ----------
    results = []
    for g_label, _ in genres:
        y = dislike_df[g_label]

        # Tolerance_{-g}: count of other genres NOT disliked (0/1), require valid responses on those genres
        others = [lab for lab, _c in genres if lab != g_label]
        others_dislike = dislike_df[others]

        # not disliked indicator is 1 if dislike==0, 0 if dislike==1
        not_disliked = 1.0 - others_dislike

        # require all 17 others non-missing to compute tolerance, plus y and educ non-missing
        tol = not_disliked.sum(axis=1, min_count=len(others))
        # tol will be NaN if any missing among others due to min_count
        educ = df[educ_col]

        model_df = pd.DataFrame({
            "y": y,
            "tolerance": tol,
            "educ": educ
        }).dropna()

        # ensure y is binary and has variation
        if model_df["y"].nunique() < 2:
            beta = np.nan
            pval = np.nan
        else:
            X = sm.add_constant(model_df[["tolerance", "educ"]], has_constant="add")
            try:
                fit = sm.Logit(model_df["y"], X).fit(disp=False, maxiter=200)
                beta = float(fit.params["tolerance"])
                pval = float(fit.pvalues["tolerance"])
            except Exception:
                # fallback to regularized fit if separation occurs
                try:
                    fit = sm.Logit(model_df["y"], X).fit_regularized(disp=False, maxiter=500)
                    beta = float(fit.params["tolerance"])
                    pval = np.nan
                except Exception:
                    beta = np.nan
                    pval = np.nan

        # MeanEduAudience_g among likers (like very much / like it => 1)
        like_ind = like_df[g_label]
        aud_df = pd.DataFrame({"like": like_ind, "educ": educ}).dropna()
        mean_edu_audience = float(aud_df.loc[aud_df["like"] == 1, "educ"].mean()) if (aud_df["like"] == 1).any() else np.nan

        results.append({
            "genre": g_label,
            "beta_tolerance": beta,
            "pval": pval,
            "mean_edu_audience": mean_edu_audience
        })

    res = pd.DataFrame(results)

    # Drop genres that failed
    res = res.dropna(subset=["beta_tolerance", "mean_edu_audience"]).copy()

    # Sort by beta (closest to 0 on left => descending)
    res = res.sort_values("beta_tolerance", ascending=False).reset_index(drop=True)

    # Sample mean education (among those with non-missing educ)
    sample_mean_educ = float(df[educ_col].dropna().mean())

    # ---------- Plot ----------
    out_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_scripts/output_run_all/20260119_074740/Figure1/generated_results.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(len(res))
    genres_order = res["genre"].tolist()

    fig, ax_left = plt.subplots(figsize=(11.5, 6.5))
    ax_right = ax_left.twinx()

    # Left series: tolerance coefficients (dotted)
    ax_left.plot(
        x,
        res["beta_tolerance"].values,
        linestyle=":",
        color="black",
        linewidth=2.0
    )

    # Right series: mean education (solid)
    ax_right.plot(
        x,
        res["mean_edu_audience"].values,
        linestyle="-",
        color="black",
        linewidth=1.8
    )

    # Sample mean education line (right axis)
    ax_right.axhline(sample_mean_educ, color="black", linewidth=1.0, linestyle="-")

    # Axis labels/titles comparable to Figure 1
    ax_left.set_xlabel("Type of Music")
    ax_left.set_ylabel("Coefficients for Musical Tolerance\n(Log Odds Probability of Disliking Each Musical Genre)")
    ax_right.set_ylabel("Mean Educational Level of Respondents Who Reported Liking Each Type of Music")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(genres_order, rotation=45, ha="right")

    # Set y-lims to be comparable to paper when possible, but not hard-coded to data
    # Keep left axis with top near 0 if coefficients are negative
    betas = res["beta_tolerance"].values
    if np.isfinite(betas).all():
        top = min(0.05, float(np.nanmax(betas) + 0.03))
        bottom = float(np.nanmin(betas) - 0.05)
        ax_left.set_ylim(bottom, top)

    # Right axis range around 12-15 if possible, but data-driven
    edu_vals = res["mean_edu_audience"].values
    if np.isfinite(edu_vals).all():
        r_bottom = float(np.nanmin([np.nanmin(edu_vals) - 0.3, sample_mean_educ - 1.0]))
        r_top = float(np.nanmax([np.nanmax(edu_vals) + 0.3, sample_mean_educ + 1.0]))
        ax_right.set_ylim(r_bottom, r_top)

    # Inline labels (approximate placement)
    if len(res) >= 3:
        ax_right.text(
            x=max(0, len(res)//3),
            y=float(res["mean_edu_audience"].iloc[max(0, len(res)//3)]),
            s="Mean Education of Genre Audience",
            fontsize=10,
            ha="left",
            va="bottom",
            color="black"
        )
        ax_left.text(
            x=max(0, len(res)//2),
            y=float(res["beta_tolerance"].iloc[max(0, len(res)//2)]),
            s="Coefficient for Musical Tolerance",
            fontsize=10,
            ha="left",
            va="top",
            color="black"
        )

    ax_right.text(
        0.02,
        sample_mean_educ,
        "Sample Mean Education",
        transform=ax_right.get_yaxis_transform(),
        fontsize=10,
        va="bottom",
        ha="left",
        color="black"
    )

    # Subtle grid similar to print readability
    ax_left.grid(axis="y", color="0.85", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, format="jpg")
    plt.close(fig)

    return os.path.abspath(out_path)