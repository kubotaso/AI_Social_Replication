import pandas as pd
from typing import Union, Dict, Any


def run_analysis(data_source, sep=None, na_values=None) -> pd.DataFrame:
    """
    Read the GSS 1993 music-module CSV and produce a Table-3-style descriptive
    frequency table: counts of response categories (1–5, DK, NA) and mean rating
    for each of the 18 music genres.

    Parameters
    ----------
    data_source : str or file-like
        Path or buffer to the CSV file.
    sep : str, optional
        Field separator passed to pandas.read_csv. If None, defaults to ','.
    na_values : scalar, str, list-like, or dict, optional
        Additional strings to recognize as NA/NaN.

    Returns
    -------
    table3 : pandas.DataFrame
        Descriptive table with one row per genre and columns:
        - Genre
        - N
        - Like very much (1)
        - Like it (2)
        - Mixed feelings (3)
        - Dislike (4)
        - Dislike very much (5)
        - Don't know much about it
        - No answer
        - Mean rating (1–5)
    """
    # Read data robustly
    if sep is None:
        sep = ","
    df = pd.read_csv(data_source, sep=sep, na_values=na_values)

    # Expected raw column names in the CSV (GSS variable names)
    genre_cols = [
        "latin", "jazz", "blues", "musicals", "oldies", "classicl",
        "bigband", "newage", "opera", "blugrass", "folk", "moodeasy",
        "conrock", "rap", "hvymetal", "country", "gospel", "reggae"
    ]

    # Map raw GSS names to paper's genre labels, in the paper's order
    paper_order = [
        "latin",    # Latin/Salsa
        "jazz",     # Jazz
        "blues",    # Blues/R&B
        "musicals", # Show Tunes
        "oldies",   # Oldies
        "classicl", # Classical/Chamber
        "reggae",   # Reggae
        "bigband",  # Swing/Big Band
        "newage",   # New Age/Space music
        "opera",    # Opera
        "blugrass", # Bluegrass
        "folk",     # Folk
        "moodeasy", # Pop/Easy listening
        "conrock",  # Contemporary rock
        "rap",      # Rap
        "hvymetal", # Heavy metal
        "country",  # Country/Western
        "gospel"    # Gospel
    ]

    genre_label_map = {
        "latin": "Latin/Salsa",
        "jazz": "Jazz",
        "blues": "Blues/R&B",
        "musicals": "Show Tunes",
        "oldies": "Oldies",
        "classicl": "Classical/Chamber",
        "reggae": "Reggae",
        "bigband": "Swing/Big Band",
        "newage": "New Age/Space music",
        "opera": "Opera",
        "blugrass": "Bluegrass",
        "folk": "Folk",
        "moodeasy": "Pop/Easy listening",
        "conrock": "Contemporary rock",
        "rap": "Rap",
        "hvymetal": "Heavy metal",
        "country": "Country/Western",
        "gospel": "Gospel"
    }

    # Check which expected columns are actually present
    available_cols = [c for c in paper_order if c in df.columns]
    if not available_cols:
        raise ValueError("None of the expected music-genre columns are present in the data.")

    # Prepare list to collect per-genre summaries
    rows = []

    # For robustness, treat any non-NA values outside {1,2,3,4,5}
    # as non-substantive responses (we cannot separate DK vs NA by code
    # without explicit metadata, so we count them together).
    for col in paper_order:
        if col not in df.columns:
            # Skip missing columns silently; table will only include available genres
            continue

        series = df[col]

        # Total N for this genre = number of rows in the dataset
        # (including those who did not answer this item; they will
        # appear in DK/NA or be counted as missing depending on coding).
        N_total = len(series)

        # Substantive categories 1–5
        counts = {}
        for k in [1, 2, 3, 4, 5]:
            counts[k] = series.eq(k).sum()

        # Non-substantive: anything non-NA that is not 1–5
        # We cannot distinguish "Don't know" from "No answer" without
        # explicit separate codes, so we put them all into "Don't know much
        # about it" and set "No answer" to zero to preserve structure.
        # This matches the descriptive spirit while reflecting the data we have.
        mask_non_na = series.notna()
        mask_substantive = series.isin([1, 2, 3, 4, 5])
        dk_na_count = (mask_non_na & ~mask_substantive).sum()

        dont_know_count = dk_na_count
        no_answer_count = 0

        # Mean rating (1–5) over substantive responses only
        valid_ratings = series.where(mask_substantive)
        mean_rating = valid_ratings.mean()

        row = {
            "Genre": genre_label_map.get(col, col),
            "N": N_total,
            "Like very much (1)": int(counts[1]),
            "Like it (2)": int(counts[2]),
            "Mixed feelings (3)": int(counts[3]),
            "Dislike (4)": int(counts[4]),
            "Dislike very much (5)": int(counts[5]),
            "Don't know much about it": int(dont_know_count),
            "No answer": int(no_answer_count),
            "Mean rating (1–5)": float(mean_rating) if pd.notna(mean_rating) else float("nan"),
        }
        rows.append(row)

    table3 = pd.DataFrame(rows)

    # Ensure rows are in the paper's order (filtered to those actually present)
    genre_order_labels = [genre_label_map[g] for g in paper_order if g in df.columns]
    table3 = table3.set_index("Genre").loc[genre_order_labels].reset_index()

    return table3