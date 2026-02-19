from __future__ import annotations

import pandas as pd

REQUIRED_COLS = [
    "acousticness", "danceability", "duration_ms", "energy",
    "instrumentalness", "key", "liveness", "loudness", "mode",
    "speechiness", "tempo", "time_signature", "valence",
    "song_title", "artist"
]

DEFAULT_FEATURE_COLS = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]

def load_spotify_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Drop rows with missing required numeric values or display fields
    keep_cols = list(set(DEFAULT_FEATURE_COLS + ["song_title", "artist", "key", "mode", "time_signature"]))
    df = df[keep_cols].copy()
    df = df.dropna().reset_index(drop=True)

    # Ensure types
    df["song_title"] = df["song_title"].astype(str)
    df["artist"] = df["artist"].astype(str)

    return df
