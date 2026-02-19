from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClusterArtifacts:
    df: pd.DataFrame                # df with cluster + pcs
    X_scaled: np.ndarray            # scaled feature matrix
    silhouette: float
    pca2_var: np.ndarray            # explained variance ratios (2D)
    pca3_var: np.ndarray            # explained variance ratios (3D)
    nn: NearestNeighbors            # fitted neighbors model


def fit_kmeans_and_project(
    df_in: pd.DataFrame,
    feature_cols: List[str],
    k: int,
    random_state: int = 42
) -> ClusterArtifacts:
    X = df_in[feature_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X_scaled)

    sil = float(silhouette_score(X_scaled, labels))

    pca2 = PCA(n_components=2, random_state=random_state)
    pcs2 = pca2.fit_transform(X_scaled)

    pca3 = PCA(n_components=3, random_state=random_state)
    pcs3 = pca3.fit_transform(X_scaled)

    out = df_in.copy()
    out["cluster"] = labels
    out["pc1"] = pcs2[:, 0]
    out["pc2"] = pcs2[:, 1]
    out["pc3"] = pcs3[:, 2]

    nn = NearestNeighbors(n_neighbors=min(11, len(out)))
    nn.fit(X_scaled)

    return ClusterArtifacts(
        df=out,
        X_scaled=X_scaled,
        silhouette=sil,
        pca2_var=pca2.explained_variance_ratio_,
        pca3_var=pca3.explained_variance_ratio_,
        nn=nn,
    )


def compute_elbow_inertia(
    df_in: pd.DataFrame,
    feature_cols: List[str],
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    X = df_in[feature_cols].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)

    ks = list(range(k_min, k_max + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))

    return pd.DataFrame({"K": ks, "Inertia": inertias})


def cluster_profiles(df_clustered: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    prof = df_clustered.groupby("cluster")[feature_cols].mean().round(3)
    prof["n_songs"] = df_clustered["cluster"].value_counts().sort_index().values
    return prof.reset_index()


def descriptive_cluster_label(
    cluster_id: int,
    df_clustered: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 2
) -> str:
    """
    Generate a neutral description based on feature z-scores.
    """

    global_means = df_clustered[feature_cols].mean()
    global_stds = df_clustered[feature_cols].std(ddof=0)

    cluster_means = (
        df_clustered[df_clustered["cluster"] == cluster_id][feature_cols]
        .mean()
    )

    # Compute z-scores
    z_scores = (cluster_means - global_means) / global_stds

    # Sort by magnitude
    sorted_feats = z_scores.sort_values(ascending=False)

    high_feats = sorted_feats.head(top_n)
    low_feats = sorted_feats.tail(top_n)

    label_parts = []

    for feat, val in high_feats.items():
        if val > 0.5:  # meaningful deviation
            label_parts.append(f"High {feat}")

    for feat, val in low_feats.items():
        if val < -0.5:
            label_parts.append(f"Low {feat}")

    if not label_parts:
        return "No strong distinguishing features"

    return " · ".join(label_parts[:3])



def top_examples(df_clustered: pd.DataFrame, cluster_id: int, n: int = 5) -> List[Tuple[str, str]]:
    sub = df_clustered[df_clustered["cluster"] == cluster_id][["song_title", "artist"]].head(n)
    return list(sub.itertuples(index=False, name=None))
