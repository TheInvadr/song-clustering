from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_elbow(elbow_df: pd.DataFrame):
    return px.line(elbow_df, x="K", y="Inertia", markers=True, title="Elbow method (inertia vs K)")


def plot_pca_3d(df_clustered: pd.DataFrame):
    return px.scatter_3d(
        df_clustered,
        x="pc1", y="pc2", z="pc3",
        color="cluster",
        hover_data=["song_title", "artist"],
        opacity=0.75,
        title="3D PCA (rotate/zoom)",
    )


def plot_feature_scatter(df_clustered: pd.DataFrame, x_feat: str, y_feat: str):
    return px.scatter(
        df_clustered,
        x=x_feat, y=y_feat,
        color="cluster",
        hover_data=["song_title", "artist"],
        opacity=0.75,
        title=f"{x_feat} vs {y_feat}",
    )


def heatmap_cluster_zscores(df_clustered: pd.DataFrame, feature_cols: List[str]):
    means = df_clustered.groupby("cluster")[feature_cols].mean()
    z = (means - means.mean()) / means.std(ddof=0)
    fig = px.imshow(z.round(2), aspect="auto", title="Cluster feature heatmap (z-scored)")
    return fig
