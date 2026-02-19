import streamlit as st
import pandas as pd

from spotify_cluster.data import load_spotify_csv, DEFAULT_FEATURE_COLS

from spotify_cluster.model import (
    fit_kmeans_and_project,
    compute_elbow_inertia,
    cluster_profiles,
    descriptive_cluster_label,
    top_examples,
)

from spotify_cluster.ui import (
    plot_elbow,
    plot_pca_3d,
    plot_feature_scatter,
    heatmap_cluster_zscores,
)

from spotify_cluster.previews import fetch_itunes_preview_url

DATA_PATH = "data/spotify.csv"

st.set_page_config(page_title="Spotify Audio Clustering", layout="wide")
st.title("🎵 Spotify Audio Space Explorer")
st.caption("K-Means clustering of Spotify audio features (unsupervised).")

# ---- Session state ----
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "k" not in st.session_state:
    st.session_state.k = 4

# ---- Load data (cached) ----
@st.cache_data
def get_df():
    return load_spotify_csv(DATA_PATH)

df = get_df()
feature_cols = DEFAULT_FEATURE_COLS

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Controls")
    k = st.slider("Clusters (K)", 2, 10, int(st.session_state.k))
    show_3d = st.toggle("3D PCA", value=True)
    st.markdown("---")
    if st.button("Analyze", type="primary"):
        st.session_state.analyzed = True
        st.session_state.k = k  # persist chosen K

# If user changes K after analyzing, keep analyzed True but recompute
if st.session_state.analyzed and k != st.session_state.k:
    st.session_state.k = k

# ---- Cached compute keyed only by K ----
@st.cache_data
def compute_for_k(k_val: int):
    elbow = compute_elbow_inertia(df, feature_cols, 2, 10)
    artifacts = fit_kmeans_and_project(df, feature_cols, k_val)
    profiles = cluster_profiles(artifacts.df, feature_cols)
    return elbow, artifacts, profiles

if not st.session_state.analyzed:
    st.info("Pick **K** and click **Analyze**.")
    st.stop()

elbow_df, artifacts, profiles_df = compute_for_k(int(st.session_state.k))

dfc = artifacts.df

# ---- Layout tabs ----
tab_overview, tab_visual, tab_clusters = st.tabs(["Overview", "Visualize", "Cluster Details"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Songs", f"{len(dfc):,}")
    c2.metric("K", int(st.session_state.k))
    c3.metric("Silhouette", f"{artifacts.silhouette:.3f}")
    c4.metric("PCA variance (2D)", f"{(artifacts.pca2_var.sum()*100):.1f}%")

    st.plotly_chart(plot_elbow(elbow_df), use_container_width=True)

    st.markdown("### Cluster cards")
    means = dfc.groupby("cluster")[feature_cols].mean()
    counts = dfc["cluster"].value_counts().sort_index()

    cols = st.columns(2)
    for i, cl in enumerate(sorted(dfc["cluster"].unique())):
        col = cols[i % 2]
        with col:
            with st.container(border=True):
                label = descriptive_cluster_label(cl, dfc, feature_cols)
                st.markdown(f"#### Cluster {cl} — {label}")
                st.caption(f"{int(counts.loc[cl])} songs")
                ex = top_examples(dfc, cl, n=4)
                st.write("**Examples (tap to preview):**")
                for title, artist in ex:
                    st.write(f"• {title} — {artist}")
                    preview = fetch_itunes_preview_url(title, artist)
                    if preview:
                        st.audio(preview)
                    else:
                        st.caption("No preview available.")

with tab_visual:
    left, right = st.columns([2, 1])

    with left:
        if show_3d:
            st.plotly_chart(plot_pca_3d(dfc), use_container_width=True)
            st.caption(f"3D PCA variance explained: {(artifacts.pca3_var.sum()*100):.1f}%")
        else:
            # 2D PCA fallback
            import plotly.express as px
            fig = px.scatter(
                dfc, x="pc1", y="pc2",
                color="cluster",
                hover_data=["song_title", "artist"],
                opacity=0.75,
                title="2D PCA",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Feature-space view")
        x_feat = st.selectbox("X feature", feature_cols, index=feature_cols.index("energy"))
        y_feat = st.selectbox("Y feature", feature_cols, index=feature_cols.index("danceability"))
        st.plotly_chart(plot_feature_scatter(dfc, x_feat, y_feat), use_container_width=True)

    with right:
        st.markdown("### Find a song")
        query = st.text_input("Search title", value="")
        hits = dfc[dfc["song_title"].str.contains(query, case=False, na=False)].head(15) if query.strip() else dfc.head(15)

        pick = st.selectbox(
            "Pick a song",
            options=list(hits.index),
            format_func=lambda idx: f"{dfc.loc[idx,'song_title']} — {dfc.loc[idx,'artist']}"
        )

        row = dfc.loc[pick]
        st.write(f"**Selected:** {row['song_title']} — {row['artist']}")
        st.write(f"**Cluster:** {int(row['cluster'])}")

        # Nearest neighbors in scaled space
        distances, indices = artifacts.nn.kneighbors([artifacts.X_scaled[pick]])
        neighbors = [int(i) for i in indices[0] if int(i) != int(pick)][:8]
        st.markdown("**Similar songs (nearest in feature space):**")
        for i in neighbors:
            r = dfc.loc[i]
            st.write(f"• {r['song_title']} — {r['artist']} (cluster {int(r['cluster'])})")

with tab_clusters:
    st.plotly_chart(heatmap_cluster_zscores(dfc, feature_cols), use_container_width=True)

    st.markdown("### Cluster profiles")
    st.dataframe(profiles_df, use_container_width=True)

    sel = st.selectbox("Inspect cluster", sorted(dfc["cluster"].unique()))
    st.dataframe(
        dfc[dfc["cluster"] == sel][["song_title", "artist"] + feature_cols].head(25),
        use_container_width=True,
    )

    st.download_button(
        "Download clustered CSV",
        data=dfc.to_csv(index=False).encode("utf-8"),
        file_name="spotify_clustered.csv",
        mime="text/csv",
    )
