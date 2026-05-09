import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans

st.set_page_config(
    page_title="EconoSpatia Intelligence Dashboard",
    layout="wide"
)

st.title("EconoSpatia Intelligence Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    df = pd.read_csv(
        "GAMMAFEST KICAU MANIA - Sheet1.csv"
    )

    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Laju Pertumbuhan Produk Domestik Regional Bruto  Atas Dasar Harga Konstan 2010 Menurut Provinsi (persen)": "Pertumbuhan",
        "[Metode Baru] Pengeluaran Perkapita Disesuaikan": "Pengeluaran",
        "Gini_Ratio": "Gini"
    })

    df["Provinsi"] = df["Provinsi"].str.upper()

    num_cols = [
        "Tahun",
        "Stunting",
        "Pertumbuhan",
        "Pengeluaran",
        "TPT",
        "Kemiskinan",
        "Gini",
        "IPM"
    ]

    for c in num_cols:

        if c in df.columns:

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

    df = df.fillna(df.mean(numeric_only=True))

    return df

df = load_data()

# =========================
# LOAD GEOJSON
# =========================
@st.cache_data
def load_geo():

    url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"

    indo = gpd.read_file(url)

    indo["Propinsi"] = indo["Propinsi"].str.upper()

    return indo

indo = load_geo()

# =========================
# SIDEBAR MENU
# =========================
menu = st.sidebar.radio(
    "Menu Analisis",
    [
        "Peta Stunting",
        "GWPR",
        "DTW Clustering"
    ]
)

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("Filter")

prov_list = sorted(df["Provinsi"].unique())

prov_pilih = st.sidebar.selectbox(
    "Pilih Provinsi",
    ["Semua"] + prov_list
)

tahun_list = sorted(df["Tahun"].dropna().unique())

tahun_pilih = st.sidebar.selectbox(
    "Pilih Tahun",
    tahun_list
)

# =========================================================
# 1. PETA STUNTING
# =========================================================
if menu == "Peta Stunting":

    st.subheader("Peta Stunting Indonesia")

    temp = df[df["Tahun"] == tahun_pilih]

    if prov_pilih != "Semua":

        temp = temp[
            temp["Provinsi"] == prov_pilih
        ]

    map_data = indo.merge(
        temp,
        left_on="Propinsi",
        right_on="Provinsi",
        how="left"
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    map_data.plot(
        column="Stunting",
        cmap="Reds",
        legend=True,
        ax=ax,
        edgecolor="black"
    )

    ax.set_title(
        f"Peta Stunting Indonesia {tahun_pilih}"
    )

    ax.axis("off")

    st.pyplot(fig)

    st.dataframe(temp)

# =========================================================
# 2. GWPR
# =========================================================
elif menu == "GWPR":

    st.subheader(
        "Geographically Weighted Panel Regression"
    )

    gwpr = pd.read_csv(
        "gwr_result.csv"
    )

    gwpr.columns = gwpr.columns.str.strip()

    gwpr["Provinsi"] = gwpr["Provinsi"].str.upper()

    # =========================
    # FILTER
    # =========================
    if prov_pilih != "Semua":

        gwpr = gwpr[
            gwpr["Provinsi"] == prov_pilih
        ]

    # =========================
    # DATAFRAME
    # =========================
    st.write("### Hasil GWPR")

    st.dataframe(gwpr)

    # =========================
    # TOP LOCAL R2
    # =========================
    if "LocalR2" in gwpr.columns:

        st.write("### Top Local R²")

        top = gwpr.sort_values(
            "LocalR2",
            ascending=False
        ).head(10)

        st.bar_chart(
            top.set_index("Provinsi")["LocalR2"]
        )

    # =========================
    # PETA LOCAL R2
    # =========================
    if "LocalR2" in gwpr.columns:

        st.write("### Peta Local R²")

        map_data = indo.merge(
            gwpr,
            left_on="Propinsi",
            right_on="Provinsi",
            how="left"
        )

        fig, ax = plt.subplots(figsize=(14, 8))

        map_data.plot(
            column="LocalR2",
            cmap="YlGn",
            legend=True,
            ax=ax,
            edgecolor="black"
        )

        ax.set_title("Peta Local R² GWPR")

        ax.axis("off")

        st.pyplot(fig)

    # =========================
    # INTERPRETASI
    # =========================
    if "Growth" in gwpr.columns:

        st.write("### Insight Kebijakan")

        for _, row in gwpr.iterrows():

            if row["Growth"] < 0:

                insight = (
                    "Pertumbuhan ekonomi efektif "
                    "menurunkan stunting"
                )

                rekom = (
                    "Perkuat sektor ekonomi lokal "
                    "dan penciptaan kerja"
                )

            else:

                insight = (
                    "Pertumbuhan ekonomi "
                    "belum inklusif"
                )

                rekom = (
                    "Fokus intervensi kesehatan "
                    "dan redistribusi pendapatan"
                )

            st.markdown(f"""
            ### {row['Provinsi']}

            - Local R² : {row['LocalR2']:.3f}
            - Insight : {insight}
            - Rekomendasi : {rekom}
            """)

# =========================================================
# 3. DTW CLUSTERING
# =========================================================
elif menu == "DTW Clustering":

    st.subheader(
        "Dynamic Time Warping Clustering"
    )

    # =========================
    # SIDEBAR DTW
    # =========================
    st.sidebar.subheader("Pengaturan DTW")

    available_features = [
        c for c in [
            "Pertumbuhan",
            "Pengeluaran",
            "TPT",
            "Kemiskinan",
            "Gini",
            "IPM"
        ]
        if c in df.columns
    ]

    fitur = st.sidebar.multiselect(
        "Pilih Variabel",
        available_features,
        default=available_features[:3]
    )

    n_cluster = st.sidebar.slider(
        "Jumlah Cluster",
        min_value=2,
        max_value=6,
        value=3
    )

    # =========================
    # VALIDASI
    # =========================
    if len(fitur) == 0:

        st.warning(
            "Pilih minimal satu variabel"
        )

        st.stop()

    # =========================
    # TIME SERIES
    # =========================
    pivot_data = {}

    for fitur_col in fitur:

        pivot = df.pivot_table(
            index="Provinsi",
            columns="Tahun",
            values=fitur_col
        )

        pivot = pivot.sort_index(axis=1)

        pivot_data[fitur_col] = pivot

    provinsi_valid = (
        pivot_data[fitur[0]]
        .index
        .tolist()
    )

    X = []

    for prov in provinsi_valid:

        series_list = []

        valid = True

        for fitur_col in fitur:

            vals = (
                pivot_data[fitur_col]
                .loc[prov]
                .values
            )

            if np.isnan(vals).any():

                valid = False

                break

            series_list.append(vals)

        if valid:

            arr = np.array(series_list).T

            X.append(arr)

    X = np.stack(X).astype(np.float64)


# =========================
# =========================
# SCALING
# =========================
scaler = TimeSeriesScalerMeanVariance()

X_scaled = scaler.fit_transform(X)

# =========================
# MODEL DTW
# =========================
model = TimeSeriesKMeans(
    n_clusters=n_cluster,
    metric="dtw",
    random_state=42
)

cluster = model.fit_predict(X_scaled)

hasil = pd.DataFrame({

        "Provinsi": provinsi_valid,

        "Cluster": cluster

    })

    # =========================
    # FILTER
    # =========================
    if prov_pilih != "Semua":

        hasil = hasil[
            hasil["Provinsi"] == prov_pilih
        ]

    # =========================
    # DATAFRAME
    # =========================
    st.write("### Hasil Clustering")

    st.dataframe(hasil)

    # =========================
    # DISTRIBUSI
    # =========================
    st.write("### Distribusi Cluster")

    distribusi = (
        hasil["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(distribusi)

    # =========================
    # PETA CLUSTER
    # =========================
    st.write("### Peta Cluster DTW")

    map_data = indo.merge(
        hasil,
        left_on="Propinsi",
        right_on="Provinsi",
        how="left"
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    map_data.plot(
        column="Cluster",
        cmap="Set2",
        legend=True,
        ax=ax,
        edgecolor="black"
    )

    ax.set_title(
        "Cluster DTW Provinsi Indonesia"
    )

    ax.axis("off")

    st.pyplot(fig)

    # =========================
    # INTERPRETASI
    # =========================
    st.write("### Interpretasi Cluster")

    for c in sorted(
        hasil["Cluster"].unique()
    ):

        anggota = hasil[
            hasil["Cluster"] == c
        ]["Provinsi"].tolist()

        st.markdown(f"""
        ## Cluster {c}

        - Jumlah Provinsi:
        {len(anggota)}

        - Anggota:
        {", ".join(anggota)}
        """)
