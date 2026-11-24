from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "seeds": {"label": "Graines de blé — UCI Seeds", "file": "wheat_seeds_dataset.csv", "entity": "graine", "source": "https://archive.ics.uci.edu/dataset/236/seeds", "description": "210 graines décrites par sept mesures géométriques. Le CSV fourni ne contient pas les étiquettes de variété.", "default_clusters": 3, "default_method": "CAH"},
    "heart": {"label": "Heart Disease — patients", "file": "heart_disease.csv", "entity": "patient", "source": "https://archive.ics.uci.edu/dataset/45/heart+disease", "description": "Variables cliniques numériques. Le CSV fourni ne contient pas la variable de diagnostic.", "default_clusters": 5, "default_method": "K-means"},
    "bank": {"label": "Bank Marketing — clients", "file": "bank_marketing.csv", "entity": "client", "source": "https://archive.ics.uci.edu/dataset/222/bank+marketing", "description": "Sous-ensemble de sept variables numériques du jeu Bank Marketing.", "default_clusters": 5, "default_method": "K-means"},
}

st.set_page_config(page_title="Segmentation — trois jeux de données", page_icon="🧩", layout="wide")
st.title("🧩 Segmentation de données")
st.write("Explorez les groupes trouvés par K-means ou par classification hiérarchique (CAH).")
dataset_key = st.sidebar.selectbox("Jeu de données", list(DATASETS), format_func=lambda k: DATASETS[k]["label"])
meta = DATASETS[dataset_key]
if dataset_key == "seeds":
    st.info("Pour le jeu des graines, l’application s’ouvre sur la CAH, comme demandé dans les consignes. Le fichier ne contient pas les variétés réelles.")
elif dataset_key == "heart":
    st.info("Dans ton notebook, le score de silhouette était 0,302 pour K-means et 0,258 pour la CAH. L’application propose donc K-means par défaut.")
else:
    st.info("Dans ton notebook, le score de silhouette était 0,561 pour K-means et 0,497 pour la CAH. L’application propose donc K-means par défaut.")

@st.cache_data
def load_dataset(filename):
    frame = pd.read_csv(ROOT / filename)
    frame = frame.replace(["?", "NA", "N/A", ""], np.nan)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(how="all").reset_index(drop=True)
    frame = frame.select_dtypes(include="number").dropna(axis=1, how="all")
    frame = frame.loc[:, frame.nunique(dropna=True) > 1]
    return frame.fillna(frame.median(numeric_only=True))

raw = load_dataset(meta["file"])
if raw.shape[1] < 2 or len(raw) < 3:
    st.error("Il faut au moins deux colonnes numériques et trois lignes pour lancer la segmentation.")
    st.stop()
st.caption(meta["description"])
st.markdown(f"Source : [UCI Machine Learning Repository]({meta['source']})")

with st.expander("Aperçu des données et nettoyage"):
    st.write(f"Dimensions après nettoyage : **{len(raw):,} lignes × {raw.shape[1]} variables**")
    st.write("Les marqueurs de valeurs manquantes sont convertis, les lignes entièrement vides et colonnes inutilisables sont retirées, puis les autres valeurs manquantes sont imputées par la médiane.")
    st.dataframe(raw.head(10), width="stretch")

method_options = ["K-means", "CAH"]
default_index = method_options.index(meta["default_method"])
method = st.sidebar.radio("Méthode", method_options, index=default_index)
max_rows = min(len(raw), 1500) if method == "CAH" else len(raw)
sample_rows = st.sidebar.slider("Lignes à analyser", min_value=min(100, max_rows), max_value=max_rows, value=min(max_rows, 1500), step=1)
max_clusters = min(10, sample_rows - 1)
n_clusters = st.sidebar.slider("Nombre de groupes", min_value=2, max_value=max_clusters, value=min(meta["default_clusters"], max_clusters))
@st.cache_data
def fit_segmentation(frame, algorithm, groups, row_count):
    sample = frame.sample(n=row_count, random_state=42) if row_count < len(frame) else frame.copy()
    x = normalize(sample.to_numpy(dtype=float))
    if algorithm == "K-means":
        model = KMeans(n_clusters=groups, init="random", max_iter=500, n_init=10, random_state=42)
        labels = model.fit_predict(x)
        centers = model.cluster_centers_
    else:
        model = AgglomerativeClustering(n_clusters=groups, linkage="average")
        labels = model.fit_predict(x)
        centers = np.vstack([x[labels == group].mean(axis=0) for group in range(groups)])
    score = silhouette_score(x, labels, sample_size=min(1000, len(x)), random_state=42)
    pca = PCA(n_components=2, random_state=42)
    projection = pca.fit_transform(x)
    return sample, x, labels, centers, score, projection, pca.explained_variance_ratio_.sum()

sample, x, labels, centers, score, projection, explained_variance = fit_segmentation(raw, method, n_clusters, sample_rows)
left, right = st.columns([1, 2])
with left:
    st.subheader("Résultats")
    st.metric("Score de silhouette", f"{score:.3f}" if np.isfinite(score) else "Indisponible")
    st.caption("Un score plus proche de 1 indique des groupes mieux séparés; interprétez-le avec prudence.")
    st.dataframe(pd.Series(labels, name="Groupe").value_counts().sort_index().rename("Observations").to_frame(), width="stretch")
with right:
    st.subheader("Visualisation des groupes")
    chart_data = pd.DataFrame({"Composante 1": projection[:, 0], "Composante 2": projection[:, 1], "Groupe": labels.astype(str)})
    st.plotly_chart(px.scatter(chart_data, x="Composante 1", y="Composante 2", color="Groupe", opacity=0.75), width="stretch")
    st.caption(f"Les deux axes représentent {explained_variance:.1%} de la variance des données normalisées.")

st.subheader("Profil moyen des groupes")
profile = sample.assign(Groupe=labels).groupby("Groupe").mean(numeric_only=True).round(2)
st.dataframe(profile, width="stretch")
st.subheader(f"Classer un nouveau {meta['entity']}")
st.caption("Pour la CAH, le groupe proposé est celui dont le centre moyen est le plus proche; la CAH ne prédit pas directement de nouvelles lignes.")
with st.form(f"new_{dataset_key}"):
    values = {}
    columns = st.columns(3)
    for index, feature in enumerate(sample.columns):
        series = raw[feature].dropna()
        values[feature] = columns[index % 3].number_input(feature, value=float(series.median()), key=f"{dataset_key}_{feature}")
    submitted = st.form_submit_button("Attribuer un groupe")
if submitted:
    row = pd.DataFrame([values], columns=sample.columns)
    new_x = normalize(row.to_numpy(dtype=float))
    group = int(np.argmin(((centers - new_x[0]) ** 2).sum(axis=1)))
    st.success(f"Ce {meta['entity']} est attribué au groupe **{group}**.")

result = sample.copy()
result["Groupe"] = labels
st.download_button("Télécharger les lignes segmentées (CSV)", result.to_csv(index=False).encode("utf-8"), file_name=f"segmentation_{dataset_key}.csv", mime="text/csv")
