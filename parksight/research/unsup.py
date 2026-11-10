"""PCA and K-Means on each station's hourly ticket-timing profile.

Hypothesis (to test, not assume): stations mostly differ in WHEN they record
tickets, not how many. PCA on the normalised 24-hour profile is the test.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from parksight import config as C


def hourly_profile_matrix(tickets: pd.DataFrame) -> pd.DataFrame:
    """[police_station x hour 0..23], each row sums to 1 (a station's share of its own tickets by hour)."""
    counts = (tickets.assign(hour=tickets["created_ist"].dt.hour)
             .groupby(["police_station", "hour"]).size().unstack(fill_value=0))
    counts = counts.reindex(columns=range(24), fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


def run_pca(profile: pd.DataFrame):
    """scores: [police_station, PC1, PC2, ...]; explained_variance_ratio_: array over components."""
    pca = PCA(n_components=min(profile.shape), random_state=0)
    scores = pca.fit_transform(profile.values)   # PCA centres internally
    cols = [f"PC{i + 1}" for i in range(scores.shape[1])]
    out = pd.DataFrame(scores, index=profile.index, columns=cols).reset_index()
    return out, pca.explained_variance_ratio_, pca.components_


def run_kmeans(profile: pd.DataFrame, k_range=range(2, 7), seed: int = 0) -> dict:
    """silhouette per k (on the profile rows), the best k by silhouette, and its labels."""
    sil = {}
    labels_by_k = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(profile.values)
        sil[k] = silhouette_score(profile.values, labels)
        labels_by_k[k] = labels
    best_k = max(sil, key=sil.get)
    return dict(silhouette=sil, best_k=best_k,
               labels=pd.Series(labels_by_k[best_k], index=profile.index, name="cluster"))
