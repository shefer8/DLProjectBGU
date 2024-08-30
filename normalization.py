import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# Preprocessing and normalization functions
def normalize_data_3d(data):
    mean_global = np.mean(data)
    std_global = np.std(data)

    #data_3d_normalized = (data - mean_global) / std_global

    mean_channel = np.mean(data, axis=(0, 1))
    std_channel = np.std(data, axis=(0, 1))
    data_3d_channel_normalized = (data - mean_channel) / std_channel
    return data_3d_channel_normalized



# Evaluate clustering quality
def evaluate_clustering(normalized_array, n_clusters=3):
    new = normalized_array.reshape(-1, normalized_array.shape[-1])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(new)
    silhouette_avg = silhouette_score(new, cluster_labels)
    return silhouette_avg

