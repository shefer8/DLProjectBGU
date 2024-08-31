import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
import matplotlib.pyplot as plt
import scipy.stats as stats



def evaluate_normalization(normalized_array, original_array):
    if isinstance(normalized_array, dict):
        normalized_array = normalized_array.get('clustering', None)
        if normalized_array is None:
            print("No 'clustering' key found in imputed_data.")
            return

    if isinstance(original_array, dict):
        original_array = original_array.get('X', None)
        if original_array is None:
            print("No 'X' key found in original_data.")
            return

    # Flatten the arrays for comparison
    original_array_flat = original_array.flatten()
    normalized_array_flat = normalized_array.flatten()

    # Distribution Matching
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(original_array_flat, bins=30, color='blue', alpha=0.7)
    plt.title('Original Data Distribution')

    plt.subplot(1, 2, 2)
    plt.hist(normalized_array_flat, bins=30, color='green', alpha=0.7)
    plt.title('Normalized Data Distribution')

    plt.show()

    # Q-Q plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    stats.probplot(original_array_flat, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Original Data')

    plt.subplot(1, 2, 2)
    stats.probplot(normalized_array_flat, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Normalized Data')

    plt.show()

    # Comparing the variance between the original and the normalized
    original_variance = np.var(original_array_flat)
    normalized_variance = np.var(normalized_array_flat)
    print(f"Original Variance: {original_variance:.4f}")
    print(f"Normalized Variance: {normalized_variance:.4f}")


# Evaluate clustering quality
def evaluate_clustering(data, labels, ground_truth=None):
    """
    Evaluates the quality of clustering using multiple metrics and summarizes them in a table.

    Parameters:
    X (numpy array): The dataset used for clustering.
    labels (numpy array): The predicted labels from the clustering algorithm.
    ground_truth (numpy array, optional): The true labels for the data. If provided, ARI will be computed.

    Returns:
    pandas DataFrame: A summary table of clustering evaluation metrics.
    """
    if isinstance(data, dict):
        data = data.get('X', None)  # Adjust the key as necessary
        if data is None:
            raise ValueError("Data dictionary does not contain the expected key 'X'.")

        # Ensure the data is in the correct shape (n_samples, n_features)
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)  # Flatten the data if it's 3D

        # Ensure labels is a 1D array
    if labels.ndim != 1:
        raise ValueError(f"Expected labels to be a 1D array, but got shape {labels.shape}")

        # Check if the number of samples in data and labels match
    if data.shape[0] != labels.shape[0]:
        raise ValueError("The number of samples in data and labels do not match.")

        # Proceed with the evaluation
    metrics = {}

    # Silhouette Score
    silhouette_avg = silhouette_score(data, labels)
    metrics['Silhouette Score'] = silhouette_avg
    metrics['Silhouette Evaluation'] = "Good" if silhouette_avg > 0.5 else "Bad"

    # Davies-Bouldin Index
    davies_bouldin_avg = davies_bouldin_score(data, labels)
    metrics['Davies-Bouldin Index'] = davies_bouldin_avg
    metrics['Davies-Bouldin Evaluation'] = "Good" if davies_bouldin_avg < 1 else "Bad"

    # Calinski-Harabasz Index
    calinski_harabasz_avg = calinski_harabasz_score(data, labels)
    metrics['Calinski-Harabasz Index'] = calinski_harabasz_avg
    metrics['Calinski-Harabasz Evaluation'] = "Good" if calinski_harabasz_avg > 1000 else "Bad"

    # Adjusted Rand Index (ARI)
    if ground_truth is not None:
        ari_score = adjusted_rand_score(ground_truth, labels)
        metrics['Adjusted Rand Index (ARI)'] = ari_score
        metrics['ARI Evaluation'] = "Good" if ari_score > 0.5 else "Bad"

    # Convert metrics to a pandas DataFrame
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

    return metrics_df

