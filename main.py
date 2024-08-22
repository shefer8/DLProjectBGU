import logging
import os
import pickle
import random
import time

import tensorflow as tf
import torch
# from pypots.clustering.vader import VaDER
from model_modified import VaDER
from analytics import *
from config import *  # Import global parameters

pd.reset_option('display.max_columns')

CURRENT_TIME = time.strftime("%Y%m%d-%H%M%S")

# Configure logging
logging.basicConfig(filename=f'training_log_{CURRENT_TIME}.log', level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')

# logging all config global parameters values
print_variables_from_module('config')
# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example seed
set_seed(42)

# Check GPU availability and set memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("GPUs are available and memory growth is set.")
    except RuntimeError as e:
        logging.error(f"Error setting memory growth: {e}")
else:
    logging.info("No GPUs found, using CPU.")

# Load and preprocess data
file_path = 'ToStim.csv'  # Replace with the actual path to your CSV file
data = load_and_preprocess_data(file_path)

# for ran's data
# data = pd.read_csv(file_path)
# data = data[['Medical_Record', 'DayInCycle', 'Estrogen', 'LH', 'Progesterone']]
# print(data.head(10), data.shape)

print_data_statistics(data)

# Pivot data
pivoted_data = pivot_data(data)

# Check for any NaN values in the pivoted data
logging.info(f"Number of NaN values in the pivoted data: {pivoted_data.isna().sum().sum()}")

# Convert the pivoted DataFrame to a numpy array with the required shape
data_array = pivoted_data.values.reshape(pivoted_data.shape[0], MAX_DAYINCYCLE, N_FEATURES)
logging.info(f"Shape of data_array: {data_array.shape}")

# Create a mask array where 1 indicates missing values and 0 indicates observed values
mask = np.isnan(data_array).astype(int)

# Initialize missing values
data_array = initialize_missing_values(data_array, USE_HEURISTIC_MEAN)

# Check for any NaN values in the initialized data
check_for_nan(data_array)

# Scale the data (with the intialized values)
# Normalize the data
if NORMALIZE_DATA:
    data_array = normalize_data(data_array)

# Prepare the data in the expected format
dataset = {'X': data_array, 'missing_mask': mask}

# Use GPU device if available
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

# Constructing features means and cov for the GMM, change the parameter to False to use gmm's approximated stats
gmm_means, gmm_covariances = means_covs_for_gmm(USE_PRIOR_MEAN_AND_STD)

# find the .pkl file in the current directory, which is named like '.pkl'
def find_pkl_file():
    for file in os.listdir():
        if file.endswith('.pkl'):
            return file
    return None

model_path = find_pkl_file()
if model_path is not None:
    # Load the model if it exists
    with open(model_path, 'rb') as file:
        vader = pickle.load(file)
    logging.info("Model loaded from disk.")
else:
    # Initialize and fit the VaDER model if it doesn't exist
    with tf.device(device):
        vader = VaDER(n_steps=N_STEPS, n_features=N_FEATURES, n_clusters=N_CLUSTERS,
                      rnn_hidden_size=RNN_HIDDEN_SIZE, d_mu_stddev=D_MU_STDDEV,
                      epochs=EPOCHS, gmm_means = gmm_means, gmm_covs=gmm_covariances) #,saving_path = f'run_of_{CURRENT_TIME}'

        # Fit the VaDER model
        vader.fit(dataset)

        # Save the fitted model to disk
        with open(f'vader_model_{CURRENT_TIME}.pkl', 'wb') as file:
            pickle.dump(vader, file)
        logging.info("Model saved to disk.")


# Predict the imputed data
try:
    imputed_data = vader.predict(dataset, return_latent_vars = True)
except AssertionError as e:
    logging.error(f"AssertionError during prediction: {e}")
    print(f"AssertionError during prediction: {e}")

# Inspect the keys of the returned dictionary
print("Keys in imputed_data:", imputed_data.keys())

# Extract and print cluster assignments
if 'clustering' in imputed_data:
    cluster_assignments = imputed_data['clustering']
    print("Cluster assignments (first 5):", cluster_assignments[:5])

    # Create a DataFrame with unique medical records as and their cluster assignments
    unique_records = pd.DataFrame({
        'Medical_Record': pivoted_data.index,
        'Cluster': cluster_assignments
    })

    # Save the unique_records DataFrame with cluster assignments to a CSV file
    unique_records.to_csv(f'unique_records_with_clusters_{CURRENT_TIME}.csv', index=False)
    print(f"Unique records with clusters saved to 'unique_records_with_clusters{CURRENT_TIME}.csv'")

    # Calculate cluster sizes
    cluster_sizes = unique_records['Cluster'].value_counts()
    print("Cluster sizes:\n", cluster_sizes)

else:
    print("No clustering key found in imputed_data")

latent_vars = imputed_data['latent_vars']
print('imputations = ', latent_vars['imputation_latent'])
imputation_data_df = latent_vars['imputation_latent']
latent_vars_shape = imputation_data_df.shape
print(f"""shape of latent vars = {latent_vars_shape}""")
# Print the shapes to debug
print(f"Shape of pivoted_data: {pivoted_data.shape}")

# Create the index
first_dim_index = data['Medical_Record'].unique()
second_dim_index = np.arange(1, latent_vars_shape[1]+1)

# Create a MultiIndex
index = pd.MultiIndex.from_product([first_dim_index, second_dim_index], names=['Medical_Record', 'day_in_cycle'])

# Flatten the array
flattened_array = imputation_data_df.reshape(-1, N_FEATURES)

# Create the DataFrame
columns = ['estro', 'prog', 'lh'] if WITH_LH else ['estro', 'prog']
imputation_data_df = pd.DataFrame(flattened_array, index=index, columns=columns)
# Ensure that the MultiIndex aligns by reindexing 'unique_records' to match 'imputation_data_df'
# Create a MultiIndex for unique_records to match imputation_data_df
unique_records_multiindex = unique_records.set_index('Medical_Record').reindex(imputation_data_df.index.levels[0])

# Now assign the cluster based on the reindexed unique_records
imputation_data_df['cluster'] = unique_records_multiindex['Cluster'].values.repeat(MAX_DAYINCYCLE)

# Save the DataFrame to a CSV file
imputation_data_df.to_csv(f'imputation_data_df_{CURRENT_TIME}.csv')

# Plot data (optional)
# plot_data(pivoted_data)

logging.info("Training completed and data saved.")
