# config.py
MIN_DAYINCYCLE = 1
MAX_DAYINCYCLE = 6

# configurations for using priors
USE_PRIOR_MEAN_AND_STD = False
USE_HEURISTIC_MEAN = False # if false, then use prior mean to fill missing values before the model

# normalize data
NORMALIZATION_METHOD = ['z_score', 'Min_Max', 'feature_wise_normalize', 'baseline_correction', 'baseline_correction_and_z_score','baseline_correction_and_Min_Max', None][3] # so far, best results with z_score

# With or without LH
WITH_LH = False

# Vader model parameters:
N_STEPS = MAX_DAYINCYCLE  # Number of days in the cycle
N_FEATURES = 3 if WITH_LH else 2 # Number of features (Estrogen, LH, Prog
N_CLUSTERS = 3  # Example number of clusters
RNN_HIDDEN_SIZE = 32 # 64  # Size of the RNN hidden state
D_MU_STDDEV = 12 #MAX_DAYINCYCLE - MIN_DAYINCYCLE + 2  # Latent dimension # 12
EPOCHS = 10

