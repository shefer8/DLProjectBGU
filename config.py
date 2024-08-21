# config.py
MIN_DAYINCYCLE = 1
MAX_DAYINCYCLE = 6

# configurations for using priors
USE_PRIOR_MEAN_AND_STD = False
USE_HEURISTIC_MEAN = True

# normalize data
NORMALIZE_DATA = False

# With or without LH
WITH_LH = False

# Vader model parameters:
N_STEPS = MAX_DAYINCYCLE  # Number of days in the cycle
N_FEATURES = 3 if WITH_LH else 2 # Number of features (Estrogen, LH, Prog
N_CLUSTERS = 3  # Example number of clusters
RNN_HIDDEN_SIZE = 64  # Size of the RNN hidden state
D_MU_STDDEV = 7  # Latent dimension
EPOCHS = 2