"""
    This file contains defaults for running a stan model
"""

DATA_PICKLE = "stan_data.pkl"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the data file
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pickle

with open(DATA_PICKLE, 'r') as f:
    stan_data = pickle.load(f)

