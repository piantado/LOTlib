# -*- coding: utf-8 -*-
"""
        A quick script to load some large data and re-run-evaluate it to generate a file readable by plot_learning_curve.R
"""

import pickle
from Shared import *

LARGE_DATA_SIZE = 1000

#now evaluate on different amounts of data too:
huge_data = generate_data(LARGE_DATA_SIZE)
print "# Generated data!"

allfs = pickle.load(open("runs/2014Feb10_small.pkl")) # for now, use data from the run on February 10
print "# Loaded!"

# save this with a huge data set -- eval with average ll
H = allfs.get_all()

[h.compute_posterior(huge_data) for h in H]

# show the *average* ll for each hypothesis
for h in H:
    if h.prior > float("-inf"):
        print h.prior, h.likelihood/float(LARGE_DATA_SIZE), q(get_knower_pattern(h)),  q(h)
