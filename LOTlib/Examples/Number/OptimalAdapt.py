# -*- coding: utf-8 -*-

"""
	Use optimal adaptation code to adapt show possible adpatations to the Number grammar

"""

from Shared import *
from LOTlib.sandbox.OptimalGrammarAdaptation import print_subtree_adaptations
	

allfs = pickle.load(open("runs/2014Feb10.pkl", 'r'))
hyps = allfs.get_all()
print "# Loaded hypotheses"

# Set up how much data we want
data = generate_data(200)
print "# Generated data!"

# And evaluate each hypothesis on it
for h in hyps: h.compute_posterior(data) # update everyone's posterior score for thsi amount of data
print "# Rescored hypotheses!"

# And call from OptimalGrammarAdaptation
print_subtree_adaptations(hyps)
