# -*- coding: utf-8 -*-

"""
	Use optimal adaptation code to adapt show possible adpatations to the NAND grammar

"""
from Shared import *
from LOTlib.sandbox.OptimalGrammarAdaptation import print_subtree_adaptations

NDATA = 150

# Make the data(s)
datas = [generate_data(NDATA, f) for f in TARGET_CONCEPTS]

# Load hypotheses from previous run
hypotheses = pickle_load("hypotheses.pkl").get_all()
print "# Loaded hypotheses"

# Clean out ones with 0 probability, or else KL computation in print_subtree_adaptations goes to hell
hypotheses = filter(lambda h: sum(h.compute_posterior(datas[0])) > -Infinity,  hypotheses)

## And evaluate each hypothesis on it
posteriors = map( lambda d: [ sum(h.compute_posterior(d)) for h in hypotheses], datas)
print "# Rescored hypotheses!"

# If you want to see some:
#for h,p in zip(hypotheses, posteriors[1]):
	#print p, h

## And call from OptimalGrammarAdaptation
print_subtree_adaptations(hypotheses, posteriors)
