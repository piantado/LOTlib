# -*- coding: utf-8 -*-

"""
	Use optimal adaptation code to adapt show possible adpatations to the NAND grammar

"""
from Shared import *
from LOTlib.Subtrees import *

NDATA = 100
N_SUBTREES_PER_NODE = 50
SUBTREE_P = 0.5 # when we generate a partial subtree, how likely are we to take each kid?

# Make the data(s)
datas = [generate_data(NDATA, f) for f in TARGET_CONCEPTS]

# Load hypotheses from previous run
hypotheses = pickle_load("hypotheses.pkl").get_all()
print "# Loaded hypotheses ", len(hypotheses)

# Clean out ones with 0 probability, or else KL computation in print_subtree_adaptations goes to hell
hypotheses = filter(lambda h: sum(h.compute_posterior(datas[0])) > -Infinity,  hypotheses)

## And evaluate each hypothesis on each data point
posteriors = map( lambda d: [ sum(h.compute_posterior(d)) for h in hypotheses], datas)
print "# Rescored hypotheses!"

## Generate a set of subtrees
subtrees = set()
for h in lot_iter(hypotheses):
	for x in h.value: # for each subtree
		for i in xrange(N_SUBTREES_PER_NODE):  #take subtree_multiplier random partial subtrees
			subtrees.add(   x.random_partial_subtree(p=SUBTREE_P)   )
print "# Generated", len(subtrees), "subtrees"

# If you want to see some:
#for h,p in zip(hypotheses, posteriors[1]):
	#print p, h

from LOTlib.sandbox.OptimalGrammarAdaptation import print_subtree_adaptations
## And call from OptimalGrammarAdaptation
print_subtree_adaptations(hypotheses, posteriors, subtrees)
