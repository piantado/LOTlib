# -*- coding: utf-8 -*-

"""
Use optimal adaptation code to adapt show possible adpatations to the NAND grammar

"""
import pickle
from LOTlib.Miscellaneous import Infinity
from LOTlib.Subtrees import *
from Model import *
from TargetConcepts import TargetConcepts
from Model.Grammar import grammar

if __name__ == "__main__":

    NDATA = 10
    N_SUBTREES_PER_NODE = 10
    SUBTREE_P = 0.5 # when we generate a partial subtree, how likely are we to take each kid?

    # Make the data(s)
    datas = [make_data(NDATA, f) for f in TargetConcepts]

    # Load hypotheses from previous run
    hypotheses = pickle.load(open("hypotheses.pkl", 'r')).get_all()
    print "# Loaded hypotheses ", len(hypotheses)

    # Clean out ones with 0 probability, or else KL computation in print_subtree_adaptations goes to hell
    hypotheses = filter(lambda h: h.compute_posterior(datas[0]) > -Infinity,  hypotheses)

    ## And evaluate each hypothesis on each data point
    posteriors = map( lambda d: [ h.compute_posterior(d) for h in hypotheses], datas)
    print "# Rescored hypotheses!"

    ## Generate a set of subtrees
    subtrees = set()
    for h in break_ctrlc(hypotheses):
        for x in h.value: # for each subtree
            for i in xrange(N_SUBTREES_PER_NODE):  #take subtree_multiplier random partial subtrees
                subtrees.add(   x.random_partial_subtree(p=SUBTREE_P)   )
    print "# Generated", len(subtrees), "subtrees"

    # If you want to see some:
    #for h,p in zip(hypotheses, posteriors[1]):
        #print p, h

    ## And call from OptimalGrammarAdaptation
    from LOTlib.sandbox.OptimalGrammarAdaptation import print_subtree_adaptations
    print_subtree_adaptations(grammar, hypotheses, posteriors, subtrees)
