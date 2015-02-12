"""
Play around with learning multiple concepts from a single primitive like NAND.
Useful for learning new chunks of concepts, a la Dechter, Malmaud, Adams & Tenenbaum (2013)

This runs inference on each concept in TARGET_CONCEPTS (defined in Shared) and saves the top
100 hypotheses from each concept into all_hypotheses, which is written to pickle file OUTFILE

You can then run Adapt.py, which reads OUTFILE and calls OptimalGrammarAdaptation.print_subtree_adaptations
to show the best subtrees to define for minimizing KL between the prior and posterior

"""
import pickle

from LOTlib import lot_iter
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from Model import *


NDATA = 50 # How many total data points?
NSTEPS = 10000
BEST_N = 100 # How many from each hypothesis to store
OUTFILE = "hypotheses.pkl"

# Where we keep track of all hypotheses (across concepts)
all_hypotheses = FiniteBestSet()

if __name__ == "__main__":
    # Now loop over each target concept and get a set of hypotheses
    for i, f in enumerate(TARGET_CONCEPTS):

        # Set up the hypothesis
        h0 = LOTHypothesis(grammar, start='START', args=['x'])

        # Set up some data
        data = generate_data(NDATA, f)

        # Now run some MCMC
        fs = FiniteBestSet(N=BEST_N, key="posterior_score")
        fs.add(lot_iter(MHSampler(h0, data, steps=NSTEPS, trace=False)))

        all_hypotheses.merge(fs)

    pickle.dump(all_hypotheses, open(OUTFILE, 'w'))
