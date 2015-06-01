"""

Run inference on each target concept and save the output

"""
import pickle
from LOTlib import break_ctrlc
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from Model import *
from TargetConcepts import TargetConcepts

NDATA = 20 # How many data points for each function?
NSTEPS = 100000
BEST_N = 500 # How many from each hypothesis to store

# Where we keep track of all hypotheses (across concepts)
all_hypotheses = FiniteBestSet()

if __name__ == "__main__":
    # Now loop over each target concept and get a set of hypotheses
    for i, f in enumerate(TargetConcepts):

        # Set up the hypothesis
        h0 = make_hypothesis()

        # Set up some data
        data = make_data(NDATA, f)

        # Now run some MCMC
        fs = FiniteBestSet(N=BEST_N, key="posterior_score")
        fs.add(break_ctrlc(MHSampler(h0, data, steps=NSTEPS, trace=False)))

        all_hypotheses.merge(fs)

    pickle.dump(all_hypotheses, open("hypotheses.pkl", 'w'))
