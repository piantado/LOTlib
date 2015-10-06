from Model import *

ALPHA = 0.95
TOP_N = 10
STEPS = 100000
OUT_HYPOTHESES = 'hypotheses-2015Oct6.pkl'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from Model.Grammar import lot_grammar as grammar

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define the running function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import LOTlib
from LOTlib import break_ctrlc
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.TopN import TopN

def myrun(observed_set):

    if LOTlib.SIG_INTERRUPTED:
        return set()

    h0 = NumberGameHypothesis(grammar=grammar)

    data = [FunctionData(input=[], output=observed_set, alpha=ALPHA)]

    tn = TopN(N=TOP_N)
    for h in break_ctrlc(MHSampler(h0, data, steps=STEPS)):
        tn.add(h)

    return set(tn.get_all())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define the running function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    # Load the concepts from the human data
    from Data import load_human_data

    human_nyes, _ = load_human_data()
    print "# Loaded human data"

    observed_sets = set([ k[0] for k in human_nyes.keys() ])

    # And now map
    from LOTlib.MPI.MPI_map import MPI_unorderedmap

    hypotheses = set()
    for s in MPI_unorderedmap(myrun, [ [s] for s in observed_sets ]):
        hypotheses.update(s)

    import pickle
    with open(OUT_HYPOTHESES, 'w') as f:
        pickle.dump(hypotheses, f)

    print "#", hypotheses
