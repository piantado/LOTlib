"""
    Script to generate a file of all hypotheses by running incrementally on each amount of data
"""

import LOTlib
import itertools
from LOTlib import break_ctrlc

from Model.Data import concept2data
from Model.Hypothesis import SFLHypothesis, make_hypothesis


TOP=1
STEPS=10000
CHAINS=1
OUT_PATH="hypotheses.pkl"

from LOTlib.Inference.Samplers.StandardSample import standard_sample
def run(concept_key, ndata, chainidx):
    """
    Return standard sampling of hypotheses on this amount of data
    """
    if LOTlib.SIG_INTERRUPTED:
        return None, set()

    myset = standard_sample(make_hypothesis,
                           lambda: concept2data[concept_key][:ndata],
                           N=TOP,
                           steps=STEPS,
                           show=False, save_top=None)

    return concept_key, set(myset.get_all())


if __name__ == "__main__":

    from LOTlib.MPI.MPI_map import MPI_unorderedmap, is_master_process
    from collections import defaultdict

    hypotheses = defaultdict(set)
    for key, s in  MPI_unorderedmap(run, itertools.product(concept2data.keys(), xrange(35), xrange(CHAINS)) ):
        hypotheses[key].update(s)
        print "# Done %s found %s hypotheses" % (key, len(s))

        # for h in s:
        #     print key, h

    import pickle
    with open(OUT_PATH, 'w') as f:
        pickle.dump(hypotheses, f)
