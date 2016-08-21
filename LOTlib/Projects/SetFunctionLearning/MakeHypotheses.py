"""
    Script to generate a file of all hypotheses by running incrementally on each amount of data
"""

import LOTlib
import itertools

from Model.Data import concept2data
from Model.Hypothesis import make_hypothesis

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Process Options
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--out", dest="OUT_PATH", type="string", default="hypotheses.pkl", help="Output file of all hypotheses")
parser.add_option("--steps", dest="STEPS", type="int", default=50000, help="Number of samples to run")
parser.add_option("--top", dest="TOP_COUNT", type="int", default=100, help="Top number of hypotheses to store")
parser.add_option("--chains", dest="CHAINS", type="int", default=1, help="Number of chains to run (new data set for each chain)")
# parser.add_option("--grammar", dest="GRAMMAR", type="str", default="lot_grammar", help="The grammar we use (defined in Model)")

(options, args) = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Process Options
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Inference.Samplers.StandardSample import standard_sample
def run(concept_key, ndata, chainidx):
    """
    Return standard sampling of hypotheses on this amount of data
    """
    if LOTlib.SIG_INTERRUPTED:
        return None, set()

    myset = standard_sample(make_hypothesis,
                           lambda: concept2data[concept_key][:ndata],
                           N=options.TOP_COUNT,
                           steps=options.STEPS,
                           show=False, save_top=None)

    return concept_key, set(myset.get_all())


if __name__ == "__main__":

    from LOTlib.MPI.MPI_map import MPI_unorderedmap, is_master_process
    from collections import defaultdict

    hypotheses = defaultdict(set)
    for key, s in  MPI_unorderedmap(run, itertools.product(concept2data.keys(), xrange(35), xrange(options.CHAINS)) ):
        hypotheses[key].update(s)
        print "# Done %s found %s hypotheses" % (key, len(s))

        # for h in s:
        #     print key, h

    import pickle
    with open(options.OUT_PATH, 'w') as f:
        pickle.dump(hypotheses, f)
