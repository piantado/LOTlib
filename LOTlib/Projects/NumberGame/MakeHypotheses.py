from Model import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Process Options
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--out", dest="OUT_PATH", type="string", default="hypotheses.pkl", help="Output file of all hypotheses")
parser.add_option("--steps", dest="STEPS", type="int", default=100000, help="Number of samples to run")
parser.add_option("--top", dest="TOP_COUNT", type="int", default=100, help="Top number of hypotheses to store")
parser.add_option("--chains", dest="CHAINS", type="int", default=1, help="Number of chains to run (new data set for each chain)")
parser.add_option("--grammar", dest="GRAMMAR", type="str", default="lot_grammar", help="The grammar we use (defined in Model)")

(options, args) = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

grammar = eval(options.GRAMMAR)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define the running function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import LOTlib
from LOTlib import break_ctrlc
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.DataAndObjects import FunctionData
from LOTlib.TopN import TopN

def myrun(observed_set):

    if LOTlib.SIG_INTERRUPTED:
        return set()

    h0 = NumberGameHypothesis(grammar=grammar)

    data = [FunctionData(input=[], output=observed_set, alpha=ALPHA)]

    tn = TopN(N=options.TOP_COUNT)
    for h in break_ctrlc(MHSampler(h0, data, steps=options.STEPS)):
        tn.add(h)
        # print "#", h

    print "# Finished %s" % str(observed_set)

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
    from LOTlib.MPI.MPI_map import MPI_unorderedmap, is_master_process
    from LOTlib.Miscellaneous import display_option_summary

    if is_master_process():
        display_option_summary(options)

    hypotheses = set()
    for s in MPI_unorderedmap(myrun, [ [s] for s in observed_sets ]*options.CHAINS ):
        hypotheses.update(s)

    import pickle
    with open(options.OUT_PATH, 'w') as f:
        pickle.dump(hypotheses, f)

