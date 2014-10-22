import Shared
from optparse import OptionParser
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Inference.MetropolisHastings import mh_sample

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse options
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser = OptionParser()
parser.add_option("--out", dest="OUT_PATH", type="string", help="Output file (a pickle of FiniteBestSet)", default="/home/piantado/Desktop/mit/Projects/BindingTheoryAcquisition/Model/run/mpi-run.pkl")
parser.add_option("--steps", dest="STEPS", type="int", default=Shared.Infinity, help="Number of Gibbs cycles to run")
parser.add_option("--top", dest="TOP_COUNT", type="int", default=10, help="Top number of hypotheses to store")
parser.add_option("--data", dest="DATA", type="float", default=10.0, help="How much effective data?")

# standard options
(options, args) = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simplified search/mcmc
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# set up an initial hypothesis
initial_hyp = Shared.BindingTheoryLexicon(Shared.make_hypothesis, likelihood_temperature=1./options.DATA)
initial_hyp.set_word("he/him")
initial_hyp.set_word("himself")
initial_hyp.set_word("<UNCHANGED>")

for h in Shared.lot_iter(mh_sample(initial_hyp, Shared.data, steps=1000)):
    print h.posterior_score, h.prior, h.likelihood, "\n", h, "\n"

