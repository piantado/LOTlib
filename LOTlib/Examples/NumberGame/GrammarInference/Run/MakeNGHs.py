
"""

Command line args
-----------------
-f --fname
    File name to save to (must be .p)
-o --domain
    Domain for NumberGameHypothesis
-a --alpha
    Alpha, the noise parameter

-g --grammar
    Which grammar do we use? [mix | indep | lot]
--grammar-scale
    Do we use a gamma to semi-randomly init grammar rule probs?
-d --data
    Location of data file

-i --iters
    Number of samples to run
-c --chains
    Number of chains to run on each data input
-n
    Only keep top N samples per MPI run (if we're doing MPI), or total (if not MPI)

--mpi
    Do we use MPI? (only if MCMC) [1 | 0]
--enum
    How deep to enumerate hypotheses? (only if not MCMC)

Example
-------
$ python Demo.py -f out/ngh_lot100k.p -o 100 -a 0.9 -g lot_grammar -d josh -i 100000 -c 10 -n 1000 -mcmc -mpi

"""

from optparse import OptionParser

from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.MPI.MPI_map import MPI_unorderedmap
from LOTlib.MCMCSummary.TopN import TopN
from LOTlib.Examples.NumberGame.GrammarInference.Model import *



# ============================================================================================================
# Parsing command-line options

parser = OptionParser()

parser.add_option("-f", "--filename",
                  dest="filename", type="string", default="ngh_default.p",
                  help="File name to save to (must be .p)")
parser.add_option("-o", "--domain",
                  dest="domain", type="int", default=100,
                  help="Domain for NumberGameHypothesis")
parser.add_option("-a", "--alpha",
                  dest="alpha", type="float", default=0.9,
                  help="Alpha, the noise parameter")

parser.add_option("-g", "--grammar",
                  dest="grammar", type="string", default="mpi-run.pkl",
                  help="Which grammar do we use? [mix_grammar | independent_grammar | lot_grammar]")
parser.add_option("--grammar-scale",
                  dest="grammar_scale", type="int", default=0,
                  help="If >0, use a gamma dist. for each rule in the grammar w/ specified scale & " +
                       "shape equal to initial rule prob.")
parser.add_option("-d", "--data",
                  dest="data", type="string", default="josh",
                  help="Which data do we use? [josh | filename.p]")

parser.add_option("-i", "--iters",
                  dest="iters", type="int", default=100000,
                  help="Number of samples to run per chain")
parser.add_option("-c", "--chains",
                  dest="chains", type="int", default=1,
                  help="Number of chains to run on each data input")
parser.add_option("-n",
                  dest="N", type="int", default=1000,
                  help="Only keep top N samples per MPI run (if we're doing MPI), or total (if not MPI)")

parser.add_option("--mpi",
                  action="store_true", dest="mpi", default=False,
                  help="Do we use MPI? (only if MCMC)")
parser.add_option("--enum",
                  action="store_true", dest="enum_depth", default=False,
                  help="How deep to enumerate hypotheses? (only if not MCMC)")

(options, args) = parser.parse_args()


# ============================================================================================================
# MPI method

def mpirun(d):
    """
    Generate NumberGameHypotheses using MPI.

    """
    if options.grammar_scale:
        grammar_ = grammar_gamma(grammar, options.grammar_scale)
    else:
        grammar_ = grammar
    h0 = NumberGameHypothesis(grammar=grammar_, domain=100, alpha=0.9)
    mh_sampler = MHSampler(h0, d.input, options.iters)
    # hypotheses = TopN(N=options.N)
    hypotheses = set()

    # This is a dict so we don't add duplicate hypotheses sets, e.g. h1() == [4],  h2() == [4]
    h_sets = {}

    for h in break_ctrlc(mh_sampler):
        h_set = str(h())
        if h_set in h_sets:
            if h.prior > h_sets[h_set].prior:
                hypotheses.remove(h_sets[h_set])
                h_sets[h_set] = h
                hypotheses.add(h)
        else:
            h_sets[h_set] = h
            hypotheses.add(h)

    top1000 = sorted(hypotheses, key=lambda h: -h.posterior_score)[0:1000]
    return top1000


# ============================================================================================================
# Sample hypotheses
# ============================================================================================================

if __name__ == "__main__":

    if options.grammar == 'mix':
        grammar = mix_grammar
    elif options.grammar == 'indep':
        grammar = independent_grammar
    elif options.grammar == 'lot':
        grammar = lot_grammar
    else:
        grammar = independent_grammar

    # Add more data options . . .
    if options.data == 'josh_data':
        data = import_josh_data()
    elif '.p' in options.data:
        import os
        path = os.getcwd() + '/'
        f = open(path + options.data)
        data = pickle.load(f)
    else:
        import os
        path = os.getcwd() + '/'
        data = import_pd_data(path + options.data + '.p')

    # --------------------------------------------------------------------------------------------------------
    # MCMC sampling

    # MPI
    if options.mpi:
        hypotheses = set()
        hypo_sets = MPI_unorderedmap(mpirun, [[d] for d in data * options.chains])
        for hypo_set in hypo_sets:
            hypotheses = hypotheses.union(hypo_set)

    # No MPI
    else:
        hypotheses = set()

        if options.grammar_scale:
            grammar = grammar_gamma(grammar, options.grammar_scale)

        for d in data * options.chains:
            h0 = NumberGameHypothesis(grammar=grammar, domain=options.domain, alpha=options.alpha)
            mh_sampler = MHSampler(h0, [d], options.iters)

            chain_hypos = TopN(N=options.N)
            for h in break_ctrlc(mh_sampler):
                chain_hypos.add(h)
            hypotheses = hypotheses.union(chain_hypos.get_all())

    # --------------------------------------------------------------------------------------------------------
    # Save hypotheses

    f = open(options.filename, "wb")
    pickle.dump(hypotheses, f)

