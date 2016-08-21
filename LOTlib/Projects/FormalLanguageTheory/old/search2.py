import sys
import codecs
import itertools
from LOTlib import break_ctrlc
from optparse import OptionParser
from pickle import dump
import time
import random
import numpy as np
import LOTlib
from LOTlib.Miscellaneous import display_option_summary, q
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str, logsumexp, qq
from LOTlib.TopN import TopN
from Model.Hypothesis import MyHypothesis
from Language import *
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
fff = sys.stdout.flush

from copy import deepcopy
from LOTlib.Projects.FormalLanguageTheory.Model.Grammar import base_grammar # passed in as kwargs

register_primitive(flatten2str)

global prefix
global suffix
prefix = ""
suffix = ""

LARGE_SAMPLE = 10000 # sample this many and then re-normalize to fractional counts

from math import log
from LOTlib.Miscellaneous import attrmem
from Levenshtein import distance
from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursiveLOTHypothesis, RecursionDepthException
from LOTlib.Hypotheses.Proposers.Sandbox.InverseInlineProposer import InverseInlineProposer
from LOTlib.Hypotheses.Proposers.InsertDeleteProposal import InsertDeleteProposal
from LOTlib.Hypotheses.Likelihoods.StochasticLikelihood import StochasticLikelihood
class AugustHypothesis(InsertDeleteProposal, StochasticLikelihood, RecursiveLOTHypothesis):
    def __init__(self, grammar=base_grammar, **kwargs):
        RecursiveLOTHypothesis.__init__(self, grammar, **kwargs)

    def __call__(self, *args):
        """
        The main calling function. Resets recursive_call_depth and then calls
        """
        try:
            return RecursiveLOTHypothesis.__call__(self, *args)
        except RecursionDepthException:
            return ''

    @attrmem('likelihood')
    def compute_likelihood(self, data, shortcut=None):
        # We'll just be overwriting this
        assert len(data) == 1
        return self.compute_single_likelihood(data[0])

    def compute_single_likelihood(self, datum):
        """
            Compute the likelihood with a Levenshtein noise model
        """

        assert isinstance(datum.output, dict), "Data supplied must be a dict (function outputs to counts)"

        llcounts = self.make_ll_counts(datum.input, nsamples=1024)

        lo = sum(llcounts.values())

        ll = 0.0 # We are going to compute a pseudo-likelihood, counting close strings as being close
        for k in datum.output.keys():
            ll += datum.output[k] * (log(llcounts.get(k, 1.0e-12)) - log(lo))
            # ll += datum.output[k] * (log(llcounts.get(k,1.0e-12)) - log(lo)) # Just have a smoothing parameter -- may work well if we use the averaged data
            # ll += datum.output[k] * logsumexp([ log(llcounts[r])-log(lo) - 1000.0 * distance(r, k) for r in llcounts.keys() ])
        return ll



def run(options, ndata):
    """
    This out on the DATA_RANGE amounts of data and returns all hypotheses in top count
    """
    if LOTlib.SIG_INTERRUPTED:
        return set()

    language = eval(options.LANG+"()")
    data = language.sample_data(LARGE_SAMPLE)
    assert len(data) == 1

    # renormalize the counts
    for k in data[0].output.keys():
        data[0].output[k] = float(data[0].output[k] * ndata) / LARGE_SAMPLE

    print data
    # Now add the rules to the grammar
    grammar = deepcopy(base_grammar)
    for t in language.terminals():  # add in the specifics
        grammar.add_rule('ATOM', q(t), None, 2)

    h0 = AugustHypothesis(grammar=grammar, display="lambda recurse_ :%s")

    tn = TopN(N=options.TOP_COUNT)

    for i, h in enumerate(break_ctrlc(MHSampler(h0, data, steps=options.STEPS))):
        print h.posterior_score, h
        print getattr(h, 'll_counts', None)

    # with open(prefix+'hypotheses_'+options.LANG+'_'+str(rank)+'_'+str(ndata)+'_'+suffix+".txt", 'a') as ofile:
    #
    #     for i, h in enumerate(break_ctrlc(MHSampler(h0, data, steps=options.STEPS))):
    #         tn.add(h)
    #         # print h.posterior_score, getattr(h, 'll_counts', None), h
    #         if i%options.SKIP == 0:
    #             print >>ofile, "\n"
    #             print >>ofile, i, ndata, h.posterior_score, h.prior, h.likelihood, h.likelihood/ndata
    #             print >>ofile, getattr(h,'ll_counts', None),
    #             print >>ofile, h # ends in \0 so we can sort with sort -g -z


    return tn


def simple_mpi_map(run, args):
    print 'rank: ', rank, 'running..'; fff()
    hypo_set = run(*(args[rank]))

    global prefix
    dump(hypo_set, open(prefix+'hypotheses_'+options.LANG+'_%i'%rank+suffix+".pkl", 'w'))


if __name__ == "__main__":
    """
        example:
            mpiexec -n 12 python my_search_stp.py --language=An --finite=10 --N=1 --bound=15
            mpiexec -n 12 python my_search_stp.py --language=AnBn --finite=20 --N=1 --terminal=b --bound=15
            mpiexec -n 12 python my_search_stp.py --language=ABn --finite=20?? --N=1 --terminal=b --bound=15
            mpiexec -n 12 python my_search_stp.py --language=AnB2n --finite=30 --N=1 --terminal=b --bound=15
            mpiexec -n 12 python my_search_stp.py --language=AnBnCn --finite=18 --N=3 --terminal=bc --bound=5
            mpiexec -n 12 python my_search_stp.py --language=Dyck --finite=8 --N=2 --terminal=b --bound=7 --steps=100000
            mpiexec -n 12 python my_search_stp.py --language=SimpleEnglish --finite=1000 --N=3 --bound=5 --steps=100000
    """
    # ========================================================================================================
    # Process command line arguments /
    # ========================================================================================================
    fff = sys.stdout.flush
    parser = OptionParser()
    parser.add_option("--language", dest="LANG", type="string", default='An', help="name of a language")
    parser.add_option("--steps", dest="STEPS", type="int", default=40000, help="Number of samples to run")
    parser.add_option("--skip", dest="SKIP", type="int", default=100, help="Print out every this many")
    parser.add_option("--top", dest="TOP_COUNT", type="int", default=20, help="Top number of hypotheses to store")
    parser.add_option("--name", dest="NAME", type="string", default='', help="name of file")
    parser.add_option("--N", dest="N", type="int", default=3, help="number of inner hypotheses")
    parser.add_option("--terminal", dest="TERMINALS", type="string", default='', help="extra terminals")
    parser.add_option("--bound", dest="BOUND", type="int", default=5, help="recursion bound")
    parser.add_option("--out", dest="OUT", type="str", default="out/", help="Output directory")
    (options, args) = parser.parse_args()


    global prefix
    prefix = options.OUT
    global suffix
    suffix = time.strftime('_' + options.NAME + '_%m%d_%H%M%S', time.localtime())

    sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    if rank == 0: display_option_summary(options); fff()

    #DATA_RANGE = np.arange(120, 264, 12)
    DATA_RANGE = [1000] #  np.arange(100, 50000, 100)
    random.shuffle(DATA_RANGE) # run in random order

    args = list(itertools.product([options], DATA_RANGE))

    hypotheses = simple_mpi_map(run, args)
