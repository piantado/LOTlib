"""
    This version incrementally adds symbols and then does not change them.

"""
import sys
import codecs
import itertools
import operator
from LOTlib import break_ctrlc
from pickle import dump
from copy import deepcopy
import random
import numpy as np
import LOTlib

from LOTlib.Miscellaneous import display_option_summary, q
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.Eval import register_primitive
from LOTlib.TopN import TopN
from Language import *

from LOTlib.MPI import is_master_process, MPI_unorderedmap

from Model import IncrementalLexiconHypothesis
from LOTlib.Projects.FormalLanguageTheory.Grammar import base_grammar # passed in as kwargs


LARGE_SAMPLE = 10000 # sample this many and then re-normalize to fractional counts

def run(options, ndata):
    if LOTlib.SIG_INTERRUPTED:
        return 0, set()

    language = eval(options.LANG+"()")
    data = language.sample_data(LARGE_SAMPLE)
    assert len(data) == 1
    # renormalize the counts
    for k in data[0].output.keys():
        data[0].output[k] = float(data[0].output[k] * ndata) / LARGE_SAMPLE

    # Now add the rules to the grammar
    grammar = deepcopy(base_grammar)
    for t in language.terminals():  # add in the specifics
        grammar.add_rule('ATOM', '{\'%s\':0.0}' % t, None, 2.0)
        grammar.add_rule('DETATOM', '\'%s\'' % t, None, 2.0)


    h0 = IncrementalLexiconHypothesis(grammar=grammar)
    tn = TopN(N=options.TOP_COUNT)

    for outer in xrange(options.N): # how many do we add?

        h0.deepen() # add one more word
        assert len(h0.value.keys())==h0.N==outer+1

        # and re-set the posterior or else it's something weird
        h0.compute_posterior(data)

        # now run mcmc
        for h in break_ctrlc(MHSampler(h0, data, steps=options.STEPS)):
            tn.add(h)

            if options.TRACE:
                print h.posterior_score, h.prior, h.likelihood, h
                v = h()
                sortedv = sorted(v.items(), key=operator.itemgetter(1), reverse=True )
                print "{" + ', '.join(["'%s':%s"% i for i in sortedv]) + "}"

        # and start from where we ended
        h0 = deepcopy(h) # must deepcopy

    return ndata, tn

if __name__ == "__main__":
    """
        example:
            mpiexec -n 12 python my_search_stp.py --language=SimpleEnglish --steps=100000
    """
    # ========================================================================================================
    # Process command line arguments /
    # ========================================================================================================
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("--language", dest="LANG", type="string", default='An', help="name of a language")
    parser.add_option("--steps", dest="STEPS", type="int", default=40000, help="Number of samples to run")
    parser.add_option("--skip", dest="SKIP", type="int", default=100, help="Print out every this many")
    parser.add_option("--top", dest="TOP_COUNT", type="int", default=10, help="Top number of hypotheses to store")
    parser.add_option("--N", dest="N", type="int", default=3, help="number of inner hypotheses")
    parser.add_option("--ndata", dest="ndata", type="int", default=1000, help="number of data steps to run")
    parser.add_option("--datamax", dest="datamax", type="int", default=100000, help="Max data to run")
    parser.add_option("--out", dest="OUT", type="str", default="out/", help="Output directory")
    parser.add_option("--trace", dest="TRACE", action="store_true", default=False, help="Show every step?")
    (options, args) = parser.parse_args()

    # Set the output
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)

    # Save options
    if is_master_process():
        display_option_summary(options)
        sys.stdout.flush()

    DATA_RANGE = np.exp(np.linspace(0, np.log(options.datamax), num=options.ndata))# [1000] # np.arange(1, 1000, 1)
    # DATA_RANGE = [10000]
    random.shuffle(DATA_RANGE) # run in random order

    args = list(itertools.product([options], DATA_RANGE))

    unq = set()
    for ndata, tn in MPI_unorderedmap(run, args):
        for h in tn:
            if h not in unq:
                unq.add(h)

                print ndata, h.posterior_score, h.prior, h.likelihood, h.likelihood / ndata
                print h(),
                print h  # must add \0 when not Lexicon
        sys.stdout.flush()

    print "# Finishing"

    with open(options.OUT+"/hypotheses-"+options.LANG+".pkl", 'w') as f:
        dump(unq, f)