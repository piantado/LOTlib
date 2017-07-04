"""
    This version incrementally adds symbols and then does not change them.

    Todo: show highest prob "mistake" strings

"""
import sys
import codecs
import itertools
import operator
from copy import copy
from LOTlib import break_ctrlc
from pickle import dump
from math import log
from copy import deepcopy
import random
import numpy as np
import LOTlib

from LOTlib.Miscellaneous import display_option_summary, Infinity
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.TopN import TopN
from Language import *

from LOTlib.MPI import is_master_process, MPI_unorderedmap

from Model import IncrementalLexiconHypothesis
from LOTlib.Projects.FormalLanguageTheory.Grammar import base_grammar

LARGE_SAMPLE = 100000 # sample this many and then re-normalize to fractional counts

def run(options, ndata):
    if LOTlib.SIG_INTERRUPTED: return 0, set()

    language = eval(options.LANG+"()")
    data = language.sample_data(LARGE_SAMPLE)

    assert len(data) == 1
    # renormalize the counts
    for k in data[0].output.keys():
        data[0].output[k] = float(data[0].output[k] * ndata) / LARGE_SAMPLE

    z = sum(data[0].output.values())
    if z > 0:
        best_ll = sum([ (p/z)*log(p/z) for p in data[0].output.values() ])
    else:
        best_ll = 0.0

    # Now add the rules to the grammar
    grammar = deepcopy(base_grammar)
    for t in language.terminals():  # add in the specifics
        grammar.add_rule('ATOM', "'%s'" % t, None, 1.0)

    # set up the hypothesis
    h0 = IncrementalLexiconHypothesis(grammar=grammar, alphabet_size=len(language.terminals()))
    h0.set_word(0, h0.make_hypothesis(grammar=grammar)) # make the first word at random
    h0.N = 1

    tn = TopN(N=options.TOP_COUNT)

    for outer in xrange(options.N): # how many do we add?
        if LOTlib.SIG_INTERRUPTED: return 0, set()

        # and re-set the posterior or else it's something weird
        h0.compute_posterior(data)

        # now run mcmc
        for h in break_ctrlc(MHSampler(h0, data, steps=options.STEPS)):
            h.best_ll = best_ll # just store this
            tn.add(copy(h))

            if options.TRACE:
                print h.posterior_score, h.prior, h.likelihood, h.likelihood / ndata, h
                v = h()
                sortedv = sorted(v.items(), key=operator.itemgetter(1), reverse=True )
                print "{" + ', '.join(["'%s':%s"% i for i in sortedv]) + "}"


        # and start from where we ended
        h0 = copy(h)
        h0.deepen()

    return ndata, tn

if __name__ == "__main__":
    """
        example:
            mpiexec -n 12 python search.py --language=SimpleEnglish --steps=100000
    """
    # ========================================================================================================
    # Process command line arguments /
    # ========================================================================================================
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("--language", dest="LANG", type="string", default='An', help="name of a language")
    parser.add_option("--steps", dest="STEPS", type="int", default=40000, help="Number of samples to run")
    parser.add_option("--skip", dest="SKIP", type="int", default=100, help="Print out every this many")
    parser.add_option("--top", dest="TOP_COUNT", type="int", default=100, help="Top number of hypotheses to store")
    parser.add_option("--N", dest="N", type="int", default=3, help="number of inner hypotheses")
    parser.add_option("--ndata", dest="ndata", type="int", default=1000, help="number of data steps to run")
    parser.add_option("--datamin", dest="datamin", type="float", default=0.01, help="Min data to run (>0 due to log)")
    parser.add_option("--datamax", dest="datamax", type="float", default=100, help="Max data to run")
    parser.add_option("--out", dest="OUT", type="str", default="out/", help="Output directory")
    parser.add_option("--trace", dest="TRACE", action="store_true", default=False, help="Show every step?")
    (options, args) = parser.parse_args()

    # Set the output
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)

    # Save options
    if is_master_process():
        display_option_summary(options)
        sys.stdout.flush()

    # DATA_RANGE = np.exp(np.linspace(np.log(options.datamin), np.log(options.datamax), num=options.ndata))
    DATA_RANGE = np.linspace(options.datamin, options.datamax, num=options.ndata)
    DATA_RANGE = np.append(DATA_RANGE, [0,0,0,0,1,1,1,1]) # include a bunch from the prior
    random.shuffle(DATA_RANGE) # run in random order

    args = list(itertools.product([options], DATA_RANGE))

    hypotheses = set()
    for ndata, tn in MPI_unorderedmap(run, args):
        for h in tn:
            hypotheses.add(h)
            print ndata, h.posterior_score, h.prior, h.likelihood, h.likelihood / ndata, h.best_ll
            v = h()
            sortedv = sorted(v.items(), key=operator.itemgetter(1), reverse=True)
            print "{" + ', '.join(["'%s':%s" % i for i in sortedv]) + "}"
            print h  # must add \0 when not Lexicon
            sys.stdout.flush()

        # Dump after ach so that we save incremental hypotheses
        with open(options.OUT+"/"+options.LANG+".pkl", 'wb') as f:
            dump(hypotheses, f)


    print "# Finishing"


