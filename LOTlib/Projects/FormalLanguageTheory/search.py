"""
    This version incrementally adds symbols and then does not change them.

"""
import sys
import codecs
import itertools
from LOTlib import break_ctrlc
from pickle import dump
from copy import deepcopy
import random
import numpy as np
import LOTlib

from LOTlib.Miscellaneous import display_option_summary, q

from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str, logsumexp, qq, Infinity
from LOTlib.TopN import TopN
from Language import *

from LOTlib.MPI import is_master_process, MPI_unorderedmap

from Model import IncrementalLexiconHypothesis
from LOTlib.Projects.FormalLanguageTheory.Grammar import base_grammar # passed in as kwargs

register_primitive(flatten2str)

LARGE_SAMPLE = 10000 # sample this many and then re-normalize to fractional counts


def run(options, ndata):
    """
    This out on the DATA_RANGE amounts of data and returns all hypotheses in top count
    """
    if LOTlib.SIG_INTERRUPTED:
        return 0, set()

    language = eval(options.LANG+"()")
    data = language.sample_data(LARGE_SAMPLE)
    assert len(data) == 1

    # renormalize the counts
    for k in data[0].output.keys():
        data[0].output[k] = float(data[0].output[k] * ndata) / LARGE_SAMPLE
    # print data

    # Now add the rules to the grammar
    grammar = deepcopy(base_grammar)
    for t in language.terminals():  # add in the specifics
        grammar.add_rule('ATOM', q(t), None, 2)

    h0 = IncrementalLexiconHypothesis(grammar=grammar)
    print "# Starting on ", h0

    tn = TopN(N=options.TOP_COUNT)

    for outer in xrange(options.N): # how many do we add?
        for h in break_ctrlc(MHSampler(h0, data, steps=options.STEPS)):
            tn.add(h)

            # print h.posterior_score, h
            # print getattr(h, 'll_counts', None)

        h0 = deepcopy(h)
        h0.add_new_word()

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
    parser.add_option("--out", dest="OUT", type="str", default="out/", help="Output directory")
    (options, args) = parser.parse_args()

    # Set the output
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)

    # Save options
    if is_master_process():
        display_option_summary(options);
        sys.stdout.flush()

    DATA_RANGE = np.arange(1, 1000, 10)
    random.shuffle(DATA_RANGE) # run in random order

    args = list(itertools.product([options], DATA_RANGE))

    unq = set()
    for ndata, tn in MPI_unorderedmap(run, args):
        for h in tn:
            if h not in unq:
                # unq.add(h)

                print ndata, h.posterior_score, h.prior, h.likelihood, h.likelihood / ndata
                print getattr(h, 'll_counts', None),
                print h  # must add \0 when not Lexicon
        sys.stdout.flush()

    print "# Finishing"

    # with open(options.OUT+"/all-hypotheses"+options.LANG+".pkl", 'w') as f:
    #     dump(unq, f)