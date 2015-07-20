import sys
import codecs
import itertools
import time
from optparse import OptionParser

import numpy as np

import LOTlib
from LOTlib.Miscellaneous import display_option_summary
from LOTlib.MPI.MPI_map import is_master_process
from LOTlib.Examples.Demo import standard_sample
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str, logsumexp, qq
from LOTlib.MPI.MPI_map import MPI_map
from LOTlib.Examples.FormalLanguageTheory.RegularLanguage import Regularlanguage


register_primitive(flatten2str)


def run(make_hypothesis, language, data_size):
    """
    This out on the DATA_RANGE amounts of data and returns all hypotheses in top count
    """
    if LOTlib.SIG_INTERRUPTED:
        return set()

    return standard_sample(make_hypothesis,
                           lambda: language.sample_data(data_size),
                           N=options.TOP_COUNT,
                           steps=options.STEPS,
                           show=False, save_top=None)


def run_make_data(make_hypothesis, make_data, data_size):
    """
    This out on the DATA_RANGE amounts of data and returns all hypotheses in top count
    """
    if LOTlib.SIG_INTERRUPTED:
        return set()

    return standard_sample(make_hypothesis,
                           lambda: make_data(data_size),
                           N=options.TOP_COUNT,
                           steps=options.STEPS,
                           show=False, save_top=None)


if __name__ == "__main__":
    from cog_test import make_hypothesis, make_data
    original = False
    suffix = time.strftime('_%m%d_%H%M%S', time.localtime())

    # ========================================================================================================
    # Process command line arguments /
    # ========================================================================================================
    fff = sys.stdout.flush
    parser = OptionParser()
    parser.add_option("--steps", dest="STEPS", type="int", default=1000, help="Number of samples to run")
    parser.add_option("--top", dest="TOP_COUNT", type="int", default=10, help="Top number of hypotheses to store")
    (options, args) = parser.parse_args()


    # set the output codec -- needed to display lambda to stdout
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    if is_master_process():
        display_option_summary(options); fff()

    # Make a list of arguments to map over, and Set up a reasonable range of data
    if original:
        DATA_RANGE = np.arange(0.0, 5.0, 0.25)
        args = list(itertools.product([make_hypothesis], [make_data], DATA_RANGE))
        # run on MPI
        results = MPI_map(run_make_data, args)
    else:
        # you need to run 12 machine on that
        DATA_RANGE = np.arange(1, 64, 6)
        # TODO language = Regularlanguage(max_length=15)
        language = Regularlanguage()
        args = list(itertools.product([make_hypothesis], [language], DATA_RANGE))
        # run on MPI
        results = MPI_map(run, args)


    # collapse all returned sets
    hypotheses = set()
    for r in results:
        hypotheses.update(r) # add the ith's results to the set

    # Now go through each hypothesis and print out some summary stats
    for data_size in DATA_RANGE:

        if original: evaluation_data = make_data(data_size)
        else: evaluation_data = language.sample_data(data_size)

        # Now update everyone's posterior
        for h in hypotheses:
            h.compute_posterior(evaluation_data)

        # compute the normalizing constant. This is the log of the sum of the probabilities
        Z = logsumexp([h.posterior_score for h in hypotheses])

        f = open('out_' + str(original) + suffix, 'a')
        for h in hypotheses:
            #compute the number of different strings we generate
            generated_strings = set([h() for _ in xrange(1000)])
            if original:
                print >> f, data_size, np.exp(h.posterior_score-Z), h.posterior_score, h.prior, h.likelihood, len(generated_strings), qq(h)
            else:
                precision, recall = language.estimate_precision_and_recall(h)
                print >> f, data_size, np.exp(h.posterior_score-Z), h.posterior_score, h.prior, h.likelihood, len(generated_strings), qq(h), precision, recall
        f.close()
