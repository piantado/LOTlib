import sys
import codecs
import itertools
from optparse import OptionParser
from pickle import dump
import time
import numpy as np
import LOTlib
from LOTlib.Miscellaneous import display_option_summary
from LOTlib.MPI.MPI_map import is_master_process
from LOTlib.Examples.Demo import standard_sample
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str, logsumexp, qq
from LOTlib.MPI.MPI_map import MPI_map
from Model.Hypothesis import make_hypothesis
from Language.Index import instance

register_primitive(flatten2str)


def run(mk_hypothesis, lang, size, finite):
    """
    This out on the DATA_RANGE amounts of data and returns all hypotheses in top count
    """
    if LOTlib.SIG_INTERRUPTED:
        return set()

    return standard_sample(lambda: mk_hypothesis(options.LANG),
                           lambda: lang.sample_data_as_FuncData(size, max_length=finite),
                           N=options.TOP_COUNT,
                           steps=options.STEPS,
                           show=False, save_top=None)


if __name__ == "__main__":
    # ========================================================================================================
    # Process command line arguments /
    # ========================================================================================================
    fff = sys.stdout.flush
    parser = OptionParser()
    parser.add_option("--language", dest="LANG", type="string", default='An', help="name of a language")
    parser.add_option("--steps", dest="STEPS", type="int", default=10000, help="Number of samples to run")
    parser.add_option("--top", dest="TOP_COUNT", type="int", default=20, help="Top number of hypotheses to store")
    parser.add_option("--finite", dest="FINITE", type="int", default=10, help="specify the max_length to make language finite")
    parser.add_option("--name", dest="NAME", type="string", default='', help="name of file")
    (options, args) = parser.parse_args()

    suffix = time.strftime('_' + options.NAME + '_%m%d_%H%M%S', time.localtime())

    # set the output codec -- needed to display lambda to stdout
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    if is_master_process():
        display_option_summary(options); fff()

    # you need to run 5 machine on that
    DATA_RANGE = np.arange(20, 50, 10)

    language = instance(options.LANG)
    args = list(itertools.product([make_hypothesis], [language], DATA_RANGE, [options.FINITE]))
    # run on MPI
    results = MPI_map(run, args)

    # ========================================================================================================
    # Get stats
    # ========================================================================================================
    # collapse all returned sets
    hypotheses = set()
    for r in results:
        hypotheses.update(r)  # add the ith's results to the set

    dump(hypotheses, open('hypotheses'+suffix, 'w'))

    # get precision and recall for h
    pr_data = language.sample_data_as_FuncData(1024, max_length=options.FINITE)
    p = []
    r = []
    print 'compute precision and recall..'
    for h in hypotheses:
        precision, recall = language.estimate_precision_and_recall(h, pr_data)
        p.append(precision)
        r.append(recall)

    # Now go through each hypothesis and print out some summary stats
    for data_size in DATA_RANGE:
        print 'get stats from size : ', data_size

        evaluation_data = language.sample_data_as_FuncData(data_size, max_length=options.FINITE)

        # Now update everyone's posterior
        for h in hypotheses:
            h.compute_posterior(evaluation_data)

        # compute the normalizing constant. This is the log of the sum of the probabilities
        Z = logsumexp([h.posterior_score for h in hypotheses])

        f = open('out' + suffix, 'a')
        cnt = 0
        for h in hypotheses:
            #compute the number of different strings we generate
            generated_strings = set([h() for _ in xrange(1000)])
            print >> f, data_size, np.exp(h.posterior_score-Z), h.posterior_score, h.prior, \
                h.likelihood, len(generated_strings), qq(h), p[cnt], r[cnt]
            cnt += 1
        f.close()