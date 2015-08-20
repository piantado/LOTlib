from utils import *
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
from LOTlib.Examples.FormalLanguageTheory.Model.Hypothesis import make_hypothesis
from LOTlib.Examples.FormalLanguageTheory.Language.AnBn import AnBn
import time
from mpi4py import MPI
import os
from LOTlib.Examples.FormalLanguageTheory.my_search_stp import run

register_primitive(flatten2str)

"""
In this case, We study how max_length of data can influence the convergence.
"""


def probe_sampler(h0, language, options, name, length=None):
    stuck_cnt = 0
    size = 50
    staged = length is not None
    length = (lambda: length if staged else options.FINITE)()

    get_data = language.sample_data_as_FuncData
    evaluation_data = get_data(144, max_length=options.FINITE)
    pr_data = get_data(1024, max_length=options.FINITE)
    data = get_data(n=size, max_length=length)

    sampler = MHSampler(h0, data)
    best_hypotheses = TopN(N=options.TOP_COUNT)

    iter = 0
    for h in sampler:

        if iter == options.STEPS * 12: break

        best_hypotheses.add(h)

        if iter % 200 == 0 and iter != 0:
            print '---->', iter
            Z, weighted_score, score, s, rec = probe(best_hypotheses, evaluation_data, pr_data, language.estimate_precision_and_recall)
            to_file([[iter, Z, weighted_score, score, s, rec]], name)
            stuck_cnt += 1 if -1e-6 < weighted_score - 0.181818181818 < 1e-6 else 0

        if iter % options.STEPS == 0 and iter != 0:
            if staged: length += 4 * (size % 48 == 0)
            size += 12
            sampler.data = get_data(n=size, max_length=length)

        iter += 1

        if stuck_cnt == 10:
            print '*'*10, 'get stuck, try again', '*'*10
            os.remove(name)
            probe_sampler(h0.propose()[0], language, options, name, length)
            return

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # ========================================================================================================
    # Process command line arguments
    # ========================================================================================================
    (options, args) = parser.parse_args()

    suffix = time.strftime('_' + options.NAME + '_%m%d_%H%M%S', time.localtime())
    prefix = '../out/simulations/staged/'
    # ========================================================================================================
    # Running
    # ========================================================================================================
    language = AnBn()

    # x = [10, 20, 30, 40, 50]
    # run(make_hypothesis, language, x[rank], 20)

    show_info('running staged input case..')
    probe_sampler(make_hypothesis('AnBn'), language, options, prefix + 'staged_out_' + str(rank) + suffix, length=4)

    show_info('running normal input case..')
    probe_sampler(make_hypothesis('AnBn'), language, options, prefix + 'normal_out_' + str(rank) + suffix)