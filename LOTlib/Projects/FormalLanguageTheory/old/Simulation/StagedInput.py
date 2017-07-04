from utils import *
from LOTlib.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
from LOTlib.Projects.FormalLanguageTheory.Model.Hypothesis import make_hypothesis
from LOTlib.Projects.FormalLanguageTheory.Language.AnBn import AnBn
import time
from mpi4py import MPI
from LOTlib.Inference.Samplers.MultipleChainMCMC import MultipleChainMCMC
import sys

fff = sys.stdout.flush
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

register_primitive(flatten2str)

"""
In this case, We study how max_length of data can influence the convergence.
"""


def probe_sampler(make_h0, language, options, name, length=None):
    size = 12
    staged = length is not None
    length = (lambda: length if staged else options.FINITE)()

    get_data = language.sample_data_as_FuncData
    evaluation_data = get_data(144, max_length=options.FINITE)
    pr_data = get_data(1024, max_length=options.FINITE)
    data = get_data(n=size, max_length=length)

    # sampler = MHSampler(make_h0(), data)
    sampler = MultipleChainMCMC(make_h0, data, steps=options.STEPS * 12, nchains=5)
    best_hypotheses = TopN(N=options.TOP_COUNT)

    iter = 0
    for h in sampler:

        if iter == options.STEPS * 12: break

        best_hypotheses.add(h)

        if iter % 200 == 0 and iter != 0:
            print rank, '---->', iter
            fff()
            Z, weighted_score, score, s, rec = probe(best_hypotheses, evaluation_data, pr_data, language.estimate_precision_and_recall)
            to_file([[iter, Z, weighted_score, score, s, rec]], name)

        if iter % options.STEPS == 0 and iter != 0:
            size += 12
            if staged: length += 4 * (size % 48 == 0)
            for e in sampler.chains: e.data = get_data(n=size, max_length=length)

        iter += 1

if __name__ == '__main__':
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

    show_info('running staged input case..')
    probe_sampler(lambda: make_hypothesis('AnBn', N=options.N), language, options, prefix + 'staged_out_' + str(rank) + suffix, length=4)

    show_info('running normal input case..')
    probe_sampler(lambda: make_hypothesis('AnBn', N=options.N), language, options, prefix + 'normal_out_' + str(rank) + suffix)