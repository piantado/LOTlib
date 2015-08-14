from utils import *
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
from LOTlib.Examples.FormalLanguageTheory.Model.Hypothesis import make_hypothesis
from LOTlib.Examples.FormalLanguageTheory.Language.AnBn import AnBn
import time
from mpi4py import MPI

register_primitive(flatten2str)

"""
In this case, We study how max_length of data can influence the convergence.
"""

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # ========================================================================================================
    # Process command line arguments /
    # ========================================================================================================
    (options, args) = parser.parse_args()

    suffix = time.strftime('_' + options.NAME + '_%m%d_%H%M%S', time.localtime())

    init_size = 2
    iters_per_stage = 2 * options.STEPS / (options.FINITE-init_size)
    # ========================================================================================================
    # Process command line arguments /
    # ========================================================================================================

    language = AnBn()

    # show_info('running staged input case..')
    # rec = probe_MHsampler(make_hypothesis('AnBn'), language.sample_data_as_FuncData, options, 'staged_out_' + str(rank) + suffix, init_size=init_size, iters_per_stage=iters_per_stage)

    show_info('running normal input case..')
    rec1 = probe_MHsampler(make_hypothesis('AnBn'), language.sample_data_as_FuncData, options, 'normal_out_' + str(rank) + suffix)