from utils import *
from LOTlib.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
from LOTlib.Projects.FormalLanguageTheory.Model.Hypothesis import make_hypothesis
from LOTlib.Projects.FormalLanguageTheory.Language.LongDependency import LongDependency
import time
from mpi4py import MPI

register_primitive(flatten2str)

"""
In this case, we investigate the effect of different observed data distributions on training convergence.
"""

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ========================================================================================================
    # Process command line arguments
    # ========================================================================================================
    (options, args) = parser.parse_args()

    suffix = time.strftime('_' + options.NAME + '_%m%d_%H%M%S', time.localtime())
    prefix = '../out/simulations/nonadjacent/'

    # ========================================================================================================
    # Running
    # ========================================================================================================
    # case 1
    show_info('running predictable input case..')
    language = LongDependency(C=['c'])
    probe_MHsampler(make_hypothesis('LongDependency', terminals=['c', 'd', 'e', 'f']), language, options, prefix + 'c_long_' + str(rank) + suffix)

    show_info('running predictable input case..')
    CASE += 1
    language = LongDependency(C=['c', 'd', 'e', 'f'])
    probe_MHsampler(make_hypothesis('LongDependency', terminals=['c', 'd', 'e', 'f']), language, options, prefix + 'cdef_' + str(rank) + suffix)

    # --------------------------------------------------------------------------------------------------------
    # case 2
    show_info('running predictable input case..')
    options.FINITE = 4
    CASE += 1
    language = LongDependency(C=['c', 'd', 'e', 'f'])
    probe_MHsampler(make_hypothesis('LongDependency', terminals=['c', 'd', 'e', 'f']), language, options, prefix + 'c_short_' + str(rank) + suffix)


