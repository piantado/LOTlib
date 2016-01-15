from utils import *
from LOTlib.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
from LOTlib.Projects.FormalLanguageTheory.Model.Hypothesis import make_hypothesis
from LOTlib.Projects.FormalLanguageTheory.Language.AnBn import AnBn
import time
from mpi4py import MPI

from LOTlib.DataAndObjects import FunctionData
from collections import Counter

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
    prefix = '../out/simulations/skewed/'

    # ========================================================================================================
    # Running
    # ========================================================================================================
    language = AnBn()

    show_info('running skewed input case..')
    rec = probe_MHsampler(make_hypothesis('AnBn'), language, options, prefix + 'skewed_out_' + str(rank) + suffix)

    show_info('running normal input case..')
    CASE += 1
    cnt = Counter()
    num = 64.0 * 2 / options.FINITE
    for i in xrange(1, options.FINITE/2+1):
        cnt['a'*i+'b'*i] = num

    rec1 = probe_MHsampler(make_hypothesis('AnBn'), language, options, prefix + 'normal_out' + str(rank) + suffix, data=[FunctionData(input=[], output=cnt)])