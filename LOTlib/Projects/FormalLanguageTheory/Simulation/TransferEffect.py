from utils import *
from LOTlib.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
from LOTlib.Projects.FormalLanguageTheory.Model.Hypothesis import make_hypothesis
from LOTlib.Projects.FormalLanguageTheory.Language.An import An
import time
from mpi4py import MPI

register_primitive(flatten2str)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ========================================================================================================
    # Process command line arguments
    # ========================================================================================================
    (options, args) = parser.parse_args()

    suffix = time.strftime('_' + options.NAME + '_%m%d_%H%M%S', time.localtime())
    prefix = '../out/simulations/transfer/'

    # ========================================================================================================
    # Running
    # ========================================================================================================
    language = An()

    show_info('running normal input case..')
    sampler = probe_MHsampler(make_hypothesis('An', terminals=['b']), language, options, prefix + 'without_prior_out_' + str(rank) + suffix, ret_sampler=True)

    show_info('running with input using different letter case..')
    CASE += 1
    language = An(atom='b')
    probe_MHsampler(make_hypothesis('An', terminals=['b']), language, options, prefix + 'with_prior_out_' + str(rank) + suffix, sampler=sampler)