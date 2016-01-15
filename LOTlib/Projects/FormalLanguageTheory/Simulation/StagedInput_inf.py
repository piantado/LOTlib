from utils import *
from LOTlib.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
from LOTlib.Projects.FormalLanguageTheory.Model.Hypothesis import make_hypothesis
import time
from mpi4py import MPI
import sys
from pickle import dump

fff = sys.stdout.flush
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

register_primitive(flatten2str)


def slice_list(input, size):
    input_size = len(input)
    slice_size = input_size / size
    remain = input_size % size
    result = []
    iterator = iter(input)
    for i in range(size):
        result.append([])
        for j in range(int(slice_size)):
            result[i].append(iterator.next())
        if remain:
            result[i].append(iterator.next())
            remain -= 1
    return result


def sq(i):
    return '_' + str(i)


def no_data(a, max_length=None):
    return [FunctionData(input=[], output=Counter())]

if __name__ == '__main__':
    """
    run with script: mpiexec -n 12 python StagedInput_inf.py --steps=40000 --language=AnBn --finite=12 --N=1 --mode=0/1
    """
    # ========================================================================================================
    # Process command line arguments
    # ========================================================================================================
    parser.add_option("--mode", dest="MODE", type="int", default=0, help="staged or normal")
    parser.add_option("--uniform", dest="UNI", type="int", default=0, help="geometry or uniform")
    (options, args) = parser.parse_args()

    suffix = time.strftime('_' + str(rank) + '_' + options.NAME + '_%m%d_%H%M%S', time.localtime())
    prefix = './'
    # prefix = '../out/simulations/staged/'
    # prefix = '/ho.me/lijm/WORK/yuan/lot/staged/'
    # ========================================================================================================
    # Running
    # ========================================================================================================
    # language = AnBn(max_length=options.FINITE)
    #
    # work_list = slice_list([[make_hypothesis, 12*(i+1), options.FINITE if options.MODE else 4*(1+i/4), options,
    #                          uniform_data if options.UNI else None] for i in xrange(12)], size)

    work_list = [[make_hypothesis, 12, options.FINITE, options, no_data]]

    # for e in work_list[rank]:
    for e in work_list:
        topn = run(*e)
        dump(topn, open(prefix + ('normal' + str(options.UNI) if options.MODE else 'staged') + sq(e[1]) + sq(e[2]) + suffix,'w'))