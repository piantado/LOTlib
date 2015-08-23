from utils import *
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
from LOTlib.Examples.FormalLanguageTheory.Model.Hypothesis import make_hypothesis
from LOTlib.Examples.FormalLanguageTheory.Language.AnBn import AnBn
import time
from mpi4py import MPI
import sys
from pickle import dump
import LOTlib
from LOTlib.Examples.Demo import standard_sample

fff = sys.stdout.flush
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

register_primitive(flatten2str)


def run(mk_hypothesis, get_data, size, finite):
    """
    This out on the DATA_RANGE amounts of data and returns all hypotheses in top count
    """
    if LOTlib.SIG_INTERRUPTED:
        return set()

    return standard_sample(lambda: mk_hypothesis(options.LANG, N=options.N),
                           lambda: get_data(size, max_length=finite),
                           N=options.TOP_COUNT,
                           steps=options.STEPS,
                           show=False, save_top=None)


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
    prefix = '../out/simulations/staged/'

    # ========================================================================================================
    # Running
    # ========================================================================================================
    language = AnBn()

    work_list = slice_list([[make_hypothesis, uniform_data if options.UNI else language.sample_data_as_FuncData,
                             12*(i+1), options.FINITE if options.MODE else 4*(1+i/4)] for i in xrange(12)], size)
    for e in work_list[rank]:
        topn = run(*e)
        dump(topn, open(prefix + ('normal' + str(options.UNI) if options.MODE else 'staged') + sq(e[2]) + sq(e[3]) + suffix,'w'))