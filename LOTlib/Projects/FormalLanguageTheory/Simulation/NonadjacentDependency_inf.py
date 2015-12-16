from utils import *
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
from LOTlib.Examples.FormalLanguageTheory.Model.Hypothesis import make_hypothesis
from LOTlib.Examples.FormalLanguageTheory.Language.LongDependency import LongDependency
import time
from mpi4py import MPI
from StagedInput_inf import sq
from pickle import dump
import sys
import codecs

register_primitive(flatten2str)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def no_data(a, max_length=None):
    return [FunctionData(input=[], output=Counter())]

if __name__ == '__main__':
    """
    run with script: mpiexec -n 12 python NonadjacentDependency_inf.py --steps=50000 --language=LongDependency --N=1 --mode=0/1
    """
    # ========================================================================================================
    # Process command line arguments
    # ========================================================================================================
    parser.add_option("--mode", dest="MODE", type="int", default=0, help="long or short")
    (options, args) = parser.parse_args()

    suffix = time.strftime('_' + options.NAME + '_%m%d_%H%M%S', time.localtime())
    prefix = './'
    # prefix = '../out/simulations/nonadjacent/'
    # prefix = '/home/lijm/WORK/yuan/lot/'

    # ========================================================================================================
    # Running
    # ========================================================================================================
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    work_list = [make_hypothesis, 144, np.arange(2, 12, 2)[rank], options, None, ['c', 'd', 'e', 'f', 'h', 'i', 'j', 'k']]
    # work_list = [make_hypothesis, 144, 2, options, no_data, ['c', 'd', 'e', 'f']]

    topn = run(*work_list)
    dump(topn, open(prefix + sq(rank) + suffix, 'w'))