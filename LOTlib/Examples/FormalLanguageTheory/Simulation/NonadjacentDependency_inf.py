from utils import *
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
from LOTlib.Examples.FormalLanguageTheory.Model.Hypothesis import make_hypothesis
from LOTlib.Examples.FormalLanguageTheory.Language.LongDependency import LongDependency
import time
from mpi4py import MPI
from StagedInput_inf import sq
from pickle import dump

register_primitive(flatten2str)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if __name__ == '__main__':
    """
    run with script: mpiexec -n 12 python NonadjacentDependency_inf.py --steps=50000 --language=LongDependency --N=1 --mode=0/1
    """
    # ========================================================================================================
    # Process command line arguments
    # ========================================================================================================
    parser.add_option("--mode", dest="MODE", type="int", default=0, help="long or short")
    (options, args) = parser.parse_args()
    options.FINITE = 5
    suffix = time.strftime('_' + options.NAME + '_%m%d_%H%M%S', time.localtime())
    prefix = '../out/simulations/nonadjacent/'

    # ========================================================================================================
    # Running
    # ========================================================================================================
    language = LongDependency(C=['c', 'd', 'e', 'f'] if options.MODE else ['c', 'd'])
    work_list = [make_hypothesis, language.sample_data_as_FuncData, 144, options.FINITE, options]

    topn = run(*work_list)
    dump(topn, open(prefix + ('long' if options.MODE else 'short') + sq(rank) + suffix, 'w'))