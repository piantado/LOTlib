import sys
import codecs
import itertools
from optparse import OptionParser
from pickle import dump
import time
import numpy as np
import LOTlib
from LOTlib.Miscellaneous import display_option_summary
from LOTlib.Inference.Samplers.StandardSample import standard_sample
from LOTlib.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str, logsumexp, qq
from Model.Hypothesis import make_hypothesis
from Language.Index import instance
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
fff = sys.stdout.flush

register_primitive(flatten2str)


def run(mk_hypothesis, lang, size):
    """
    This out on the DATA_RANGE amounts of data and returns all hypotheses in top count
    """
    if LOTlib.SIG_INTERRUPTED:
        return set()

    return standard_sample(lambda: mk_hypothesis(options.LANG, N=options.N, rank=rank, terminals=options.TERMINALS, bound=options.BOUND),
                           lambda: lang.sample_data_as_FuncData(size),
                           N=options.TOP_COUNT,
                           steps=options.STEPS,
                           show=True, skip=200, save_top=None)


def simple_mpi_map(run, args):
    print 'rank: ', rank, 'running..'; fff()
    hypo_set = run(*(args[rank]))

    dump(hypo_set, open(prefix+'hypotheses_'+options.LANG+'_%i'%rank+suffix, 'w'))


if __name__ == "__main__":
    """
        example:
            mpiexec -n 12 python my_search_stp.py --language=An --finite=10 --N=1 --bound=15
            mpiexec -n 12 python my_search_stp.py --language=AnBn --finite=20 --N=1 --terminal=b --bound=15
            mpiexec -n 12 python my_search_stp.py --language=ABn --finite=20?? --N=1 --terminal=b --bound=15
            mpiexec -n 12 python my_search_stp.py --language=AnB2n --finite=30 --N=1 --terminal=b --bound=15
            mpiexec -n 12 python my_search_stp.py --language=AnBnCn --finite=18 --N=3 --terminal=bc --bound=5
            mpiexec -n 12 python my_search_stp.py --language=Dyck --finite=8 --N=2 --terminal=b --bound=7 --steps=100000
            mpiexec -n 12 python my_search_stp.py --language=SimpleEnglish --finite=8 --N=3 --bound=5 --steps=100000
    """
    # ========================================================================================================
    # Process command line arguments /
    # ========================================================================================================
    fff = sys.stdout.flush
    parser = OptionParser()
    parser.add_option("--language", dest="LANG", type="string", default='An', help="name of a language")
    parser.add_option("--steps", dest="STEPS", type="int", default=40000, help="Number of samples to run")
    parser.add_option("--top", dest="TOP_COUNT", type="int", default=20, help="Top number of hypotheses to store")
    parser.add_option("--finite", dest="FINITE", type="int", default=10, help="specify the max_length to make language finite")
    parser.add_option("--name", dest="NAME", type="string", default='', help="name of file")
    parser.add_option("--N", dest="N", type="int", default=3, help="number of inner hypotheses")
    parser.add_option("--terminal", dest="TERMINALS", type="string", default='', help="extra terminals")
    parser.add_option("--bound", dest="BOUND", type="int", default=5, help="recursion bound")
    (options, args) = parser.parse_args()

    prefix = 'out/'
    # prefix = '/home/lijm/WORK/yuan/lot/'
    suffix = time.strftime('_' + options.NAME + '_%m%d_%H%M%S', time.localtime())

    sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    if rank == 0: display_option_summary(options); fff()

    DATA_RANGE = np.arange(0, 70, 6)

    language = instance(options.LANG, options.FINITE)
    args = list(itertools.product([make_hypothesis], [language], DATA_RANGE))

    hypotheses = simple_mpi_map(run, args)