import sys
import codecs
from cog_test import make_hypothesis, is_valid, gen_rotate_data_zoom
import LOTlib
from LOTlib.Miscellaneous import display_option_summary
from LOTlib.MPI.MPI_map import is_master_process
from optparse import OptionParser
from LOTlib.Examples.Demo import standard_sample
from mpi4py import MPI
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
import numpy as np

register_primitive(flatten2str)


def slice_list(input, size):
    input_size = len(input)
    slice_size = input_size / size
    remain = input_size % size
    result = []
    iterator = iter(input)
    for i in xrange(size):
        result.append([])
        for j in xrange(int(slice_size)):
            result[i].append(iterator.next())
        if remain:
            result[i].append(iterator.next())
            remain -= 1
    return result


def run(make_hypothesis, make_data, data_size):
    """
    This out on the DATA_RANGE amounts of data and returns all hypotheses in top count
    """
    if LOTlib.SIG_INTERRUPTED:
        return set()

    return standard_sample(make_hypothesis,
                           lambda: make_data(data_size),
                           N=options.TOP_COUNT,
                           steps=options.STEPS,
                           show=False, save_top=None)


if __name__ == "__main__":

    # ========================================================================================================
    # Process command line arguments
    # ========================================================================================================
    fff = sys.stdout.flush
    parser = OptionParser()
    parser.add_option("--steps", dest="STEPS", type="int", default=2000, help="Number of samples to run")
    parser.add_option("--top", dest="TOP_COUNT", type="int", default=100, help="Top number of hypotheses to store")
    parser.add_option("--data", dest="DATA", type="int", default=5, help="Amount of data")
    parser.add_option("--inf_len", dest="LEN", type="int", default=16, help="length at which we decide whether hypothesis is infinite")
    (options, args) = parser.parse_args()

    if is_master_process():
        display_option_summary(options); fff()

    # set the output codec -- needed to display lambda to stdout
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank != 0:
        # ========================================================================================================
        # worker
        # ========================================================================================================
        for data_size in xrange(31):
            print '%i samples on data_size %i' % (rank, data_size);fff()

            topN = run(make_hypothesis, gen_rotate_data_zoom, data_size)

            seen = set()
            out = []
            cnt = 0
            for h in topN:
                if h not in seen:
                    seen.add(h)
                    out.append([0, h])
                    for i in xrange(10000):
                        str_gen = h()
                        if str_gen is not None and len(str_gen) > options.LEN:
                            out[-1][0] = 1
                            cnt += 1
                            break
            print rank, data_size, cnt
            comm.send(out, dest=0)
            comm.barrier()
    else:
        # ========================================================================================================
        # master
        # ========================================================================================================
        for data_size in xrange(31):

            out = []
            for i in xrange(1, size):
                out += comm.recv(source=i)

            data = gen_rotate_data_zoom(data_size)
            all_sum = 0
            inf_sum = 0
            prob = 0
            seen = set()
            for flag, h in out:
                if h not in seen:
                    seen.add(h)
                    poster_mass = h.compute_posterior(data)
                    if not is_valid(poster_mass): continue
                    poster_mass = np.exp(poster_mass)
                    all_sum += poster_mass
                    if flag:
                        inf_sum += poster_mass

            if all_sum != 0: prob = inf_sum / all_sum
            else: prob = 0
            print '******************* data_size: %i, prob of inf %.5f *****************' % (data_size, prob)
            comm.barrier()