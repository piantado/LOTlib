# -*- coding: utf-8 -*-
"""
This out the number model on the laptop or an MPI cluster.

This is the primary implementation intended for replication of Piantadosi, Tenebaum & Goodman
To install on my system, I had to build mpich2, mpi4py and set up ubuntu with the following:
        https://help.ubuntu.com/community/MpichCluster

To run on MPI:
$time mpiexec -hostfile /home/piantado/Libraries/LOTlib/hosts.mpich2 -n 36 python Run.py --steps=10000
              --top=50 --chains=25 --large=1000 --dmin=0 --dmax=300 --dstep=10 --mpi --out=/path/to/file.pkl

"""
import numpy
import sys
import LOTlib
from LOTlib import lot_iter
from LOTlib.MCMCSummary.TopN import TopN
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Miscellaneous import q, display_option_summary, qq
from LOTlib.MPI.MPI_map import MPI_unorderedmap, is_master_process
from LOTlib.Examples.Number.Model import *

## Parse command line options:
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--out",
                  dest="OUT_PATH", type="string", default="mpi-run.pkl",
                  help="Output file (a pickle of FiniteBestSet)")
parser.add_option("--steps",
                  dest="STEPS", type="int", default=200000,
                  help="Number of samples to run")
parser.add_option("--top",
                  dest="TOP_COUNT", type="int", default=100,
                  help="Top number of hypotheses to store")
parser.add_option("--chains",
                  dest="CHAINS", type="int", default=1,
                  help="Number of chains to run (new data set for each chain)")
parser.add_option("--large",
                  dest="LARGE_DATA_SIZE", type="int", default=100,
                  help="If > 0, recomputes the likelihood on a sample of data this size")

parser.add_option("--data",
                  dest="DATA", type="int",default=-1,
                  help="Amount of data")
parser.add_option("--dmin",
                  dest="DATA_MIN", type="int",default=20,
                  help="Min data to run")
parser.add_option("--dmax",
                  dest="DATA_MAX", type="int", default=500,
                  help="Max data to run")
parser.add_option("--dstep",
                  dest="DATA_STEP", type="int", default=20,
                  help="Step size for varying data")

# standard options
parser.add_option("-q",
                  "--quiet", action="store_true", dest="QUIET", default=False,
                  help="Don't print status messages to stdout")

(options, args) = parser.parse_args()

if options.DATA == -1:
    options.DATA_AMOUNTS = range(options.DATA_MIN, options.DATA_MAX, options.DATA_STEP)
else:
    options.DATA_AMOUNTS = [options.DATA]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# the main sampling function to run

def run(data_size):
    """
    This out on the DATA_RANGE amounts of data and returns *all* hypothese in the top options.TOP_COUNT
    """

    if LOTlib.SIG_INTERRUPTED: return TopN()  # So we don't waste time making data for everything that isn't run

    # initialize the data
    data = generate_data(data_size)

    # starting hypothesis -- here this generates at random
    h0 = Utilities.make_h0()

    hyps = TopN(N=options.TOP_COUNT)
    
    hyps.add(lot_iter(MHSampler(h0, data, options.STEPS, trace=False)))

    return hyps


if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Main running

    if is_master_process():
        display_option_summary(options)
        huge_data = generate_data(options.LARGE_DATA_SIZE)

    # choose the appropriate map function
    argarray = map(lambda x: [x], options.DATA_AMOUNTS * options.CHAINS)

    seen = set()
    for fs in MPI_unorderedmap(run, numpy.random.permutation(argarray)):
        for h in fs.get_all():
            if h not in seen:
                seen.add(h)
                h.compute_posterior(huge_data)

                if h.prior > float("-inf"):
                    print h.prior, \
                        h.likelihood /float(options.LARGE_DATA_SIZE), \
                        q(get_knower_pattern(h)), \
                        qq(h)

            sys.stdout.flush()

    import pickle
    with open(options.OUT_PATH, 'w') as f:
        pickle.dump(seen, f)