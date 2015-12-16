"""
Run an example on MPI, saving a bunch of chains and data amounts.

NOTE: Not all examples take data amounts as an argument.

e.g. to run on MPI:
$time mpiexec -n 36 python Run.py
"""

import numpy
import itertools

import sys
import codecs

import LOTlib
from LOTlib.Miscellaneous import q, display_option_summary, qq
from LOTlib.FunctionNode import cleanFunctionNodeString
from LOTlib.MPI.MPI_map import MPI_unorderedmap, is_master_process

from LOTlib.Inference.Samplers.StandardSample import standard_sample

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# the main sampling function to run

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
                           show=False,save_top=None)


if __name__ == "__main__":

    # ========================================================================================================
    # Process command line arguments
    # ========================================================================================================

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--out", dest="OUT_PATH", type="string", default="mpi-run.pkl", help="Output file (a pickle of FiniteBestSet)")
    parser.add_option("--steps", dest="STEPS", type="int", default=100000, help="Number of samples to run")
    parser.add_option("--top", dest="TOP_COUNT", type="int", default=100, help="Top number of hypotheses to store")
    parser.add_option("--chains", dest="CHAINS", type="int", default=1, help="Number of chains to run (new data set for each chain)")
    parser.add_option("--data", dest="DATA", type="int",default=1, help="Amount of data")
    parser.add_option("--dmin", dest="DATA_MIN", type="int",default=0, help="Min data to run")
    parser.add_option("--dmax", dest="DATA_MAX", type="int", default=0, help="Max data to run")
    parser.add_option("--dstep", dest="DATA_STEP", type="int", default=0, help="Step size for varying data")
    parser.add_option("--evaldata", dest="EVAL_DATA", type="int", default=1000, help="If specified, we'll print everything evaled on this amount.")
    parser.add_option("--model", dest="MODEL", type="string", default="Number", help="Which model do we run? (e.g. 'Number', 'Magnetism.Simple', etc.")
    parser.add_option("--alsoprint", dest="ALSO_PRINT", type="string", default="None",
                      help="A function of a hypothesis we can also print at the start of a line to see things we "
                           "want. E.g. --alsoprint='lambda h: h.get_knower_pattern()' ")
    (options, args) = parser.parse_args()

    alsoprint = eval(options.ALSO_PRINT)

    if options.DATA_STEP > 0:
        data_amounts = range(options.DATA_MIN, options.DATA_MAX, options.DATA_STEP)
    else:
        assert options.DATA > 0, "*** Must specify either data or dmin, dmax, dstep"
        data_amounts = [options.DATA]

    # ========================================================================================================
    # Load the model specified on the command line
    # ========================================================================================================

    from LOTlib.Examples import load_example

    make_hypothesis, make_data = load_example(options.MODEL)

    # ========================================================================================================
    # Run, MPI friendly
    # ========================================================================================================

    if is_master_process():
        display_option_summary(options)

        eval_data = None
        if options.EVAL_DATA > 0:
            eval_data = make_data(options.EVAL_DATA)


    # choose the appropriate map function
    args = list(itertools.product([make_hypothesis],[make_data], data_amounts * options.CHAINS) )

    # set the output codec -- needed to display lambda to stdout
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)

    seen = set()
    for fs in MPI_unorderedmap(run, numpy.random.permutation(args)):
        assert is_master_process()

        for h in fs:

            if h not in seen:
                seen.add(h)

                if eval_data is not None:
                    h.compute_posterior(eval_data) # evaluate on the big data
                    print h.posterior_score, h.prior, h.likelihood / options.EVAL_DATA, \
                            alsoprint(h) if alsoprint is not None else '',\
                            qq(cleanFunctionNodeString(h))


    import pickle
    with open(options.OUT_PATH, 'w') as f:
        pickle.dump(seen, f)

