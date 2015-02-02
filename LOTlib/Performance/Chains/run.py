# -*- coding: utf-8 -*-
"""
        Simple evaluation of number schemes -- read LOTlib.Performance.Evaluation to see what the output is
"""

import LOTlib
import re
from LOTlib.Performance.Evaluation import load_model
from itertools import product
from LOTlib.MPI.MPI_map import MPI_map, get_rank

from optparse import OptionParser

parser = OptionParser()
parser.add_option("--out", dest="OUT", type="string", help="Output prefix", default="output-NumberOfChains")
parser.add_option("--samples", dest="SAMPLES", type="int", default=10000, help="Samples to run (total)")
parser.add_option("--repetitions", dest="REPETITONS", type="int", default=100, help="Number of repetitions to run")
parser.add_option("--print-every", dest="PRINTEVERY", type="int", default=1000, help="Evaluation prints every this many")
parser.add_option("--models", dest="MODELS", type="str", default='Number2015:100,Number2015:300', help="Which models do we run on?")
options, _ = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# These get defined for each process
from LOTlib.Performance.Evaluation import evaluate_sampler
from LOTlib.Inference.MultipleChainMCMC import MultipleChainMCMC

def run_one(iteration, model, chains):
    if LOTlib.SIG_INTERRUPTED: # do this so we don't create (big) hypotheses
        return

    data, make_h0 = load_model(model)

    h0 = make_h0()
    sampler = MultipleChainMCMC(make_h0, data, steps=options.SAMPLES, nchains=chains)

    with open(options.OUT+"/aggregate.%s" % get_rank(), 'a') as out_aggregate:
        with open(options.OUT+"/hypotheses.%s" % get_rank(),'a')  as out_hypotheses:
            # Run evaluate on it, printing to the right locations
            evaluate_sampler(sampler, trace=False, prefix="\t".join(map(str, [model, iteration, chains])),
                             out_hypotheses=out_hypotheses, out_aggregate=out_aggregate, print_every=options.PRINTEVERY)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create all parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# For each process, create the lsit of parameter
params = [list(g) for g in product(range(options.REPETITONS),\
                                    re.split(r',', options.MODELS),
                                    [1, 10, 50, 100, 500, 1000, 5000])]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Actually run
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MPI_map(run_one, params, random_order=False)

