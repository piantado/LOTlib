# -*- coding: utf-8 -*-
"""
        Simple evaluation of number schemes -- read LOTlib.Performance.Evaluation to see what the output is
"""

import re
import os
from itertools import product
from optparse import OptionParser

import LOTlib
from LOTlib.MPI.MPI_map import MPI_map, get_rank
from LOTlib.Examples.ExampleLoader import load_example
from LOTlib.Performance.EvaluateSampler import evaluate_sampler
from LOTlib.Inference.Samplers.MultipleChainMCMC import MultipleChainMCMC

parser = OptionParser()
parser.add_option("--out", dest="OUT", type="string", help="Output prefix", default="output")
parser.add_option("--samples", dest="SAMPLES", type="int", default=1000000, help="Samples to run (total)")
parser.add_option("--repetitions", dest="REPETITONS", type="int", default=100, help="Number of repetitions to run")
parser.add_option("--print-every", dest="PRINTEVERY", type="int", default=1000, help="Evaluation prints every this many")
parser.add_option("--models", dest="MODELS", type="str", default='SymbolicRegression.Galileo,Magnetism.Simple,Magnetism.Complex,RationalRules,RegularExpression,Number:100,Number:300,Number:1000,FOL', help="Which models do we run on?")
options, _ = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_one(iteration, model, model2data, llt):
    if LOTlib.SIG_INTERRUPTED: return

    # Take model and load the function to create hypotheses
    # Data is passed in to be constant across runs
    if re.search(r":", model):
        m, d = re.split(r":", model)
        make_hypothesis, _ = load_example(m)
    else:
        make_hypothesis, _ = load_example(model)

    with open(options.OUT+"/aggregate.%s" % get_rank(), 'a') as out_aggregate:
        evaluate_sampler(MultipleChainMCMC(make_hypothesis, model2data[model], steps=options.SAMPLES, nchains=1, likelihood_temperature=llt),
                         trace=False, prefix="\t".join(map(str, [model, iteration, llt])),
                         out_aggregate=out_aggregate, print_every=options.PRINTEVERY)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":



    # Parse the models from the input
    models = re.split(r',', options.MODELS)

    # define the data for each model.
    # Right now, we do this so that every iteration and method uses THE SAME data in order to minimize the variance
    model2data = dict()
    for model in models:
        if re.search(r":", model): # split model string by : to handle different amounts of data
            m, d = re.split(r":", model)
            _, make_data = load_example(m)
            model2data[model] = make_data(int(d))
        else:
            _, make_data = load_example(model)
            model2data[model] = make_data()


    # For each process, create the list of parameter
    params = [list(g) for g in product(range(options.REPETITONS),
                                       models, [model2data],
                                       [0.1, 0.5, 1.0, 2.0, 10.0])]

    MPI_map(run_one, params)

