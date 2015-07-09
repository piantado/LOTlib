# -*- coding: utf-8 -*-
"""
    Simple evaluation of proposal schemes. Test various mixtures of them.
"""
import re
from itertools import product
from optparse import OptionParser

import LOTlib
from LOTlib.MPI.MPI_map import MPI_map, get_rank
from LOTlib.Miscellaneous import q
from LOTlib.Examples.ExampleLoader import load_example
from LOTlib.Performance.EvaluateSampler import evaluate_sampler

from LOTlib.Inference.Samplers.MultipleChainMCMC import MultipleChainMCMC
from LOTlib.Hypotheses.Proposers.MixtureProposer import MixtureProposer

parser = OptionParser()
parser.add_option("--out", dest="OUT", type="string", help="Output prefix", default="output")
parser.add_option("--samples", dest="SAMPLES", type="int", default=100000, help="Number of samples to run")
parser.add_option("--chains", dest="CHAINS", type="int", default=10, help="Number of chains to run in parallel")
parser.add_option("--repetitions", dest="REPETITONS", type="int", default=100, help="Number of repetitions to run")
parser.add_option("--print-every", dest="PRINTEVERY", type="int", default=1000, help="Evaluation prints every this many")
parser.add_option("--models", dest="MODELS", type="str", default='SymbolicRegression.Galileo,Magnetism.Simple,Magnetism.Complex,RationalRules,RegularExpression,Number:100,Number:300,Number:1000,FOL', help="Which models do we run on?")
options, _ = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_one(iteration, model, model2data, probs):
    if LOTlib.SIG_INTERRUPTED: # do this so we don't create (big) hypotheses
        return

    # Take model and load the function to create hypotheses
    # Data is passed in to be constant across runs
    if re.search(r":", model):
        m, d = re.split(r":", model)
        make_hypothesis, _ = load_example(m)
    else:
        make_hypothesis, _ = load_example(model)

    htmp = make_hypothesis() # just use this to get the grammar

    # Make a new class to wrap our mixture in
    class WrappedClass(MixtureProposer, type(htmp)):
        pass

    # define a wrapper to set this proposal
    def wrapped_make_hypothesis(**kwargs):
        h = WrappedClass(**kwargs)
        print ">>", htmp, model,  h, kwargs
        h.set_proposal_probabilities(probs)
        return h

    sampler = MultipleChainMCMC(wrapped_make_hypothesis,  model2data[model], steps=options.SAMPLES, nchains=options.CHAINS)

    with open(options.OUT+"/aggregate.%s" % get_rank(), 'a') as out_aggregate:
        evaluate_sampler(sampler, trace=False, prefix="\t".join(map(str, [model, iteration, q(str(probs)) ])),
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
    params = map(list, product( range(options.REPETITONS),
                                models, [model2data],
                                [ (1., 1., 1.),
                                  (2., 1., 1.),
                                  (1., 2., 1.),
                                  (1., 1., 2.),
                                  (2., 2., 1.),
                                  (1., 2., 2.),
                                  (2., 1., 2.),
                                  (10., 1., 1.),
                                  (1., 10., 1.),
                                  (1., 1., 10.),
                                  (10., 10., 1.),
                                  (1., 10., 10.),
                                  (10., 1., 10.)
                                ]))

    MPI_map(run_one, params)


