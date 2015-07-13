# -*- coding: utf-8 -*-
"""
        Simple evaluation of number schemes -- read LOTlib.Performance.Evaluation to see what the output is
"""

import os
import re
from itertools import product
from optparse import OptionParser

import LOTlib
from LOTlib.MPI.MPI_map import MPI_map, get_rank
from LOTlib.Examples.ExampleLoader import load_example
from LOTlib.Performance.EvaluateSampler import evaluate_sampler

parser = OptionParser()
parser.add_option("--out", dest="OUT", type="string", help="Output prefix", default="output-InfereceSchemes")
parser.add_option("--samples", dest="SAMPLES", type="int", default=100000, help="Number of samples to run")
parser.add_option("--repetitions", dest="REPETITONS", type="int", default=100, help="Number of repetitions to run")
parser.add_option("--print-every", dest="PRINTEVERY", type="int", default=1000, help="Evaluation prints every this many")
parser.add_option("--models", dest="MODELS", type="str", default='SymbolicRegression.Galileo,Magnetism.Simple,Magnetism.Complex,RationalRules,RegularExpression,Number:100,Number:300,Number:1000,FOL', help="Which models do we run on?")
options, _ = parser.parse_args()


# Define all of the samplers
from LOTlib.Inference.Samplers.ParallelTempering import ParallelTemperingSampler
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.Inference.Samplers.ParticleSwarm import ParticleSwarmPriorResample
from LOTlib.Inference.Samplers.MultipleChainMCMC import MultipleChainMCMC
from LOTlib.Inference.Samplers.ParticleSwarm import ParticleSwarm
from LOTlib.Inference.Samplers.TabooMCMC import TabooMCMC
from LOTlib.Inference.EnumerationInference import EnumerationInference
# from LOTlib.Inference.Samplers.PartitionMCMC import PartitionMCMC

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_one(iteration, model, model2data, sampler_type):
    """
    Run one iteration of a sampling method
    """

    if LOTlib.SIG_INTERRUPTED: return

    # Take model and load the function to create hypotheses
    # Data is passed in to be constant across runs
    if re.search(r":", model):
        m, d = re.split(r":", model)
        make_hypothesis, _ = load_example(m)
    else:
        make_hypothesis, _ = load_example(model)


    h0 = make_hypothesis()
    grammar = h0.grammar
    data = model2data[model]

    # Create a sampler
    if sampler_type == 'mh_sample_A':               sampler = MHSampler(h0, data, options.SAMPLES,  likelihood_temperature=1.0)
    # elif sampler_type == 'mh_sample_B':             sampler = MHSampler(h0, data, options.SAMPLES,  likelihood_temperature=1.1)
    # elif sampler_type == 'mh_sample_C':             sampler = MHSampler(h0, data, options.SAMPLES,  likelihood_temperature=1.25)
    # elif sampler_type == 'mh_sample_D':             sampler = MHSampler(h0, data, options.SAMPLES,  likelihood_temperature=2.0 )
    # elif sampler_type == 'mh_sample_E':             sampler = MHSampler(h0, data, options.SAMPLES,  likelihood_temperature=5.0 )
    elif sampler_type == 'particle_swarm_A':        sampler = ParticleSwarm(make_hypothesis, data, steps=options.SAMPLES, within_steps=10)
    elif sampler_type == 'particle_swarm_B':        sampler = ParticleSwarm(make_hypothesis, data, steps=options.SAMPLES, within_steps=100)
    elif sampler_type == 'particle_swarm_C':        sampler = ParticleSwarm(make_hypothesis, data, steps=options.SAMPLES, within_steps=200)
    elif sampler_type == 'particle_swarm_prior_sample_A':        sampler = ParticleSwarmPriorResample(make_hypothesis, data, steps=options.SAMPLES, within_steps=10)
    elif sampler_type == 'particle_swarm_prior_sample_B':        sampler = ParticleSwarmPriorResample(make_hypothesis, data, steps=options.SAMPLES, within_steps=100)
    elif sampler_type == 'particle_swarm_prior_sample_C':        sampler = ParticleSwarmPriorResample(make_hypothesis, data, steps=options.SAMPLES, within_steps=200)
    elif sampler_type == 'multiple_chains_A':       sampler = MultipleChainMCMC(make_hypothesis, data, steps=options.SAMPLES, nchains=10)
    elif sampler_type == 'multiple_chains_B':       sampler = MultipleChainMCMC(make_hypothesis, data, steps=options.SAMPLES, nchains=100)
    elif sampler_type == 'multiple_chains_C':       sampler = MultipleChainMCMC(make_hypothesis, data, steps=options.SAMPLES, nchains=1000)
    elif sampler_type == 'parallel_tempering_A':    sampler = ParallelTemperingSampler(make_hypothesis, data, steps=options.SAMPLES, within_steps=10, temperatures=[1.0, 1.025, 1.05], swaps=1, yield_only_t0=False)
    elif sampler_type == 'parallel_tempering_B':    sampler = ParallelTemperingSampler(make_hypothesis, data, steps=options.SAMPLES, within_steps=10, temperatures=[1.0, 1.25, 1.5], swaps=1, yield_only_t0=False)
    elif sampler_type == 'parallel_tempering_C':    sampler = ParallelTemperingSampler(make_hypothesis, data, steps=options.SAMPLES, within_steps=10, temperatures=[1.0, 2.0, 5.0], swaps=1, yield_only_t0=False)
    elif sampler_type == 'taboo_A':                 sampler = TabooMCMC(h0, data, steps=options.SAMPLES, skip=0, penalty= 0.001)
    elif sampler_type == 'taboo_B':                 sampler = TabooMCMC(h0, data, steps=options.SAMPLES, skip=0, penalty= 0.010)
    elif sampler_type == 'taboo_C':                 sampler = TabooMCMC(h0, data, steps=options.SAMPLES, skip=0, penalty= 0.100)
    elif sampler_type == 'taboo_D':                 sampler = TabooMCMC(h0, data, steps=options.SAMPLES, skip=0, penalty= 1.000)
    elif sampler_type == 'taboo_E':                 sampler = TabooMCMC(h0, data, steps=options.SAMPLES, skip=0, penalty=10.000)
    # elif sampler_type == 'partitionMCMC_A':         sampler = PartitionMCMC(grammar, make_hypothesis, data, 10, steps=options.SAMPLES)
    # elif sampler_type == 'partitionMCMC_B':         sampler = PartitionMCMC(grammar, make_hypothesis, data, 100, steps=options.SAMPLES)
    # elif sampler_type == 'partitionMCMC_C':         sampler = PartitionMCMC(grammar, make_hypothesis, data, 1000, steps=options.SAMPLES)
    elif sampler_type == 'enumeration_A':           sampler = EnumerationInference(grammar, make_hypothesis, data, steps=options.SAMPLES)
    else: assert False, "Bad sampler type: %s" % sampler_type

    # And open our output and evaluate
    with open("output/out-aggregate.%s" % get_rank(), 'a') as out_aggregate:
        evaluate_sampler(sampler, trace=False, prefix="\t".join(map(str, [model, iteration, sampler_type])),
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


    # For each process, create the list of parameters
    params = [list(g) for g in product(range(options.REPETITONS),\
                                        models,
                                        [model2data],
                                        ['multiple_chains_A', 'multiple_chains_B', 'multiple_chains_C',
                                         'taboo_A', 'taboo_B', 'taboo_C', 'taboo_D',
                                         'particle_swarm_A', 'particle_swarm_B', 'particle_swarm_C',
                                         'particle_swarm_prior_sample_A', 'particle_swarm_prior_sample_B', 'particle_swarm_prior_sample_C',
                                         'mh_sample_A',#, 'mh_sample_B', 'mh_sample_C', 'mh_sample_D', 'mh_sample_E',
                                         'parallel_tempering_A', 'parallel_tempering_B', 'parallel_tempering_C',
                                         #'partitionMCMC_A', 'partitionMCMC_B', 'partitionMCMC_C', ## Super slow to make partitions
                                         'enumeration_A'])]

    # Actually run
    MPI_map(run_one, params)

