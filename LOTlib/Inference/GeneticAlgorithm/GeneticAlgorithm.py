from random import randint
from LOTlib.Inference.Samplers.Sampler import MH_acceptance
from LOTlib.Hypotheses.Proposers import ProposalFailedException
from LOTlib.Hypotheses.Proposers.RegenerationProposal import RegenerationProposal
from LOTlib.Miscellaneous import lambdaOne, Infinity, weighted_sample
from LOTlib.Hypotheses.Hypothesis import Hypothesis
from LOTlib.FunctionNode import NodeSamplingException
from copy import copy

def crossover_lot(x,y):
    t = copy(x.value)
    n1, _ = t.sample_subnode(resampleProbability=lambdaOne)
    n2, _ = y.value.sample_subnode(resampleProbability=lambda t: 1*(t.returntype==n1.returntype))

    n1.setto(n2) # assign the value!

    return Hypothesis.__copy__(x, value=t)

def mutate_lot(x):
    h, _ = x.propose()
    return h

def genetic_algorithm(make_hypothesis, data, mutate, crossover, population_size=100, generations=100000):

    population = [make_hypothesis() for _ in xrange(population_size)]
    for h in population:
        h.compute_posterior(data)

    for g in xrange(generations):

        nextpopulation = []

        while len(nextpopulation) < population_size:
            # sample proportional to fitness
            mom = weighted_sample(population, probs=[v.posterior_score for v in population], log=True)
            dad = weighted_sample(population, probs=[v.posterior_score for v in population], log=True)

            try:
                kid = mutate(crossover(mom, dad))
            except (ProposalFailedException, NodeSamplingException):
                continue

            kid.compute_posterior(data)
            yield kid

            nextpopulation.append(kid)

            # # if MH_acceptance(population[i].posterior_score, kid.posterior_score, 0.0):
            # if kid.posterior_score > population[i].posterior_score:
            #     population[i] = kid
            #     yield kid
        population = nextpopulation

if __name__ == "__main__":
    from LOTlib import break_ctrlc
    from LOTlib.Examples.Number.Model import make_hypothesis, make_data
    from LOTlib.Miscellaneous import qq
    data = make_data(400)

    for h in break_ctrlc(genetic_algorithm(make_hypothesis, data, mutate_lot, crossover_lot)):
        print h.posterior_score, h.get_knower_pattern(), qq(h)



