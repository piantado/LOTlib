# -*- coding: utf-8 -*-
"""
A quick demo of the number model.

Note:

    CTRL-C breaks out of the MCMC loop, and the processes at the bottom with average likelihood for each
    hypothesis.

"""
import LOTlib
from LOTlib import lot_iter
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Inference.MetropolisHastings import MHSampler
import LOTlib.Inference.ParallelTempering
import LOTlib.Inference.TemperedTransitions
from LOTlib.Miscellaneous import q, qq
from LOTlib.Examples.Number.Model import *
from LOTlib.MCMCSummary.TopN import TopN

LARGE_DATA_SIZE = 10000 # this is what we compute the average LL on
DATA_SIZE = 300
TRACE = True
STEPS = 1000000
SKIP = 1

if __name__ == "__main__":

    # ========================================================================================================
    #  Generate some data
    data = generate_data(DATA_SIZE)


    # A starting hypothesis (later ones are created by .propose, called in LOTlib.MetropolisHastings
    h0 = NumberExpression(grammar)

    '''
    from LOTlib.Inference.Proposals.InsertDeleteProposal import InsertDeleteProposal
    h0 = NumberExpression(grammar, proposal_function=InsertDeleteProposal(grammar))
    '''

    # store hypotheses we've found
    allhyp = TopN(N=1000)

    # ========================================================================================================
    #  A bunch of different MCMC algorithms to try. mh_sample is from the Rational Rules paper and
    #   it generally works very well.

    # tempered_transitions = LOTlib.Inference.TemperedTransitions.tempered_transitions_sample(
    #     h0, data, 500000, skip=0, temperatures=[1.0, 1.25, 1.5])
    '''
    parallel_tempering = LOTlib.Inference.ParallelTempering.ParallelTemperingSampler(
        h0, data, STEPS, within_steps=10, yield_all=True, temperatures=[1.0,1.05, 1.1])
    '''
    mh_sampler = MHSampler(h0, data, STEPS, skip=SKIP)

    for h in lot_iter(mh_sampler):
        if TRACE:
            print q(get_knower_pattern(h)), h.posterior_score, h.compute_prior(), h.compute_likelihood(data), qq(h)

        # add h to our priority queue, with priority of its log probability, h.posterior_score
        allhyp.add(h)

    # ========================================================================================================
    #  now re-evaluate everything we found on new data
    '''
    huge_data = generate_data(LARGE_DATA_SIZE)

    save this with a huge data set -- eval with average ll
    H = allhyp.get_sorted()

    compute the posterior for each hypothesis
    [ h.compute_posterior(huge_data) for h in H]

    show the *average* ll for each hypothesis, at this data size
    for h in H:
        print h.prior, h.likelihood/float(LARGE_DATA_SIZE), q(get_knower_pattern(h)),  q(h) # a quoted x
    '''
