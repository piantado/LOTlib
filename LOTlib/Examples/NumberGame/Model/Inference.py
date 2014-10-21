from math import factorial, log
from collections import defaultdict
import Hypothesis
from LOTlib.Examples.NumberGame.Model import Grammar
from LOTlib.Miscellaneous import logsumexp, exp, logplusexp, Infinity, log1mexp
from LOTlib.Inference.MetropolisHastings import MHSampler

from LOTlib.Inference.PriorSample import prior_sample


def mhSample(grammar, data, num_iters=10000, alpha=0.9):
    hypotheses = set()
    h0 = make_h0(alpha=alpha)
    for h in MHSampler(h0, data, steps=num_iters):
        #print h        # with this you can see how hill climbing moves towards maxima
        hypotheses.add(h)
    return hypotheses


# Estimate normalizing constant Z
def normalizingConstant(hypotheses):
    return logsumexp([h.posterior_score for h in hypotheses])


# Print top 10 hypotheses
def printBestHypotheses(hypotheses, n=10):
    best_hypotheses = sorted(hypotheses, key=lambda x: x.posterior_score)
    best_hypotheses.reverse()
    Z = normalizingConstant(hypotheses)

    print 'printing top '+str(n)+' hypotheses\n'
    print '================================================================'
    for i in range(n):
        h = best_hypotheses[i].__call__()
        print (i+1), ' ~\t', 'Hypothesis:\t', str(h)
        print '\tPosterior:\t%.5f' % exp(h.posterior_score - Z)
        print '\t\tPrior:\t\t%.3f' % h.prior
        print '\t\tLikehd:\t\t%.3f' % h.likelihood
        print '================================================================\n'


def make_h0(**kwargs):
    return Hypothesis.NumberSetHypothesis(Grammar.grammar, **kwargs)


def randomSample(grammar, data, num_iters=10000, alpha=0.9):
    hypotheses = set()
    for i in xrange(num_iters):
        t = Hypothesis.NumberSetHypothesis(grammar, alpha=alpha)
        print '%'*70+'\n'
        print t
        print '#'*70+'\n'
        t.compute_posterior(data) # Get prior, likelihood, posterior
        hypotheses.add(t)
    return hypotheses


#######################################################################################################################
# Inference with human data                                                                                           #
#######################################################################################################################
'''
# maps output number (e.g. 8) to a number of yes/no's (e.g. [10/2] )
human_in_data = [2, 4, 6]
human_out_data = {
    8: (10, 2),
    12: (5, 7),
    14: (8, 4)
}
'''


# TODO: make this generate some hypotheses
def generateHypotheses(grammar, input_data):
    return set()


# return likelihood of generating data given a grammar, summed over all hypotheses generated
# ==> returns a dictionary with each output key returning the summed likelihood of that single data point
def likelihoodGivenGrammar(grammar, input_data, output_data):
    hypotheses = generateHypotheses(grammar, input_data)
    Z = normalizingConstant(hypotheses)

    likelihoods = defaultdict(lambda: -Infinity)
    for h in hypotheses:
        w = h.posterior_score - Z
        for o in output_data.keys():
            weighted_likelihood = h.compute_likelihood(o) + w
            likelihoods[0] = logplusexp(likelihoods[o], weighted_likelihood)
    return likelihoods


# for fixed grammar and model parameters (e.g. for a fixed model you could import) compute the match to human data
def probabilityOfHumanData(grammar, input_data, output_data):
    model_likelihoods = likelihoodGivenGrammar(grammar, input_data, output_data)

    p_gen_human_data = {}
    for o in output_data.keys():
        p = model_likelihoods[o]
        k = output_data[o][0]       # num. yes responses
        n = k + output_data[o][1]   # num. trials
        bc = factorial(n) / (factorial(k) * factorial(n-k))   # binomial coefficient
        p_gen_human_data[o] = log(bc) + (k*p) + (n-k)*log1mexp(p)       # log version
        # p_gen_human_data[o] = bc * pow(p, k) * pow(1-p, n-k)          # linear version

    return p_gen_human_data

