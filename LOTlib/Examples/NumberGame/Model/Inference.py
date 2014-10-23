from math import factorial, log
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from LOTlib.Examples.NumberGame.Model import Grammar as G
from LOTlib.Miscellaneous import logsumexp, exp, logplusexp, Infinity, log1mexp
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Inference.PriorSample import prior_sample
import Hypothesis


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~ Generate hypotheses                                                         ~~~~~#

# Generate new hypotheses by sampling w/ metro hastings
def mhSample(data, grammar=G.grammar, num_iters=10000, alpha=0.9):
    hypotheses = set()
    h0 = make_h0(grammar, alpha=alpha)
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


def make_h0(grammar=G.grammar, **kwargs):
    return Hypothesis.NumberSetHypothesis(grammar, **kwargs)


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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~ Infer grammar rule probabilities with human data                            ~~~~~#

# return likelihood of generating data given a grammar, summed over all hypotheses generated
# ==> returns a dictionary with each output key returning the summed likelihood of that single data point
def likelihoodGivenGrammar(grammar, input_data, output_data, num_iters=10000, alpha=0.9):
    hypotheses = mhSample(input_data, grammar=grammar, num_iters=num_iters, alpha=alpha)
    Z = normalizingConstant(hypotheses)

    likelihoods = defaultdict(lambda: -Infinity)
    for h in hypotheses:
        w = h.posterior_score - Z
        for o in output_data.keys():                # TODO: is this loop updating posterior_score each time?
            old_likelihood = h.likelihood
            weighted_likelihood = h.compute_likelihood([o]) + w
            h.likelihood = old_likelihood
            likelihoods[0] = logplusexp(likelihoods[o], weighted_likelihood)
    return likelihoods


# for fixed grammar and model parameters (e.g. for a fixed model you could import) compute the match to human data
def probabilityOfHumanData(grammar, input_data, output_data, num_iters=10000, alpha=0.9):
    model_likelihoods = likelihoodGivenGrammar(grammar, input_data, output_data, num_iters, alpha)

    p_gen_human_data = {}
    for o in output_data.keys():
        p = model_likelihoods[o]
        k = output_data[o][0]       # num. yes responses
        n = k + output_data[o][1]   # num. trials
        bc = factorial(n) / (factorial(k) * factorial(n-k))   # binomial coefficient
        p_gen_human_data[o] = log(bc) + (k*p) + (n-k)*log1mexp(p)       # log version
        # p_gen_human_data[o] = bc * pow(p, k) * pow(1-p, n-k)          # linear version

    return p_gen_human_data


# return the distribution of probability of human data given dist. of probabilities for this rule
def probHDataGivenRuleProbs(grammar, rule_nt, rule_name, input_data, output_data,
                            probs=np.arange(0, 2, 0.2), num_iters=10000, alpha=0.9):
    nt_rules = grammar.rules[rule_nt]
    name_rules = filter(lambda r: (r.name == rule_name), nt_rules)   # there should only be 1 item in this list
    rule = name_rules[0]
    dist = []
    orig_p = rule.p
    for p in probs:
        rule.p = p
        dist.append(probabilityOfHumanData(grammar, input_data, output_data, num_iters, alpha))

    rule.p = orig_p
    return dist


# visualize results from probHDataGivenRule
def visualizeRuleProbs(probs, dist, rule_name='RULE_'):
    fig, ax = plt.subplots()
    rects = plt.bar(probs, dist)

    plt.xlabel('Grammar Rule Probability')
    plt.ylabel('Pr. of human data')
    plt.title('Prob. of human data given prob. for rule: '+rule_name)
    plt.show()
