from math import factorial, log
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from LOTlib.Miscellaneous import logsumexp, logplusexp, Infinity, log1mexp
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Inference.PriorSample import prior_sample
import Grammar as G, Hypothesis


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~ Generate number set hypotheses
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def normalizing_constant(hypotheses):
    """Estimate normalizing constant Z.

    Calculated as logsumexp(posterior scores for all hypotheses).
    """
    return logsumexp([h.posterior_score for h in hypotheses])


def make_h0(grammar=G.grammar, **kwargs):
    """Default make initial hypothesis method.

    Return type: NumberGame.Hypothesis.NumberSetHypothesis
    """
    return Hypothesis.NumberSetHypothesis(grammar, **kwargs)


def mh_sample(data, grammar=G.grammar, num_iters=10000, alpha=0.9):
    """Generate new hypotheses by sampling w/ metro hastings."""
    hypotheses = set()
    h0 = make_h0(grammar, alpha=alpha)
    for h in MHSampler(h0, data, steps=num_iters):
        #print h        # with this you can see how hill climbing moves towards maxima
        hypotheses.add(h)
    return hypotheses


def random_sample(grammar, input_data, n=10000, alpha=0.9):
    """Randomly sample `n` hypotheses from the grammar using prior_sample generator."""
    hypotheses = set()
    h0 = make_h0(grammar, alpha=alpha)
    for h in prior_sample(h0, input_data, n):
        hypotheses.add(h)
    return hypotheses


def random_sample_old(grammar, input_data, n=10000, alpha=0.9):
    """Randomly sample `n` hypotheses from the grammar -- old version."""
    hypotheses = set()
    for i in xrange(n):
        t = Hypothesis.NumberSetHypothesis(grammar, alpha=alpha)
        print '%'*70 + '\n'
        print str(t) + '\n'                 # Show the last hypothesis we tried
        t.compute_posterior(input_data)     # Compute prior, likelihood, posterior
        hypotheses.add(t)
    return hypotheses


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~ Infer grammar rule probabilities with human data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def prob_data_rule(grammar, rule, data, p, num_iters=10000, alpha=0.9):
    """Return the probabilities of set of data given a single p value for a rule."""
    orig_p = rule.p
    rule.p = p
    p_human = 0
    for d in data:
        # get probability of producing this data pair, add to total
        p_human_d = prob_data(grammar, d[0], d[1], num_iters, alpha)
        p_human = logplusexp(p_human, p_human_d)
    rule.p = orig_p
    return p_human


def probs_data_rule(grammar, rule, data, probs=np.arange(0, 2, 0.2), num_iters=10000, alpha=0.9):
    """Return the probabilities of set of data given distribution of probabilities for a given rule

    Args:
        grammar (LOTlib.Grammar): The grammar.
        rule (LOTlib.GrammarRule): Specify a specific rule of which to vary the probability. Use get_rule
            to get the GrammarRule for a name, e.g. 'union_'.

    Returns:
        list: Probability of human data for each value in `probs`.

    Example:
        >> data = [([2, 8, 16], {4: (10, 2), 6: (4, 8), 12: (7, 5)}),      # (data set 1)
        ..         ([3, 9, 13], {6: (11, 1), 5: (3, 9), 12: (8, 4)})]      # (data set 2)
        >> probHDataGivenRuleProbs(G.grammar, 'SET', 'union_', data, probs=[0.,1.,2.,3.,4.])
        [-0.923, -2.48, -5.12, -0.44, -6.36]

    """
    dist = []
    orig_p = rule.p
    for p in probs:
        rule.p = p
        p_human = 0
        for d in data:
            # get probability of producing this data pair, add to total
            p_d = prob_data(grammar, d[0], d[1], num_iters, alpha)
            p_human = logplusexp(p_human, p_d)
            print '%'*50
            print p_human
        dist.append(p_human)
        print '!'*70
    rule.p = orig_p
    return dist


def prob_data(grammar, input_data, output_data, num_iters=10000, alpha=0.9):
    """Compute the probability of generating human data given our grammar & input data.

    Args:
        grammar (LOTlib.Grammar): The grammar.
        input_data (list): List of numbers, the likelihood of the model is initially computed with these.
        output_data (list): List of tuples corresponding to (# yes, # no) responses in human data.

    Returns:
         float: Estimated probability of generating human data.

    """
    model_likelihoods = likelihood_data(grammar, input_data, output_data, num_iters, alpha)
    p_output = 0

    for o in output_data.keys():
        p = model_likelihoods[o]
        k = output_data[o][0]       # num. yes responses
        n = k + output_data[o][1]   # num. trials
        bc = factorial(n) / (factorial(k) * factorial(n-k))             # binomial coefficient
        p_o = log(bc) + (k*p) + (n-k)*log1mexp(p)                       # log version
        p_output = logplusexp(p_output, p_o)
        # p_gen_human_data[o] = bc * pow(p, k) * pow(1-p, n-k)          # linear version

    return p_output


def likelihood_data(grammar, input_data, output_data, num_iters=10000, alpha=0.9):
    """Generate a set of hypotheses, and use these to estimate likelihood of generating the human data.

    This is taken as a weighted sum over all hypotheses.

    Returns:
        dict: Each output key returns the summed likelihood of that single data point. Keys are the same as
        those of argument `output_data`.

    """
    hypotheses = mh_sample(input_data, grammar=grammar, num_iters=num_iters, alpha=alpha)
    Z = normalizing_constant(hypotheses)
    likelihoods = defaultdict(lambda: -Infinity)

    for h in hypotheses:
        w = h.posterior_score - Z
        for o in output_data.keys():                # TODO: is this loop updating posterior_score each time?
            old_likelihood = h.likelihood
            weighted_likelihood = h.compute_likelihood([o]) + w
            h.likelihood = old_likelihood
            likelihoods[0] = logplusexp(likelihoods[o], weighted_likelihood)

    return likelihoods


def get_rule(rule_name, rule_nt=None, grammar=G.grammar):
    """Return the GrammarRule associated with this rule name."""
    if rule_nt is None:
        rules = [rule for sublist in grammar.rules.values() for rule in sublist]
    else:
        rules = grammar.rules[rule_nt]
    rules = filter(lambda r: (r.name == rule_name), rules)   # there should only be 1 item in this list
    return rules[0]


def visualize_probs(probs, dist, rule_name='RULE_'):
    """Visualize results from probs_data_rule."""
    fig, ax = plt.subplots()
    rects = plt.bar(probs, dist)

    plt.xlabel('Grammar Rule Probability')
    plt.ylabel('Pr. of human data')
    plt.title('Prob. of human data given prob. for rule: '+rule_name)
    plt.show()


# # Print top 10 hypotheses
# def printBestHypotheses(hypotheses, n=10):
#     best_hypotheses = sorted(hypotheses, key=lambda x: x.posterior_score)
#     best_hypotheses.reverse()
#     Z = normalizingConstant(hypotheses)
#
#     print 'printing top '+str(n)+' hypotheses\n'
#     print '================================================================'
#     for i in range(n):
#         h = best_hypotheses[i].__call__()
#         print (i+1), ' ~\t', 'Hypothesis:\t', str(h)
#         print '\tPosterior:\t%.5f' % exp(h.posterior_score - Z)
#         print '\t\tPrior:\t\t%.3f' % h.prior
#         print '\t\tLikehd:\t\t%.3f' % h.likelihood
#         print '================================================================\n'
