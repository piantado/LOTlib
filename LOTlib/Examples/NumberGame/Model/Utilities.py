
import numpy as np
import matplotlib.pyplot as plt
from LOTlib import MHSampler
from LOTlib.Miscellaneous import logsumexp, logplusexp
import Grammar as G, Hypothesis


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~ Generate number set hypotheses
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def normalizing_constant(hypotheses):
    """Estimate normalizing constant Z by logsumexp(posterior scores for all hypotheses)."""
    return logsumexp([h.posterior_score for h in hypotheses])


def make_h0(grammar=G.grammar, **kwargs):
    """Make initial NumberGameHypothesis."""
    return Hypothesis.NumberGameHypothesis(grammar, **kwargs)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~ Infer grammar rule probabilities with human data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# h0 = make_h0(grammar, alpha=alpha)
# trees = set(MHSampler(h0, data, steps=num_iters))
# h = Hypothesis.GrammarProbHypothesis(grammar, trees)



def visualize_probs(probs, dist, rule_name='RULE_'):
    """Visualize results from probs_data_rule."""
    fig, ax = plt.subplots()
    rects = plt.bar(probs, dist)

    plt.xlabel('Grammar Rule Probability')
    plt.ylabel('Pr. of human data')
    plt.title('Prob. of human data given prob. for rule: '+rule_name)
    plt.show()








#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~ scrap
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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

def get_rule(grammar, rule_name, rule_nt=None):
        """Get the GrammarRule associated with this rule name."""
        if rule_nt is None:
            rules = [rule for sublist in grammar.rules.values() for rule in sublist]
        else:
            rules = grammar.rules[rule_nt]
        rules = filter(lambda r: (r.name == rule_name), rules)   # there should only be 1 item in this list
        return rules[0]