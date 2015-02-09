

from LOTlib.MCMCSummary.VectorSummary import *


def sample_grammar_hypotheses(sampler, skip, cap, print_=False):
    summary = VectorSummary(skip=skip, cap=cap)
    if print_:
        i = 0
        print '^*'*60, '\nGenerating GrammarHypothesis Samples\n', '^*'*60
        for h in summary(sampler):
            i += 1
            if i % (sampler.steps/20) == 0:
                print ['%.3f' % v for v in h.value]
                print i, '-'*100
                print h.prior, h.likelihood, h.posterior_score
    else:
        for h in summary(sampler):
            pass
    return summary


def print_dist(vals, posteriors):
    print '@'*120
    for val, post in zip(vals, posteriors):
        print val, '\t', post
    print '@'*120


# ============================================================================================================


def visualize_dist(probs, dist, rule_name='RULE_'):
    """Visualize results from VectorHypothesis.conditional_distribution.

    The x-axis is the prior values we're inputting for the given rule (probs).
    The y-axis is the summed posterior of the human data given these prior values

    """
    fig, ax = plt.subplots()
    rects = plt.bar(probs, dist)

    plt.xlabel('Grammar Rule Probability')
    plt.ylabel('Pr. of human data')
    plt.title('Prob. of human data given prob. for rule: '+rule_name)
    plt.show()
