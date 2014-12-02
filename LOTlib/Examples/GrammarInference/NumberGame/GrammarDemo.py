"""
Inference with grammar rule probabilities.

"""
import numpy as np
from LOTlib import lot_iter
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Inference.PriorSample import prior_sample
from LOTlib.Examples.NumberGame.Model import *

# Parameters for number game inference
domain = 100
alpha = 0.9
num_iters = 10000

# Parameters for grammar hypothesis inference
num_grammar = 1000


# ============================================================================================================

if __name__ == "__main__":

    """Generate some number game hypotheses."""
    number_data = [2, 4, 8, 16, 32, 64, 5, 10, 15, 25, 35]
    h0 = make_h0(grammar=grammar, domain=domain, alpha=alpha)
    hypotheses = set(prior_sample(h0, number_data, N=num_iters))
    # hypotheses = set()
    # for h in MHSampler(h0, number_data, num_iters):
    #     hypotheses.add(h)

    print '%'*120
    print '%'*120

    """What grammar probabilities will best model our human data (grammar_data)?"""
    grammar_h0 = GrammarHypothesis(grammar, hypotheses, proposal_step=.1, proposal_n=1)
    grammar_hypotheses = []

    for d in grammar_h0.rule_distribution(grammar_data, 'ipowf_', np.arange(0.1, 5., 0.1)):
        print d

    hypos = [h for h in hypotheses]

    hypos = sorted(set(hypos), key=lambda x: x.posterior_score)
    for h in hypos[-10:]:
        print str(h), h.posterior_score, h.likelihood, h.prior



    # i = 0
    # for grammar_h in lot_iter(MHSampler(grammar_h0, grammar_data, num_grammar, trace=False)):
    #
    #     print ['%.3f' % v for v in grammar_h.value]
    #     i += 1
    #     print i, '!'*120
    #     print grammar_h.prior, grammar_h.likelihood, grammar_h.posterior_score
    #     grammar_hypotheses.append(grammar_h)



    # vals = np.arange(0.2, 2, .2)
    # rule_dist = grammar_h.rule_distribution(data, 'range_set_', vals=vals)
    # print rule_dist

    # visualize_probs(vals, rule_dist, 'union_')

