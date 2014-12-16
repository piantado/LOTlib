
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Inference.PriorSample import prior_sample
from LOTlib.Examples.NumberGame.NewVersion.Model import *
from Model import *


# ============================================================================================================
# Parameters
data = toy_3n

# for NumberGameHypothesis inference
alpha = 0.99
n = 1000
domain = 20

# for GrammarHypothesis inference
grammar_n = 10000
cap = 1000


def run():
    # ========================================================================================================
    # Sample some NumberGameHypotheses

    h0 = make_h0(grammar=simple_grammar_2, domain=domain, alpha=alpha)
    mh_sampler = MHSampler(h0, data[0].input, n)

    hypotheses = []
    for fn in simple_grammar_2.enumerate_at_depth(5):
        h = NumberGameHypothesis(grammar=simple_grammar_2, domain=domain, alpha=alpha)
        h.set_value(fn)
        hypotheses.append(h)

    print '%'*100, '\nNumberGameHypotheses:'
    for h in hypotheses:
        print h, h(), h.domain, h.alpha

    # ========================================================================================================
    # Sample some GrammarHypotheses

    grammar_h0 = GrammarHypothesis(simple_grammar_2, hypotheses, proposal_step=.1, proposal_n=1)
    for r in grammar_h0.rules:
        print r

    mh_grammar_sampler = MHSampler(grammar_h0, data, grammar_n, trace=False)
    mh_grammar_summary = sample_grammar_hypotheses(mh_grammar_sampler, skip=grammar_n/cap, cap=cap)
    mh_grammar_summary.print_top_samples()
    mh_grammar_summary.graph_samples()


if __name__ == "__main__":
    run()


#
#
# '''print distribution over power rule:  [prior, likelihood, posterior]'''
# # vals, posteriors = grammar_h0.rule_distribution(data, 'ipowf_', np.arange(0.1, 5., 0.1))
# # print_dist(vals, posteriors)
# #visualize_dist(vals, posteriors, 'union_')
#
#

