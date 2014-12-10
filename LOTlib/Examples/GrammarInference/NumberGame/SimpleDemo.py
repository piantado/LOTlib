
import numpy as np
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Inference.PriorSample import prior_sample
from LOTlib.Examples.NumberGame.NewVersion.Model import *
from Model import *

# Parameters for number game inference
alpha = 0.99
n = 1000
domain = 20

# Parameters for GrammarHypothesis inference
grammar_n = 1000
data = toy_2n

# Variables for NumberGameHypothesis inference
h0 = make_h0(grammar=simple_test_grammar, domain=domain, alpha=alpha)
prior_sampler = prior_sample(h0, data[0].input, N=n)
mh_sampler = MHSampler(h0, data[0].input, n)


# ============================================================================================================

def run():
    """Run demo"""
    '''Generate number game hypotheses'''
    hypotheses = set()
    for h in lot_iter(mh_sampler):
        hypotheses.add(h)

    for h in hypotheses:
        print h, h(), h.domain, h.alpha

    '''What grammar probabilities will best model our human data?'''
    grammar_h0 = GrammarHypothesis(simple_test_grammar, hypotheses, proposal_step=.1, proposal_n=1)

    '''print distribution over power rule:  [prior, likelihood, posterior]'''
    # vals, posteriors = grammar_h0.rule_distribution(data, 'ipowf_', np.arange(0.1, 5., 0.1))
    # print_dist(vals, posteriors)
    #visualize_dist(vals, posteriors, 'union_')

    '''grammar hypothesis inference'''
    #prior_grammar_sampler = prior_sample(grammar_h0, data, grammar_n)
    mh_grammar_sampler = MHSampler(grammar_h0, data, grammar_n, trace=False)
    grammar_hypotheses = sample_grammar_hypotheses(mh_grammar_sampler)


if __name__ == "__main__":
    run()
