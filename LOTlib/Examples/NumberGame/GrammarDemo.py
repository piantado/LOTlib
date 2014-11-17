"""
Inference with grammar rule probabilities.

"""
import numpy as np
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Inference.PriorSample import prior_sample
from Model import *

# Global parameters for inference
domain = 100
alpha = 0.9
num_iters = 10000
h0 = make_h0(grammar=grammar, alpha=alpha)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
if __name__ == "__main__":

    hypotheses = set(prior_sample(h0, data, N=num_iters))
    grammar_h = GrammarHypothesis(grammar, hypotheses)

    vals = np.arange(0.2, 2, .2)    # NOTE: Cannot start at 0! (why??)
    rule_dist = grammar_h.rule_distribution(data, 'range_set_', vals=vals)
    print rule_dist

    # visualize_probs(vals, rule_dist, 'union_')

