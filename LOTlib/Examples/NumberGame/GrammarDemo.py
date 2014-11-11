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

hypotheses = set(prior_sample(h0, data, N=num_iters))
grammar_h = GrammarHypothesis(grammar, hypotheses)

vals = np.arange(0, 2, .2)
rule_dist = grammar_h.m_distribution(data, 'union_', vals=vals)
print rule_dist

visualize_probs(vals, rule_dist, 'union_')

