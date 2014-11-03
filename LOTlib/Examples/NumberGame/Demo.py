
import numpy as np
from LOTlib.Hypotheses.GrammarProbHypothesis import GrammarProbHypothesis
from LOTlib.Inference.MetropolisHastings import MHSampler
from Model import *

# Global parameters for inference
domain = 100
alpha = 0.9
num_iters = 100


'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~ the basic stuff (i.e. sampling hypotheses)                                  ~~~~~#
h0 = I.make_h0(alpha=alpha)
# hypotheses = I.prior_sample(h0, initial_data, num_iters)

hypotheses = I.randomSample(G.grammar, initial_data, num_iters=num_iters, alpha=alpha)
# Inference.printBestHypotheses(hypotheses)

'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~ inference with grammar rule probabilities


hypotheses = set(MHSampler(make_h0(), in_data1, steps=10000))
grammar_h = GrammarProbHypothesis(grammar, hypotheses)

probs = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
dist = grammar_h.dist_over_rule(grammar, 'union_', data, probs)
print dist

visualize_probs(probs, dist, 'union_')



