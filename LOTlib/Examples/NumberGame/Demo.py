
import numpy as np
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Hypotheses.GrammarProbHypothesis import GrammarProbHypothesis
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Inference.PriorSample import prior_sample
from Model import *

# Global parameters for inference
domain = 100
alpha = 0.9
num_iters = 100
h0 = make_h0(grammar=grammar, alpha=alpha)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~ find 10 best number game hypotheses

prior_sampler = prior_sample(h0, in_data1, num_iters)
hypotheses = FiniteBestSet(generator=prior_sampler, N=10)
print str([h for h in hypotheses])



'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~ inference with grammar rule probabilities

hypotheses = set(MHSampler(h0, in_data1, steps=num_iters))
grammar_h = GrammarProbHypothesis(grammar, hypotheses)

probs = np.arange(0, 2, .2)
dist = grammar_h.dist_over_rule(grammar, 'union_', data, probs=probs)
print dist

visualize_probs(probs, dist, 'union_')


'''