
import numpy as np
from Model import *

# Global parameters for inference
domain = 100
alpha = 0.9
num_iters = 100



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~ the basic stuff (i.e. sampling hypotheses)                                  ~~~~~#
h0 = Utilities.make_h0(alpha=alpha)
# hypotheses = I.prior_sample(h0, initial_data, num_iters)

hypotheses = Utilities.randomSample(G.grammar, [2,6,8], num_iters=num_iters, alpha=alpha)
# Inference.printBestHypotheses(hypotheses)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~ just one rule . . .                                                         ~~~~~#

# rule = get_rule('union_', rule_nt='SET', grammar=grammar)
# probs = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
# dist = probs_data_rule(grammar, rule, data, probs, num_iters, alpha)
# print dist
# visualize_probs(probs, dist, rule.name)
#

'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~ all rules! (and which probabilities?)                                       ~~~~~#

use_this_class = H.GrammarProbHypothesis(G.grammar, alpha, domain=domain)
cool_data = use_this_class.compute_likelihood(data, num_iters, alpha)

'''