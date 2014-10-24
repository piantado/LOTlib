import numpy as np
from Model import Hypothesis as H, Inference as I, Grammar as G

# Global parameters for inference
domain = 100
alpha = 0.9
num_iters = 10000

# maps output number (e.g. 8) to a number of yes/no's (e.g. [10/2] )
in_data1 = [2, 4, 6]
out_data1 = {
    8: (10, 2),
    12: (5, 7),
    14: (8, 4)
}
in_data2 = [2, 8, 16]
out_data2 = {
    8: (10, 2),
    12: (5, 7),
    14: (8, 4)
}
data = [(in_data1, out_data1), (in_data2, out_data2)]

'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~ the basic stuff (i.e. sampling hypotheses)                                  ~~~~~#
h0 = I.make_h0(alpha=alpha)
# hypotheses = I.prior_sample(h0, initial_data, num_iters)

hypotheses = I.randomSample(G.grammar, initial_data, num_iters=num_iters, alpha=alpha)
# Inference.printBestHypotheses(hypotheses)

'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~ just one rule . . .                                                         ~~~~~#

rule = I.get_rule('union_', rule_nt='SET')
probs = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
dist = I.prob_data_rule(G.grammar, rule, data, probs, num_iters, alpha)
I.visualize_probs(probs, dist, rule.name)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~ all rules! (and which probabilities?)                                       ~~~~~#

use_this_class = H.GrammarProbHypothesis(G.grammar, alpha, domain=domain)
cool_data = use_this_class.compute_likelihood(data, num_iters, alpha)