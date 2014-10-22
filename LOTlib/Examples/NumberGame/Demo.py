import matplotlib as plt
from Model import Inference as I, Grammar as G

# Global parameters for inference
alpha = 0.9
num_iters = 10000

# maps output number (e.g. 8) to a number of yes/no's (e.g. [10/2] )
initial_data = [2, 4, 6]
human_out_data = {
    8: (10, 2),
    12: (5, 7),
    14: (8, 4)
}


h0 = I.make_h0(alpha=alpha)
hypotheses = I.prior_sample(h0, initial_data, num_iters)

hypotheses = I.randomSample(G.grammar, initial_data, num_iters=num_iters, alpha=alpha)
# Inference.printBestHypotheses(hypotheses)

rule_nt = 'SET'
rule_name = 'union_'
probs = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
dist = I.probHDataGivenRule(G.grammar, rule_nt, rule_name,
                                    initial_data, human_out_data, probs)
I.visualizeRuleProbs(dist)


