from Model import Inference

# Global parameters for inference
alpha = 0.9
num_iters = 10000
data = [2, 8, 16]

h0 = Inference.make_h0(alpha=alpha)
hypotheses = Inference.prior_sample(h0, data, num_iters)

# hypotheses = Inference.randomSample(Grammar.grammar, data, num_iters=num_iters, alpha=alpha)
# Inference.printBestHypotheses(hypotheses)
