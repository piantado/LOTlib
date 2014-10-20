import Inference

# Global parameters for inference
ALPHA = 0.9
NUM_ITERS = 10000
DATA = []

h0 = Inference.make_h0(alpha=ALPHA)
hypotheses = Inference.prior_sample(h0, DATA, NUM_ITERS)

Inference.printBestHypotheses(hypotheses)

