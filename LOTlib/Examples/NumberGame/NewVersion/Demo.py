"""
Find N best number game hypotheses.

"""
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Inference.PriorSample import prior_sample
from Model import *


# ============================================================================================================

# Parameters for inference
domain = 100
alpha = 0.99
num_iters = 1000
N = 10
# demo_data = [2, 4, 8, 16, 32, 32, 64, 64]
demo_data = [1, 3, 7, 15, 31, 31, 63, 63]


# ============================================================================================================


def run():
    h0 = make_h0(grammar=grammar, domain=domain, alpha=alpha)
    prior_sampler = prior_sample(h0, demo_data, num_iters)
    mh_sampler = MHSampler(h0, demo_data, num_iters)

    # hypotheses = save_hypotheses(prior_sampler)
    hypotheses = load_hypotheses()
    print_top_hypotheses(hypotheses)


# ============================================================================================================

if __name__ == "__main__":
    run()