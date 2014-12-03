"""
Find N best number game hypotheses.

"""
from LOTlib import lot_iter
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Inference.PriorSample import prior_sample
from Model import *

# Global parameters for inference
domain = 100
alpha = 0.99
N = 10
h0 = JoshConceptsHypothesis(grammar=grammar, domain=domain, alpha=alpha)
# TODO: enumerate JoshConceptHypotheses
hypotheses = grammar.enumerate()
# demo_data = [2, 4, 8, 16, 32, 32, 64, 64]
demo_data = [1, 3, 7, 15, 31, 31, 63, 63]


#=============================================================================================================

if __name__ == "__main__":

    hypos = sorted(hypotheses, key=lambda x: x.posterior_score)
    for h in hypos[-10:]:
        print str(h)
        print h.prior, h.likelihood, h.posterior_score
    # hypotheses = FiniteBestSet(generator=prior_sampler, N=N, key="posterior_score")
    # for h in hypotheses:
    #     print str(h), h.posterior_score

