"""
Define a new kind of LOTHypothesis, that gives regex strings.

These have a special interpretation function that compiles differently than straight python eval.

"""
from LOTlib import lot_iter
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Miscellaneous import qq
from Model import *

if __name__ == "__main__":
    for h in lot_iter(MHSampler(make_h0(), data, steps=10000)):
        print h.posterior_score, h.prior, h.likelihood, qq(h)
