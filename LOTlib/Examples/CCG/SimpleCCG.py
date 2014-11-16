"""
A simple case of CCG-ish learning for a toy domain

This just uses brute force parsing.


TODO: Learn that MAN is JOHN or BILL

"""
from LOTlib.Inference.MetropolisHastings import MHSampler
from Model import *


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SAMPLES = 100000

if __name__ == "__main__":
    h0 = CCGLexicon(make_hypothesis, words=all_words, alpha=0.9, palpha=0.9, likelihood_temperature=1.0)

    for h in MHSampler(h0, data, 1000, skip=100):
        print h.posterior_score, h
        print "\n\n"
