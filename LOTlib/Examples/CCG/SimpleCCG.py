"""
        A simple case of CCG-ish learning for a toy domain

        This just uses brute force parsing.


        TODO: Learn that MAN is JOHN or BILL

"""

import re

from LOTlib import lot_iter
from LOTlib.Miscellaneous import qq
from LOTlib.FiniteBestSet import FiniteBestSet
from Specification import CCGLexicon
from Grammar import make_hypothesis
from LOTlib.Inference.MetropolisHastings import MHSampler


from Data import all_words, data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SAMPLES = 100000


h0 = CCGLexicon(make_hypothesis, words=all_words, alpha=0.9, palpha=0.9, likelihood_temperature=1.0)

for h in MHSampler(h0, data, 1000, skip=100):
    print h.posterior_score, h
    print "\n\n"
