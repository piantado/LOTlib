"""
    This currently will learn mutually recursive definitions, things like even/odd. It makes a version
    of a SimpleLexicon that allows word definitinos to call other word definitions.

    In the future, we should see how we could invent predicates from nothing in order to explain data (via Ystar)
"""

from LOTlib import lot_iter
from Model import *

from LOTlib.DataAndObjects import FunctionData
from LOTlib.FunctionNode import cleanFunctionNodeString
from LOTlib.Inference.MetropolisHastings import MHSampler

data = []
for x in xrange(1, 10):
    data.append(FunctionData(input=['even', x], output=(x % 2 == 0)))
    data.append(FunctionData(input=['odd',  x], output=(x % 2 == 1)))
# print data

for h in lot_iter(MHSampler(make_h0(), data, skip=100)):
    print cleanFunctionNodeString(h)
    print h.posterior_score, h.prior, h.likelihood




