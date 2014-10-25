from Primitives import LOTlib_primitive
from LOTlib.Miscellaneous import flip
import numpy

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Stochastic Primitives
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@LOTlib_primitive
def flip_(p=0.5):
    return flip(p)

@LOTlib_primitive
def binomial_(n, p):
    if isinstance(n, int) and n > 0 and 0. <= p <= 1:
        return numpy.random.binomial(n, p)
    else:
        return float("nan")