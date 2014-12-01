from math import log, exp, isnan
from random import random


def MH_acceptance(cur, prop, fb, acceptance_temperature=1.0):
    """
            Handle all the weird corner cases for computing MH acceptance ratios
            And then returns a true/false for acceptance
    """

    # If we get infs or are in a stupid state, let's just sample from the prior so things don't get crazy
    if isnan(cur) or (cur == float("-inf") and prop == float("-inf")):
        # Just choose at random -- we can't sample priors since they may be -inf both
        r = -log(2.0)
    elif isnan(prop):
        # Never accept
        r = float("-inf")
    else:
        r = (prop-cur-fb) / acceptance_temperature

    # And flip
    return r >= 0.0 or random() < exp(r)
