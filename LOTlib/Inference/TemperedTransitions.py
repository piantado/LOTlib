import LOTlib
from random import random
from math import exp

## TODO: DEBUG THIS -- especially for asmmetric proposals.. ALSO TO BE SURE IT IS RIGHT
def tempered_transitions_sample(inh, data, steps, proposer=None, skip=0, temperatures=[1.0, 1.05, 1.1], stats=None):
    current_sample = inh

    LT = len(temperatures)

    ## TODO: CHECK THIS--STILL NOT SURE THIS IS RIGHT
    # a helper function for temperature transitions -- one single MH step, returning a new sample
    # this allows diff. temps for top and bottom
    def tt_helper(xi, data, tnew, told, proposer):
        if proposer is None: xinew, fb = xi.propose()
        else:                xinew, fb = proposer(xi)
        xinew.compute_posterior(data)
        r = (xinew.prior + xinew.likelihood) / tnew - (xi.prior + xi.likelihood) / told - fb
        if r > 0.0 or random() < exp(r):
            return xinew
        else:   return xi


    for mhi in xrange(steps):
        for skp in xrange(skip+1):

            xi = current_sample # do not need to copy this
            totlp = 0.0 #(xi.lp / temperatures[1]) - (xi.lp / temperatures[0])

            for i in xrange(0,LT-2): # go up
                xi = tt_helper(xi, data, temperatures[i+1], temperatures[i], proposer)
                totlp = totlp + (xi.prior + xi.likelihood) / temperatures[i+1] - (xi.prior + xi.likelihood) / temperatures[i]

            # do the top:
            xi = tt_helper(xi, data, temperatures[LT-1], temperatures[LT-1], proposer)

            for i in xrange(len(temperatures)-2, 0, -1): # go down
                xi = tt_helper(xi, data, temperatures[i], temperatures[i], proposer)
                totlp = totlp + (xi.prior + xi.likelihood) / temperatures[i] - (xi.prior + xi.likelihood) / temperatures[i+1]

            if random() < exp(totlp):
                current_sample = xi # copy this over

        yield current_sample
