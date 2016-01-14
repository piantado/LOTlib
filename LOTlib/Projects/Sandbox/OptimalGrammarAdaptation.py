# -*- coding: utf-8 -*-

"""
        An experimental adaptive method for finding tree pieces that tend to be used in good posteriors
        TODO: FIX MULTIPLE COUNTS ON PARTIAL MATCHES MATCHING THEMSELVES. Here we restrict to one match per tree, which is stupid.

"""

from collections import defaultdict
import itertools

from copy import copy

import numpy
import scipy
import scipy.optimize

import LOTlib
from LOTlib import break_ctrlc
from LOTlib.Miscellaneous import *
from numpy import log, exp, sqrt, sum

from LOTlib.Subtrees import *

def print_subtree_adaptations(grammar, hypotheses, posteriors, subtrees, relative_KL=True):
    """
            Determine how useful it would be to explicitly define each subtree in H across
            all of the (corresponding) posteriors, as measured by KL from prior to posterior

            - hypotheses - a list of LOThypotheses
            - posteriors - [ [P(h|data) for h in hypotheies] x problems ]
            - subtrees   - a collection of (possibly partial) subtrees to try adapting

            We treat hyps as a fixed finite hypothesis space, and assume every subtree considered
            is *not* derived compositionally (although this could change in future versions)

            p - the probability of going to kids in randomly generating a subtree
            subtree_multiplier - how many times we sample a subtree from *each* node in each hypothesis
            relative_KL - compute summed KL divergence absolutely, or relative to the h.compute_prior()?

    """

    # compute the normalized posteriors
    Ps = map(lognormalize, posteriors)

    # Compute the baseline KL divergence so we can score relative to this
    if relative_KL:
        oldpriors = lognormalize(numpy.array([h.compute_prior() for h in hypotheses]))
        KL0s = [ sum(exp(oldpriors)*(oldpriors-P)) for P in Ps ]
    else:
        KL0s = [ 1.0 for P in Ps ] # pretend everything just had KL of 1, so we score relatively

    ## Now process each, starting with the most simple
    for t in break_ctrlc(sorted(subtrees, key=lambda t: grammar.log_probability(t), reverse=True)):

        # Get some stats on t:
        tlp = grammar.log_probability(t)
        tnt = count_identical_nonterminals( t.returntype, t) # How many times is this nonterminal used?

        # How many matches of t are there in each H?
        m = numpy.array([ count_subtree_matches(t, h.value) for h in hypotheses])
        ## TODO: There is a complication: partial patterns matching themselves.
        ##       For simplicity, we'll just take the *first* match, seetting max(m)=1
        ##       In the future, we should change this to correctly handle and count
        ##       partial matches matching themselves
        m = (m>=1)*1
        assert max(m)==1, "Error: "+str(t)+"\t"+str(m)

        # How many times is the nonterminal used, NOT counting in t?
        nt = numpy.array([ count_identical_nonterminals( t.returntype, h.value ) for h in hypotheses]) - (tnt-1)*m
        assert min(nt)>=0, "Error: "+str(t)

        # And the PCFG prior *not* counting t
        q = lognormalize(numpy.array([ grammar.log_probability(h.value) for h in hypotheses]) - tlp * m)

        # The function to optimize
        def fnc(p):
            if p <= 0. or p >= 1.: return float("inf") # enforce bounds

            newprior = lognormalize( q + log(p)*m + log(1.-p)*nt )

            kl = 0.0
            for P, kl0 in zip(Ps, KL0s):
                kl += sum( numpy.exp(newprior) * (newprior-P) ) / kl0

            return kl


        ### TODO: This optimization should be analytically tractable...
        ###       but we need to check that it is convex! Any ideas?
        o = scipy.optimize.fmin(fnc, numpy.array([0.1]), xtol=0.0001, ftol=0.0001, disp=0)

        print fnc(o[0]), o[0], log(o[0]), grammar.log_probability(t), qq(t)


if __name__ == '__main__':

    print "*** For an example, see Examples.NAND"
    raise NotImplementedError
