"""
        Routines for evaluating MCMC runs

        TODO: use LOTlib.Hypothesis.POSTERIOR_CALL_COUNTER and report how many calls we've made
"""

import sys
import re
from collections import defaultdict
from time import time
from numpy import mean, diff
from math import log

from LOTlib import break_ctrlc
from LOTlib.Miscellaneous import logsumexp, r3, r5

def evaluate_sampler(my_sampler, print_every=1000, out_aggregate=sys.stdout, trace=False, pthreshold=0.999, prefix=""):
    """
            Print the stats for a single sampler run

            *my_sampler* -- a generator of samples
            print_every -- display the output every this many steps
            out_hypothesis -- where we put hypothesis stats
            out_aggregate  -- where we put aggregate stats

            trace -- print every sample
            prefix -- display before lines
    """
    visited_at = defaultdict(list)

    startt = time()
    for n, s in break_ctrlc(enumerate(my_sampler)): # each sample should have an .posterior_score defined
        if trace: print "#", n, s

        visited_at[s].append(n)

        if (n%print_every)==0 and n>0:
            post =  sorted([x.posterior_score for x in visited_at.keys()], reverse=True) # the unnormalized posteriors of everything found
            ll   =  sorted([x.likelihood for x in visited_at.keys()], reverse=True)
            Z = logsumexp(post) # just compute total probability mass found -- the main measure

            # determine how many you need to get pthreshold of the posterior mass
            J=0
            while J < len(post):
                if logsumexp(post[J:]) < Z + log(1.0-pthreshold):
                    break
                J += 1

            out_aggregate.write('\t'.join(map(str, [prefix, n, r3(time()-startt), r5(Z), r5(post[0]), J, len(post)] )) + '\n')
            out_aggregate.flush()

    return



"""
OLD Code for analyzing visits to hypotheses. May be incorporated lated
    # Now once we're done, output the hypothesis stats
    for k,v in visited_at.items():

        mean_diff = "NA"
        if len(v) > 1: mean_diff = mean(diff(v))

        out_hypotheses.write('\t'.join(map(str, [prefix, k.posterior_score, k.prior, k.likelihood, len(v), min(v), max(v), mean_diff, sum(diff(v)==0) ])) +'\n') # number of rejects from this
        out_hypotheses.flush()
"""