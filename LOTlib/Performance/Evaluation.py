"""
        Routines for evaluating MCMC runs

        TODO:

                MAYBE USE: #LOTlib.Hypothesis.POSTERIOR_CALL_COUNTER and report how many calls we've made
"""

import sys
import re
from collections import defaultdict
from time import time
from numpy import mean, diff

from LOTlib import lot_iter
from LOTlib.Miscellaneous import logsumexp, r3, r5


def load_model(modelname):
    """
    Loads a model, returning (data, make_h0)
    If modelname has a :<number> (e.g. Number.200) then it calls generate_data with that amount of data
    :param modelName:
    :return:
    """
    try:
        if re.search(r"\:", modelname):
            model, ndata = re.split(r":", modelname)
            exec("from LOTlib.Examples.%s import grammar, generate_data, make_h0" % model)
            data = generate_data(int(ndata))
        else:
            model = modelname
            exec("from LOTlib.Examples.%s import grammar, data, make_h0" % model)
    except ImportError as e:
        print >>sys.stderr, "*** Import error with %s" % modelname
        raise e

    return data, make_h0

def mydisplay(lst,n=10):
    # A nice display of the first n, guaranteeing that there will be n
    ret = map(r3, lst[:n])

    # Make it the right length, in case its too short
    while len(ret) < n:
        ret.append("NA")

    return ret

def evaluate_sampler(my_sampler, print_every=1000, out_hypotheses=sys.stdout, out_aggregate=sys.stdout, trace=False, prefix=""):
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
    for n, s in lot_iter(enumerate(my_sampler)): # each sample should have an .posterior_score defined
        if trace: print "#", n, s

        visited_at[s].append(n)

        if (n%print_every)==0 and n>0:
            post =  sorted([x.posterior_score for x in visited_at.keys()], reverse=True) # the unnormalized posteriors of everything found
            ll   =  sorted([x.likelihood for x in visited_at.keys()], reverse=True)
            Z = logsumexp(post) # just compute total probability mass found -- the main measure

            out_aggregate.write('\t'.join(map(str, [prefix, n, r3(time()-startt), r5(Z), len(post)]+mydisplay(post))) + '\n')
            out_aggregate.flush()

    # Now once we're done, output the hypothesis stats
    for k,v in visited_at.items():

        mean_diff = "NA"
        if len(v) > 1: mean_diff = mean(diff(v))

        out_hypotheses.write('\t'.join(map(str, [prefix, k.posterior_score, k.prior, k.likelihood, len(v), min(v), max(v), mean_diff, sum(diff(v)==0) ])) +'\n') # number of rejects from this
        out_hypotheses.flush()

    return 0.0


#
