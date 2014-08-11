"""
        Move out some functions from Shared -- things that aren't part of the core model specification
"""
from collections import defaultdict
from LOTlib.Miscellaneous import *

def show_baseline_distribution(testing_set, N=1000):
    from Shared import generate_data, target

    d = generate_data(N)

    frq = defaultdict(int)
    for di in d:
        frq[di.utterance] += 1

    for w in frq.keys():

        # figure out how often its true:
        pct = float(sum(map(lambda s: ifelse(target.value[w](s), 1.0, 0.0), testing_set) )) / len(testing_set)

        print frq[w], "\t", q(w), "\t", pct

def is_conservative(h,testing_set):
    """
            Check if a hypothesis (funciton node) is conservative or not
    """
    f = evaluate_expression(h, ['context'])
    for x in testing_set:
        a,b,s = x
        if f(a,b,s) != f(a, b.intersection(a), s.intersection(a) ): # HMM: is this right? We intersect s with a?
            return False
    return True



def extract_presup(resp):
    """
            From a bunch of responses, extract the T/F presups
    """
    out = []
    for k in resp:
        if is_undef(k): out.append(False)
        else:           out.append(True)
    return out

def extract_literal(resp):
    """
            From a bunch of responses, extract the T/F literals
    """
    out = []
    for k in resp:
        if (k is True) or (k == "undefT"): out.append(True)
        else:                              out.append(False)
    return out

def collapse_undefs(resp):
    """
            Collapse together our multiple kinds of undefs so that we can compare vectors
    """
    out = []
    for k in resp:
        if is_undef(k): out.append("undef")
        else:           out.append(k)
    return out
