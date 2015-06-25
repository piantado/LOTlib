"""
    Maximize grammar entropy for constraind length
"""
from collections import defaultdict
from math import log
import numpy
import numpy.linalg

from LOTlib.Miscellaneous import Infinity

def H(p):
    z = float(sum(p))
    return -sum([ (pi/z) * log(pi/z) for pi in p ])

def entropy(grammar):
    """ For now, we'll maximize the entropy over each nonterminal, NOT trees """
    h = 0.0

    for nt in grammar.nonterminals():
        p = [r.p for r in grammar.get_rules(nt)]
        if min(p) < 0.0:
            return -1e99 # just get us out of here
        h += H(p)
    return h

def branching_factor(grammar, ts="<TERMINAL>"):
    """
    Returns the branching factor of the grammar.
     # TODO: THIS DOESN'T COUNT REACHABILITY, OR BV
    """

    nts = grammar.nonterminals()

    assert ts not in nts, "*** Terminal symbol cannot be in nonterminals!"

    name2idx = { n:i for i,n in enumerate(nts+[ts])}

    n = len(name2idx.keys())

    m = numpy.zeros( (n,n) )
    for nt in nts:
        z = float(sum([r.p for r in grammar.get_rules(nt)]))

        # nt will go to each n2 how many times?
        cnt = defaultdict(float)
        for r in grammar.get_rules(nt):

            if r.to is None or len(r.to) == 0:
                cnt[ts] += r.p / z
            else:
                for a in r.to:
                    if grammar.is_nonterminal(a):
                        cnt[a] += r.p / z

        for nt2 in name2idx.keys():
            m[name2idx[nt]][name2idx[nt2]] = cnt[nt2]

    m[name2idx[ts]][name2idx[ts]] = 1 # terminals always go to terminals

    # convert to a matrix or else power does not work right!
    m = numpy.matrix(m) # add 1 to diagonal to give the total number

    e, v = numpy.linalg.eig(m)

    # print (m**100)[:,-1]
    # print e
    #print v

    # for nt, x in zip(nts+[ts],  v[:,numpy.argmax(e)]):
    #     print "\t", nt, x

    # We really want to know how many STARTs will go to terminals
    mp = m**100 ## A HACK
    return mp[name2idx[grammar.start], name2idx[ts]]


def maximize_entropy(grammar, branching_bound):
    """
    Take a grammar and change it so that its parameters were maximum entropy, with the given branching bound.
    """

    def objective_function(x):
        assert len([r for r in grammar]) == len(x)
        for r,x in zip(grammar, x):
            r.p = x

        # Max entropy with a squared penalty on branching factor
        # print branching_factor(grammar)
        return -entropy(grammar) + 10000.0 * (branching_factor(grammar) - branching_bound)**2.

    from random import random

    p = [r.p+random() for r in grammar]

    from scipy.optimize import minimize

    res = minimize(objective_function, p, method="Powell") # Wow very sensitive to the method. Powell works well

    if not res.success:
        print res
        raise RuntimeError("Fit did not converge!")

    # Set to the solution
    for r,x in zip(grammar, res.x):
            r.p = x

    # and renormalize it
    grammar.renormalize()

    return grammar


if __name__ == "__main__":

    # from LOTlib.Examples.Number import grammar

    # from LOTlib.Examples.RationalRules.Model import grammar
    # from LOTlib.Examples.Number2015.Model.Grammars import grammarnton as grammar
    from LOTlib.Examples.AnBnCn.Model.Grammar import grammar as grammar
    # grammar = Grammar()
    # grammar.add_rule('START', 'f', ['WORD'], 1.0)
    # grammar.add_rule('START', 'f', None, 2.0)
    # grammar.add_rule('WORD',  'g', ['WORD', 'WORD'], 2.0)
    # grammar.add_rule('WORD',  'g2', [],      1.000)


    # print branching_factor(grammar)
    grammar = maximize_entropy(grammar, 5.0)

    for _ in xrange(1000):
        print grammar.generate()

    grammar.display_rules()

    print entropy(grammar)
    print branching_factor(grammar)

    import pickle
    pickle.dump(grammar, open("fit-grammar.pkl", 'w'))

