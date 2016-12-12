"""

Versions of scheme that return dictionaries for all possible outcomes, where the outcomes are strings

Here, the probabilities that are stored in the dictionary are log probabilities
"""


from LOTlib.Primitives import primitive
from collections import defaultdict
from LOTlib.Miscellaneous import logplusexp, Infinity, nicelog, log1mexp

@primitive
def cons_d(x,y):
    out = defaultdict(lambda: -Infinity)

    for a, av in x.items():
        for b, bv in y.items():
            out[a+b] = logplusexp(out[a+b], av + bv)
    return out

@primitive
def cdr_d(x):
    out = defaultdict(lambda: -Infinity)
    for a, av in x.items():
        v = a[1:] if len(a) > 1 else ''
        out[v] = logplusexp(out[v], av)
    return out


@primitive
def car_d(x):
    out = defaultdict(lambda: -Infinity)
    for a, av in x.items():
        v = a[1] if len(a) > 1 else ''
        out[v] = logplusexp(out[v], av)
    return out

@primitive
def flip_d(p):
    return {True: nicelog(p), False: nicelog(1.-p)}

@primitive
def empty_d(x):
    p = x.get('', -Infinity)
    return {True: p, False:log1mexp(p)}

@primitive
def if_d(prb,x,y):
    out = defaultdict(lambda: -Infinity)
    pt = prb[True]
    pf = prb[False]
    for a, av in x.items():
        out[a] = av + pt
    for b, bv in y.items():
        out[b] = logplusexp(out[b], bv + pf)

    return out


