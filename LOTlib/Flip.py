"""

In LOTlib expressions we may sometimes want to use a random flip. This makes the outcomes of evaluation stochastic. A simple
way to handle this is to use the Hypotheses.StochasticSimulation code, which will compute the distribution of outcomes
by simplying running the program forward many times.

The alternative is this code, which uses a special "random context" and "flip" which allows you to, up to a certain depth,
enumerate all of the possible program traces and their associated probabilities. Thisi is based on many ideas from
probabilistic programming.
"""
from math import log
from collections import defaultdict
from LOTlib.Miscellaneous import logplusexp, lambdaMinusInfinity

MAX_CONTEXTS = 5000

class TooManyContextsException(Exception):
    """ Called when a ContextSet has too many contexts in it
    """

class ContextSizeException(Exception):
    """ Called when a context has too many choices in it
    """
    pass


class ContextSet(set):
    """ Store a set of contexts """
    pass


class RandomContext(object): # manage uncertainty
    """
    This stores a list of random choices we have made, to allow us to evaluate a stochastic hypothesis in a deterministic way,
    by calling RandomContext.flip().

    """

    def __init__(self, cs, choices=(), max_size=80):
        self.choices = choices
        self.contextset = cs # who we update
        self.idx = 0
        self.lp = 0.0
        self.max_size = max_size

    def __str__(self):
        return ''.join([ '1' if x else '0' for x in self.choices])
    def __repr__(self):
        return str(self)

    def flip(self, p=0.5):
        """ Flip a coin according to the context. This is somewhat complicated. If we have outcomes stored in the
        context, return the right one, accumulating the probability. If we don't have an outcome determined, then
        return default (True), and push the *other* outcome (and its whole context onto contextset, so that we
        visit that route later.

        This can be used in a grammar like
        C.flip(p=0.8)
        and then when we use the clases here we can enumerate all program traces

        """

        ret = None

        if self.idx < len(self.choices): # if we are on the first choice,
            ret = self.choices[self.idx]
            self.idx += 1
        else:
            ret = True # which way we choose when its unspecified

            if len(self.choices) > self.max_size:
                raise ContextSizeException

            # The choice we make later
            self.contextset.add(RandomContext(self.contextset, self.choices + (not ret,)))

            # the choice we make now
            self.choices = self.choices + (ret,)

        # and update my lps
        if ret:
            self.lp += log(p)  # update the context lp (summing all our random choices)
        else:
            self.lp += log(1.0 - p)

        return ret

def compute_outcomes(f, *args, **kwargs):
    """
    Return a dictionary of outcomes using our RandomContext tools, giving each possible trace (up to the given depth)
    and its probability.
    f here is a function of context, as in f(context, *args, **kwargs)

    In kwargs you can pass "alsocatch" as a tuple of exceptions to catch
    """

    out = defaultdict(lambdaMinusInfinity)  # dict from strings to lps that we accumulate

    cs = ContextSet() # this is the "open" set of contexts we need to explore
    cs.add(RandomContext(cs)) # add a single context with no history

    while len(cs) > 0:
        context = cs.pop()  # pop an element from Context set.
        # print "CTX", context, "  \t", cs

        try:
            v = f(context, *args) # when we call context.flip, we may update cs with new paths to explore

            out[v] = logplusexp(out[v], context.lp)  # add up the lp for this outcomem

        except kwargs.get('alsocatch', None) as e:
            pass
        except ContextSizeException:
            pass

        if len(cs) > MAX_CONTEXTS:
            raise TooManyContextsException

    return out