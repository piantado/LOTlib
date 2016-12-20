"""

In LOTlib expressions we may sometimes want to use a random flip. This makes the outcomes of evaluation stochastic. A simple
way to handle this is to use the Hypotheses.StochasticSimulation code, which will compute the distribution of outcomes
by simplying running the program forward many times.

The alternative is this code, which uses a special "random context" and "flip" which allows you to, up to a certain depth,
enumerate all of the possible program traces and their associated probabilities. Thisi is based on many ideas from
probabilistic programming.
"""


class ContextSizeException(Exception):
    pass

class ContextSet(set):
    """ Store a set of contexts """
    pass

from math import log
class RandomContext(object): # manage uncertainty
    def __init__(self, cs, choices=()):
        self.choices = choices
        self.contextset = cs # who we update
        self.idx = 0
        self.lp = 0.0

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
        # print ">>POP>>", self
        ret = None

        if self.idx < len(self.choices): # if we are on the first choice,
            ret = self.choices[self.idx]
            self.idx += 1
        else:
            ret = True # which way we choose when its unspecified

            if len(self.choices) > 25:
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
