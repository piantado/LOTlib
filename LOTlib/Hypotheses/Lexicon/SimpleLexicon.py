# -*- coding: utf-8 -*-

"""

    TODO:
        - Make the.valueicon be indexable like an array/dict, rather than having to say h.value[...] say h[..]
"""
from copy import copy
from inspect import isroutine

from LOTlib.Miscellaneous import flip, Infinity, qq, weighted_sample
from LOTlib.Hypotheses.Hypothesis import Hypothesis

class SimpleLexicon(Hypothesis):
    """
        A class for mapping words to hypotheses.

        This defaultly assumes that the data comes from sampling with probability alpha from
        the true utteranecs
    """

    def __init__(self, make_hypothesis, words=(), propose_p=0.25, value=None, **kwargs):
        """
            hypothesis - a function to generate hypotheses
            words -- words to initially add (sampling from the prior)
            propose_p -- the probability of proposing to each word
        """


        Hypothesis.__init__(self, value=dict() if value is None else value, **kwargs)
        self.__dict__.update(locals())

        assert isroutine(make_hypothesis) # check that we can call

        # update with the supplied words, each generating from the grammar
        for w in words:
            self.set_word(w, v=None)

    def __copy__(self):
        """ Copy a.valueicon. We don't re-create the fucntions since that's unnecessary and slow"""
        new = type(self)(self.make_hypothesis, words=copy(self.words))
        for w in self.value.keys():
            new.value[w] = copy(self.value[w])

        # And copy everything else
        for k in self.__dict__.keys():
            if k not in ['self', 'value', 'make_hypothesis']:
                new.__dict__[k] = copy(self.__dict__[k])

        return new

    def shallowcopy(self):
        """
        Copy but leave values pointing to old values
        """
        new = type(self)(self.make_hypothesis) # create the right class, but don't give words or else it tries to initialize them
        for w in self.value.keys():
            new.set_word(w, self.value[w])  # set to this, shallowly, since these will get proposed to

        # And copy everything else
        for k in self.__dict__.keys():
            if k not in ['self', 'value', 'make_hypothesis']:
                new.__dict__[k] = copy(self.__dict__[k])

        return new

    def __str__(self):
        """
            This defaultly puts a \0 at the end so that we can sort -z if we want (e.g. if we print out a posterior first)
        """
        return '\n'.join([ "%-15s: %s"% (qq(w), str(v)) for w,v in sorted(self.value.iteritems())]) + '\0'

    def __hash__(self): return hash(str(self))
    def __eq__(self, other):   return (str(self)==str(other)) # simple but there are probably better ways

    def __call__(self, word, *args):
        """
        Just a wrapper so we can call like SimpleLexicon('hi', 4)
        """
        return self.value[word](*args)

    # this sets the word and automatically compute its function
    def set_word(self, w, v=None):
        """
            This sets word w to value v. v can be either None, a FunctionNode or a  Hypothesis, and
            in either case it is copied here. When it is a Hypothesis, the value is extracted. If it is
            None, we generate from the grammar
        """

        # Conver to standard expressiosn
        if v is None:
            v = self.make_hypothesis()

        assert isinstance(v, Hypothesis)

        self.value[w] = v

    def all_words(self):
        return self.value.keys()

    def count_nodes(self):
        return sum([v.count_nodes() for v in self.value.values()])

    def force_function(self, w, f):
        """
            Allow force_function
        """
        self.value[w].force_function(f)

    ###################################################################################
    ## MH stuff
    ###################################################################################

    def propose(self):
        """
        Default proposal to a lexicon -- now at least one, plus some coin flips
        :return:
        """

        new = copy(self) ## Now we just copy the whole thing

        # Propose one for sure
        w = weighted_sample(self.value.keys()) # the word to change
        p, fb = self.value[w].propose()
        new.set_word(w, p)

        for x in self.all_words():
            if w != x and flip(self.propose_p):
                xp, xfb = self.value[x].propose()
                new.set_word(x, xp)
                fb += xfb

        return new, fb


    def compute_prior(self):
        self.prior = sum([x.compute_prior() for x in self.value.values()]) / self.prior_temperature
        self.posterior_score = self.prior + self.likelihood
        return self.prior











