# -*- coding: utf-8 -*-

from copy import copy
from inspect import isroutine

from LOTlib.Miscellaneous import flip, Infinity, qq, weighted_sample, attrmem
from LOTlib.Hypotheses.Hypothesis import Hypothesis


class SimpleLexicon(Hypothesis):
    """
        A class for mapping words to hypotheses.

        This defaultly assumes that the data comes from sampling with probability alpha from
        the true utteranecs
    """

    def __init__(self, make_hypothesis, words=None, propose_p=0.25, value=None, **kwargs):
        """
            make_hypothesis -- a function to make each individual word meaning. None will leave it empty (for copying)
            words -- words to initially add (sampling from the prior)
            propose_p -- the probability of proposing to each word
        """

        if value is None:
            value = dict()
        else:
            assert isinstance(self.value, dict)

        Hypothesis.__init__(self, value=value, **kwargs)

        self.propose_p = propose_p

        # update with the supplied words, each generating from the grammar
        if make_hypothesis is not None and words is not None:
            for w in words:
                self.set_word(w, make_hypothesis())

    def __copy__(self):
        """ Copy our values. We will copy all the functions defaultly for simplicity """
        new = type(self)(None)
        for w in self.all_words():
            new.set_word(w, copy(self.get_word(w)))

        # And copy everything else
        for k in self.__dict__.keys():
            if k not in ['self', 'value', 'make_hypothesis']:
                new.__dict__[k] = copy(self.__dict__[k])

        return new

    def shallowcopy(self):
        """
        Copy but leave values pointing to old values
        """
        new = type(self)(
            self.make_hypothesis)  # create the right class, but don't give words or else it tries to initialize them
        for w in self.value.keys():
            new.set_word(w, self.get_word(w))  # set to this, shallowly, since these will get proposed to

        # And copy everything else
        for k in self.__dict__.keys():
            if k not in ['self', 'value', 'make_hypothesis']:
                new.__dict__[k] = copy(self.__dict__[k])

        return new

    def __str__(self):
        """
            This defaultly puts a \0 at the end so that we can sort -z if we want (e.g. if we print out a posterior first)
        """
        return '\n'+'\n'.join(["%-15s: %s" % (qq(w), str(v)) for w, v in sorted(self.value.iteritems())]) + '\0'

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return (str(self) == str(other))  # simple but there are probably better ways

    def __call__(self, word, *args):
        """
        Just a wrapper so we can call like SimpleLexicon('hi', 4)
        """
        return self.value[word](*args)

    # this sets the word and automatically compute its function
    def set_word(self, w, v):
        """
            This sets word w to value v. v can be either None, a FunctionNode or a  Hypothesis, and
            in either case it is copied here.
        """
        assert isinstance(v, Hypothesis)

        self.value[w] = v

    def get_word(self, w):
        return self.value[w]

    def all_words(self):
        return self.value.keys()

    def count_nodes(self):
        return sum([v.count_nodes() for v in self.value.values()])

    def force_function(self, w, f):
        """
            Allow force_function
        """
        self.value[w].force_function(f)

    # ##################################################################################
    ## MH stuff
    ###################################################################################

    def propose(self):
        """
        Propose to the lexicon by flipping a coin for each word and proposing to it.
        """

        new = copy(self)  ## Now we just copy the whole thing
        fb = 0.0

        for w in self.all_words():
            if flip(self.propose_p):
                xp, xfb = self.get_word(w).propose()
                new.set_word(w, xp)
                fb += xfb

        return new, fb

    @attrmem('prior')
    def compute_prior(self):
        return sum([x.compute_prior() for x in self.value.values()]) / self.prior_temperature












