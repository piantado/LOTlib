from copy import copy
from LOTlib.Miscellaneous import Infinity
from random import random
from math import log

class LOTProposal(object):
    """
            A class of LOT proposals. This wraps calls with copying of the hypothesis
            so that we can implement only propose_t classes for subclasses, that generate trees
    """
    def __init__(self, grammar):
        self.__dict__.update(locals())

    def __call__(self, h, **kwargs):
        # A wrapper that calls propose_tree (defined in subclasses) on our tree value
        # so this manages making LOTHypotheses (or the relevant subclass), and proposal subclasses
        # can just manage trees
        p = h.__copy__(copy_value=False) ## Don't copy the value -- we get this from propose_tree
        ret = self.propose_tree(h.value, **kwargs) # don't unpack, since we may return [newt,fb] or [newt,f,b]
        p.set_value(ret[0])
        ret[0] = p
        return ret


