from copy import copy

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
        p = copy(h) ## TODO: This copies twice
        ret = self.propose_tree(h.value, **kwargs) # don't unpack, since we may return [newt,fb] or [newt,f,b]
        p.set_value(ret[0])
        ret[0] = p
        return ret
