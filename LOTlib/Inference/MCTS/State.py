import LOTlib
from math import log, sqrt
from LOTlib.Miscellaneous import Infinity, argmax

class StatePruneException(Exception):
    """
    This gets raised when we try to expand_children and we decide to prune (too big, too many nt, etc.)
    """
    pass

class State(object):
    """
    Abstract State class for MCTS.
    This stores a value, and a weight, where the weight is called for re-sampling

    If this is an end state, expand_children should yeild an empty list
    """

    def __init__(self, value, parent=None, C=1.0, V=10.0):
        """

        :param value:
        :param parent:
        :param C: Exploration constant. This scales the UTC search counts relative to the expected posterior. Higher explores more.
        :param V: pseudocounts on the prior
        :return:
        """

        self.value = value # what object do I store?
        self.parent = parent # who was my prior state?
        self.nsteps = 0 # How many simulations have I done from here?
        self.sumscore = 0.0 # What was the sum of scores of my children? (with nsteps, used to get expected score)
        self.children = None # each of the states I can get to from here. Initially None until expand_children() is called
        self.is_complete = False # If complete, we have -inf weight
        self.stored_prior = None # Store the prior, for use in computing the expected scores
        self.C = C
        self.V = V

    def __repr__(self):
        return "<STATE: %s>" % str(self.value)

    def make_children(self):
        """
        Returns a list of all of the child states
        """
        raise NotImplementedError

    def get_score(self):
        """
        Return the score from this state, if you can. Else None
        """
        raise NotImplementedError

    def is_terminal_state(self):
        """
        Called to check if this state has any possible children or not
        :return:
        """
        raise NotImplementedError

    def get_expected_score(self):
        """ A Bayesian estimate of *my* expected score, using sumscore and nsteps. V pesudocounts of the prior. """

        if self.is_complete:
            return -Infinity
        else:

            # We'll compute a cross-entropy for the prior, getting it right only p proportion of the time
            a = self.value.ALPHA
            p = 0.9 * a

            u0 = self.stored_prior + len(self.data)*( p*log(a)  + (1.-p)*log(1.-a))# prior plus best likelihood
            n = self.nsteps
            if n == 0:
                return u0
            else:
                xbar = float(self.sumscore)/float(n)
                return ( self.V*u0 + n*xbar) / (self.V+n)

    def get_weights(self):
        """
        The weights of all kids below, giving the probability of expanding each child.
        This currently uses UCT.

        NOTE: This must be computed for kids (rather than self) because it requires the total number of steps

        """
        assert self.children is not None

        N = sum([x.nsteps for x in self.children])+1
        return map(lambda x: x.get_expected_score() + self.C * sqrt(2.0 * log(N)/float(x.nsteps+1)), self.children)

    def show_graph(self, maxdepth=2, depth=0):
        """ Show the current state graph up to maxdepth """
        if self.children is not None:
            if maxdepth > depth:
                weights = self.get_weights()
                for c, w in zip(self.children, weights):
                    print "\t"*(depth+1), w, c.get_expected_score(), c.nsteps, c
                    c.show_graph(maxdepth=maxdepth, depth=depth+1)

    def __iter__(self, break_SIG_INTERRUPTED=True):
        """ Iteration is defaultly how we access MCTS """

        ## Sometimes our next() will return None. So we have to wrap it
        def wrapped_iter():
            while (break_SIG_INTERRUPTED and not LOTlib.SIG_INTERRUPTED):
                n = self.next()
                if n is not None:
                    yield n

        return wrapped_iter()

    def next(self, depth=0):
        """
        Give the next state we'll yield.
        """
        # print "# next call on: ", self.nsteps, self

        if self.is_terminal_state():
            score = self.get_score()

            ## And propagate up the tree
            curparent = self
            while curparent != None:
                assert isinstance(curparent, type(self))
                curparent.sumscore += score
                curparent.nsteps   += 1

                curparent = curparent.parent

            # and make sure we don't expand this again, since its a complete tree
            self.complete()

            # And we're done
            return self.value

        else: # It's a non-terminal, so expand the children

            try: # Catch StatePruneExceptions, where we just return None

                if self.children is None:
                    self.children = list(self.make_children())
                assert len(self.children) > 0, "*** Zero length children in %s" % self

                weights = self.get_weights()

                if max(weights) == -Infinity: # we are all done with the kids
                    raise StatePruneException
                else:
                    # and expand below
                    return self.children[argmax(weights)].next(depth=depth+1)

            except StatePruneException:
                self.complete() # don't return here

                return None

    def complete(self):
        """ This is called when there is nothing more below this state. It sets self.complete and mercilessly kills the children """
        self.is_complete = True
        del self.children
        self.children = []


