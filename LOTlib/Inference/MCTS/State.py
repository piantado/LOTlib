import LOTlib
from math import log, sqrt, exp
from LOTlib.Miscellaneous import Infinity, argmax, logsumexp

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

    def __init__(self, value, parent=None, C=1.0):
        """

        :param value: The value in this node
        :param parent: Who our parent is. None is root
        :param C: Exploration constant. This scales the UTC search counts relative to the expected posterior. Higher explores more.
        """

        self.value = value # what object do I store?
        self.parent = parent # who was my prior state?
        self.nsteps = 0 # How many simulations have I done from here?
        self.summarystat = 0.0 # A summary statistic for get_expected_score
        self.children = None # each of the states I can get to from here. Initially None until expand_children() is called
        self.is_complete = False # If complete, we have -inf weight
        self.C=C

    def make_children(self):
        """
        Returns a list of all of the child states
        """
        raise NotImplementedError

    def score_terminal_state(self):
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

    def __repr__(self):
        return "<STATE: %s>" % str(self.value)

    def get_xbar(self):
        """ The score I expect given my summarystat and nsteps. """

        if self.is_complete:
            return -Infinity
        elif self.nsteps == 0:
            return None # This should get over-written by Infinity compute_weights
        else:
            return float(self.summarystat)/float(self.nsteps)

    def update_score(self, score):
        """ Take a sample from one of my children and update my score. My children call this """
        self.summarystat += score
        self.nsteps   += 1

    def compute_weights(self):
        """
        The default UCT weighting scheme
        """

        N = sum([c.nsteps for c in self.children])

        return [c.get_xbar() + self.C * sqrt(2.0 * log(N)/float(c.nsteps+1)) if c.nsteps > 0 else Infinity for c in self.children ]


    def show_graph(self, maxdepth=2, depth=0):
        """ Show the current state graph up to maxdepth """
        if self.children is not None:
            if maxdepth > depth:
                weights = self.compute_weights()
                for c, w in zip(self.children, weights):
                    print "#", "\t"*(depth+1), w, c.get_xbar(), c.nsteps, c
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

    def next(self, depth=0, return_state=False):
        """
        Give the value of the next state we'll yeild
        (or the state, if return_state)
        """
        # print "# next call on: ", self.nsteps, self

        if self.is_terminal_state():

            score = self.score_terminal_state()

            ## And propagate up the tree. But don't do this for -inf scores, which can occur (for instance) when
            ## we propose something zero prior. The reason is that this reigns havoc on the average scores
            if score > -Infinity:
                curparent = self
                while curparent != None:
                    curparent.update_score(score)
                    curparent = curparent.parent

            # and make sure we don't expand this again, since its a complete tree
            self.complete()

            # And we're done
            if return_state:
                return self
            else:
                return self.value

        else: # It's a non-terminal, so expand the children

            try: # Catch StatePruneExceptions, where we just return None

                if self.children is None:
                    self.children = list(self.make_children()) # This may raise StatePruneException if the children are too deep, for instance
                assert len(self.children) > 0, "*** Zero length children in %s" % self

                weights = self.compute_weights()

                if max(weights) == -Infinity: # we are all done with the kids
                    raise StatePruneException
                else:
                    return self.children[argmax(weights)].next(depth=depth+1, return_state=return_state)

            except StatePruneException:
                self.complete() # don't return anything here

                return None

    def complete(self):
        """ This is called when there is nothing more below this state. It sets self.complete and mercilessly kills the children """
        self.is_complete = True
        del self.children
        self.children = []


