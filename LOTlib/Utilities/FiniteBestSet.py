# -*- coding: utf-8 -*-

"""

        This is a version of what was called "PriorityQueue.py" in LOTlib.VERSION < 0.3.

        NOTE: This is terrible. Let's re-do this so everything is nice, we can easily add and remove, max/min is more clear, etc.
                -- have it initialize to extract a certain key from hypotheses, or call a certain function

"""
import heapq
import collections
from LOTlib.Miscellaneous import Infinity

class QueueItem(object):
    """
            A wrapper to hold items and scores in the queue--just wraps "cmp" on a priority value
    """
    def __init__(self, x, p):
        self.x = x
        self.priority = p

    def __cmp__(self, y):
        # Comparisons are based on priority
        return cmp(self.priority, y.priority)

    def __repr__(self): return repr(self.x)
    def __str__(self):  return str(self.x)


class FiniteBestSet(object):
    """
            This class stores the top N (possibly infinite) hypotheses it observes, keeping only unique ones.
            It works by storing a priority queue (in the opposite order), and popping off the worst as we need to add more
    """

    def __init__(self, generator=None, N=Infinity, max=True, key='posterior_score'):
        """
                N - the number of hypotheses to store
                max - True/False -- do we keep the ones closes to +inf (or -inf)
                key - if a string (attribute) or function, we used this to access a hypothesis' priority score
        """

        self.N = N
        self.max = max
        self.key = key

        self.max_multiplier = (1 if self.max else -1) # invert sign

        self.Q = [] # we use heapq to
        self.unique_set = set()

        # if we can, add from here
        if generator is not None:
            for g in generator:
                self.add(g)


    def __contains__(self, y):
        for r in self.Q:
            if r.x == y: return True
        return False

    def __iter__(self):
        for x in self.get_all(): yield x

    def __len__(self):
        return len(self.Q)

    # Another name since I can't stop using it
    def push(self, x, p=None):
        self.add(x,p)

    def add(self, x, p=None, store_iterator=False):
        """
                Add *x* with priority p to the set. If x is an iterable, we add everything in it.

                If p=None, we use self.key to get the value.

                *store_iterator* - if we are supposed to store an iterator (rather than elements from it)
        """

        if isinstance(x, collections.Iterable) and not store_iterator:
            assert p is None, "FiniteBestSet.add must have p=None for use with an iterator"

            for xi in x: self.add(xi)

        else:

            if p is None:
                assert self.key is not None
                if isinstance(self.key, str): p = getattr(x,self.key)
                else:                         p = self.key(x)


            if (x in self.unique_set):
                return
            else:
                heapq.heappush(self.Q, QueueItem(x, self.max_multiplier*p))
                self.unique_set.add(x) # add to the set

                # if we have too many elements
                if len(self) > self.N:
                    rr = heapq.heappop(self.Q)

                    if rr.x in self.unique_set:
                        self.unique_set.remove(rr.x) # clean out the removed from the set

    def get_all(self, **kwargs):
        """ Return all elements (arbitrary order). Does NOT return a copy. This uses kwargs so that we can call one 'sorted' """
        if kwargs.get('sorted', False):
            return [ c.x for c in sorted(self.Q, reverse=kwargs.get('decreasing',False))]
        else:
            return [ c.x for c in self.Q]

    def merge(self, y):
        """
                Copy over everything from y. Here, y may be a list of things to merge (e.g. other FiniteBestSets)
                This is slightly inefficient because we create all new QueueItems, but it's easiest to deal with min/max
        """
        if isinstance(y, list) or isinstance(y, tuple) or isinstance(y, set):
            for yi in y: self.merge(yi)
        elif isinstance(y, FiniteBestSet):
            for yi in y.Q:
                self.add(yi.x, yi.priority*y.max_multiplier) # mult y by y.max_multiplier to convert it back to the original scale
        else:
            raise NotImplementedError

if __name__ == "__main__":

    import random

    # Check the max
    for i in xrange(100):
        Q = FiniteBestSet(N=10)

        ar = range(100)
        random.shuffle(ar)
        for x in ar: Q.add(x,x)

        assert set(Q.get_all()).issuperset( set([90,91,92,93,94,95,96,97,98,99]))
        print Q.get_all()

    # check the min
    for i in xrange(100):
        Q = FiniteBestSet(N=10, max=False)

        ar = range(100)
        random.shuffle(ar)
        for x in ar: Q.add(x,x)

        assert set(Q.get_all()).issuperset( set([0,1,2,3,4,5,6,7,8,9]))
        print Q.get_all()
