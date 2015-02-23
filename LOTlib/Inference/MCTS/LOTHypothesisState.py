"""

To implement Monte Carlo Tree Search, we'll define a new kind of FunctionNode that stores the weights for each rule
that could be applied to each of its args
"""
from math import log
from copy import copy

from LOTlib.Miscellaneous import Infinity, lambdaAssertFalse
from LOTlib.BVRuleContextManager import BVRuleContextManager
from LOTlib.FunctionNode import FunctionNode
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Grammar import MissingNonterminalException
from LOTlib.Hypotheses.Hypothesis import Hypothesis

from State import State, StatePruneException

class LoopsBreakException(Exception):
    """ Get us out of that goddamn loop """
    pass

class LOTHypothesisState(State):
    """
    A wrapper of LOTHypothese with partial values.

    When we make this normally, we use __init__. But when we make the first one (as in most code where this is used)
    we must use "make" in order to add children for each expansion of grammar.start.

    NOTE: This computes the prior on states by penalizing holes, since they must be filled
    """

    @classmethod
    def make(cls, make_h0, data, grammar, **args):
        """ This is the initializer we use to create this from a grammar, creating children for each way grammar.start can expand """
        h0 = make_h0(value=None)

        ## For each nonterminal, find out how much (in the prior) we pay for a hole that size
        hole_penalty = dict()
        for x in grammar.nonterminals():
            sm, m = 0.0, 0
            for _ in xrange(100): # sample to compute the expected prior
                try:
                    h = make_h0(value=grammar.generate(x), f=lambdaAssertFalse) # be sure not to compile
                    p = h.compute_prior()
                    if p > -Infinity:
                        sm += p
                        m += 1
                except MissingNonterminalException:
                    pass

            if m > 0:
                hole_penalty[x] = float(sm)/float(m)

        # and return the node
        return cls(make_h0(value=FunctionNode(None, '', '', [grammar.start])), data, grammar, hole_penalty=hole_penalty, parent=None, **args) # the top state

    def __init__(self, value, data, grammar, hole_penalty=None, **kwargs):
        """
        Initializer. The hole_penalty is a dictionary from nonterminals to
        """

        State.__init__(self, value, **kwargs)

        self.data = data
        self.grammar = grammar
        self.hole_penalty = hole_penalty

       # For each hole in the tree, we fill in stored_prior with an estimate of its expected influence on the prior
        if isinstance(value, Hypothesis):
            self.stored_prior = self.value.compute_prior()

            for fn in self.value.value:
                for a in fn.argStrings():
                    if grammar.is_nonterminal(a):
                         self.stored_prior += self.hole_penalty.get(a, -1.0) # the default here is for when, sometimes, we have NT that only occur after BV. This is for them.

    def is_terminal_state(self):
        return self.value.value.is_complete_tree(self.grammar)

    def make_children(self):
        assert self.children is None ## Otherwise we should never call this

        root = self.value.value

        if root.count_nodes() > self.value.maxnodes:
            raise StatePruneException

        # Now make the copy
        newfn = copy(root)

        ## find the first unfilled Node: the argi'th argument of fn
        try:
            fn, argi = None, None # the index of the fn. This will be used to find it in the copy
            for j, x in enumerate(newfn):
                if x.args is not None:
                    for i, a in enumerate(x.args):
                        if self.grammar.is_nonterminal(a):
                            fn, argi = x, i
                            raise LoopsBreakException
        except LoopsBreakException:
            pass
        assert fn is not None, "Cannot call make_children on a terminal. This must be avoided in State.next()"

        # Now make the children below
        children = []
        with BVRuleContextManager(self.grammar, fn, recurse_up=True):
            rules = self.grammar.get_rules(fn.args[argi])
            lZ = log(sum([r.p for r in rules]))

            for r in rules:
                fn.args[argi] = r.make_FunctionNodeStub(self.grammar, log(r.p)-lZ, fn)

                # copy the type in self.value
                newh = self.value.__copy__(value=None)
                newh.set_value(copy(newfn), f=lambdaAssertFalse) # Need to copy so different r give different fn; don't use set_value or it compiles

                # and make it into a State
                s = type(self)(newh, data=self.data, grammar=self.grammar, hole_penalty=self.hole_penalty, parent=self, C=self.C, V=self.V)
                children.append(s)

        return children

    def get_score(self):
        self.value.fvalue = self.value.compile_function() # make it actually compile!
        return sum(self.value.compute_posterior(self.data))






# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    from LOTlib import lot_iter

    from LOTlib.Examples.Magnetism.Simple import grammar, make_h0, data
    # USE:  C=50.0, V=1.0

    s = LOTHypothesisState.make(lambda **args: make_h0( maxnodes=100, **args), data, grammar, C=50.0, V=1.0)

    for x in lot_iter(s):
        print x.posterior_score, x.prior, x.likelihood, x

    print "<><><><><><><><><><><><><><><><><><><><>"
    s.show_graph(maxdepth=6)
    print "<><><><><><><><><><><><><><><><><><><><>"






