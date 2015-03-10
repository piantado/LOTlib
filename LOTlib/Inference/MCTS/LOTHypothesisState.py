"""

To implement Monte Carlo Tree Search, we'll define a new kind of FunctionNode that stores the weights for each rule
that could be applied to each of its args
"""
from math import log, exp, sqrt
from copy import copy

from collections import defaultdict

from LOTlib.Miscellaneous import Infinity, lambdaAssertFalse, logsumexp
from LOTlib.BVRuleContextManager import BVRuleContextManager
from LOTlib.FunctionNode import FunctionNode
from LOTlib.GrammarRule import GrammarRule

from State import State, StatePruneException
from MaxScoreState import MaxScoreState

class LoopsBreakException(Exception):
    """ Get us out of that goddamn loop """
    pass


class LOTHypothesisState(MaxScoreState):
    """
    A wrapper of LOTHypothese with partial values.

    When we make this normally, we use __init__. But when we make the first one (as in most code where this is used)
    we must use "make" in orderqq to add children for each expansion of grammar.start.

    NOTE: This computes the prior on states by penalizing holes, since they must be filled
    """

    @classmethod
    def make(cls, make_h0, data, grammar, **args):
        """ This is the initializer we use to create this from a grammar, creating children for each way grammar.start can expand """
        h0 = make_h0(value=None)

        ## For each nonterminal, find out how much (in the prior) we pay for a hole that size
        hole_penalty = dict()
        dct = defaultdict(list) # make a lit of the log_probabilities below
        for _ in xrange(1000): # generate this many trees
            t = grammar.generate()
            for fn in t:
                dct[fn.returntype].append(grammar.log_probability(fn))
        hole_penalty = { nt : sum(dct[nt]) / len(dct[nt]) for nt in dct }

        #We must modify the grammar to include one more nonterminal here
        mynt = "<hsmake>"
        myr = GrammarRule(mynt, '', [grammar.start], p=1.0)
        grammar.rules[mynt].append(myr)
        return cls(make_h0(value=FunctionNode(None, mynt,  '', [grammar.start], rule=myr)), data, grammar, hole_penalty=hole_penalty, parent=None, **args) # the top state

    def __init__(self, value, data, grammar, hole_penalty=None, **kwargs):
        """
        Initializer. The hole_penalty is a dictionary from nonterminals to
        """

        State.__init__(self, value, **kwargs)

        self.data = data
        self.grammar = grammar
        self.hole_penalty = hole_penalty
        assert self.hole_penalty is not None # Need this!

    def compute_weights(self):
        """
        Here we compute weights defaultly and then add an extra penalty for unfilled holes to decide which to use next.
        Returning a tuple lets these weights get sorted by each successive element.

        This also exponentiates and re-normalizes the posterior among children, keeping it within [0,1]
        """

        # Here what we call x_bar is really the mean log posterior. So we convert it out of that.
        es = [c.get_xbar() if c.nsteps > 0 else Infinity for c in self.children]

        Z = logsumexp(es) ## renormalize, for converting to logprob

        # We need to preserve -inf here as well as +inf since these mean something special
        # -inf means we should never ever visit; +inf means we can't not visit
        es = [ exp(x-Z) if abs(x) < Infinity else x for x in es]

        N = sum([c.nsteps for c in self.children])

        # the weights we return
        weights = [None] * len(self.children)

        for i, c in enumerate(self.children):

            v = 0.0 # the adjustment
            if es[i] == Infinity: # so break the ties.
                # This must prevent us from wandering off to infinity. To do that, we impose a penalty for each nonterminal
                for fn in c.value.value:
                    for a in fn.argStrings():
                        if self.grammar.is_nonterminal(a):
                            v += self.hole_penalty.get(a, -1.0) # pay this much for this hole. -1 is for those weird nonterminals that need bv introduced

            weights[i] = (es[i] + self.C * sqrt(2.0 * log(N)/float(c.nsteps+1)) if c.nsteps > 0 else Infinity, v)

        return weights


    def score_terminal_state(self):
        """ Get the score here """
        # print "SCORING", self

        # We must make this compile the function since it is told not to compile in newh within self.make_children
        self.value.fvalue = self.value.compile_function() # make it actually compile!
        return sum(self.value.compute_posterior(self.data))

    def is_terminal_state(self):
        return self.value.value.is_complete_tree(self.grammar)

    def make_children(self):
        assert self.children is None ## Otherwise we should never call this

        root = self.value.value

        if root.count_nodes() >= self.value.maxnodes:
            raise StatePruneException

        # Now make the copy
        newfn = copy(root)

        ## find the first unfilled Node: the argi'th argument of fn in our new copy
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
                fn.args[argi] = r.make_FunctionNodeStub(self.grammar, fn)

                # copy the type in self.value
                newh = self.value.__copy__(value=None)
                newh.set_value(copy(newfn), f=lambdaAssertFalse) # Need to copy so different r give different fn; don't use set_value or it compiles

                # and make it into a State
                s = type(self)(newh, data=self.data, grammar=self.grammar, hole_penalty=self.hole_penalty, parent=self)
                children.append(s)

        return children




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    from LOTlib import break_ctrlc

    from LOTlib.Examples.Magnetism.Simple import grammar, make_h0, data
    # USE:  C=50.0, V=1.0

    s = LOTHypothesisState.make(lambda **args: make_h0( maxnodes=100, **args), data, grammar, C=50.0, V=1.0)

    for x in break_ctrlc(s):
        print x.posterior_score, x.prior, x.likelihood, x

    print "<><><><><><><><><><><><><><><><><><><><>"
    s.show_graph(maxdepth=6)
    print "<><><><><><><><><><><><><><><><><><><><>"






