
import unittest
import re
from collections import Counter
from math import exp
from scipy.stats import chisquare

from LOTlib import lot_iter
from LOTlib.Miscellaneous import logsumexp
from LOTlib.FunctionNode import FunctionNode, BVUseFunctionNode, BVAddFunctionNode

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

# Critical p value for rejecting tests
PVALUE = 0.001

class InfiniteTreeTester(unittest.TestCase):
    """
    A tree testing class for infinite grammars, testing only the high probability ones
    """
    def make_h0(self, **kwargs):
         return LOTHypothesis(self.grammar, **kwargs)

    def setUp(self, max_depth=5): ## or 4, depending
        from InfiniteGrammar import grammar
        self.grammar = grammar
        self.max_depth = max_depth
        self.trees = [t for t in grammar.enumerate(self.max_depth)]

        # for t in self.trees:
        #     print t

    def check_tree(self, t):
        # correct overall return type
        self.assertTrue(t.returntype == self.grammar.start)

        # correct argument types for each subnode
        for ti in t:
            if ti.args is None:
                self.assertTrue( ti.rule.to is None)
            else:
                for ri, ai in zip(ti.rule.to, ti.args):
                    if isinstance(ai, FunctionNode):
                        self.assertTrue(ai.returntype == ri)
                        # and check parent refs
                        self.assertTrue(ai.parent == ti)
                    else:
                        self.assertTrue(ai == ri)

        # Check that the bv function nodes are of the right type
        # And that we added and removed rules appropriately
        added_rules = [] # just see what we added
        for ti in t.iterate_subnodes(self.grammar):
            if re.match(r'bv_', ti.name):
                self.assertTrue(isinstance(ti, BVUseFunctionNode))

                # NOTE: We cannot use "in" here since that uses rule "is", but we've created
                # a new thing that is equivalent to the rule. So instead, we check the bv name
                self.assertTrue(ti.rule.name in [r.name for r in self.grammar.rules[ti.returntype]])
                added_rules.append(ti.rule)

            if re.match(r'lambda', ti.name):
                self.assertTrue(isinstance(ti, BVAddFunctionNode))

                # assert that this rule isn't already there
                self.assertTrue(ti.added_rule.name not in [r.name for r in self.grammar.rules[ti.returntype]])

        # Then assert that none of the rules are still in the grammar
        for therule in added_rules:
            self.assertTrue(therule.name not in [r.name for r in self.grammar.rules[ti.returntype]])

    def evaluate_sampler(self, sampler):

        cnt = Counter()
        for h in lot_iter(sampler):
            cnt[h.value] += 1

        ## TODO: When the MCMC methods get cleaned up for how many samples they return, we will assert that we got the right number here
        # assert sum(cnt.values()) == NSAMPLES # Just make sure we aren't using a sampler that returns fewer samples! I'm looking at you, ParallelTempering

        Z = logsumexp([t.log_probability() for t in self.trees]) # renormalize to the trees in self.trees
        obsc = [cnt[t] for t in self.trees]
        expc = [exp(t.log_probability()-Z)*sum(obsc) for t in self.trees]
        csq, pv = chisquare(obsc, expc)
        assert abs(sum(obsc) - sum(expc)) < 0.01

        # assert min(expc) > 5 # or else chisq sux

        for t, c, s in zip(self.trees, obsc, expc):
            print c, s, t
        print (csq, pv), sum(obsc)

        self.assertGreater(pv, PVALUE, msg="Sampler failed chi squared!")

        return csq, pv

    def plot_sampler(self, opath, sampler):
        """
        Plot the sampler, for cases with many zeros where chisquared won't work well
        """
        cnt = Counter()
        for h in lot_iter(sampler):
            cnt[h.value] += 1

        Z = logsumexp([t.log_probability() for t in self.trees]) # renormalize to the trees in self.trees
        obsc = [cnt[t] for t in self.trees]
        expc = [exp(t.log_probability()-Z)*sum(obsc) for t in self.trees]

        for t, c, s in zip(self.trees, obsc, expc):
            print c, "\t", s, "\t", t


        expc, obsc, trees = zip(*sorted(zip(expc, obsc, self.trees), reverse=True))

        import matplotlib.pyplot as plt
        from numpy import log
        plt.subplot(111)
        # Log here spaces things out at the high end, where we can see it!
        plt.scatter(log(range(len(trees))), expc, color="red", alpha=1.)
        plt.scatter(log(range(len(trees))), obsc, color="blue", marker="x", alpha=1.)
        plt.savefig(opath)
        plt.clf()





class FiniteTreeTester(InfiniteTreeTester):
    """
    When the grammar is finite, we can test a little more
    """
    # initialization that happens before each test is carried out
    def setUp(self):
        from FiniteGrammar import grammar

        self.grammar = grammar
        self.trees = [t for t in grammar.enumerate()]

        # for t in self.trees:
        #     print t

    def check_tree(self, t):
        """
        A bunch of checking functions for individual trees. This uses self.grammar
        """

        # call the superclass test
        InfiniteTreeTester.check_tree(self, t)

        # and if its finite, it must be in our list
        # assert that its a valid tree
        self.assertTrue(t in self.trees)
        ee = [v for v in self.trees if v==t]
        self.assertTrue(len(ee) == 1) # only one thing can be equal -- no multiple derivations are possible in our grammar

        # and they have the same log probability
        self.assertAlmostEquals(t.log_probability(), ee[0].log_probability())