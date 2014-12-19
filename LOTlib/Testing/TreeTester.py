import unittest
import re
from LOTlib.FunctionNode import FunctionNode, BVUseFunctionNode, BVAddFunctionNode


from TestGrammar import grammar


class TreeTester(unittest.TestCase):
    """
    A superclass where we have defined a check_tree function that can verify the goodness of trees
    """
    # initialization that happens before each test is carried out
    def setUp(self):
        self.grammar = grammar
        self.trees = [t for t in grammar.enumerate()]


    def check_tree(self, t):
        """
        A bunch of checking functions for individual trees. This uses self.grammar
        """

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
        for ti in self.grammar.iterate_subnodes(t):
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

        # assert that its a valid tree
        self.assertTrue(t in self.trees)
        ee = [v for v in self.trees if v==t]
        self.assertTrue(len(ee) == 1) # only one thing can be equal -- no multiple derivations are possible in our grammar

        # and they have the same log probability
        self.assertAlmostEquals(t.log_probability(), ee[0].log_probability())


